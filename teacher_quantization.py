"""Qwen3.5-9B 원본의 text-only NF4 로딩/추론 확인. 학습이나 벤치마크는 수행하지 않는다."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CACHE = ROOT / ".cache" / "huggingface" / "hub"
SNAPSHOT = CACHE / "models--Qwen--Qwen3.5-9B" / "snapshots" / REVISION
RESULT = ROOT / "outputs" / "teacher_nf4_check.json"


def check_source():
    raw = (SNAPSHOT / "config.json").read_bytes()
    config = json.loads(raw)
    text = config["text_config"]
    assert text["model_type"] == "qwen3_5_text"
    assert text["hidden_size"] == 4096 and text["num_hidden_layers"] == 32
    assert text["layer_types"] == (["linear_attention"] * 3 + ["full_attention"]) * 8
    assert len(text["layer_types"]) == 32
    assert text["vocab_size"] == 248320
    assert len(list(SNAPSHOT.glob("model*.safetensors"))) == 4
    return config, hashlib.sha256(raw).hexdigest()


def load_text_model(model_id, revision, quantization_config=None):
    """Shared native text-only loader for the actual Teacher and Student."""
    import torch
    from transformers import Qwen3_5ForCausalLM
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        model_id, revision=revision, cache_dir=str(CACHE),
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, quantization_config=quantization_config,
        device_map={"": 0}, attn_implementation="sdpa", output_loading_info=True,
    )
    assert not any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert not hasattr(model, "peft_config") and not hasattr(model.model, "visual")
    return model, loading


def run_check(report):
    os.environ.setdefault("HF_HOME", str(CACHE.parent))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import bitsandbytes as bnb
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16을 지원하는 CUDA GPU가 필요합니다.")
    report["versions"] = {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "accelerate", "bitsandbytes")
    }
    report["gpu"] = torch.cuda.get_device_name(0)
    report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.reset_peak_memory_stats()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    report["quantization_config"] = quantization.to_dict()
    load_start = perf_counter()
    print("원본 checkpoint를 text-only NF4로 로딩합니다.", flush=True)
    model, loading = load_text_model(MODEL_ID, REVISION, quantization)
    report["loading_info"] = {key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()}
    assert not loading.get("missing_keys"), "누락된 language model weight"
    assert not loading.get("mismatched_keys"), "shape이 다른 weight"
    assert not loading.get("error_msgs"), "checkpoint 로딩 오류"
    assert not loading.get("unexpected_keys"), "처리되지 않은 checkpoint weight"
    assert model.is_loaded_in_4bit
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert not hasattr(model, "peft_config"), "기존 adapter가 포함됨"
    assert not hasattr(model.model, "visual"), "vision encoder가 포함됨"
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    layers = [module for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)]
    assert layers and all(layer.weight.quant_state is not None for layer in layers)
    assert all(layer.weight.quant_state.quant_type == "nf4" for layer in layers)
    assert all(layer.weight.quant_state.nested for layer in layers)
    assert all(layer.compute_dtype == torch.bfloat16 for layer in layers)
    torch.cuda.synchronize()
    report["load_seconds"] = round(perf_counter() - load_start, 3)
    report["quantized_linear_layers"] = len(layers)
    report["model_memory_footprint_bytes"] = model.get_memory_footprint()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, cache_dir=str(CACHE),
        local_files_only=True, trust_remote_code=False,
    )
    prompt = "한국의 수도는 어디인가요? 한 단어로 답하세요."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    report["probe"] = {"prompt": prompt, "thinking": False, "input_tokens": inputs.input_ids.shape[1]}
    inference_start = perf_counter()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits
        assert logits.shape == (1, 1, 248320) and torch.isfinite(logits).all().item()
        generated = model.generate(
            **inputs, max_new_tokens=8, do_sample=False, use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    report["probe"].update({
        "finite_logits": True,
        "generated_tokens": generated.shape[1] - inputs.input_ids.shape[1],
        "response": tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True),
        "seconds": round(perf_counter() - inference_start, 3),
    })
    report["memory"] = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    report["status"] = "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="GPU 로딩 없이 cached checkpoint 구성을 검사")
    args = parser.parse_args()
    config, config_hash = check_source()
    if args.self_check:
        print("SELF-CHECK PASS: pinned checkpoint, 32-layer hybrid layout, vocabulary, four weight shards")
    else:
        report = {
            "status": "failed", "started_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "model_id": MODEL_ID, "revision": REVISION, "scope": "text_only",
            "adapter": None, "training": False, "benchmark": False,
            "source_config_sha256": config_hash, "source_config": config,
            "checkpoint_weights_bytes": sum(path.stat().st_size for path in SNAPSHOT.glob("model*.safetensors")),
        }
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_check(report)
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            report["finished_at_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
            RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"검사 상태: {report['status']} | 결과: {RESULT}", flush=True)
