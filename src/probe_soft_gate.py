"""Isolated full-input GPU check of the paper Soft loss; never saves a trained adapter."""
import gc
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import kl_experiment as legacy
import logit_kd
import pilot
import soft_gate
import train_student as trainer

OUTPUT = pilot.ROOT / "outputs/gate_comparison_v4/soft_gpu_probe.json"


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {**legacy.code_hashes(), **{name: legacy.sha(folder/name)
        for name in ("soft_gate.py", "probe_soft_gate.py")}}


def probe():
    import torch
    legacy.require_protocol()
    if OUTPUT.exists():
        saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if saved.get("status") != "passed" or saved["code_sha256"] != source_hashes():
            raise RuntimeError("Existing probe is preserved and needs inspection")
        print(json.dumps(saved,ensure_ascii=False),flush=True)
        return
    report = {"status":"running", "started_at_kst":trainer.now(),
        "code_sha256":source_hashes(), "margin":soft_gate.MARGIN,
        "soft_gate_detached":False, "separate_gate_temperature":None,
        "probe_model_reused_for_training":False, "adapter_saved":False,
        "test_evaluation":False, "optimizer_steps":0,
        "versions":{p:importlib.metadata.version(p) for p in ("torch","transformers","peft","safetensors")}}
    pilot.write_json(OUTPUT,report)
    model = None
    try:
        ctx = legacy.context("full","adaptive_mix",42)
        report["soft_targets_sha256"] = ctx["soft_targets_sha256"]
        model,tokenizer,parameters,optimizer = trainer.load_student(42)
        examples = [trainer.encode_example(row,"adaptive_mix",tokenizer) for row in ctx["rows"]]
        assert ctx["target_ids"] == [e["source_id"] for e in examples]
        assert ctx["prefix_sha256"] == [logit_kd.prefix_hash(e) for e in examples]
        index = max(range(len(examples)),key=lambda i:len(examples[i]["input_ids"]))
        example = examples[index]
        teacher = ctx["targets"][index]
        assert not teacher.requires_grad
        before = trainer.adapter_hash(model)
        report.update({"gpu":torch.cuda.get_device_name(0),
            "gpu_total_bytes":torch.cuda.get_device_properties(0).total_memory,
            "source_id":example["source_id"], "sequence_tokens":len(example["input_ids"]),
            "input_prefixes_verified":len(examples),
            "trainable_parameters":sum(p.numel() for p in parameters),
            "base_model_quantized":bool(getattr(model,"is_loaded_in_4bit",False))})
        assert not report["base_model_quantized"]
        ids = torch.tensor([example["input_ids"]],dtype=torch.long,device="cuda")
        positions = torch.tensor(example["prediction_positions"],device="cuda")
        targets = torch.tensor(example["target_ids"],device="cuda")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = perf_counter()
        accumulation = ctx["config"]["gradient_accumulation"]
        for step in range(accumulation):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits = model(input_ids=ids,attention_mask=torch.ones_like(ids),
                    logits_to_keep=positions,use_cache=False).logits[0]
            signal = {}
            loss = soft_gate.distillation_loss(logits,targets,teacher,signal)
            (loss/accumulation).backward()
            print(f"Soft GPU probe: accumulation {step+1}/{accumulation}, loss={loss.item():.6f}",flush=True)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters,ctx["config"]["max_grad_norm"],error_if_nonfinite=True)
        assert norm.item() > 0
        optimizer.step()
        torch.cuda.synchronize()
        assert trainer.adapter_hash(model) != before
        assert model.get_input_embeddings().weight.grad is None and not teacher.requires_grad
        report.update({"status":"passed", "optimizer_steps":1,
            "loss":loss.item(), "gradient_norm":norm.item(), "signals":signal,
            "optimizer_check_seconds":perf_counter()-start,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "base_embedding_frozen":True, "teacher_frozen":True,
            "lora_updated":True})
    except Exception as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at_kst"] = trainer.now()
        pilot.write_json(OUTPUT,report)
        del model
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    probe()
