"""Task-label KD preparation and disposable GPU checks; never evaluates test or saves adapters."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as F
import kl_experiment as legacy
import logit_kd
import pilot
import soft_gate
import train_student
from teacher_quantization import ROOT, CACHE, MODEL_ID, REVISION, load_text_model

SETUP = ROOT / "results/protocols/choice_kd_setup.json"
OUT = ROOT / "outputs/choice_kd_v1"
METHODS = ("fixed_mix", "adaptive_hard", "adaptive_soft")
TEMPERATURE = logit_kd.TEMPERATURE


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return legacy.sha(path)


def models():
    return read(SETUP)["selected_models"]


def recipe_sha():
    payload = {"models": models(), "temperature": TEMPERATURE, "margin": soft_gate.MARGIN}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def tokenizer_for(spec):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"],
        cache_dir=str(CACHE), local_files_only=True, trust_remote_code=False)


def encode_question(row, tokenizer, spec):
    labels = [c["label"] for c in row["choices"]]
    if not 3 <= len(labels) <= 5 or labels != list("ABCDE"[:len(labels)]):
        raise ValueError("Expected original-order semantic labels A..E")
    messages = [{"role": "user", "content": pilot.prompt(row)}]
    if spec.get("system_message"):
        messages.insert(0, {"role": "system", "content": spec["system_message"]})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, **spec["chat_template_kwargs"])
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    choice_ids = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != label:
            raise ValueError(f"Unsupported answer token: {spec['id']} / {label}")
        if tokenizer.encode(rendered + label, add_special_tokens=False) != prefix + ids:
            raise ValueError(f"Answer crosses tokenizer prefix boundary: {row['source_id']}")
        choice_ids.append(ids[0])
    if not prefix or len(prefix) > train_student.CONFIG["max_sequence_length"]:
        raise ValueError("Input exceeds frozen budget; do not truncate")
    if len(set(choice_ids)) != len(labels):
        raise ValueError("Answer labels must have distinct token IDs")
    return {"source_id": row["source_id"], "input_ids": prefix, "choice_ids": choice_ids,
            "labels": labels, "prompt_tokens": len(prefix),
            "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}


def choice_log_probs(scores, temperature=TEMPERATURE):
    if scores.ndim != 1 or not 3 <= scores.numel() <= 5:
        raise ValueError("Choice scores must cover exactly the valid 3..5 options")
    if not math.isfinite(temperature) or temperature <= 0 or not torch.isfinite(scores).all():
        raise ValueError("Finite scores and positive temperature required")
    return F.log_softmax(scores.float() / temperature, dim=-1)


def distillation_loss(choice_logits, teacher_log_probs, method, metrics=None):
    if method not in METHODS:
        raise ValueError("Unknown task-label KD method")
    if teacher_log_probs.requires_grad or teacher_log_probs.shape != choice_logits.shape:
        raise ValueError("Teacher must be frozen in the same semantic label space")
    if not torch.isfinite(teacher_log_probs).all():
        raise ValueError("Nonfinite Teacher targets")
    if not torch.allclose(teacher_log_probs.logsumexp(-1),
                          torch.zeros((), device=teacher_log_probs.device), atol=1e-5):
        raise ValueError("Teacher targets must be normalized over valid choices")
    log_q = choice_log_probs(choice_logits)
    log_p = teacher_log_probs.to(device=log_q.device, dtype=torch.float32)
    forward = F.kl_div(log_q, log_p, reduction="sum", log_target=True)
    reverse = (log_q.exp() * (log_q - log_p)).sum()
    cp, cq, hard = logit_kd.confidence_gate(log_p, log_q)
    weight = (torch.full_like(hard, .5) if method == "fixed_mix" else hard
              if method == "adaptive_hard" else torch.sigmoid(cp - cq - soft_gate.MARGIN))
    loss = TEMPERATURE**2 * (weight * forward + (1 - weight) * reverse)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite KD loss")
    if metrics is not None:
        metrics.update(dict(zip(("forward_kl", "reverse_kl", "teacher_confidence",
                                 "student_confidence", "hard_gate", "forward_weight"),
            torch.stack((forward, reverse, cp, cq, hard, weight)).detach().cpu().tolist())))
    return loss


def require_preservation():
    frozen = read(ROOT / "results/protocols/gate_test_manifest.json")
    original = read(ROOT / "results/protocols/gate_training_protocol.json")
    for name, expected in frozen["code_sha256"].items():
        assert file_sha(ROOT / "src" / name) == expected, name
    for name, expected in original["protected_artifacts_sha256"].items():
        assert file_sha(ROOT / name) == expected, name
    return {"frozen_sources": len(frozen["code_sha256"]),
            "protected_artifacts": len(original["protected_artifacts_sha256"]), "all_match": True}


def prepare():
    from safetensors.torch import load_file, save_file
    target = OUT / "teacher_choice_scores_T2.safetensors"
    summary_path = OUT / "preparation.json"
    if target.exists() or summary_path.exists():
        raise FileExistsError("Existing preparation preserved; inspect rather than overwrite")
    preservation = require_preservation()
    rows, path = legacy.rows_for("full")
    dev, dev_path = legacy.rows_for("full", "validation")
    teacher_dir = legacy.OUT / "targets/full"
    source = read(teacher_dir / "summary.json")
    assert source["status"] == "passed" and source["source_split"] == "train"
    assert source["model_id"] == MODEL_ID and source["revision"] == REVISION
    assert source["temperature"] == TEMPERATURE and source["data_sha256"] == file_sha(path)
    assert source["sample_ids"] == [r["source_id"] for r in rows]
    teacher_spec = {"id": MODEL_ID, "revision": REVISION,
                    "chat_template_kwargs": {"enable_thinking": False}}
    teacher_tok = tokenizer_for(teacher_spec)
    teacher_examples = [encode_question(r, teacher_tok, teacher_spec) for r in rows]
    assert source["prefix_sha256"] == [e["prefix_sha256"] for e in teacher_examples]
    label_ids = [teacher_tok.encode(a, add_special_tokens=False)[0] for a in "ABCDE"]
    source_file = teacher_dir / "teacher_log_probs.safetensors"
    assert file_sha(source_file) == source["output_sha256"]
    full = load_file(str(source_file), device="cpu")["answer_log_probs"]
    assert full.shape == (1118, source["vocabulary_size"]) and torch.isfinite(full).all()
    assert torch.allclose(full.logsumexp(-1), torch.zeros(1118), atol=1e-5)
    scores = full[:, label_ids].contiguous()
    # Cached values already have T=2; renormalize the valid labels without dividing by T again.
    projected = [choice_log_probs(scores[i, :len(r["choices"])], temperature=1)
                 for i, r in enumerate(rows)]
    assert all(torch.isfinite(p).all() and abs(p.logsumexp(-1).item()) < 1e-5 for p in projected)
    token_checks = {}
    for spec in models():
        tok = tokenizer_for(spec)
        examples = [encode_question(r, tok, spec) for r in rows + dev]
        checks = {"status": "passed", "train_n": len(rows), "validation_n": len(dev),
                  "test_read": False, "max_prompt_tokens": max(e["prompt_tokens"] for e in examples),
                  "label_token_ids": {a: tok.encode(a, add_special_tokens=False)[0] for a in "ABCDE"},
                  "prefix_sha256": [e["prefix_sha256"] for e in examples],
                  "prefix_set_sha256": hashlib.sha256(json.dumps([e["prefix_sha256"] for e in examples]).encode()).hexdigest(),
                  "chat_template_kwargs": spec["chat_template_kwargs"],
                  "fixed_system_message": spec.get("system_message"),
                  "tokenizer_file_sha256": spec["tokenizer_file_sha256"]}
        token_checks[spec["key"]] = checks
        print(f"Tokenizer passed: {spec['key']} / {len(examples)} inputs", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    save_file({"answer_log_scores_T2": scores}, str(target))
    summary = {"status": "passed", "source_ids": source["sample_ids"], "rows": len(rows),
        "train_sha256": file_sha(path), "validation_sha256": file_sha(dev_path),
        "teacher_source_sha256": source["output_sha256"], "teacher_source_temperature": TEMPERATURE,
        "teacher_labels": list("ABCDE"), "teacher_token_ids": label_ids,
        "projected_shape": list(scores.shape), "projected_bytes": target.stat().st_size,
        "projected_sha256": file_sha(target), "recipe_sha256": recipe_sha(),
        "code_sha256": file_sha(Path(__file__)), "tokenizers": token_checks,
        "preservation": preservation, "test_read": False, "training_started": False}
    pilot.write_json(summary_path, summary)
    public = {**summary, "tokenizers": {key: {k: v for k, v in values.items() if k != "prefix_sha256"}
                                      for key, values in token_checks.items()}}
    public.pop("source_ids")
    pilot.write_json(ROOT / "results/checks/choice_kd_preparation.json", public)
    print("Choice targets ready; no test or model training performed", flush=True)


def require_prepared():
    from safetensors.torch import load_file
    summary = read(OUT / "preparation.json")
    assert summary["status"] == "passed" and summary["code_sha256"] == file_sha(Path(__file__))
    assert summary["recipe_sha256"] == recipe_sha()
    rows, path = legacy.rows_for("full")
    assert file_sha(path) == summary["train_sha256"]
    assert [r["source_id"] for r in rows] == summary["source_ids"]
    target = OUT / "teacher_choice_scores_T2.safetensors"
    assert file_sha(target) == summary["projected_sha256"]
    scores = load_file(str(target), device="cpu")["answer_log_scores_T2"]
    assert scores.shape == (len(rows), 5) and torch.isfinite(scores).all()
    return rows, scores, summary


def probe(key):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    require_preservation()
    rows, scores, preparation = require_prepared()
    spec = next(m for m in models() if m["key"] == key)
    tokenizer = tokenizer_for(spec)
    examples = [encode_question(row, tokenizer, spec) for row in rows]
    assert [e["prefix_sha256"] for e in examples] == preparation["tokenizers"][key]["prefix_sha256"][:len(rows)]
    report = {"status": "running", "student": key, "model_id": spec["id"],
              "revision": spec["revision"], "precision": "BF16 base / FP32 LoRA",
              "source_split": "train", "test_read": False, "adapter_saved": False,
              "probe_reused_for_training": False, "code_sha256": file_sha(Path(__file__)),
              "gpu": torch.cuda.get_device_name(0), "recipe_sha256": recipe_sha()}
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    report_path = ROOT / "results/checks" / f"choice_gpu_{key}_{stamp}.json"
    try:
        if key == "qwen2b":
            base, loading = load_text_model(spec["id"], spec["revision"])
        else:
            base, loading = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"],
                cache_dir=str(CACHE), local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=torch.bfloat16, device_map={"": 0},
                attn_implementation="sdpa", output_loading_info=True)
        assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        assert all(p.device.type == "cuda" and p.dtype == torch.bfloat16 for p in base.parameters())
        assert sum(p.numel() for p in base.parameters()) == spec["text_model_parameters"]
        base.config.use_cache = False
        set_seed(42)
        cfg = train_student.CONFIG
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"], bias="none",
            base_model_name_or_path=spec["id"], revision=spec["revision"]))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        assert params and all(p.dtype == torch.float32 for p in params)
        assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                                      betas=tuple(cfg["betas"]), eps=cfg["eps"])
        indices = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]), reverse=True)[:8]
        report.update(base_parameters=spec["text_model_parameters"],
            trainable_parameters=sum(p.numel() for p in params),
            probe_input_ids=[rows[i]["source_id"] for i in indices],
            longest_train_tokens=len(examples[indices[0]]["input_ids"]), methods={})
        torch.cuda.reset_peak_memory_stats()
        for method in METHODS:
            initial = train_student.adapter_hash(model)
            optimizer.zero_grad(set_to_none=True)
            start = perf_counter()
            observations = []
            for index in indices:
                example = examples[index]
                ids = torch.tensor([example["input_ids"]], dtype=torch.long, device="cuda")
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               logits_to_keep=1, use_cache=False).logits[0, -1].float()
                teacher = choice_log_probs(scores[index, :len(example["labels"])], temperature=1).to("cuda")
                metrics = {}
                loss = distillation_loss(logits[example["choice_ids"]], teacher, method, metrics)
                (loss / len(indices)).backward()
                observations.append({"loss": loss.item(), **metrics})
            norm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            assert train_student.adapter_hash(model) != initial
            assert model.get_input_embeddings().weight.grad is None
            report["methods"][method] = {"status": "passed", "optimizer_steps": 1,
                "micro_batches": len(indices), "grad_norm": float(norm),
                "seconds": perf_counter() - start, "observations": observations}
            print(f"GPU check passed: {key} / {method}", flush=True)
        report.update(status="passed", peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      note="Disposable sequential checks share one probe adapter; not independent training experiments")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        pilot.write_json(report_path, report)
        print(f"GPU check report: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "probe"))
    parser.add_argument("--student", choices=("qwen2b", "smollm2", "granite2b"))
    args = parser.parse_args()
    if args.command == "prepare":
        if args.student:
            parser.error("--student is only for probe")
        prepare()
    else:
        if not args.student:
            parser.error("probe requires --student")
        probe(args.student)
