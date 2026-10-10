
"""BF16 LoRA Student: isolated memory probe, independent Gold SFT/Response KD pilots."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import pilot
from teacher_quantization import ROOT, CACHE, load_text_model

TRAINING = ROOT / "outputs" / "training"
CONFIG = {
    "model_id": pilot.STUDENT_ID, "model_revision": pilot.STUDENT_REVISION,
    "precision": "BF16 base, FP32 LoRA parameters", "quantization": None,
    "seed": 42, "epochs": 3, "micro_batch_size": 1, "gradient_accumulation": 8,
    "learning_rate": 0.0002, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
    "scheduler": "linear", "warmup_steps": 10, "max_sequence_length": 1024,
    "gradient_checkpointing": True, "use_reentrant": False, "use_cache": False,
    "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
    "target_modules": "all-linear", "bias": "none",
    "loss": "mean cross-entropy on assistant answer-letter and EOS only",
    "checkpoint_selection": "fixed final epoch; no validation-based selection",
}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()


def run_config(condition, seed=42):
    config = {**CONFIG, "seed": seed}
    if condition == "logit_kd":
        import logit_kd
        config.update({"loss":logit_kd.LOSS,"distillation":{"temperature":logit_kd.TEMPERATURE,
                      "scope":"full vocabulary at answer position; shared hard EOS","gold_in_loss":False}})
    return config, hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def load_training_rows():
    manifest = json.loads((pilot.OUTPUT / "data_manifest.json").read_text(encoding="utf-8"))
    teacher = json.loads((pilot.OUTPUT / "teacher_responses_summary.json").read_text(encoding="utf-8"))
    train_path = pilot.DATA / "train.jsonl"
    response_path = pilot.OUTPUT / "teacher_responses.jsonl"
    assert teacher["status"] == "passed" and teacher["train_targets_ready"]
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest["export_sha256"]["train.jsonl"]
    assert hashlib.sha256(response_path.read_bytes()).hexdigest() == teacher["output_sha256"]
    rows, responses = pilot.read_rows(train_path), pilot.read_rows(response_path)
    assert [r["source_id"] for r in rows] == teacher["sample_ids"] == manifest["pilot_train_ids"]
    assert len(rows) == len(responses) == 256 and len(rows) % CONFIG["gradient_accumulation"] == 0
    for row, response in zip(rows, responses):
        assert all(response[key] == value for key, value in row.items())
        assert row["source_split"] == "train"
        target = pilot.parse_answer(response["response"], row)
        assert target is not None and target == response["teacher_target"] == response["predicted"]
        assert not response["hit_token_limit"]
        row["teacher_target"] = target
    assert sum(r["gold"] != r["teacher_target"] for r in rows) == teacher["gold_disagreements"]
    return rows, teacher


def encode_example(row, condition, tokenizer):
    if condition not in {"gold_sft", "response_kd", "logit_kd", "reverse_kl", "fixed_mix", "adaptive_mix"}:
        raise ValueError("Unknown training condition")
    target = row["gold"] if condition == "gold_sft" else row["teacher_target"]
    assert target in {c["label"] for c in row["choices"]}
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": pilot.prompt(row)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    answer = tokenizer.encode(target, add_special_tokens=False)
    assert len(answer) == 1 and answer[0] == ord(target) - ord("A") + 32
    target_ids = answer + [tokenizer.eos_token_id]
    input_ids = prefix + target_ids
    assert len(input_ids) <= CONFIG["max_sequence_length"], "Do not silently truncate"
    return {"source_id": row["source_id"], "input_ids": input_ids,
            "prediction_positions": list(range(len(prefix) - 1, len(input_ids) - 1)),
            "target_ids": target_ids, "prompt_tokens": len(prefix)}


def adapter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_student(seed=42):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, set_seed
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(pilot.STUDENT_ID, revision=pilot.STUDENT_REVISION,
        cache_dir=str(CACHE), local_files_only=True, trust_remote_code=False)
    base, _ = load_text_model(pilot.STUDENT_ID, pilot.STUDENT_REVISION)
    base.config.use_cache = False
    set_seed(seed)
    lora = LoraConfig(task_type="CAUSAL_LM", r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"], bias="none",
        base_model_name_or_path=pilot.STUDENT_ID, revision=pilot.STUDENT_REVISION)
    model = get_peft_model(base, lora, revision=pilot.STUDENT_REVISION)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters and all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    assert all(p.dtype == torch.float32 for p in parameters)
    assert not getattr(base, "is_loaded_in_4bit", False)
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"], betas=tuple(CONFIG["betas"]), eps=CONFIG["eps"])
    return model, tokenizer, parameters, optimizer


def answer_loss(model, example, teacher_log_probs=None, variant="forward", metrics=None):
    import torch
    import torch.nn.functional as F
    ids = torch.tensor([example["input_ids"]], dtype=torch.long, device="cuda")
    positions = torch.tensor(example["prediction_positions"], dtype=torch.long, device="cuda")
    targets = torch.tensor(example["target_ids"], dtype=torch.long, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # ponytail: batch size 1 and answer+EOS loss; use padding/masks for larger batches or longer targets.
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
            logits_to_keep=positions, use_cache=False).logits[0]
    assert logits.shape[0] == targets.numel() == 2
    if teacher_log_probs is None:
        loss = F.cross_entropy(logits.float(), targets)
    else:
        import logit_kd
        loss = logit_kd.distillation_loss(logits.float(), targets, teacher_log_probs, variant, metrics)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite training loss")
    return loss


def memory_probe():
    import torch
    path = TRAINING / "memory_probe.json"
    if path.exists():
        raise FileExistsError(f"Existing probe preserved: {path}")
    rows, _ = load_training_rows()
    report = {"status": "running", "purpose": "isolated probe; no saved adapter or reused updated model",
              "config": CONFIG, "config_sha256": CONFIG_HASH, "started_at_kst": now()}
    pilot.write_json(path, report)
    try:
        model, tokenizer, parameters, optimizer = load_student()
        examples = [encode_example(r, "gold_sft", tokenizer) for r in rows]
        longest = max(examples, key=lambda x: len(x["input_ids"]))
        initial = adapter_hash(model)
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        for _ in range(CONFIG["gradient_accumulation"]):
            loss = answer_loss(model, longest)
            (loss / CONFIG["gradient_accumulation"]).backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        assert adapter_hash(model) != initial, "Probe optimizer did not update LoRA"
        assert model.get_input_embeddings().weight.grad is None
        report.update({"status": "passed", "sample_id": longest["source_id"],
            "sequence_tokens": len(longest["input_ids"]), "max_sequence_tokens_train": len(longest["input_ids"]),
            "loss": loss.item(), "grad_norm": float(norm), "seconds": round(perf_counter()-start, 3),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "base_embedding_frozen": True, "probe_model_reused_for_training": False})
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        report["finished_at_kst"] = now()
        pilot.write_json(path, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


def train(condition, seed=42, *, output_dir=None, monitor=None, experiment=None):
    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed
    config, config_hash = run_config(condition, seed) if experiment is None else (
        experiment["config"], hashlib.sha256(json.dumps(experiment["config"], sort_keys=True).encode()).hexdigest())
    run = pilot.training_run_path(condition, seed) if output_dir is None else Path(output_dir)
    if run.exists():
        raise FileExistsError(f"Existing training run preserved: {run}")
    probe_path = ROOT/"outputs/logit_kd/memory_probe.json" if condition == "logit_kd" else TRAINING/"memory_probe.json"
    if experiment is not None:
        probe_path = experiment["probe_path"]
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    assert probe["status"] == "passed"
    assert {k: v for k, v in probe["config"].items() if k != "seed"} == {k: v for k, v in config.items() if k != "seed"}
    rows, teacher = load_training_rows() if experiment is None else (experiment["rows"], experiment["teacher"])
    steps = math.ceil(len(rows) / config["gradient_accumulation"]) * config["epochs"]
    metadata = {"status": "running", "condition": condition, "config": config, "config_sha256": config_hash,
        "model_id": pilot.STUDENT_ID, "model_revision": pilot.STUDENT_REVISION,
        "source_split": "train", "sample_ids": [r["source_id"] for r in rows], "train_rows": len(rows),
        "train_sha256": teacher["data_sha256"], "teacher_responses_sha256": teacher["output_sha256"],
        "target_source": "dataset gold" if condition == "gold_sft" else (
            "Teacher full-vocabulary answer distribution + shared hard EOS" if condition == "logit_kd" else "original Teacher answer letter"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_checkpoint_start": True, "previous_adapter": None, "validation_used_for_training": False,
        "test_evaluation": False, "optimizer_steps_planned": steps, "started_at_kst": now(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}}
    if condition == "logit_kd" and experiment is None:
        import logit_kd
        metadata.update({"distillation_script_sha256":hashlib.sha256(Path(logit_kd.__file__).read_bytes()).hexdigest(),
            "distillation_protocol_sha256":hashlib.sha256((logit_kd.OUTPUT/"protocol.json").read_bytes()).hexdigest()})
    if experiment is not None:
        metadata.update({"experiment_scope": experiment["scope"], "protocol_sha256": experiment["protocol_sha256"],
                         "target_source": "dataset gold" if condition == "gold_sft" else (
                             "original Teacher answer letter" if condition == "response_kd" else
                             "Teacher full-vocabulary answer distribution + shared hard EOS"),
                         "soft_targets_sha256": experiment.get("soft_targets_sha256")})
    pilot.write_json(run / "summary.json", metadata)
    try:
        start = perf_counter()
        model, tokenizer, parameters, optimizer = load_student(seed)
        metadata["load_seconds"] = round(perf_counter()-start, 3)
        metadata["initial_adapter_sha256"] = adapter_hash(model)
        metadata["trainable_parameters"] = sum(p.numel() for p in parameters)
        examples = [encode_example(r, condition, tokenizer) for r in rows]
        soft_targets = None if experiment is None else experiment["targets"]
        if condition == "logit_kd" and experiment is None:
            import logit_kd
            soft_targets, target_summary = logit_kd.load_targets(examples)
            assert target_summary["train_sha256"] == metadata["train_sha256"]
            assert target_summary["teacher_responses_sha256"] == metadata["teacher_responses_sha256"]
            assert target_summary["output_sha256"] == probe["soft_targets_sha256"]
            metadata["soft_targets_sha256"] = target_summary["output_sha256"]
            metadata["soft_positions_with_repeats"] = len(rows)*config["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, config["warmup_steps"], steps)
        set_seed(config["seed"])
        if monitor is not None:
            monitor(model, tokenizer, 0, 0, run)
        monitor_seconds = 0.0
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        global_step = total_input = total_target = 0
        epoch_losses, order_hashes = [], []
        optimizer.zero_grad(set_to_none=True)
        with (run / "train_log.jsonl").open("x", encoding="utf-8") as stream:
            for epoch in range(config["epochs"]):
                order = random.Random(config["seed"]+epoch).sample(range(len(examples)), len(examples))
                order_hashes.append(hashlib.sha256(json.dumps([examples[i]["source_id"] for i in order]).encode()).hexdigest())
                epoch_loss = batch_loss = 0.0
                batch_ids = []
                for step, index in enumerate(order, 1):
                    example = examples[index]
                    signal = {} if experiment is not None and soft_targets is not None else None
                    variant = condition if condition in {"reverse_kl", "fixed_mix", "adaptive_mix"} else "forward"
                    loss = answer_loss(model, example, None if soft_targets is None else soft_targets[index], variant, signal)
                    value = loss.item()
                    actual_batch = min(config["gradient_accumulation"],
                                       len(examples) - ((step-1)//config["gradient_accumulation"])*config["gradient_accumulation"])
                    (loss / actual_batch).backward()
                    if signal is not None:
                        signal.update({"source_id": example["source_id"], "epoch": epoch+1,
                                       "optimizer_step": global_step+1, "loss": value})
                        with (run/"signal_log.jsonl").open("a",encoding="utf-8") as signals:
                            signals.write(json.dumps(signal)+"\n")
                    epoch_loss += value
                    batch_loss += value
                    batch_ids.append(example["source_id"])
                    total_input += len(example["input_ids"])
                    total_target += len(example["target_ids"])
                    if step % config["gradient_accumulation"] == 0 or step == len(examples):
                        norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
                        lr = optimizer.param_groups[0]["lr"]
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        record = {"epoch": epoch+1, "optimizer_step": global_step,
                            "mean_loss": batch_loss/actual_batch, "grad_norm": float(norm),
                            "learning_rate": lr, "sample_ids": batch_ids, "seconds": round(perf_counter()-start-monitor_seconds, 3)}
                        stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                        stream.flush()
                        batch_loss, batch_ids = 0.0, []
                        if global_step % 8 == 0:
                            metadata.update({"optimizer_steps_completed": global_step, "epoch": epoch+1,
                                "elapsed_training_seconds": round(perf_counter()-start-monitor_seconds, 3),
                                "last_batch_loss": record["mean_loss"]})
                            pilot.write_json(run / "summary.json", metadata)
                            print(f"seed {seed} {condition}: step {global_step}/{steps}, loss={record['mean_loss']:.4f}, elapsed={record['seconds']:.1f}s", flush=True)
                epoch_losses.append(epoch_loss / len(examples))
                if monitor is not None:
                    begin_monitor = perf_counter()
                    monitor(model, tokenizer, epoch+1, global_step, run)
                    monitor_seconds += perf_counter()-begin_monitor
        torch.cuda.synchronize()
        assert global_step == steps and adapter_hash(model) != metadata["initial_adapter_sha256"]
        assert model.get_input_embeddings().weight.grad is None
        adapter = run / "adapter"
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        (adapter / "README.md").unlink(missing_ok=True)
        metadata.update({"status": "passed", "optimizer_steps_completed": global_step,
            "epoch_mean_loss": epoch_losses, "epoch_order_sha256": order_hashes,
            "training_seconds": round(perf_counter()-start-monitor_seconds, 3), "processed_input_tokens": total_input,
            "supervised_target_tokens": total_target, "trained_examples_with_repeats": len(rows)*config["epochs"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "adapter_path": str(adapter),
            "adapter_bytes": sum(p.stat().st_size for p in adapter.glob("*") if p.is_file()),
            "adapter_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in adapter.glob("*") if p.is_file()}})
        if monitor is not None:
            metadata["epoch_monitoring_seconds"] = round(monitor_seconds, 3)
    except Exception as exc:
        metadata.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        metadata["finished_at_kst"] = now()
        pilot.write_json(run / "summary.json", metadata)


def self_check():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pilot.STUDENT_ID, revision=pilot.STUDENT_REVISION,
        cache_dir=str(CACHE), local_files_only=True, trust_remote_code=False)
    rows, _ = load_training_rows()
    gold = [encode_example(r, "gold_sft", tokenizer) for r in rows]
    kd = [encode_example(r, "response_kd", tokenizer) for r in rows]
    assert sum(g["target_ids"] != k["target_ids"] for g, k in zip(gold,kd)) == 18
    for g,k in zip(gold,kd):
        length=g["prompt_tokens"]
        assert g["input_ids"][:length] == k["input_ids"][:length]
        assert g["prediction_positions"] == [length-1,length]
        assert g["input_ids"][length:] == g["target_ids"]
    logits = torch.randn(1,6,7, generator=torch.Generator().manual_seed(42),requires_grad=True)
    inputs = torch.tensor([[0,1,2,3,4,5]])
    labels=inputs.clone()
    labels[:,:4]=-100
    standard=F.cross_entropy(logits[:,:-1,:].reshape(-1,7),labels[:,1:].reshape(-1),ignore_index=-100)
    sparse=F.cross_entropy(logits[0,[3,4],:],inputs[0,4:])
    assert torch.allclose(standard,sparse)
    grad_a=torch.autograd.grad(standard,logits,retain_graph=True)[0]
    grad_b=torch.autograd.grad(sparse,logits)[0]
    assert torch.allclose(grad_a,grad_b)
    print("SELF-CHECK PASS: 256 matched inputs, 18 target differences, assistant-only causal loss and gradients match masked full loss")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["self-check","probe","train"])
    parser.add_argument("--condition",choices=["gold_sft","response_kd","logit_kd"])
    parser.add_argument("--seed", type=int, help="Training seed; default 42, fixed data sampling seed 42")
    args=parser.parse_args()
    if args.seed is not None and args.action != "train":
        parser.error("--seed is only valid for train")
    if args.action=="self-check":
        self_check()
    elif args.action=="probe":
        memory_probe()
    else:
        if args.condition is None:
            parser.error("train requires --condition")
        train(args.condition, 42 if args.seed is None else args.seed)

