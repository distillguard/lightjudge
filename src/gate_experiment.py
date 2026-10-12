"""Frozen v4 training/validation; legacy results remain immutable, test stays sealed."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import kl_experiment as legacy
import logit_kd
import monitor_training as diagnostic
import pilot
import soft_gate
import train_student as trainer
from teacher_quantization import ROOT

OUT = ROOT / "outputs/gate_comparison_v4"
CONDITIONS = (*legacy.CONDITIONS, "adaptive_soft")
COHORT = (("fixed_mix", 42), ("adaptive_mix", 42), ("adaptive_soft", 42),
          ("adaptive_mix", 7), ("adaptive_soft", 7), ("fixed_mix", 7),
          ("adaptive_soft", 123), ("fixed_mix", 123), ("adaptive_mix", 123))
sha = diagnostic.sha


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hashes():
    return {**legacy.code_hashes(), **{name: sha(ROOT / "src" / name) for name in
        ("soft_gate.py", "probe_soft_gate.py", "gate_experiment.py")}}


def configuration(condition, seed):
    assert condition in CONDITIONS and seed in legacy.SEEDS
    config, _ = trainer.run_config("logit_kd" if condition in CONDITIONS[2:] else condition, seed)
    if condition in CONDITIONS[2:]:
        variant = "forward" if condition == "logit_kd" else condition
        config["distillation"]["variant"] = variant
        config["loss"] = f"0.5 * (T^2 {variant} answer divergence + shared EOS CE)"
    config.update(dataset_scope="full",
        partial_batch_normalization="divide each sample by actual accumulation-group size")
    if condition == "adaptive_soft":
        config["distillation"].update(margin=soft_gate.MARGIN, gate_detached=False,
                                     separate_gate_temperature=None)
    return config


def epoch_order(n, seed, epoch):
    return random.Random(seed + epoch).sample(range(n), n)


def batches(order, size):
    for start in range(0, len(order), size):
        yield order[start:start + size]


def order_hashes(rows, seed):
    return [hashlib.sha256(json.dumps([rows[i]["source_id"] for i in
        epoch_order(len(rows), seed, epoch)]).encode()).hexdigest()
        for epoch in range(trainer.CONFIG["epochs"])]


def checked_files(files):
    for name, checksum in files.items():
        assert sha(ROOT / name) == checksum, f"Changed artifact: {name}"


def freeze():
    if (OUT / "training_protocol.json").exists():
        require_protocol()
        return
    _, old_hash = legacy.require_protocol()
    assert not (legacy.OUT / "test_manifest.json").exists()
    rows, train_path = legacy.rows_for("full")
    _, validation_path = legacy.rows_for("full", "validation")
    targets = read(legacy.OUT / "targets/full/summary.json")
    assert targets["status"] == "passed" and targets["data_sha256"] == sha(train_path)
    checked_files({str((legacy.OUT / "targets/full" / filename).relative_to(ROOT)): checksum
        for filename, checksum in (("teacher_log_probs.safetensors", targets["output_sha256"]),
                                    ("teacher_responses.jsonl", targets["response_sha256"]))})
    protected = {}
    for filename, key in (("preservation.json", "original_artifacts_sha256"),
                          ("scope_change.json", "completed_extension_pilot_artifacts_sha256")):
        protected.update(read(ROOT / "results/protocols" / filename)[key])
    assert len(protected) == 130
    checked_files(protected)
    reuse, validations, initial = {}, {}, {}
    for condition in legacy.CONDITIONS[:5]:
        for seed in legacy.SEEDS:
            run = legacy.run_path("full", condition, seed)
            if not (run / "summary.json").exists():
                continue
            summary = read(run / "summary.json")
            assert summary["status"] == "passed" and summary["config"] == configuration(condition, seed)
            assert summary["protocol_sha256"] == old_hash
            assert summary["script_sha256"] == sha(ROOT / "src/train_student.py")
            assert summary["train_rows"] == 1118 and summary["optimizer_steps_completed"] == 420
            assert summary["train_sha256"] == sha(train_path)
            assert summary["teacher_responses_sha256"] == targets["response_sha256"]
            if condition in legacy.SOFT:
                assert summary["soft_targets_sha256"] == targets["output_sha256"]
            assert summary["sample_ids"] == [r["source_id"] for r in rows]
            assert summary["epoch_order_sha256"] == order_hashes(rows, seed)
            assert summary["base_checkpoint_start"] and summary["previous_adapter"] is None
            assert not summary["test_evaluation"] and not summary["validation_used_for_training"]
            assert summary["versions"] == {p: importlib.metadata.version(p) for p in summary["versions"]}
            initial.setdefault(str(seed), summary["initial_adapter_sha256"])
            assert initial[str(seed)] == summary["initial_adapter_sha256"]
            files = {str((run / "adapter" / name).relative_to(ROOT)): checksum
                     for name, checksum in summary["adapter_files_sha256"].items()}
            files[str((run / "summary.json").relative_to(ROOT))] = sha(run / "summary.json")
            checked_files(files)
            tag = f"{condition}/{seed}"
            reuse[tag] = files
            score = legacy.OUT / "scores/validation" / f"full_{condition}_{seed}_summary.json"
            if condition == "fixed_mix":
                continue
            saved = read(score)
            output = score.with_name(score.name.replace("_summary.json", ".jsonl"))
            records = pilot.read_rows(output)
            validation_rows, _ = legacy.rows_for("full", "validation")
            assert saved["status"] == "passed" and saved["n"] == len(records) == 299
            assert saved["data_sha256"] == sha(validation_path) and sha(output) == saved["output_sha256"]
            assert [r["source_id"] for r in records] == [r["source_id"] for r in validation_rows]
            assert all(a["gold"] == b["gold"] and a["correct"] == (a["predicted"] == b["gold"])
                       for a, b in zip(records, validation_rows))
            assert abs(sum(r["correct"] for r in records) / 299 - saved["metrics"]["accuracy"]) < 1e-12
            validations[tag] = {str(p.relative_to(ROOT)): sha(p) for p in (score, output)}
    assert len(reuse) == 13 and len(validations) == 12 and len(initial) == 3
    probe = read(OUT / "soft_gpu_probe.json")
    assert probe["status"] == "passed" and probe["soft_targets_sha256"] == targets["output_sha256"]
    assert all(code_hashes()[name] == checksum for name, checksum in probe["code_sha256"].items())
    for condition in ("fixed_mix", "adaptive_mix"):
        saved = read(legacy.OUT / "probes/full" / f"{condition}.json")
        assert saved["status"] == "passed" and saved["config"] == configuration(condition, 42)
        assert legacy.frozen_code_matches(saved["code_sha256"])
    sources = code_hashes()
    folder = OUT / "sources_frozen"
    folder.mkdir(exist_ok=False)
    for name in sources:
        shutil.copyfile(ROOT / "src" / name, folder / name)
    protocol = {"status": "passed", "version": "4.3", "frozen_at_kst": trainer.now(),
        "code_sha256": sources, "legacy_protocol_sha256": old_hash,
        "protected_artifacts_sha256": protected, "reused_training": reuse,
        "reused_validation": validations, "initial_adapter_sha256_by_seed": initial,
        "train_sha256": sha(train_path), "validation_sha256": sha(validation_path),
        "teacher_targets_sha256": targets["output_sha256"],
        "teacher_responses_sha256": targets["response_sha256"],
        "configs": {f"{c}/{s}": configuration(c, s) for c in CONDITIONS for s in legacy.SEEDS},
        "training_order": [list(job) for job in COHORT], "measurement_only": ["fixed_mix/42"],
        "main_cells": 21, "additional_main_training": 8, "additional_timing_replays": 1,
        "planned_test_models": 23, "test_evaluated": False,
        "checkpoint_selection": "fixed epoch 3; test blocked until separate 23-model seal",
        "monitor": {"validation_n": 299, "epochs": [0, 1, 2, 3], "steps": [0, 140, 280, 420],
                    "rng_mode_weights_preserved": True, "selection_or_updates": False},
        "timing": {"primary": "first training loop through final adapter save and checks, excluding epoch diagnostics",
                   "cuda_synchronize": "phase and epoch boundaries", "quality_target": None,
                   "setup_monitor_final_scoring_separate": True,
                   "backend": "unchanged PyTorch reference kernels; no causal_conv1d/fla/Triton"},
        "soft": {"formula": "sigmoid(c_teacher-c_student-0.05)", "detach": False,
                 "source": "https://arxiv.org/html/2609.36734v1", "equations": [3, 7],
                 "margin_basis": "reported best section 4.5 ablation, not ARC optimum/default",
                 "official_config_verified": False, "full_paper_reproduction": False},
        "control_values": "Inherited matched ARC settings; not asserted paper defaults",
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes", "safetensors")}}
    pilot.write_json(OUT / "training_protocol.json", protocol)
    pilot.write_json(ROOT / "results/protocols/gate_training_protocol.json", protocol)
    print("V4 FREEZE PASS: 13 reusable trainings, 12 validations, 130 preserved artifacts; 8 new cells + 1 timing replay", flush=True)


def require_protocol():
    legacy.require_protocol()
    protocol = read(OUT / "training_protocol.json")
    assert protocol["status"] == "passed" and protocol["code_sha256"] == code_hashes()
    assert protocol["versions"] == {p: importlib.metadata.version(p) for p in protocol["versions"]}
    assert protocol["legacy_protocol_sha256"] == sha(legacy.OUT / "training_protocol.json")
    assert protocol["train_sha256"] == sha(pilot.DATA / "train_full.jsonl")
    assert protocol["validation_sha256"] == sha(pilot.DATA / "validation.jsonl")
    assert protocol["teacher_targets_sha256"] == sha(legacy.OUT / "targets/full/teacher_log_probs.safetensors")
    assert protocol["teacher_responses_sha256"] == sha(legacy.OUT / "targets/full/teacher_responses.jsonl")
    for name, checksum in protocol["code_sha256"].items():
        assert sha(OUT / "sources_frozen" / name) == checksum
    for files in (*protocol["reused_training"].values(), *protocol["reused_validation"].values()):
        checked_files(files)
    checked_files(protocol["protected_artifacts_sha256"])
    return protocol


def run_path(condition, seed):
    assert (condition, seed) in COHORT
    return OUT / "training" / f"seed_{seed}" / condition


def loss(model, example, target, condition, signal):
    if condition != "adaptive_soft":
        return trainer.answer_loss(model, example, target, condition, signal)
    import torch
    ids = torch.tensor([example["input_ids"]], device="cuda")
    positions = torch.tensor(example["prediction_positions"], device="cuda")
    targets = torch.tensor(example["target_ids"], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       logits_to_keep=positions, use_cache=False).logits[0]
    return soft_gate.distillation_loss(logits.float(), targets, target, signal)


def monitor(model, tokenizer, epoch, step, run, training_seconds):
    import torch
    path = run / f"epoch_{epoch}.json"
    assert not path.exists()
    rows, data_path = legacy.rows_for("full", "validation")
    torch.cuda.synchronize()
    start = perf_counter()
    before = trainer.adapter_hash(model)
    with diagnostic.evaluation_scope(model):
        records, metrics = diagnostic.evaluate(model, tokenizer, rows, "validation", epoch)
    assert trainer.adapter_hash(model) == before
    pilot.write_rows(run / f"epoch_{epoch}_validation.jsonl", records)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    pilot.write_json(path, {"status": "passed", "epoch": epoch, "optimizer_step": step,
        "training_seconds": training_seconds, "monitor_seconds": seconds,
        "data_sha256": sha(data_path), "adapter_parameters_sha256": before,
        "output_sha256": sha(run / f"epoch_{epoch}_validation.jsonl"),
        "metrics": metrics, "test_evaluation": False, "validation_used_for_selection": False})
    print(f"{run.name} epoch {epoch}: validation={metrics['accuracy']:.4%}, training={training_seconds:.1f}s", flush=True)
    return seconds


def train(condition, seed):
    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed
    protocol = require_protocol()
    run = run_path(condition, seed)
    if run.exists():
        if legacy.ready(run / "summary.json"):
            saved = read(run / "summary.json")
            assert saved["protocol_sha256"] == sha(OUT / "training_protocol.json")
            assert saved["optimizer_steps_completed"] == 420 and saved["config"] == configuration(condition, seed)
            checked_files({str((run / "adapter" / n).relative_to(ROOT)): h
                           for n, h in saved["adapter_files_sha256"].items()})
            for epoch in range(4):
                point = read(run / f"epoch_{epoch}.json")
                assert point["status"] == "passed" and point["metrics"]["n"] == 299
                assert point["output_sha256"] == sha(run / f"epoch_{epoch}_validation.jsonl")
            return
        raise FileExistsError(f"Existing incomplete run preserved: {run}")
    config = configuration(condition, seed)
    assert config == protocol["configs"][f"{condition}/{seed}"]
    metadata = {"status": "running", "condition": condition, "seed": seed, "config": config,
        "protocol_sha256": sha(OUT / "training_protocol.json"), "started_at_kst": trainer.now(),
        "measurement_only": (condition, seed) == ("fixed_mix", 42),
        "test_evaluation": False, "validation_used_for_training": False,
        "base_checkpoint_start": True, "previous_adapter": None, "optimizer_steps_planned": 420}
    pilot.write_json(run / "summary.json", metadata)
    job_start = perf_counter()
    try:
        ctx = legacy.context("full", "adaptive_mix" if condition == "adaptive_soft" else condition, seed)
        assert ctx["soft_targets_sha256"] == protocol["teacher_targets_sha256"]
        model, tokenizer, parameters, optimizer = trainer.load_student(seed)
        examples = [trainer.encode_example(r, "response_kd", tokenizer) for r in ctx["rows"]]
        assert [e["source_id"] for e in examples] == ctx["target_ids"]
        assert [logit_kd.prefix_hash(e) for e in examples] == ctx["prefix_sha256"]
        initial = trainer.adapter_hash(model)
        assert initial == protocol["initial_adapter_sha256_by_seed"][str(seed)]
        scheduler = get_linear_schedule_with_warmup(optimizer, config["warmup_steps"], 420)
        set_seed(seed)
        torch.cuda.synchronize()
        metadata.update(setup_seconds=perf_counter() - job_start, initial_adapter_sha256=initial,
                        train_rows=len(examples), trainable_parameters=sum(p.numel() for p in parameters))
        zero_seconds = monitor(model, tokenizer, 0, 0, run, 0.)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = perf_counter()
        metadata["training_started_at_kst"] = trainer.now()
        excluded = 0.
        global_step = total_tokens = 0
        epochs = []
        peak_allocated = peak_reserved = 0
        optimizer.zero_grad(set_to_none=True)
        with (run / "train_log.jsonl").open("x", encoding="utf-8") as stream, \
             (run / "signal_log.jsonl").open("x", encoding="utf-8") as signals:
            for epoch in range(config["epochs"]):
                order = epoch_order(len(examples), seed, epoch)
                epoch_total = 0.
                for group in batches(order, config["gradient_accumulation"]):
                    batch_total = 0.
                    for index in group:
                        signal = {}
                        value = loss(model, examples[index], ctx["targets"][index], condition, signal)
                        number = value.item()
                        (value / len(group)).backward()
                        signal.update(source_id=examples[index]["source_id"], epoch=epoch + 1,
                                      optimizer_step=global_step + 1, loss=number)
                        signals.write(json.dumps(signal) + "\n")
                        batch_total += number
                        total_tokens += len(examples[index]["input_ids"])
                    norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
                    lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    epoch_total += batch_total
                    elapsed = perf_counter() - start - excluded
                    record = {"epoch": epoch + 1, "optimizer_step": global_step,
                        "mean_loss": batch_total / len(group), "grad_norm": float(norm), "learning_rate": lr,
                        "sample_ids": [examples[i]["source_id"] for i in group], "seconds": elapsed}
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    signals.flush()
                    if global_step % 8 == 0:
                        metadata.update(optimizer_steps_completed=global_step, epoch=epoch + 1,
                                        elapsed_training_seconds=elapsed, last_batch_loss=record["mean_loss"])
                        pilot.write_json(run / "summary.json", metadata)
                        print(f"seed {seed} {condition}: step {global_step}/420, loss={record['mean_loss']:.4f}, training={elapsed:.1f}s", flush=True)
                epochs.append(epoch_total / len(examples))
                torch.cuda.synchronize()
                elapsed = perf_counter() - start - excluded
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                begin = perf_counter()
                monitor(model, tokenizer, epoch + 1, global_step, run, elapsed)
                torch.cuda.synchronize()
                excluded += perf_counter() - begin
                torch.cuda.reset_peak_memory_stats()
        assert global_step == 420 and trainer.adapter_hash(model) != initial
        assert model.get_input_embeddings().weight.grad is None
        before_save = perf_counter()
        adapter = run / "adapter"
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        (adapter / "README.md").unlink(missing_ok=True)
        files = {p.name: sha(p) for p in adapter.iterdir() if p.is_file()}
        assert set(files) == {"adapter_model.safetensors", "adapter_config.json"}
        torch.cuda.synchronize()
        metadata.update(status="passed", optimizer_steps_completed=global_step,
            training_seconds=perf_counter() - start - excluded, epoch_mean_loss=epochs,
            epoch_order_sha256=order_hashes(ctx["rows"], seed), processed_input_tokens=total_tokens,
            trained_examples_with_repeats=len(examples) * 3, supervised_target_tokens=len(examples) * 6,
            epoch_zero_monitor_seconds=zero_seconds, epoch_monitoring_seconds=excluded,
            final_save_and_check_seconds=perf_counter() - before_save,
            total_job_seconds=perf_counter() - job_start,
            peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
            adapter_path=str(adapter), adapter_files_sha256=files,
            training_finished_at_kst=trainer.now())
        assert metadata["epoch_order_sha256"] == order_hashes(ctx["rows"], seed)
        if metadata["measurement_only"]:
            original = read(legacy.run_path("full", condition, seed) / "summary.json")
            metadata["replay_adapter_tensor_sha256_matches_original"] = (
                files["adapter_model.safetensors"] == original["adapter_files_sha256"]["adapter_model.safetensors"])
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_at_kst"] = trainer.now()
        pilot.write_json(run / "summary.json", metadata)


def main_run(condition, seed):
    protocol = require_protocol()
    if f"{condition}/{seed}" in protocol["reused_training"]:
        return legacy.run_path("full", condition, seed)
    return run_path(condition, seed)


def score(condition, seed):
    import torch
    protocol = require_protocol()
    assert f"{condition}/{seed}" not in protocol["reused_validation"]
    run = main_run(condition, seed)
    summary = read(run / "summary.json")
    assert summary["status"] == "passed" and summary["optimizer_steps_completed"] == 420
    checked_files({str((run / "adapter" / n).relative_to(ROOT)): h
                   for n, h in summary["adapter_files_sha256"].items()})
    folder = OUT / "scores/validation"
    path = folder / f"{condition}_{seed}_summary.json"
    output = folder / f"{condition}_{seed}.jsonl"
    if legacy.ready(path):
        saved = read(path)
        assert sha(output) == saved["output_sha256"] and saved["adapter_files_sha256"] == summary["adapter_files_sha256"]
        assert saved["protocol_sha256"] == sha(OUT / "training_protocol.json")
        return
    assert not output.exists()
    metadata = {"status": "running", "condition": condition, "seed": seed, "split": "validation",
        "n": 299, "protocol_sha256": sha(OUT / "training_protocol.json"),
        "adapter_files_sha256": summary["adapter_files_sha256"], "test_evaluation": False,
        "started_at_kst": trainer.now()}
    pilot.write_json(path, metadata)
    try:
        rows, data_path = legacy.rows_for("full", "validation")
        model, tokenizer, loading = legacy.load_eval_model("student", run / "adapter")
        with diagnostic.evaluation_scope(model):
            records, metrics = diagnostic.evaluate(model, tokenizer, rows, "validation", 3)
        curve = run_path(condition, seed) / "epoch_3_validation.jsonl"
        previous = pilot.read_rows(curve)
        mismatches = sum(a["predicted"] != b["predicted"] for a, b in zip(previous, records))
        assert [r["source_id"] for r in previous] == [r["source_id"] for r in records]
        pilot.write_rows(output, records)
        metadata.update(status="passed", metrics=metrics, output_sha256=sha(output),
            data_sha256=sha(data_path), loading=loading, curve_prediction_mismatches=mismatches,
            comparison_scope="original Fixed42 final adapter vs timing replay" if (condition, seed) == ("fixed_mix", 42) else "reloaded final adapter vs epoch3")
        if (condition, seed) != ("fixed_mix", 42):
            assert mismatches == 0, "Reloaded checkpoint changed final answers"
        torch.cuda.synchronize()
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_at_kst"] = trainer.now()
        pilot.write_json(path, metadata)
    print(f"FINAL VALIDATION {condition} seed {seed}: {metrics['accuracy']:.4%}", flush=True)


def pipeline():
    require_protocol()
    path = OUT / "status.json"
    state = {"status": "running", "stage": "training_validation", "pid": __import__("os").getpid(),
             "started_at_kst": trainer.now(), "test_evaluation": False}
    try:
        for index, (condition, seed) in enumerate(COHORT):
            for action in ("train", "score"):
                state.update(current_action=action, condition=condition, seed=seed,
                             cohort_index=index + 1, cohort_total=9, updated_at_kst=trainer.now())
                pilot.write_json(path, state)
                subprocess.run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                                action, "--condition", condition, "--seed", str(seed)], check=True)
        state.update(status="training_validation_complete", stage="awaiting_test_seal",
                     main_training_complete=21, final_validation_complete=21, timing_runs_complete=9,
                     test_evaluation=False, no_test_or_checkpoint_selection=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["updated_at_kst"] = trainer.now()
        pilot.write_json(path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "train", "score", "pipeline"))
    parser.add_argument("--condition", choices=("fixed_mix", "adaptive_mix", "adaptive_soft"))
    parser.add_argument("--seed", type=int, choices=legacy.SEEDS)
    args = parser.parse_args()
    if args.action in ("train", "score") and (args.condition is None or args.seed is None):
        parser.error("train/score require --condition and --seed")
    if args.action == "freeze":
        freeze()
    elif args.action == "pipeline":
        pipeline()
    else:
        globals()[args.action](args.condition, args.seed)
