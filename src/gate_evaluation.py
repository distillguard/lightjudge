"""Seal 23 final models, then evaluate test once; wait for the existing training stage."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np
import gate_experiment as gate
import kl_experiment as legacy
import monitor_training as diagnostic
import pilot
import train_student as trainer
from teacher_quantization import CACHE, MODEL_ID, REVISION, ROOT

OUT = gate.OUT
SPEC = {"primary": "adaptive_mix_minus_adaptive_soft", "pairs": [
    ["adaptive_mix", "adaptive_soft"], ["adaptive_mix", "fixed_mix"], ["adaptive_soft", "fixed_mix"]],
    "bootstrap_replicates": 5000, "bootstrap_seed": 42,
    "ci_scope": "paired evaluation questions conditional on the three observed training seeds",
    "secondary_comparisons": "prespecified exploratory unadjusted intervals; no equivalence claim",
    "final_epoch": 3, "generation": "thinking=False, greedy, max_new_tokens=8, batch=1, use_cache=False",
    "warmup_questions": 3, "choice_probabilities": "T=1 normalized over valid first-step answer letters",
    "time_cohort": "nine newly matched Fixed/Hard/Soft runs; Fixed42 replay is measurement only",
    "quality_target": None, "no_post_test_selection_or_retraining": True}


def code_hashes():
    return {**gate.code_hashes(), **{name: gate.sha(ROOT / "src" / name) for name in
                                    ("gate_evaluation.py", "gate_reporting.py")}}


def prepare():
    gate.require_protocol()
    path = OUT / "evaluation_protocol.json"
    if path.exists():
        require_prepared()
        return
    assert not (OUT / "test_manifest.json").exists() and not (OUT / "scores/test").exists()
    folder = OUT / "evaluation_sources_prepared"
    folder.mkdir(exist_ok=False)
    for name in ("gate_evaluation.py", "gate_reporting.py"):
        shutil.copyfile(ROOT / "src" / name, folder / name)
    protocol = {"status": "passed", "prepared_at_kst": trainer.now(), "code_sha256": code_hashes(),
        "analysis_spec": SPEC, "test_seen": False, "final_checkpoint_seal": "awaiting all 21 final validations",
        "training_protocol_sha256": gate.sha(OUT / "training_protocol.json"),
        "setting_basis": "same 5000 paired question bootstrap/RNG42 and three-question warmup as the original evaluation; no new gate/target settings"}
    pilot.write_json(path, protocol)
    pilot.write_json(ROOT / "results/protocols/gate_evaluation_protocol.json", protocol)


def require_prepared():
    protocol = gate.read(OUT / "evaluation_protocol.json")
    assert protocol["status"] == "passed" and protocol["code_sha256"] == code_hashes()
    assert protocol["analysis_spec"] == SPEC
    assert protocol["training_protocol_sha256"] == gate.sha(OUT / "training_protocol.json")
    for name in ("gate_evaluation.py", "gate_reporting.py"):
        assert gate.sha(OUT / "evaluation_sources_prepared" / name) == protocol["code_sha256"][name]
    return protocol


def model_run(protocol, condition, seed):
    return legacy.run_path("full", condition, seed) if f"{condition}/{seed}" in protocol["reused_training"] else gate.run_path(condition, seed)


def validation_path(protocol, condition, seed):
    if f"{condition}/{seed}" in protocol["reused_validation"]:
        return legacy.OUT / "scores/validation" / f"full_{condition}_{seed}_summary.json"
    return OUT / "scores/validation" / f"{condition}_{seed}_summary.json"


def records_checked(path, rows, split):
    summary = gate.read(path)
    output = path.with_name(path.name.replace("_summary.json", ".jsonl"))
    assert summary["status"] == "passed" and summary["split"] == split
    assert summary["n"] == len(rows) and gate.sha(output) == summary["output_sha256"]
    records = pilot.read_rows(output)
    assert len(records) == len(rows)
    assert [r["source_id"] for r in records] == [r["source_id"] for r in rows]
    for record, row in zip(records, rows):
        assert record["gold"] == row["gold"]
        assert record["correct"] == (record["predicted"] == row["gold"])
        assert record["predicted"] is None or record["predicted"] in {c["label"] for c in row["choices"]}
        assert set(record["choice_probabilities"]) == {c["label"] for c in row["choices"]}
        assert np.isfinite(list(record["choice_log_probabilities"].values())).all()
        assert np.isclose(sum(record["choice_probabilities"].values()), 1.)
    assert np.isclose(sum(r["correct"] for r in records) / len(rows), summary["metrics"]["accuracy"])
    return records, summary


def freeze():
    protocol = gate.require_protocol()
    path = OUT / "test_manifest.json"
    if path.exists():
        require_manifest()
        return
    state = gate.read(OUT / "status.json")
    assert state["status"] == "training_validation_complete", "All training/validation must complete before test seal"
    require_prepared()
    assert not (OUT / "scores/test").exists(), "Unsealed test outputs require inspection"
    assert not (legacy.OUT / "test_manifest.json").exists()
    validation_rows, validation_data = legacy.rows_for("full", "validation")
    models, validation_files, timing_files = {}, {}, {}
    for role in ("teacher", "student"):
        baseline_path = legacy.OUT / "scores/validation" / f"{role}_baseline_summary.json"
        _, saved = records_checked(baseline_path, validation_rows, "validation")
        assert saved["data_sha256"] == gate.sha(validation_data)
        for p in (baseline_path, baseline_path.with_name(baseline_path.name.replace("_summary.json", ".jsonl"))):
            validation_files[str(p.relative_to(ROOT))] = gate.sha(p)
    for condition in gate.CONDITIONS:
        for seed in legacy.SEEDS:
            run = model_run(protocol, condition, seed)
            summary = gate.read(run / "summary.json")
            assert summary["status"] == "passed" and summary["optimizer_steps_completed"] == 420
            assert summary["train_rows"] == 1118 and summary["config"] == gate.configuration(condition, seed)
            assert not summary["validation_used_for_training"] and not summary["test_evaluation"]
            assert summary["initial_adapter_sha256"] == protocol["initial_adapter_sha256_by_seed"][str(seed)]
            train_rows, _ = legacy.rows_for("full")
            assert summary["epoch_order_sha256"] == gate.order_hashes(train_rows, seed)
            assert summary["trained_examples_with_repeats"] == 3354
            expected_protocol = protocol["legacy_protocol_sha256"] if f"{condition}/{seed}" in protocol["reused_training"] else gate.sha(OUT / "training_protocol.json")
            assert summary["protocol_sha256"] == expected_protocol
            adapter_files = {str((run / "adapter" / name).relative_to(ROOT)): digest
                             for name, digest in summary["adapter_files_sha256"].items()}
            gate.checked_files(adapter_files)
            score_path = validation_path(protocol, condition, seed)
            _, scored = records_checked(score_path, validation_rows, "validation")
            assert scored["data_sha256"] == gate.sha(validation_data)
            if f"{condition}/{seed}" not in protocol["reused_validation"]:
                assert scored["adapter_files_sha256"] == summary["adapter_files_sha256"]
                assert scored["protocol_sha256"] == gate.sha(OUT / "training_protocol.json")
            for p in (score_path, score_path.with_name(score_path.name.replace("_summary.json", ".jsonl"))):
                validation_files[str(p.relative_to(ROOT))] = gate.sha(p)
            files = {**adapter_files, str((run / "summary.json").relative_to(ROOT)): gate.sha(run / "summary.json")}
            models[f"{condition}_{seed}"] = {"role": "student", "condition": condition, "seed": seed,
                "adapter": str((run / "adapter").relative_to(ROOT)), "files_sha256": files}
    assert len(models) == 21
    for condition, seed in gate.COHORT:
        run = gate.run_path(condition, seed)
        summary = gate.read(run / "summary.json")
        assert summary["status"] == "passed" and summary["optimizer_steps_completed"] == 420
        assert summary["measurement_only"] == ((condition, seed) == ("fixed_mix", 42))
        assert summary["training_seconds"] > 0
        for epoch in range(4):
            point = gate.read(run / f"epoch_{epoch}.json")
            assert point["status"] == "passed" and point["metrics"]["n"] == 299
            assert point["optimizer_step"] == epoch * 140
            assert point["data_sha256"] == gate.sha(validation_data)
            assert not point["test_evaluation"] and not point["validation_used_for_selection"]
            assert point["output_sha256"] == gate.sha(run / f"epoch_{epoch}_validation.jsonl")
        logs = pilot.read_rows(run / "train_log.jsonl")
        signals = pilot.read_rows(run / "signal_log.jsonl")
        assert [r["optimizer_step"] for r in logs] == list(range(1, 421)) and len(signals) == 3354
        assert np.isfinite([r["mean_loss"] for r in logs]).all()
        for p in [run / "summary.json", run / "train_log.jsonl", run / "signal_log.jsonl", *run.glob("epoch_*.json*"), * (run / "adapter").iterdir()]:
            if p.is_file():
                timing_files[str(p.relative_to(ROOT))] = gate.sha(p)
    checkpoint_files = {}
    for role, model_id, revision in (("teacher", MODEL_ID, REVISION), ("student", pilot.STUDENT_ID, pilot.STUDENT_REVISION)):
        snapshot = CACHE / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
        files = [p for p in snapshot.iterdir() if p.is_file() and (p.suffix in (".safetensors", ".json", ".jinja", ".txt"))]
        assert any(p.suffix == ".safetensors" for p in files)
        checkpoint_files.update({str(p.relative_to(ROOT)): gate.sha(p) for p in files})
        models[f"{role}_baseline"] = {"role": role, "condition": None, "seed": 42,
            "adapter": None, "model_id": model_id, "revision": revision, "files_sha256": {}}
    assert len(models) == 23
    # This is a new evaluation seal; the training sources and protocol stay unchanged.
    sources = code_hashes()
    folder = OUT / "test_sources_frozen"
    folder.mkdir(exist_ok=False)
    for name in sources:
        shutil.copyfile(ROOT / "src" / name, folder / name)
    manifest = {"status": "passed", "frozen_at_kst": trainer.now(), "code_sha256": sources,
        "training_protocol_sha256": gate.sha(OUT / "training_protocol.json"),
        "evaluation_protocol_sha256": gate.sha(OUT / "evaluation_protocol.json"),
        "test_data_sha256": gate.sha(pilot.DATA / "test.jsonl"), "test_n": 1172,
        "test_seen_before_freeze": False, "models": models, "main_models": 21,
        "timing_replay_used_as_test_model": False, "analysis_spec": SPEC,
        "validation_files_sha256": validation_files, "timing_files_sha256": timing_files,
        "public_checkpoint_files_sha256": checkpoint_files,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "numpy", "matplotlib")}}
    pilot.write_json(path, manifest)
    pilot.write_json(ROOT / "results/protocols/gate_test_manifest.json", manifest)
    print("TEST SEAL PASS: 21 final adapters + Teacher + original Student; Fixed42 replay excluded", flush=True)


def require_manifest(tag=None):
    gate.require_protocol()
    require_prepared()
    path = OUT / "test_manifest.json"
    manifest = gate.read(path)
    assert manifest["status"] == "passed" and manifest["code_sha256"] == code_hashes()
    assert manifest["analysis_spec"] == SPEC and len(manifest["models"]) == 23
    assert manifest["versions"] == {p: importlib.metadata.version(p) for p in manifest["versions"]}
    assert manifest["training_protocol_sha256"] == gate.sha(OUT / "training_protocol.json")
    assert manifest["evaluation_protocol_sha256"] == gate.sha(OUT / "evaluation_protocol.json")
    assert manifest["test_data_sha256"] == gate.sha(pilot.DATA / "test.jsonl")
    for name, digest in manifest["code_sha256"].items():
        assert gate.sha(OUT / "test_sources_frozen" / name) == digest
    gate.checked_files(manifest["validation_files_sha256"])
    gate.checked_files(manifest["timing_files_sha256"])
    if tag is not None:
        assert tag in manifest["models"]
        gate.checked_files(manifest["models"][tag]["files_sha256"])
    return manifest


def score(tag):
    import torch
    manifest = require_manifest(tag)
    selected = manifest["models"][tag]
    role, condition = selected["role"], selected["condition"]
    folder = OUT / "scores/test"
    path, output = folder / f"{tag}_summary.json", folder / f"{tag}.jsonl"
    rows, data_path = legacy.rows_for("full", "test")
    if legacy.ready(path):
        _, saved = records_checked(path, rows, "test")
        assert saved["test_manifest_sha256"] == gate.sha(OUT / "test_manifest.json")
        return
    assert not output.exists(), "Existing incomplete evaluation preserved"
    # Verify every actual checkpoint byte immediately before its first test load.
    gate.checked_files(manifest["public_checkpoint_files_sha256"])
    metadata = {"status": "running", "tag": tag, "role": role, "condition": condition,
        "seed": selected["seed"], "split": "test", "n": len(rows), "data_sha256": gate.sha(data_path),
        "test_manifest_sha256": gate.sha(OUT / "test_manifest.json"), "test_evaluation": True,
        "checkpoint_selection": "frozen epoch 3", "generation": SPEC["generation"],
        "warmup_questions": SPEC["warmup_questions"], "started_at_kst": trainer.now()}
    pilot.write_json(path, metadata)
    start = perf_counter()
    try:
        adapter = None if selected["adapter"] is None else ROOT / selected["adapter"]
        model, tokenizer, loading = legacy.load_eval_model(role, adapter)
        if role == "student" and adapter is None:
            loading["precision"] = "BF16 original public Student; no adapter"
        for row in rows[:SPEC["warmup_questions"]]:
            example = trainer.encode_example(row, "gold_sft", tokenizer)
            ids = torch.tensor([example["input_ids"][:example["prompt_tokens"]]], device="cuda")
            with torch.inference_mode():
                model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8,
                               do_sample=False, use_cache=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with diagnostic.evaluation_scope(model):
            records, metrics = diagnostic.evaluate(model, tokenizer, rows, "test", 3 if condition else 0)
        times = np.array([r["generation_seconds"] for r in records])
        metrics["inference"] = {"generation_seconds": float(times.sum()),
            "latency_mean_seconds": float(times.mean()), "latency_p50_seconds": float(np.median(times)),
            "latency_p95_seconds": float(np.quantile(times, .95)),
            "questions_per_second": len(records) / float(times.sum()),
            "generated_tokens_per_second": sum(r["generated_tokens"] for r in records) / float(times.sum()),
            "includes_score_output_overhead": True}
        pilot.write_rows(output, records)
        metadata.update(status="passed", metrics=metrics, loading=loading, output_sha256=gate.sha(output),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            total_evaluation_job_seconds=perf_counter() - start)
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_at_kst"] = trainer.now()
        pilot.write_json(path, metadata)
    print(f"TEST {tag}: {metrics['accuracy']:.4%} on {len(rows)}", flush=True)


def pipeline():
    require_prepared()
    path = OUT / "evaluation_status.json"
    lock = OUT / "evaluation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = {"status": "waiting_for_training", "pid": os.getpid(), "started_at_kst": trainer.now(),
             "test_evaluation": False, "automation_created": False}
    try:
        pilot.write_json(path, state)
        previous = None
        while True:
            training = gate.read(OUT / "status.json")
            if training["status"] == "training_validation_complete":
                break
            if training["status"] != "running":
                raise RuntimeError(f"Training stage needs inspection: {training}")
            current = (training["condition"], training["seed"], training["current_action"])
            if current != previous:
                print(f"Waiting for existing training: {current}", flush=True)
                previous = current
            sleep(30)
        freeze()
        manifest = require_manifest()
        tags = ["teacher_baseline", "student_baseline"] + [f"{c}_{s}" for c in gate.CONDITIONS for s in legacy.SEEDS]
        assert set(tags) == set(manifest["models"])
        for index, tag in enumerate(tags, 1):
            state.update(status="running", stage="test", current_model=tag, model_index=index,
                         model_total=23, test_evaluation=True, updated_at_kst=trainer.now())
            pilot.write_json(path, state)
            subprocess.run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                            "score", "--tag", tag], check=True)
        import gate_reporting
        for split in ("validation", "test"):
            gate_reporting.analyze(split)
        gate_reporting.publish()
        state.update(status="passed", stage="complete", models_scored=23,
                     no_post_test_selection_or_retraining=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["updated_at_kst"] = trainer.now()
        pilot.write_json(path, state)
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "score", "pipeline"))
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.action == "score":
        if not args.tag:
            parser.error("score requires --tag")
        score(args.tag)
    else:
        globals()[args.action]()
