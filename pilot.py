"""ARC-Challenge 준비·validation 평가·train Teacher 응답 생성. 학습이나 test 평가는 수행하지 않는다."""
import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from teacher_quantization import ROOT, CACHE, MODEL_ID, REVISION, load_text_model

DATASET_ID = "allenai/ai2_arc"
DATASET_REVISION = "210d026faf9955653af8916fad021475a3f00453"
STUDENT_ID = "Qwen/Qwen3.5-2B"
STUDENT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATA = ROOT / "data" / "pilot"
OUTPUT = ROOT / "outputs" / "pilot"
SEED = 42
os.environ.setdefault("HF_HOME", str(CACHE.parent))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def question_key(question):
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def input_key(question, choice_texts):
    canonical = lambda value: " ".join(unicodedata.normalize("NFKC", value).split())
    payload = [canonical(question), sorted(canonical(value) for value in choice_texts)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def normalize(row, split):
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    if not row["id"] or not row["question"].strip() or not 2 <= len(labels) <= 5:
        raise ValueError(f"Invalid question/choice count: {row['id']}")
    if len(labels) != len(texts) or len(set(labels)) != len(labels) or row["answerKey"] not in labels:
        raise ValueError(f"Invalid labels/gold: {row['id']}")
    if any(not text.strip() for text in texts):
        raise ValueError(f"Empty choice: {row['id']}")
    return {
        "source_id": row["id"], "source_split": split, "question": row["question"].strip(),
        "choices": [{"label": chr(65 + i), "source_label": label, "text": text.strip()}
                    for i, (label, text) in enumerate(zip(labels, texts))],
        "gold": chr(65 + labels.index(row["answerKey"])), "source_gold": row["answerKey"],
        "question_sha256": question_key(row["question"]),
        "input_sha256": input_key(row["question"], texts),
    }


def prompt(row):
    options = "\n".join(f"{choice['label']}. {choice['text']}" for choice in row["choices"])
    valid = ", ".join(choice["label"] for choice in row["choices"])
    return ("Select the best answer to this science question.\n"
            "Reply with exactly one answer letter. Do not include an explanation.\n\n"
            f"Question: {row['question']}\n{options}\n\nValid answers: {valid}\nAnswer:")


def parse_answer(text, row):
    answer = text.strip()
    return answer if answer in {choice["label"] for choice in row["choices"]} else None


def prepare():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    dataset = load_dataset(DATASET_ID, "ARC-Challenge", revision=DATASET_REVISION,
                           cache_dir=str(ROOT / ".cache" / "huggingface" / "datasets"))
    expected = {"train": 1119, "validation": 299, "test": 1172}
    assert {split: len(rows) for split, rows in dataset.items()} == expected
    # Preserve held-out sets; deduplicate full inputs, keeping distinct choice sets.
    seen_ids, seen_inputs, removed, clean = {}, {}, [], {}
    for split in ("test", "validation", "train"):
        kept = []
        for raw in dataset[split]:
            row = normalize(raw, split)
            previous = seen_ids.get(row["source_id"]) or seen_inputs.get(row["input_sha256"])
            if previous:
                removed.append({"split": split, "id": row["source_id"], "same_as": previous})
                continue
            marker = {"split": split, "id": row["source_id"]}
            seen_ids[row["source_id"]] = marker
            seen_inputs[row["input_sha256"]] = marker
            kept.append(row)
        clean[split] = kept
    assert all(clean.values())
    train = random.Random(SEED).sample(clean["train"], min(256, len(clean["train"])))
    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_ID, revision=STUDENT_REVISION, cache_dir=str(CACHE),
        local_files_only=True, trust_remote_code=False,
    )
    stats = {}
    for split, rows in clean.items():
        chars = sorted(len(prompt(row)) for row in rows)
        stats[split] = {
            "usable_rows": len(rows), "answer_counts": dict(Counter(row["gold"] for row in rows)),
            "choice_counts": dict(Counter(len(row["choices"]) for row in rows)),
            "prompt_chars_median": chars[len(chars) // 2], "prompt_chars_max": max(chars),
        }
    token_lengths = []
    for row in clean["train"] + clean["validation"]:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt(row)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        token_lengths.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
    assert max(token_lengths) <= 1024, "Input too long; revise budget explicitly rather than truncate."
    for split in ("validation", "test"):
        write_rows(DATA / f"{split}.jsonl", clean[split])
    write_rows(DATA / "train_full.jsonl", clean["train"])
    write_rows(DATA / "train.jsonl", train)
    samples = clean["train"][:4] + [row for row in clean["train"] if row["source_gold"].isdigit()][:2]
    write_rows(OUTPUT / "review_samples.jsonl", samples)
    manifest = {
        "dataset_id": DATASET_ID, "revision": DATASET_REVISION, "subset": "ARC-Challenge",
        "license_metadata": "cc-by-sa-4.0", "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
        "raw_split_counts": expected, "split_statistics": stats,
        "pilot_train_count": len(train), "pilot_train_ids": [row["source_id"] for row in train],
        "seed": SEED, "removed_duplicates": removed,
        "duplicate_rule": "NFKC + whitespace normalized question and sorted choice texts, case preserved; priority test > validation > train",
        "same_stem_different_options": "Retained; a repeated stem alone is not a duplicate input",
        "answer_mapping": "Original option order preserved; source labels (letters/digits) mapped to A..E",
        "input_tokens_max_train_dev": max(token_lengths), "max_input_tokens": 1024,
        "test_evaluation": False, "test_use": "Duplicate audit and sealed export only; no predictions",
        "source_parquet_bytes": {"train": 189909, "validation": 55743, "test": 203808},
        "export_bytes": {path.name: path.stat().st_size for path in DATA.glob("*.jsonl")},
        "export_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in DATA.glob("*.jsonl")},
        "prepared_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    }
    write_json(OUTPUT / "data_manifest.json", manifest)
    print(json.dumps({"usable": {key: len(value) for key, value in clean.items()},
                      "pilot_train": len(train), "duplicates_removed": len(removed),
                      "input_tokens_max": max(token_lengths)}, ensure_ascii=False), flush=True)


def training_run_path(condition, seed=SEED):
    if condition not in {"gold_sft", "response_kd", "logit_kd"} or not 0 <= seed < 2**32:
        raise ValueError("Invalid condition or training seed")
    folder = ROOT / "outputs" / "training"
    return folder / condition if seed == SEED else folder / f"seed_{seed}" / condition


def validation_stem(condition, seed=SEED):
    training_run_path(condition, seed)
    return f"{condition}_validation" if seed == SEED else f"{condition}_seed_{seed}_validation"


def run_inference(role, limit=0, split="validation", condition=None, training_seed=SEED):
    if condition is None and training_seed != SEED:
        raise ValueError("Training seed is only valid for adapter evaluation")
    if role not in {"teacher", "student"} or split not in {"validation", "train"}:
        raise ValueError("Only validation evaluation and train Teacher generation are allowed")
    if split == "train" and (role != "teacher" or limit != 0):
        raise ValueError("Train generation must use the Teacher and the entire selected pilot")
    if condition is not None and (condition not in {"gold_sft", "response_kd", "logit_kd"}
            or role != "student" or split != "validation" or limit != 0):
        raise ValueError("Adapters are allowed only for full Student validation evaluation")
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_id, revision = (MODEL_ID, REVISION) if role == "teacher" else (STUDENT_ID, STUDENT_REVISION)
    data_path = DATA / f"{split}.jsonl"
    rows = read_rows(data_path)
    manifest = json.loads((OUTPUT / "data_manifest.json").read_text(encoding="utf-8"))
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert data_hash == manifest["export_sha256"][data_path.name], "Prepared input file changed"
    assert rows and all(row["source_split"] == split for row in rows)
    assert len({row["source_id"] for row in rows}) == len(rows)
    if split == "train":
        assert [row["source_id"] for row in rows] == manifest["pilot_train_ids"]
        assert len(rows) == manifest["pilot_train_count"]
    if limit:
        if not 1 <= limit <= len(rows):
            raise ValueError("limit must be within validation size")
        rows = random.Random(SEED).sample(rows, limit)
    stem = validation_stem(condition, training_seed) if condition else (
        "teacher_responses" if split == "train" else f"{role}_baseline")
    output = OUTPUT / f"{stem}.jsonl"
    summary_path = OUTPUT / f"{stem}_summary.json"
    error_path = OUTPUT / "teacher_response_errors.jsonl"
    preserved = [output, summary_path] + ([error_path] if split == "train" else [])
    if any(path.exists() for path in preserved):
        raise FileExistsError(f"Existing inference results preserved: {output}")
    metadata = {
        "status": "running", "role": role, "model_id": model_id, "revision": revision,
        "split": split, "sample_ids": [row["source_id"] for row in rows],
        "purpose": "teacher_training_targets" if split == "train" else "validation_baseline",
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION, "subset": "ARC-Challenge",
        "data_sha256": data_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quantization": "NF4 double quant, BF16 compute" if role == "teacher" else "BF16, no quantization",
        "thinking": False, "do_sample": False, "max_new_tokens": 8, "max_input_tokens": 1024,
        "answer_rule": "Trim whitespace; exactly one valid uppercase answer letter; invalid counts as wrong",
        "prompt_template": prompt({"question": "{question}", "choices":
            [{"label": letter, "text": "{option_" + letter + "}"} for letter in "ABCDE"]}),
        "adapter": None, "training": False, "test_evaluation": False, "gold_used_in_prompt": False,
        "started_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    }
    if split == "train":
        # ponytail: answer-only targets; use a separate run for reasoning or soft-target KD.
        metadata["target_type"] = "generated_answer_letter"
        metadata["filter_policy"] = "Keep all selected inputs; no gold-based filter, replacement or response repair"
    adapter = None
    if condition is not None:
        run = training_run_path(condition, training_seed)
        trained = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        assert trained["status"] == "passed" and trained["model_id"] == STUDENT_ID
        assert trained["config"]["seed"] == training_seed
        assert trained["model_revision"] == STUDENT_REVISION and trained["config"]["quantization"] is None
        adapter = run / "adapter"
        for name, checksum in trained["adapter_files_sha256"].items():
            assert hashlib.sha256((adapter / name).read_bytes()).hexdigest() == checksum
        metadata.update({"condition": condition, "training_seed": training_seed, "adapter": str(adapter),
            "training_summary_sha256": hashlib.sha256((run / "summary.json").read_bytes()).hexdigest(),
            "training_config_sha256": trained["config_sha256"]})
    write_json(summary_path, metadata)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=str(CACHE),
            local_files_only=True, trust_remote_code=False)
        teacher_tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=str(CACHE),
            local_files_only=True, trust_remote_code=False)
        metadata["tokenizer_vocab_equal_to_teacher"] = tokenizer.get_vocab() == teacher_tok.get_vocab()
        metadata["special_tokens_equal_to_teacher"] = tokenizer.special_tokens_map == teacher_tok.special_tokens_map
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16) if role == "teacher" else None
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        model, loading = load_text_model(model_id, revision, quantization)
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.eval().requires_grad_(False)
        model.config.use_cache = False
        metadata["load_seconds"] = round(perf_counter() - started, 3)
        metadata["model_memory_footprint_bytes"] = model.get_memory_footprint()
        metadata["loading_info"] = {key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()}
        correct = invalid = truncated = total_input = total_output = 0
        inference_start = perf_counter()
        with output.open("x", encoding="utf-8") as stream:
            for i, row in enumerate(rows, 1):
                rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt(row)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                input_count = inputs.input_ids.shape[1]
                assert input_count <= 1024, "Input exceeds the frozen budget"
                begin = perf_counter()
                with torch.inference_mode():
                    if i == 1:
                        logits = model(**inputs, logits_to_keep=1, use_cache=False).logits
                        assert torch.isfinite(logits).all().item()
                        del logits
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                        use_cache=False, pad_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                ids = generated[0, input_count:]
                response = tokenizer.decode(ids, skip_special_tokens=True)
                answer = parse_answer(response, row)
                hit = answer == row["gold"]
                eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
                eos_ids = {eos} if isinstance(eos, int) else set(eos)
                cutoff = ids.numel() == 8 and ids[-1].item() not in eos_ids
                record = {"source_id": row["source_id"], "gold": row["gold"], "predicted": answer,
                          "response": response, "correct": hit, "invalid": answer is None,
                          "hit_token_limit": cutoff, "input_tokens": input_count,
                          "generated_tokens": ids.numel(), "seconds": round(perf_counter() - begin, 4)}
                if split == "train":
                    record.update(row)
                    record["teacher_target"] = answer
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                correct += hit
                invalid += answer is None
                truncated += cutoff
                total_input += input_count
                total_output += ids.numel()
                metadata.update({"completed": i, "correct": correct, "invalid": invalid,
                                 "accuracy": correct / i, "hit_token_limit": truncated})
                if i % 25 == 0 or i == len(rows):
                    write_json(summary_path, metadata)
                    print(f"{role} {split}: {i}/{len(rows)}, accuracy={correct / i:.3f}, invalid={invalid}", flush=True)
        metadata.update({"status": "passed", "inference_seconds": round(perf_counter() - inference_start, 3),
                         "input_tokens": total_input, "generated_tokens": total_output,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        if split == "train":
            records = read_rows(output)
            assert [row["source_id"] for row in records] == metadata["sample_ids"]
            errors = [row for row in records if not row["correct"] or row["invalid"] or row["hit_token_limit"]]
            write_rows(error_path, errors)
            metadata.update({
                "gold_target_counts": dict(sorted(Counter(row["gold"] for row in records).items())),
                "teacher_target_counts": dict(sorted(Counter(row["teacher_target"] for row in records
                                                           if row["teacher_target"] is not None).items())),
                "gold_disagreements": len(records) - correct,
                "error_review_rows": len(errors), "error_review_path": str(error_path),
                "train_targets_ready": len(records) == len(rows) and invalid == 0 and truncated == 0,
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "output_bytes": output.stat().st_size,
            })
    except Exception as exc:
        metadata.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        metadata["finished_at_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        write_json(summary_path, metadata)


def analyze():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    teacher, student = read_rows(OUTPUT / "teacher_baseline.jsonl"), read_rows(OUTPUT / "student_baseline.jsonl")
    for role in ("teacher", "student"):
        assert json.loads((OUTPUT / f"{role}_baseline_summary.json").read_text(encoding="utf-8"))["status"] == "passed"
    assert [row["source_id"] for row in teacher] == [row["source_id"] for row in student]
    t = np.array([row["correct"] for row in teacher], dtype=float)
    s = np.array([row["correct"] for row in student], dtype=float)
    assert t.size > 0
    indices = np.random.default_rng(SEED).integers(0, t.size, size=(5000, t.size))
    means = np.stack([t[indices].mean(axis=1), s[indices].mean(axis=1)])
    intervals = np.percentile(means, [2.5, 97.5], axis=1).T
    gap = t - s
    interval_gap = np.percentile(gap[indices].mean(axis=1), [2.5, 97.5])
    result = {"split": "validation", "n": int(t.size), "seed": SEED, "bootstrap_replicates": 5000,
              "method": "paired percentile bootstrap over validation examples; single model run, no seed CI",
              "teacher_accuracy": float(t.mean()), "student_accuracy": float(s.mean()),
              "teacher_accuracy_ci95": intervals[0].tolist(), "student_accuracy_ci95": intervals[1].tolist(),
              "teacher_minus_student": float(gap.mean()), "difference_ci95": interval_gap.tolist(),
              "teacher_correct_student_wrong": int(((t == 1) & (s == 0)).sum()),
              "student_correct_teacher_wrong": int(((t == 0) & (s == 1)).sum()),
              "test_evaluation": False, "training": False}
    write_json(OUTPUT / "baseline_comparison.json", result)
    values = np.array([t.mean(), s.mean()])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Qwen3.5-9B NF4\nTeacher", "Qwen3.5-2B BF16\nStudent"], values * 100,
           color=["#325d88", "#4b9883"], yerr=(np.stack([values - intervals[:, 0], intervals[:, 1] - values]) * 100),
           capsize=5)
    ax.set(ylim=(0, 100), ylabel="Accuracy (%)", title=f"ARC-Challenge validation (n={t.size})")
    for i, value in enumerate(values):
        ax.text(i, min(intervals[i, 1] * 100 + 1.4, 98), f"{value * 100:.1f}%", ha="center")
    fig.text(0.5, 0.01, "95% example-bootstrap CI; before pilot learning; test not evaluated", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUTPUT / "baseline_comparison.png", dpi=180)
    plt.close(fig)
    disagreement = [{"source_id": a["source_id"], "gold": a["gold"], "teacher": a["predicted"],
                     "student": b["predicted"], "teacher_correct": a["correct"], "student_correct": b["correct"]}
                    for a, b in zip(teacher, student) if a["correct"] != b["correct"]]
    write_rows(OUTPUT / "disagreements.jsonl", disagreement)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def self_check():
    raw = {"id": "demo", "question": "Which option?", "choices": {"label": ["1", "2", "3"],
           "text": ["first", "second", "third"]}, "answerKey": "2"}
    row = normalize(raw, "train")
    assert row["gold"] == "B" and row["choices"][1]["text"] == "second"
    assert parse_answer(" B\n", row) == "B" and parse_answer("Answer: B", row) is None
    assert parse_answer("D", row) is None
    assert question_key("  Which\nOPTION?") == question_key("which option?")
    assert input_key("Q", ["first", "second"]) == input_key("Q", ["second", "first"])
    assert input_key("Q", ["first", "second"]) != input_key("Q", ["first", "third"])
    assert input_key("Q", ["CO", "Co"]) != input_key("Q", ["co", "Co"])
    assert "gold" not in prompt(row) and "second" in prompt(row)
    assert prompt(row) == prompt({**row, "gold": "A", "source_gold": "1"})
    for role, split, limit in (("teacher", "test", 0), ("student", "train", 0), ("teacher", "train", 1)):
        try:
            run_inference(role, limit, split)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid inference scope was not rejected")
    raw["answerKey"] = "9"
    try:
        normalize(raw, "train")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid answer label was not rejected")
    assert training_run_path("gold_sft", 42) == ROOT / "outputs" / "training" / "gold_sft"
    assert training_run_path("gold_sft", 7) != training_run_path("gold_sft", 42)
    assert validation_stem("response_kd", 42) == "response_kd_validation"
    assert validation_stem("response_kd", 123) == "response_kd_seed_123_validation"
    for seed in (-1, 2**32):
        try:
            training_run_path("gold_sft", seed)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid seed accepted")
    print("SELF-CHECK PASS: label mapping, strict parser, duplicates, invalid gold, gold-free prompt, train/test scope guards")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "benchmark", "generate-teacher", "evaluate-student", "analyze", "self-check"])
    parser.add_argument("--model", choices=["teacher", "student"])
    parser.add_argument("--condition", choices=["gold_sft", "response_kd", "logit_kd"])
    parser.add_argument("--limit", type=int, default=0, help="0 uses all validation; positive values select a seed-42 subset")
    parser.add_argument("--seed", type=int, help="Training seed for evaluate-student; default 42; data stays fixed")
    args = parser.parse_args()
    if args.seed is not None and args.action != "evaluate-student":
        parser.error("--seed is only valid for evaluate-student")
    if args.condition is not None and args.action != "evaluate-student":
        parser.error("--condition is only valid for evaluate-student")
    if args.action == "prepare":
        prepare()
    elif args.action == "benchmark":
        if args.model is None:
            parser.error("benchmark requires --model")
        run_inference(args.model, args.limit)
    elif args.action == "generate-teacher":
        if args.model is not None or args.limit != 0:
            parser.error("generate-teacher uses the Teacher and all selected train inputs; omit --model/--limit")
        run_inference("teacher", split="train")
    elif args.action == "evaluate-student":
        if args.condition is None or args.model is not None or args.limit != 0:
            parser.error("evaluate-student requires --condition and uses full Student validation")
        run_inference("student", condition=args.condition, training_seed=SEED if args.seed is None else args.seed)
    elif args.action == "analyze":
        analyze()
    else:
        self_check()
