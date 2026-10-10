"""Record real epoch diagnostics without changing training updates or existing runs."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pilot
import train_student as trainer

OUTPUT = pilot.ROOT / "outputs" / "diagnostics"
CONDITIONS = ("gold_sft", "response_kd", "logit_kd")
SEED = 42


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def evaluation_scope(model):
    import torch
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        model.train(training)
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


def calibration(records, bins=10):
    confidence = np.array([r["choice_confidence"] for r in records])
    correct = np.array([r["choice_predicted"] == r["gold"] for r in records], dtype=float)
    assert len(records) and np.isfinite(confidence).all()
    assert ((confidence >= 0) & (confidence <= 1)).all()
    assignments = np.minimum((confidence * bins).astype(int), bins-1)
    bucket_rows = []
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            bucket_rows.append({"bin": index, "n": int(selected.sum()),
                "confidence": float(confidence[selected].mean()),
                "accuracy": float(correct[selected].mean())})
    ece = sum(r["n"] / len(records) * abs(r["accuracy"]-r["confidence"]) for r in bucket_rows)
    brier = []
    for row in records:
        labels = list(row["choice_probabilities"])
        probabilities = np.array(list(row["choice_probabilities"].values()))
        assert row["gold"] in labels and np.isclose(probabilities.sum(), 1)
        target = np.array([label == row["gold"] for label in labels], dtype=float)
        brier.append(float(np.square(probabilities-target).sum()))
    nll = []
    for row in records:
        if "choice_log_probabilities" in row:
            nll.append(-row["choice_log_probabilities"][row["gold"]])
        else:
            probability = row["choice_probabilities"][row["gold"]]
            if probability <= 0:
                raise ValueError("Rescore underflowed choices with saved log probabilities")
            nll.append(-math.log(probability))
    return {"ece": ece, "multiclass_brier": float(np.mean(brier)),
            "choice_nll": float(np.mean(nll)), "bins": bucket_rows,
            "scope": "T=1 probabilities normalized over valid choices; first-step generation scores normalized over valid choices; choice-token argmax correctness"}


def evaluate(model, tokenizer, rows, split, epoch):
    import torch
    import torch.nn.functional as F
    records = []
    for count, row in enumerate(rows, 1):
        example = trainer.encode_example(row, "gold_sft", tokenizer)
        ids = torch.tensor([example["input_ids"]], dtype=torch.long, device="cuda")
        positions = torch.tensor(example["prediction_positions"], dtype=torch.long, device="cuda")
        targets = torch.tensor(example["target_ids"], dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                logits_to_keep=positions, use_cache=False).logits[0].float()
        gold_ce = F.cross_entropy(logits, targets).item()
        labels = [choice["label"] for choice in row["choices"]]
        choice_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
        assert all(len(tokenizer.encode(label, add_special_tokens=False)) == 1 for label in labels)
        prefix = ids[:, :example["prompt_tokens"]]
        torch.cuda.synchronize()
        generation_start = perf_counter()
        generated = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
            max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True)
        torch.cuda.synchronize()
        generation_seconds = perf_counter()-generation_start
        output_ids = generated.sequences[0, example["prompt_tokens"]:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        predicted = pilot.parse_answer(response, row)
        first_scores = generated.scores[0][0].float()
        log_probabilities = first_scores[choice_ids].log_softmax(dim=0).cpu().tolist()
        probabilities = first_scores[choice_ids].softmax(dim=0).cpu().tolist()
        choice_predicted = labels[int(np.argmax(probabilities))]
        eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        cutoff = output_ids.numel() == 8 and output_ids[-1].item() not in eos_ids
        records.append({"source_id": row["source_id"], "gold": row["gold"],
            "predicted": predicted, "response": response, "choice_predicted": choice_predicted,
            "correct": predicted == row["gold"], "gold_ce": gold_ce,
            "invalid": predicted is None, "hit_token_limit": cutoff,
            "choice_probabilities": dict(zip(labels, probabilities)),
            "choice_log_probabilities": dict(zip(labels, log_probabilities)),
            "generation_seconds": generation_seconds, "generated_tokens": int(output_ids.numel()),
            "choice_confidence": max(probabilities), "input_tokens": example["prompt_tokens"]})
        if count % 100 == 0 or count == len(rows):
            print(f"monitor epoch {epoch} {split}: {count}/{len(rows)}", flush=True)
    assert all(np.isfinite(r["gold_ce"]) for r in records)
    return records, {"n": len(records),
        "gold_ce": float(np.mean([r["gold_ce"] for r in records])),
        "accuracy": float(np.mean([r["correct"] for r in records])),
        "invalid": sum(r["invalid"] for r in records),
        "hit_token_limit": sum(r["hit_token_limit"] for r in records),
        "choice_accuracy": float(np.mean([r["choice_predicted"] == r["gold"] for r in records])),
        "calibration": calibration(records)}


def monitor(model, tokenizer, epoch, step, run):
    point = Path(run)/f"epoch_{epoch}.json"
    if point.exists():
        raise FileExistsError(f"Existing epoch diagnostics preserved: {point}")
    started = perf_counter()
    initial = trainer.adapter_hash(model)
    summaries = {}
    with evaluation_scope(model):
        for split in ("train", "validation"):
            rows = pilot.read_rows(pilot.DATA/f"{split}.jsonl")
            assert len(rows) == (256 if split == "train" else 299)
            assert all(r["source_split"] == split for r in rows)
            records, summaries[split] = evaluate(model, tokenizer, rows, split, epoch)
            pilot.write_rows(Path(run)/f"epoch_{epoch}_{split}.jsonl", records)
    assert trainer.adapter_hash(model) == initial, "Evaluation changed trainable weights"
    summary = {"status": "passed", "epoch": epoch, "optimizer_step": step,
        "training_seed": SEED, "metrics": summaries, "adapter_parameters_sha256": initial,
        "monitor_source_sha256": sha(__file__), "training_source_sha256": sha(trainer.__file__),
        "loss_definition": "Common gold answer-letter + tokenizer EOS mean CE; evaluation mode, no dropout; not KD training objective",
        "accuracy_definition": "Greedy generated response, thinking=False, max_new_tokens=8, strict valid-letter parser; same official evaluation rule",
        "validation_for_updates_or_checkpoint_selection": False, "test_evaluation": False,
        "seconds": round(perf_counter()-started, 3)}
    pilot.write_json(point, summary)
    print(json.dumps({"condition": Path(run).name, "epoch": epoch,
        "train_loss": summaries["train"]["gold_ce"], "validation_loss": summaries["validation"]["gold_ce"],
        "train_accuracy": summaries["train"]["accuracy"],
        "validation_accuracy": summaries["validation"]["accuracy"]}), flush=True)


def train(condition):
    protocol = json.loads((OUTPUT/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["training_seed"] == SEED and protocol["epochs_observed"] == [0, 1, 2, 3]
    assert protocol["monitor_source_sha256"] == sha(__file__)
    assert protocol["training_source_sha256"] == sha(trainer.__file__)
    trainer.train(condition, SEED, output_dir=OUTPUT/"runs"/condition, monitor=monitor)
    run = OUTPUT/"runs"/condition
    from safetensors.torch import load_file
    original = pilot.training_run_path(condition, SEED)
    old_weights = load_file(str(original/"adapter/adapter_model.safetensors"))
    new_weights = load_file(str(run/"adapter/adapter_model.safetensors"))
    assert old_weights.keys() == new_weights.keys()
    exact = all(np.array_equal(old_weights[k].numpy(), new_weights[k].numpy()) for k in old_weights)
    max_difference = max((old_weights[k]-new_weights[k]).abs().max().item() for k in old_weights)
    previous = pilot.read_rows(pilot.OUTPUT/f"{pilot.validation_stem(condition, SEED)}.jsonl")
    current = pilot.read_rows(run/"epoch_3_validation.jsonl")
    assert [r["source_id"] for r in previous] == [r["source_id"] for r in current]
    report = {"status": "passed", "original_adapter_parameters_identical": exact,
        "max_adapter_parameter_difference": max_difference,
        "final_generated_answers_match_original": sum(a["predicted"] == b["predicted"] for a,b in zip(previous,current)),
        "n": len(current), "training_source_sha256": sha(trainer.__file__),
        "monitor_source_sha256": sha(__file__), "test_evaluation": False}
    pilot.write_json(run/"replay_check.json", report)
    print(json.dumps(report), flush=True)


def self_check():
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5)).train()
    random.seed(23); np.random.seed(23); torch.manual_seed(23)
    expected = (random.random(), np.random.random(), torch.rand(2))
    random.seed(23); np.random.seed(23); torch.manual_seed(23)
    parameters = {key: value.clone() for key,value in model.state_dict().items()}
    with evaluation_scope(model):
        assert not model.training and not torch.is_grad_enabled()
        random.random(); np.random.random(); torch.rand(5)
        model(torch.ones(1, 2))
    actual = (random.random(), np.random.random(), torch.rand(2))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    assert model.training and all(torch.equal(parameters[k],v) for k,v in model.state_dict().items())
    try:
        with evaluation_scope(model):
            raise RuntimeError("probe")
    except RuntimeError:
        pass
    assert model.training and torch.is_grad_enabled()
    synthetic = [{"gold": "A", "choice_predicted": "A", "choice_confidence": 1.,
                  "choice_probabilities": {"A": 1., "B": 0.},
                  "choice_log_probabilities": {"A": 0., "B": -1000.}}]
    result = calibration(synthetic)
    assert result["ece"] == result["multiclass_brier"] == 0 and result["bins"][0]["bin"] == 9
    synthetic[0].update(gold="B")
    result = calibration(synthetic)
    assert result["ece"] == 1 and result["multiclass_brier"] == 2
    print("SELF-CHECK PASS: evaluation restores RNG/mode/grad state, weights unchanged, calibration boundaries and Brier", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "self-check"])
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    if args.action == "self-check":
        self_check()
    else:
        if args.condition is None:
            parser.error("train requires --condition")
        train(args.condition)
