"""Test exposure guard, checkpoint selection, paired statistics and report reuse."""
import json
import math
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gate_experiment as gate
import gate_evaluation as evaluation
import gate_reporting as reporting
import pilot


def row(correct, source_id="q"):
    return {"source_id": source_id, "gold": "A", "predicted": "A" if correct else "B",
        "correct": correct, "choice_predicted": "A" if correct else "B", "choice_confidence": .7,
        "choice_probabilities": {"A": .7 if correct else .3, "B": .3 if correct else .7},
        "choice_log_probabilities": {"A": math.log(.7 if correct else .3), "B": math.log(.3 if correct else .7)}}


class EvaluationTests(unittest.TestCase):
    def test_incomplete_training_cannot_create_test_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "status.json").write_text('{"status":"running"}')
            with patch.object(evaluation, "OUT", out), patch.object(gate, "require_protocol", return_value={}):
                with self.assertRaises(AssertionError):
                    evaluation.freeze()
            self.assertFalse((out / "test_manifest.json").exists())
            self.assertFalse((out / "scores").exists())

    def test_test_score_without_seal_never_loads_gpu_model(self):
        with patch.object(evaluation, "require_manifest", side_effect=FileNotFoundError("no seal")), patch.object(
                evaluation.legacy, "load_eval_model") as load:
            with self.assertRaises(FileNotFoundError):
                evaluation.score("adaptive_soft_42")
            load.assert_not_called()

    def test_fixed42_uses_original_final_adapter(self):
        protocol = {"reused_training": {"fixed_mix/42": {}}}
        with patch.object(gate, "run_path", return_value=Path("replay")), patch.object(
                evaluation.legacy, "run_path", return_value=Path("original")):
            self.assertEqual(evaluation.model_run(protocol, "fixed_mix", 42), Path("original"))
            self.assertEqual(evaluation.model_run(protocol, "fixed_mix", 7), Path("replay"))

    def test_identical_conditions_zero_paired_difference_and_seed_sd(self):
        data = {c: [[row(True), row(False)] for _ in range(3)] for c in gate.CONDITIONS}
        summary, comparisons = reporting.paired_statistics(data)
        for c in gate.CONDITIONS:
            self.assertEqual(summary[c]["accuracy"]["mean"], .5)
            self.assertEqual(summary[c]["accuracy"]["seed_sample_sd"], 0.)
        for pair in comparisons.values():
            self.assertEqual(pair["accuracy"]["ci95"], [0., 0.])

    def test_bootstrap_pairs_questions_after_averaging_seeds(self):
        data = {c: [[row(False, str(i)) for i in range(4)] for _ in range(3)] for c in gate.CONDITIONS}
        success = [[True, False, True, False], [False, True, True, False], [True, True, False, False]]
        data["adaptive_mix"] = [[row(v, str(i)) for i, v in enumerate(values)] for values in success]
        _, comparisons = reporting.paired_statistics(data)
        indices = np.random.default_rng(42).integers(0, 4, size=(5000, 4))
        expected = np.quantile(np.array(success, dtype=float).mean(0)[indices].mean(1), [.025, .975])
        np.testing.assert_array_equal(comparisons["adaptive_mix_minus_adaptive_soft"]["accuracy"]["ci95"], expected)

    def test_soft_gate_formula_and_hard_tie_checked_in_logs(self):
        values = [dict(forward_kl=.1, reverse_kl=.2, teacher_confidence=.5,
                       student_confidence=.5, loss=.2),
                  dict(forward_kl=.2, reverse_kl=.1, teacher_confidence=.7,
                       student_confidence=.4, loss=.3)]
        soft = [{**r, "forward_weight": 1 / (1 + np.exp(-(r["teacher_confidence"] - r["student_confidence"] - .05)))} for r in values]
        summary = reporting.describe_signals(soft, "adaptive_soft")
        self.assertEqual(summary["n_training_forwards"], 2)
        self.assertEqual(summary["counterfactual_hard_forward_fraction"], 1.)
        hard = [{**r, "forward_weight": 1.} for r in values]
        self.assertEqual(reporting.describe_signals(hard, "adaptive_mix")["forward_weight_mean"], 1.)
        hard[0]["forward_weight"] = 0.
        with self.assertRaises(AssertionError):
            reporting.describe_signals(hard, "adaptive_mix")

    def test_evaluation_records_reject_wrong_identity_or_false_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "score_summary.json"
            output = Path(folder) / "score.jsonl"
            rows = [{"source_id": "q", "gold": "A", "choices": [{"label": "A"}, {"label": "B"}]}]
            pilot.write_rows(output, [row(True)])
            summary = {"status": "passed", "split": "test", "n": 1,
                       "output_sha256": gate.sha(output), "metrics": {"accuracy": 1.}}
            pilot.write_json(path, summary)
            evaluation.records_checked(path, rows, "test")
            with self.assertRaises(AssertionError):
                evaluation.records_checked(path, [{**rows[0], "source_id": "other"}], "test")
            summary["metrics"]["accuracy"] = 0.
            pilot.write_json(path, summary)
            with self.assertRaises(AssertionError):
                evaluation.records_checked(path, rows, "test")

    def test_changed_evaluation_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "test_manifest.json").write_text(json.dumps({"status": "passed", "code_sha256": {"changed": "0"}}))
            with patch.object(evaluation, "OUT", out), patch.object(gate, "require_protocol", return_value={}), patch.object(
                    evaluation, "require_prepared", return_value={}):
                with self.assertRaises(AssertionError):
                    evaluation.require_manifest()

    def test_prepared_evaluation_source_change_rejected_before_test(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "evaluation_protocol.json").write_text(json.dumps({"status": "passed", "code_sha256": {"changed": "0"}}))
            with patch.object(evaluation, "OUT", out):
                with self.assertRaises(AssertionError):
                    evaluation.require_prepared()

    def test_successful_seal_requires_all_21_models_and_excludes_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root / "outputs/v4"; old = root / "outputs/legacy"
            cache = root / "cache"; data = root / "data"; data.mkdir()
            out.mkdir(parents=True)
            pilot.write_json(out / "status.json", {"status": "training_validation_complete"})
            pilot.write_json(out / "training_protocol.json", {"fixture": "training"})
            pilot.write_json(out / "evaluation_protocol.json", {"fixture": "evaluation"})
            (data / "test.jsonl").write_text("test hash fixture")
            train_rows = [{"source_id": str(i)} for i in range(1118)]
            val_rows = [{"source_id": str(i), "gold": "A", "choices": [{"label": "A"}, {"label": "B"}]} for i in range(299)]
            train_path, val_path = data / "train.jsonl", data / "validation.jsonl"
            pilot.write_rows(train_path, train_rows); pilot.write_rows(val_path, val_rows)
            reuse = {f"{c}/{s}": {} for c in gate.CONDITIONS[:4] for s in gate.legacy.SEEDS}
            reuse["fixed_mix/42"] = {}
            protocol = {"reused_training": reuse, "reused_validation": {k: {} for k in reuse if not k.startswith("fixed_mix")},
                "legacy_protocol_sha256": "legacy", "initial_adapter_sha256_by_seed": {str(s): f"initial{s}" for s in gate.legacy.SEEDS}}
            (root / "src").mkdir()
            for name in ("gate_evaluation.py", "gate_reporting.py"):
                shutil.copyfile(ROOT / "src" / name, root / "src" / name)
            code = {p.name: gate.sha(p) for p in (root / "src").iterdir()}
            for model_id, revision in ((evaluation.MODEL_ID, evaluation.REVISION), (pilot.STUDENT_ID, pilot.STUDENT_REVISION)):
                path = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
                path.mkdir(parents=True); (path / "model.safetensors").write_bytes(b"checkpoint fixture")
            def rows_for(scope, split="train"):
                return (train_rows, train_path) if split == "train" else (val_rows, val_path)
            def score_file(path):
                output = path.with_name(path.name.replace("_summary.json", ".jsonl"))
                pilot.write_rows(output, [row(True, str(i)) for i in range(299)])
                pilot.write_json(path, {"status": "passed", "split": "validation", "n": 299,
                    "data_sha256": gate.sha(val_path), "output_sha256": gate.sha(output),
                    "metrics": {"accuracy": 1.}, "protocol_sha256": gate.sha(out / "training_protocol.json"),
                    "adapter_files_sha256": {}})
            with patch.object(evaluation, "ROOT", root), patch.object(gate, "ROOT", root), patch.object(
                    evaluation, "OUT", out), patch.object(gate, "OUT", out), patch.object(evaluation.legacy, "OUT", old), patch.object(
                    evaluation, "CACHE", cache), patch.object(pilot, "DATA", data), patch.object(
                    gate, "require_protocol", return_value=protocol), patch.object(evaluation, "require_prepared", return_value={}), patch.object(
                    evaluation, "code_hashes", return_value=code), patch.object(evaluation.legacy, "rows_for", side_effect=rows_for):
                def create_run(run, c, s, original=False):
                    adapter = run / "adapter"; adapter.mkdir(parents=True)
                    (adapter / "adapter_model.safetensors").write_bytes(f"{c}{s}{original}".encode())
                    files = {p.name: gate.sha(p) for p in adapter.iterdir()}
                    pilot.write_json(run / "summary.json", {"status": "passed", "config": gate.configuration(c, s),
                        "optimizer_steps_completed": 420, "train_rows": 1118, "trained_examples_with_repeats": 3354,
                        "validation_used_for_training": False, "test_evaluation": False,
                        "initial_adapter_sha256": f"initial{s}", "epoch_order_sha256": gate.order_hashes(train_rows, s),
                        "adapter_files_sha256": files, "protocol_sha256": "legacy" if original else gate.sha(out / "training_protocol.json"),
                        "measurement_only": not original and (c, s) == ("fixed_mix", 42), "training_seconds": 100.})
                    return files
                for role in ("teacher", "student"):
                    score_file(old / "scores/validation" / f"{role}_baseline_summary.json")
                for c in gate.CONDITIONS:
                    for s in gate.legacy.SEEDS:
                        original = f"{c}/{s}" in reuse
                        files = create_run(evaluation.model_run(protocol, c, s), c, s, original)
                        path = evaluation.validation_path(protocol, c, s); score_file(path)
                        saved = gate.read(path); saved["adapter_files_sha256"] = files; pilot.write_json(path, saved)
                create_run(gate.run_path("fixed_mix", 42), "fixed_mix", 42)
                for c, s in gate.COHORT:
                    run = gate.run_path(c, s)
                    pilot.write_rows(run / "train_log.jsonl", [{"optimizer_step": i, "mean_loss": .1} for i in range(1, 421)])
                    pilot.write_rows(run / "signal_log.jsonl", [{"fixture": True} for _ in range(3354)])
                    for epoch in range(4):
                        output = run / f"epoch_{epoch}_validation.jsonl"; output.write_text("epoch fixture")
                        pilot.write_json(run / f"epoch_{epoch}.json", {"status": "passed", "metrics": {"n": 299},
                            "optimizer_step": epoch * 140, "data_sha256": gate.sha(val_path), "output_sha256": gate.sha(output),
                            "test_evaluation": False, "validation_used_for_selection": False})
                missing = evaluation.validation_path(protocol, "adaptive_soft", 123)
                content = missing.read_bytes(); missing.unlink()
                with self.assertRaises(FileNotFoundError):
                    evaluation.freeze()
                self.assertFalse((out / "test_manifest.json").exists())
                missing.write_bytes(content)
                baseline_path = old / "scores/validation/student_baseline_summary.json"
                baseline_before = baseline_path.read_bytes()
                evaluation.freeze()
                manifest = gate.read(out / "test_manifest.json")
                self.assertEqual(baseline_before, baseline_path.read_bytes())
                self.assertEqual(len(manifest["models"]), 23)
                self.assertFalse(manifest["timing_replay_used_as_test_model"])
                self.assertEqual(manifest["models"]["fixed_mix_42"]["adapter"], str((old / "training/full/seed_42/fixed_mix/adapter").relative_to(root)))

    def test_final_report_is_repeatable_preserves_three_documents_and_no_retraining(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            (root / "results/protocols").mkdir(parents=True)
            out = root / "outputs"
            (out / "analysis/test").mkdir(parents=True)
            pilot.write_json(out / "test_manifest.json", {"status": "passed"})
            pilot.write_json(root / "results/protocols/research_revision_v4.json", {"verification": {}})
            fixture = root / "results/gate_comparison_v4/test"
            fixture.mkdir(parents=True)
            for name in ("accuracy.png", "paired_ci.png", "learning_curves.png", "time_accuracy.png", "comparison.json"):
                (fixture / name).write_text("fixture")
            for name in ("README.md", "docs/RESEARCH_PLAN.md", "docs/RESEARCH_REPORT.md"):
                (root / name).write_bytes(("<!-- main-status:start -->old<!-- main-status:end -->\r\n\r\n"
                    "## 2026-10-12 계획 개정\r\n\r\nE7 본학습·test 결과는 아직 없다.\r\n").encode("utf-8"))
            data = {"status": "passed", "n": 1172,
                "aggregate": {c: {"accuracy": {"mean": .8, "seed_sample_sd": .01, "per_seed": [.8, .81, .79]},
                    "choice_nll": {"mean": .4}} for c in gate.CONDITIONS},
                "comparisons": {f"{a}_minus_{b}": {"accuracy": {"mean_difference": 0., "ci95": [-.01, .01]}}
                    for a, b in evaluation.SPEC["pairs"]},
                "timing": {c: [{"training_seconds": 1800.}, {"training_seconds": 1801.}, {"training_seconds": 1799.}]
                    for c in ("fixed_mix", "adaptive_mix", "adaptive_soft")},
                "baselines": {role: {"accuracy": .8} for role in ("teacher", "student")},
                "fixed42_replay": {"tensor_sha256_matches_original": True, "final_validation_prediction_mismatches": 0}}
            pilot.write_json(out / "analysis/test/comparison.json", data)
            with patch.object(reporting, "ROOT", root), patch.object(gate, "OUT", out), patch.object(
                    evaluation, "require_manifest", return_value={"public_checkpoint_files_sha256": {}}), patch.object(
                    reporting.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as commands, patch.object(
                    reporting.subprocess, "check_output", return_value="main\n"):
                reporting.publish()
                first = (root / "docs/RESEARCH_REPORT.md").read_bytes()
                reporting.publish()
                self.assertEqual(first, (root / "docs/RESEARCH_REPORT.md").read_bytes())
                self.assertTrue(all("train" not in call.args[0] for call in commands.call_args_list))
            text = first.decode("utf-8")
            self.assertEqual(text.count("<!-- gate-final-results:start -->"), 1)
            self.assertIn("동등성을 입증한 것은 아니다", text)
            self.assertIn("관측한 세 학습 seed에 조건부", text)
            self.assertIn("\r\n", text)


if __name__ == "__main__":
    unittest.main()
