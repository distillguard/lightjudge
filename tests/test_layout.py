"""Regression checks for source relocation and reuse of the unchanged frozen protocol."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import kl_experiment as experiment
import pilot
import teacher_quantization


class LayoutTests(unittest.TestCase):
    def test_project_paths_and_original_protocol(self):
        self.assertEqual(teacher_quantization.ROOT, ROOT)
        self.assertEqual(pilot.DATA, ROOT / "data/pilot")
        self.assertEqual(experiment.OUT, ROOT / "outputs/kl_extension")
        protocol, checksum = experiment.require_protocol()
        migration = json.loads((experiment.OUT / "layout_migration.json").read_text(encoding="utf-8"))
        self.assertEqual(checksum, migration["training_protocol_sha256"])
        self.assertTrue(experiment.frozen_code_matches(protocol["code_sha256"]))
        self.assertEqual(experiment.code_hashes(), migration["after_code_sha256"])

    def test_unrecorded_code_change_is_rejected(self):
        protocol, _ = experiment.require_protocol()
        changed = experiment.code_hashes()
        changed["logit_kd.py"] = "0" * 64
        with patch.object(experiment, "code_hashes", return_value=changed):
            self.assertFalse(experiment.frozen_code_matches(protocol["code_sha256"]))
            with self.assertRaises(AssertionError):
                experiment.require_protocol()

    def test_changed_protocol_is_rejected(self):
        expected, _ = experiment.require_protocol()
        actual_sha = experiment.sha
        def altered_sha(path):
            return "0" * 64 if Path(path) == experiment.OUT / "training_protocol.json" else actual_sha(path)
        with patch.object(experiment, "sha", side_effect=altered_sha):
            self.assertFalse(experiment.frozen_code_matches(expected["code_sha256"]))

    def test_missing_migration_cannot_reuse_old_protocol(self):
        expected, _ = experiment.require_protocol()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(experiment, "OUT", Path(directory)):
                self.assertFalse(experiment.frozen_code_matches(expected["code_sha256"]))

    def test_training_and_scoring_algorithms_are_unchanged(self):
        frozen = experiment.OUT / "sources_frozen"
        for name in ("pilot.py", "train_student.py", "logit_kd.py", "monitor_training.py"):
            self.assertEqual((ROOT / "src" / name).read_bytes(), (frozen / name).read_bytes(), name)
        for name, functions in {
            "kl_experiment.py": ("load_eval_model", "score", "prepare_full", "context", "train", "pipeline_steps"),
            "analyze_kl.py": ("vectors", "aggregate", "signal_summary", "analyze", "plots"),
            "teacher_quantization.py": ("load_text_model",),
        }.items():
            def definitions(path):
                return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            before = definitions(frozen / name)
            after = definitions(ROOT / "src" / name)
            for function in functions:
                self.assertEqual(before[function], after[function], name + ":" + function)


if __name__ == "__main__":
    unittest.main()
