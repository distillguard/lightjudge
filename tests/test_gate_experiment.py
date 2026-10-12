"""V4 routing, exact old update equivalence, monitoring isolation and guard checks."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gate_experiment as gate
import kl_experiment as legacy
import logit_kd
import monitor_training as diagnostic
import soft_gate
import train_student as trainer


class GateExperimentTests(unittest.TestCase):
    def test_full_order_partial_group_matches_legacy(self):
        rows, _ = legacy.rows_for("full")
        for seed in legacy.SEEDS:
            for epoch in range(3):
                order = gate.epoch_order(len(rows), seed, epoch)
                self.assertEqual(order, random.Random(seed + epoch).sample(range(1118), 1118))
                groups = list(gate.batches(order, 8))
                self.assertEqual(len(groups), 140)
                self.assertEqual(len(groups[-1]), 6)
                self.assertEqual([i for group in groups for i in group], order)

    def test_new_groups_reproduce_old_optimizer_updates(self):
        from transformers import get_linear_schedule_with_warmup
        for condition in ("fixed_mix", "adaptive_mix", "adaptive_soft"):
            a = torch.tensor([[.1, .3, -.5], [.2, -.1, .4]], requires_grad=True)
            b = a.detach().clone().requires_grad_()
            opts = [torch.optim.AdamW([x], lr=.0002, weight_decay=0.) for x in (a, b)]
            schedules = [get_linear_schedule_with_warmup(o, 2, 6) for o in opts]
            targets = torch.tensor([0, 1])
            distribution = torch.tensor([.3, .1, -.2]).log_softmax(-1)
            objective = (lambda x: soft_gate.distillation_loss(x, targets, distribution)) if condition == "adaptive_soft" else (
                lambda x: logit_kd.distillation_loss(x, targets, distribution, condition))
            for epoch in range(3):
                order = gate.epoch_order(14, 42, epoch)
                for step, index in enumerate(order, 1):
                    actual = min(8, 14 - ((step - 1) // 8) * 8)
                    (objective(a) * (index + 1) / actual).backward()
                    if step % 8 == 0 or step == 14:
                        torch.nn.utils.clip_grad_norm_([a], 1., error_if_nonfinite=True)
                        opts[0].step(); schedules[0].step(); opts[0].zero_grad(set_to_none=True)
                for group in gate.batches(order, 8):
                    for index in group:
                        (objective(b) * (index + 1) / len(group)).backward()
                    torch.nn.utils.clip_grad_norm_([b], 1., error_if_nonfinite=True)
                    opts[1].step(); schedules[1].step(); opts[1].zero_grad(set_to_none=True)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(opts[0].param_groups[0]["lr"], 0.)

    def test_fixed_and_hard_call_original_loss(self):
        sentinel = object()
        for condition in ("fixed_mix", "adaptive_mix"):
            with patch.object(trainer, "answer_loss", return_value=sentinel) as call:
                self.assertIs(gate.loss("model", "example", "P", condition, {}), sentinel)
                call.assert_called_once_with("model", "example", "P", condition, {})

    def test_legacy_config_unchanged_soft_only_adds_gate(self):
        for condition in legacy.CONDITIONS:
            saved = json.loads((legacy.OUT / "probes/full" / f"{condition}.json").read_text())
            self.assertEqual(gate.configuration(condition, 42), saved["config"])
        config = gate.configuration("adaptive_soft", 7)
        self.assertEqual(config["distillation"]["margin"], .05)
        self.assertFalse(config["distillation"]["gate_detached"])
        self.assertIsNone(config["distillation"]["separate_gate_temperature"])

    def test_evaluation_restores_rng_mode_and_weights_on_error(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.05))
        model.train()
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        weights = model[0].weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with diagnostic.evaluation_scope(model):
                self.assertFalse(model.training)
                random.random(); np.random.random(); torch.rand(3)
                model(torch.ones(3))
                raise RuntimeError("deliberate monitor failure")
        self.assertTrue(model.training)
        self.assertEqual(random.getstate(), state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], state[1][1])
        torch.testing.assert_close(torch.get_rng_state(), state[2], rtol=0, atol=0)
        torch.testing.assert_close(model[0].weight, weights, rtol=0, atol=0)

    def test_incomplete_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            run.mkdir()
            content = '{"status":"failed","error":"preserve evidence"}'
            (run / "summary.json").write_text(content)
            with patch.object(gate, "require_protocol", return_value={}), patch.object(gate, "run_path", return_value=run):
                with self.assertRaises(RuntimeError):
                    gate.train("fixed_mix", 42)
            self.assertEqual((run / "summary.json").read_text(), content)

    def test_cohort_balanced_only_one_measurement_replay_no_test_action(self):
        self.assertEqual(len(gate.COHORT), 9)
        self.assertEqual(len(set(gate.COHORT)), 9)
        self.assertEqual([c for c, _ in gate.COHORT[:3]], ["fixed_mix", "adaptive_mix", "adaptive_soft"])
        self.assertEqual(len(gate.CONDITIONS) * len(legacy.SEEDS), 21)
        self.assertFalse(hasattr(gate, "check_test"))

    def test_changed_source_or_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "training_protocol.json").write_text(json.dumps(
                {"status": "passed", "code_sha256": {"bad": "0"}}))
            with patch.object(gate, "OUT", Path(folder)):
                with self.assertRaises(AssertionError):
                    gate.require_protocol()


if __name__ == "__main__":
    unittest.main()
