"""Semantic alignment, cached temperature, malformed targets, and gate-gradient checks."""
from pathlib import Path
import math
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import choice_kd as kd


class ChoiceKDTests(unittest.TestCase):
    def setUp(self):
        self.teacher = kd.choice_log_probs(torch.tensor([3., -1., .7, -.2]))
        self.student = torch.tensor([-.3, .2, 1., -.8], requires_grad=True)

    def test_semantic_alignment_uses_each_tokenizers_own_ids(self):
        logits = torch.tensor([.5, 1.2, -.4, 2.])
        teacher_vocab = torch.zeros(100)
        student_vocab = torch.zeros(200)
        teacher_ids, student_ids = [32, 33, 34, 35], [51, 52, 53, 54]
        teacher_vocab[teacher_ids] = logits
        student_vocab[student_ids] = logits
        teacher = kd.choice_log_probs(teacher_vocab[teacher_ids])
        for method in kd.METHODS:
            loss = kd.distillation_loss(student_vocab[student_ids], teacher, method)
            self.assertLess(abs(loss.item()), 1e-6)

    def test_cached_temperature_must_not_be_applied_twice(self):
        raw = torch.tensor([2., -3., 1.5, -.5, 4., 0.])
        full_cache = torch.log_softmax(raw / kd.TEMPERATURE, -1)
        valid = torch.tensor([0, 2, 4])
        projected = kd.choice_log_probs(full_cache[valid], temperature=1)
        expected = kd.choice_log_probs(raw[valid])
        torch.testing.assert_close(projected, expected)
        self.assertFalse(torch.allclose(projected, kd.choice_log_probs(full_cache[valid])))

    def test_probability_scores_are_invariant_to_shared_offset(self):
        for offset in (-100., 0., 100.):
            torch.testing.assert_close(kd.choice_log_probs(self.student + offset),
                                       kd.choice_log_probs(self.student), atol=2e-6, rtol=1e-5)

    def test_option_permutation_preserves_loss(self):
        order = torch.tensor([2, 0, 3, 1])
        for method in kd.METHODS:
            original = kd.distillation_loss(self.student, self.teacher, method)
            permuted = kd.distillation_loss(self.student[order], self.teacher[order], method)
            torch.testing.assert_close(original, permuted)

    def test_invalid_teacher_targets_are_rejected(self):
        bad_targets = [self.teacher.clone().requires_grad_(True), self.teacher + 1.,
                       torch.tensor([0., float("nan"), 0., 0.]), self.teacher[:3]]
        for bad in bad_targets:
            with self.assertRaises(ValueError):
                kd.distillation_loss(self.student, bad, "fixed_mix")

    def test_soft_selection_gradient_is_retained(self):
        log_q = kd.choice_log_probs(self.student)
        forward = (self.teacher.exp() * (self.teacher - log_q)).sum()
        reverse = (log_q.exp() * (log_q - self.teacher)).sum()
        cp = 1 + (self.teacher.exp() * self.teacher).sum() / math.log(4)
        cq = 1 + (log_q.exp() * log_q).sum() / math.log(4)
        weight = torch.sigmoid(cp - cq - .05)
        detached = 4 * (weight.detach() * forward + (1-weight.detach()) * reverse)
        learning = torch.autograd.grad(detached, self.student, retain_graph=True)[0]
        selection = 4 * (forward.detach()-reverse.detach()) * torch.autograd.grad(weight, self.student, retain_graph=True)[0]
        actual = torch.autograd.grad(kd.distillation_loss(self.student, self.teacher, "adaptive_soft"), self.student)[0]
        torch.testing.assert_close(actual, learning + selection, atol=1e-6, rtol=1e-5)
        self.assertGreater(selection.abs().sum().item(), 1e-6)

    def test_hard_selects_correct_direction_and_equal_confidence_forward(self):
        teacher = kd.choice_log_probs(torch.tensor([20., 0., 0.]))
        metrics = {}
        kd.distillation_loss(torch.zeros(3), teacher, "adaptive_hard", metrics)
        self.assertEqual(metrics["forward_weight"], 1)
        kd.distillation_loss(torch.tensor([50., 0., 0.]), kd.choice_log_probs(torch.zeros(3)), "adaptive_hard", metrics)
        self.assertEqual(metrics["forward_weight"], 0)
        kd.distillation_loss(torch.zeros(3), kd.choice_log_probs(torch.zeros(3)), "adaptive_hard", metrics)
        self.assertEqual(metrics["forward_weight"], 1)

    def test_valid_choice_counts_and_extreme_scores_remain_finite(self):
        for n in (3, 4, 5):
            scores = torch.arange(n, dtype=torch.float32, requires_grad=True) * 1000
            teacher = kd.choice_log_probs(torch.zeros(n))
            for method in kd.METHODS:
                loss = kd.distillation_loss(scores, teacher, method)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(torch.autograd.grad(loss, scores, retain_graph=True)[0]).all())
        for n in (1, 2, 6):
            with self.assertRaises(ValueError):
                kd.choice_log_probs(torch.zeros(n))


if __name__ == "__main__":
    unittest.main()
