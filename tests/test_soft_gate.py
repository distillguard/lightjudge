"""Check the paper gate, EOS objective and Eq. 7 selection gradient."""
from pathlib import Path
import math
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import logit_kd
import soft_gate


class SoftGateTests(unittest.TestCase):
    def setUp(self):
        self.teacher = torch.log_softmax(torch.tensor([2.5, -1., .5, -.3, .7]), -1)
        self.logits = torch.tensor([[.3, .4, -.7, .8, -.2], [.5, -.1, .3, .2, -.8]], requires_grad=True)
        self.targets = torch.tensor([1, 3])

    def test_formula_and_frozen_margin(self):
        metrics = {}
        soft_gate.distillation_loss(self.logits, self.targets, self.teacher, metrics)
        gap = metrics["teacher_confidence"] - metrics["student_confidence"]
        self.assertEqual(soft_gate.MARGIN, 0.05)
        self.assertAlmostEqual(metrics["forward_weight"], 1/(1+math.exp(-(gap-.05))), places=7)

    def test_equal_confidence_is_not_half_with_nonzero_margin(self):
        teacher = torch.log_softmax(self.logits[0].detach()/2, -1)
        metrics = {}
        soft_gate.distillation_loss(self.logits, self.targets, teacher, metrics)
        self.assertAlmostEqual(metrics["forward_weight"], 1/(1+math.exp(.05)), places=7)
        self.assertEqual(metrics["counterfactual_hard_gate"], 1)

    def test_selection_gradient_matches_equation_seven(self):
        log_q = F.log_softmax(self.logits[0]/2, -1)
        forward = F.kl_div(log_q, self.teacher, reduction="sum", log_target=True)
        reverse = (log_q.exp()*(log_q-self.teacher)).sum()
        entropy_p = -(self.teacher.exp()*self.teacher).sum()
        entropy_q = -(log_q.exp()*log_q).sum()
        gate = torch.sigmoid((entropy_q-entropy_p)/math.log(log_q.numel())-.05)
        eos = F.cross_entropy(self.logits[1:], self.targets[1:])
        detached_objective = .5*(4*(gate.detach()*forward+(1-gate.detach())*reverse)+eos)
        learning = torch.autograd.grad(detached_objective, self.logits, retain_graph=True)[0]
        selection = .5*4*(forward.detach()-reverse.detach())*torch.autograd.grad(gate, self.logits, retain_graph=True)[0]
        actual = torch.autograd.grad(soft_gate.distillation_loss(self.logits,self.targets,self.teacher), self.logits)[0]
        torch.testing.assert_close(actual, learning+selection, atol=1e-6, rtol=1e-5)
        self.assertGreater(selection.abs().sum().item(), 1e-6)
        self.assertFalse(torch.allclose(actual, learning, atol=1e-7, rtol=1e-7))

    def test_identical_distributions_leave_only_eos(self):
        teacher = F.log_softmax(self.logits[0].detach()/2, -1)
        actual = soft_gate.distillation_loss(self.logits, self.targets, teacher)
        expected = .5*F.cross_entropy(self.logits[1:], self.targets[1:])
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual, self.logits)[0]
        torch.testing.assert_close(gradient[0], torch.zeros_like(gradient[0]), atol=1e-7, rtol=0)

    def test_answer_hard_label_is_not_in_the_loss(self):
        changed = torch.tensor([4, 3])
        first = soft_gate.distillation_loss(self.logits, self.targets, self.teacher)
        second = soft_gate.distillation_loss(self.logits, changed, self.teacher)
        torch.testing.assert_close(first, second)

    def test_eos_gradient_is_shared_cross_entropy(self):
        loss = soft_gate.distillation_loss(self.logits, self.targets, self.teacher)
        actual = torch.autograd.grad(loss, self.logits, retain_graph=True)[0][1]
        expected = torch.autograd.grad(.5*F.cross_entropy(self.logits[1:], self.targets[1:]), self.logits)[0][1]
        torch.testing.assert_close(actual, expected)

    def test_teacher_must_be_frozen(self):
        with self.assertRaises(AssertionError):
            soft_gate.distillation_loss(self.logits, self.targets, self.teacher.clone().requires_grad_(True))

    def test_normalized_confidence_implies_bounded_soft_weight(self):
        for teacher_sharp, student_sharp in [(True,False),(False,True)]:
            teacher = F.log_softmax(torch.tensor([1000.,0.,0.,0.,0.]) if teacher_sharp else torch.zeros(5), -1)
            logits = torch.stack([torch.tensor([2000.,0.,0.,0.,0.]) if student_sharp else torch.zeros(5),torch.zeros(5)])
            metrics={}
            soft_gate.distillation_loss(logits,self.targets,teacher,metrics)
            lower=1/(1+math.exp(1.05));upper=1/(1+math.exp(-.95))
            self.assertGreaterEqual(metrics["forward_weight"],lower-1e-6)
            self.assertLessEqual(metrics["forward_weight"],upper+1e-6)

    def test_invalid_distribution_is_rejected(self):
        bad=self.teacher.clone();bad[0]=float("nan")
        with self.assertRaises(AssertionError):
            soft_gate.distillation_loss(self.logits,self.targets,bad)


if __name__ == "__main__":
    unittest.main()
