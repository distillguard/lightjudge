"""CaRE-KD Eq. 3 soft gate, with the reported best ablation margin (0.05)."""
import logit_kd

MARGIN = 0.05


def distillation_loss(logits, targets, teacher_log_probs, metrics=None):
    import torch
    import torch.nn.functional as F
    assert logits.ndim == 2 and logits.shape[0] == targets.numel() == 2
    assert not teacher_log_probs.requires_grad
    log_p = teacher_log_probs.to(device=logits.device, dtype=torch.float32)
    assert log_p.shape == logits[0].shape and torch.isfinite(log_p).all()
    log_q = F.log_softmax(logits[0].float() / logit_kd.TEMPERATURE, dim=-1)
    c_p, c_q, hard_gate = logit_kd.confidence_gate(log_p, log_q)
    # Eq. 7 requires the selection gradient through the soft gate.
    weight = torch.sigmoid(c_p - c_q - MARGIN)
    forward = F.kl_div(log_q, log_p, reduction="sum", log_target=True)
    reverse = (log_q.exp() * (log_q - log_p)).sum()
    eos = F.cross_entropy(logits[1:].float(), targets[1:])
    loss = 0.5 * (logit_kd.TEMPERATURE**2 * (weight * forward + (1-weight) * reverse) + eos)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite paper soft-gate loss")
    if metrics is not None:
        values = torch.stack([forward, reverse, c_p, c_q, hard_gate, weight]).detach().cpu().tolist()
        metrics.update(dict(zip(("forward_kl", "reverse_kl", "teacher_confidence",
            "student_confidence", "counterfactual_hard_gate", "forward_weight"), values)))
        metrics["eos_ce"] = eos.detach().item()
    return loss
