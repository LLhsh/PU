"""The CE + Dice losses in Eqs. (12), (13), and (15)."""

import torch
from torch.nn import functional as F


def segmentation_loss(logits, labels, valid_mask=None, eps=1e-5):
    """Hard targets; rotation padding contributes to neither CE nor Dice."""
    labels = labels.detach().long()
    if logits.ndim != 4 or labels.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError("Expected [B,K,H,W] logits and [B,H,W] target labels")
    if valid_mask is None:
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
    valid_mask = valid_mask.detach().bool()
    if valid_mask.shape != labels.shape:
        raise ValueError("valid_mask and labels must have identical shapes")
    mask = valid_mask.float()
    safe_labels = labels.masked_fill(~valid_mask, 0)
    ce = F.cross_entropy(logits, safe_labels, reduction="none")
    ce = (ce * mask).sum() / mask.sum().clamp_min(1)
    probability = logits.float().softmax(dim=1) * mask.unsqueeze(1)
    target = F.one_hot(safe_labels, logits.shape[1]).permute(0, 3, 1, 2).float()
    target = target * mask.unsqueeze(1)
    axes = (0, 2, 3)
    intersection = (probability * target).sum(dim=axes)
    denominator = probability.sum(dim=axes) + target.sum(dim=axes)
    dice = 1 - ((2 * intersection + eps) / (denominator + eps)).mean()
    return ce + dice
