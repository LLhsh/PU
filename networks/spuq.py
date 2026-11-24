"""Semantic prototype uncertainty, including absent-class and padded-pixel handling."""

import torch
from torch import nn
from torch.nn import functional as F


class SPUQ(nn.Module):
    def __init__(self, num_classes, feature_channels, threshold=0.8, momentum=0.95,
                 eps=1e-6):
        super().__init__()
        if num_classes < 2 or feature_channels < 1 or not 0 < threshold <= 1 or not 0 <= momentum < 1:
            raise ValueError("Invalid SPUQ class/feature dimensions, threshold or momentum")
        self.num_classes = num_classes
        self.threshold = threshold
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_channels))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def forward(self, features, probabilities, valid_mask=None):
        if features.ndim != 4 or probabilities.ndim != 4:
            raise ValueError("SPUQ expects [B,C,H,W] features and probabilities")
        if features.shape[0] != probabilities.shape[0] or features.shape[-2:] != probabilities.shape[-2:]:
            raise ValueError("Features and probabilities must share batch/spatial dimensions")
        if features.shape[1] != self.prototypes.shape[1] or probabilities.shape[1] != self.num_classes:
            raise ValueError("SPUQ feature/class dimensions do not match its prototype bank")
        confidence, labels = probabilities.detach().max(dim=1)
        if valid_mask is None:
            valid_mask = torch.ones_like(labels, dtype=torch.bool)
        valid_mask = valid_mask.detach().bool()
        if valid_mask.shape != labels.shape:
            raise ValueError("SPUQ valid_mask must have shape [B,H,W]")
        high = (confidence >= self.threshold) & valid_mask
        # Class-wise one-hot masks: a masked-out pixel is not a background pixel.
        one_hot = F.one_hot(labels, self.num_classes).permute(0, 3, 1, 2).float()
        high_labels = one_hot * high.unsqueeze(1)
        low_labels = one_hot * ((~high) & valid_mask).unsqueeze(1)
        features = features.detach().float()

        # Eqs. (3)-(4): batch-wide weighted mean, preserving absent-class history.
        for cls in range(self.num_classes):
            weights = high_labels[:, cls:cls + 1]
            count = weights.sum()
            if count.item() == 0:
                continue
            candidate = (features * weights).sum(dim=(0, 2, 3)) / (count + self.eps)
            if self.initialized[cls]:
                self.prototypes[cls].mul_(self.momentum).add_(
                    candidate, alpha=1.0 - self.momentum)
            else:
                self.prototypes[cls].copy_(candidate)
                self.initialized[cls] = True

        # Do not use zero vectors as prototypes before their first observation.
        if self.initialized.any():
            distances = torch.stack([
                torch.linalg.vector_norm(features - prototype[None, :, None, None],
                                         ord=2, dim=1)
                for prototype in self.prototypes[self.initialized]
            ], dim=1)
            # The supplied PDF's Eq. (5) has POSITIVE distance in the exponent.
            # Follow the printed equation literally; see README for this caveat.
            similarity = distances.softmax(dim=1)
            entropy = -(similarity * (similarity + self.eps).log()).sum(dim=1, keepdim=True)
            valid = valid_mask.unsqueeze(1)
            has_valid = valid.any(dim=(-2, -1), keepdim=True)
            minimum = entropy.masked_fill(~valid, torch.inf).amin(dim=(-2, -1), keepdim=True)
            maximum = entropy.masked_fill(~valid, -torch.inf).amax(dim=(-2, -1), keepdim=True)
            minimum = torch.where(has_valid, minimum, torch.zeros_like(minimum))
            maximum = torch.where(has_valid, maximum, torch.zeros_like(maximum))
            uncertainty = (entropy - minimum) / (maximum - minimum + self.eps)
            uncertainty = uncertainty.masked_fill(~valid, 0)
        else:
            uncertainty = probabilities.new_zeros((labels.shape[0], 1, *labels.shape[-2:]))
        return {
            "labels": labels, "high_mask": high, "high_labels": high_labels,
            "low_labels": low_labels, "uncertainty": uncertainty,
        }
