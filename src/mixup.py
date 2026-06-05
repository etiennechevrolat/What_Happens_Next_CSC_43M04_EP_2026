"""
MixUp + CutMix for video clips of shape (B, T, C, H, W).

The transform draws ONE coefficient per batch and applies it identically across
the T temporal axis -- so a clip stays a coherent clip, just mixed/cut with
another one. Label smoothing is folded into the soft targets.

Usage:
    mixer = MixupCutmix(num_classes=33, ...)
    loss_fn = SoftTargetCrossEntropy()  # at training time
    ...
    x, soft_targets = mixer(x, labels)   # x is mixed in place, labels become (B, C) soft
    loss = loss_fn(model(x), soft_targets)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixupCutmix:
    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        # mixup_prob + cutmix_prob should be <= 1.0
        # The remainder is the prob of leaving the batch unchanged.
        assert mixup_prob + cutmix_prob <= 1.0 + 1e-6
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.label_smoothing = label_smoothing

    def _smooth_one_hot(self, y: torch.Tensor) -> torch.Tensor:
        eps = self.label_smoothing
        n = self.num_classes
        off = eps / n
        on = 1.0 - eps + off
        soft = torch.full((y.size(0), n), off, device=y.device, dtype=torch.float32)
        soft.scatter_(1, y.unsqueeze(1), on)
        return soft

    def _mixup(self, x: torch.Tensor, y_oh: torch.Tensor):
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1.0 - lam) * x[idx]
        y = lam * y_oh + (1.0 - lam) * y_oh[idx]
        return x, y

    def _cutmix(self, x: torch.Tensor, y_oh: torch.Tensor):
        lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        _, _, _, H, W = x.shape
        cut_rat = float(np.sqrt(1.0 - lam))
        cw, ch = int(W * cut_rat), int(H * cut_rat)
        cx, cy = int(np.random.randint(W)), int(np.random.randint(H))
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        # Same spatial box applied to every t -> preserves temporal coherence.
        x[:, :, :, y1:y2, x1:x2] = x[idx, :, :, y1:y2, x1:x2]
        # Re-compute lam from the actual (clipped) box area.
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
        y = lam * y_oh + (1.0 - lam) * y_oh[idx]
        return x, y

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: (B, T, C, H, W) float
        y: (B,) long
        returns: (x_mixed, soft_targets) where soft_targets has shape (B, num_classes)
        """
        y_oh = self._smooth_one_hot(y)
        if x.size(0) < 2:
            return x, y_oh
        r = float(np.random.rand())
        if r < self.cutmix_prob:
            return self._cutmix(x, y_oh)
        if r < self.cutmix_prob + self.mixup_prob:
            return self._mixup(x, y_oh)
        return x, y_oh


class SoftTargetCrossEntropy(nn.Module):
    """CE with soft targets (B, C). No internal label smoothing -- already folded in."""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()