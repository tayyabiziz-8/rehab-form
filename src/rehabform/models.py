"""Baseline models for skeleton-sequence classification.

FCN follows the Fully Convolutional Network of Wang et al. (2017), one of the
nine models benchmarked on Rehab-Pile, so our numbers are comparable to theirs.
For LITEMV (the Rehab-Pile winner), use the authors' official implementation
from the Rehab-Pile code release rather than re-writing it.
"""
from __future__ import annotations

import torch
from torch import nn


class FCN(nn.Module):
    """Input (batch, channels, time) -> logits (batch, n_classes)."""

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        def block(c_in: int, c_out: int, k: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding="same"),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_channels, 128, 8),
            block(128, 256, 5),
            block(256, 128, 3),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)          # (B, 128, T)
        h = h.mean(dim=-1)            # global average pooling over time
        return self.head(h)
