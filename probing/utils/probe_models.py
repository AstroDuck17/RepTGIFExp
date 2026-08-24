"""
utils/probe_models.py

Linear and MLP probe architectures for layer-wise probing.
All probes are thin classifiers on top of frozen feature vectors.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LinearProbe(nn.Module):
    """
    Depth-1 linear probe: Linear(D → 1) with a scalar logit output.
    Trained with binary cross-entropy loss.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, D] → logits: [batch]"""
        return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    """
    Depth-N MLP probe with geometric dimension halving.

    Architecture for depth=2, D=1024:
        Linear(1024 → 512) → ReLU → Linear(512 → 1)

    Architecture for depth=3, D=1024:
        Linear(1024 → 512) → ReLU → Linear(512 → 256) → ReLU → Linear(256 → 1)
    """

    def __init__(self, input_dim: int, depth: int):
        super().__init__()
        if depth < 2:
            raise ValueError(f"MLPProbe requires depth >= 2, got {depth}. Use LinearProbe for depth=1.")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth - 1):
            next_dim = max(current_dim // 2, 1)
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, D] → logits: [batch]"""
        return self.net(x).squeeze(-1)


def build_probe(input_dim: int, depth: int) -> nn.Module:
    """Factory function: returns LinearProbe (depth=1) or MLPProbe (depth>=2)."""
    if depth == 1:
        return LinearProbe(input_dim)
    return MLPProbe(input_dim, depth)


def build_optimizer(probe: nn.Module, optimizer_name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Build an optimizer for probe training.

    Args:
        probe: The probe model.
        optimizer_name: One of 'adam', 'adamw', 'sgd'.
        lr: Learning rate.
        weight_decay: L2 regularization coefficient.
    """
    name = optimizer_name.lower()
    params = probe.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: '{optimizer_name}'. Choose from: adam, adamw, sgd")


def count_parameters(probe: nn.Module) -> int:
    """Count total trainable parameters in a probe."""
    return sum(p.numel() for p in probe.parameters() if p.requires_grad)
