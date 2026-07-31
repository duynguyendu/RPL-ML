#!/usr/bin/env python3
"""
Configuration dataclass for MLP models.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List

import torch


@dataclass
class MLPConfig:
    """All hyperparameters for the MLP in one place."""

    input_dim: int = 3
    output_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.0
    activation: str = "relu"
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    task: str = "regression"
    patience: int = 10
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MLPConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)
