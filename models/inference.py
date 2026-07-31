#!/usr/bin/env python3
"""
Inference utilities for MLP models.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.config import MLPConfig
from models.model import MLP


@torch.no_grad()
def predict(model: MLP, X: np.ndarray) -> np.ndarray:
    """Run inference and return numpy predictions."""
    model.eval()
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    out = model(X_t).cpu().numpy()
    return out


def load_model(dir_path: str, device: str = "auto") -> MLP:
    """Reconstruct an MLP from a saved directory."""
    config = MLPConfig.load(os.path.join(dir_path, "config.json"))
    config.device = device
    model = MLP(config)
    state = torch.load(
        os.path.join(dir_path, "model.pt"),
        map_location=config.resolve_device(),
        weights_only=True,
    )
    model.load_state_dict(state)
    return model.to(config.resolve_device())
