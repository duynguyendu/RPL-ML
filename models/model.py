#!/usr/bin/env python3
"""
MLP model definition.
"""

import torch
import torch.nn as nn

from models.config import MLPConfig


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Simple multi-layer perceptron built from an MLPConfig."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        act_cls = _ACTIVATIONS.get(config.activation)
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{config.activation}'. "
                f"Choose from: {list(_ACTIVATIONS)}"
            )

        dims = [config.input_dim] + list(config.hidden_dims) + [config.output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
                layers.append(act_cls())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
