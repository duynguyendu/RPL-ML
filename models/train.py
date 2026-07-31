#!/usr/bin/env python3
"""
Simple configurable MLP for RPL sensor/metrics data.

Usage as CLI:
    python models/train.py --input-dim 5 --output-dim 1 --hidden-dims 64 32 --epochs 50

Usage as module:
    from models.train import train
    from models.inference import predict, load_model

    cfg = MLPConfig(input_dim=5, hidden_dims=[64, 32], task="regression")
    model = train(cfg, X_train, y_train, X_val, y_val)
    preds = predict(model, X_test)
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.config import MLPConfig
from models.model import _ACTIVATIONS, MLP
from models.inference import predict


def save_model(model: MLP, config: MLPConfig, dir_path: str) -> None:
    """Save model weights + config to a directory."""
    os.makedirs(dir_path, exist_ok=True)
    config.save(os.path.join(dir_path, "config.json"))
    torch.save(model.state_dict(), os.path.join(dir_path, "model.pt"))


def _build_loaders(
    config: MLPConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
) -> tuple[DataLoader, Optional[DataLoader]]:
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(
        y_train, dtype=torch.float32 if config.task == "regression" else torch.long
    )
    train_loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(
            y_val, dtype=torch.float32 if config.task == "regression" else torch.long
        )
        val_loader = DataLoader(
            TensorDataset(X_v, y_v),
            batch_size=config.batch_size,
            shuffle=False,
        )

    return train_loader, val_loader


def train(
    config: MLPConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model: Optional[MLP] = None,
) -> MLP:
    """Train an MLP and return the best model."""

    device = config.resolve_device()

    if model is None:
        model = MLP(config).to(device)
    else:
        model = model.to(device)

    if config.task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loader, val_loader = _build_loaders(config, X_train, y_train, X_val, y_val)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)
        train_loss = running_loss / n_samples
        history["train_loss"].append(train_loss)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_n = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    out = model(X_batch)
                    loss = criterion(out, y_batch)
                    val_running += loss.item() * X_batch.size(0)
                    val_n += X_batch.size(0)
            val_loss = val_running / val_n
            history["val_loss"].append(val_loss)

        monitor = val_loss if val_loss is not None else train_loss
        if monitor < best_val_loss:
            best_val_loss = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % max(1, config.epochs // 10) == 0 or epoch == 1:
            msg = f"Epoch {epoch:>4d}/{config.epochs}  train_loss={train_loss:.6f}"
            if val_loss is not None:
                msg += f"  val_loss={val_loss:.6f}"
            print(msg)

        if config.patience > 0 and epochs_no_improve >= config.patience:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {config.patience} epochs)"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model


def train_model(
    input_dim: int = 3,
    output_dim: int = 1,
    hidden_dims: list[int] = [64, 32],
    dropout: float = 0.0,
    activation: str = "relu",
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    task: str = "regression",
    patience: int = 10,
    device: str = "auto",
    save_dir: str | None = None,
    data: str | None = None,
) -> None:
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        task=task,
        patience=patience,
        device=device,
    )

    print(f"Config: {config}")
    device = config.resolve_device()
    print(f"Device: {device}")

    if data:
        data_np = np.load(data)
        X_train = data_np["X_train"]
        y_train = data_np["y_train"]
        X_val = data_np.get("X_val")
        y_val = data_np.get("y_val")

        config.input_dim = X_train.shape[1]
        if config.output_dim == 1 and config.task == "classification":
            config.output_dim = int(y_train.max()) + 1

        print(
            f"Train: {X_train.shape}, Val: {X_val.shape if X_val is not None else 'N/A'}"
        )
        model = train(config, X_train, y_train, X_val, y_val)
    else:
        print("No --data provided. Creating model with random weights (dry run).")
        model = MLP(config)

    print(f"\nModel:\n{model}")

    if save_dir:
        save_model(model, config, save_dir)
        print(f"Model saved to {save_dir}")

    dummy = np.random.randn(4, config.input_dim).astype(np.float32)
    out = predict(model, dummy)
    print(f"Dummy inference shape: {out.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train / run a simple MLP on tabular data."
    )
    parser.add_argument("--input-dim", type=int, default=3)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", choices=list(_ACTIVATIONS), default="relu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--task", choices=["regression", "classification"], default="regression"
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="NPZ file with keys X_train, y_train, [X_val, y_val]",
    )

    args = parser.parse_args()
    train_model(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=args.activation,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        task=args.task,
        patience=args.patience,
        device=args.device,
        save_dir=args.save_dir,
        data=args.data,
    )
