#!/usr/bin/env python3
"""
LightGBM regressor predicting PDR (packet delivery ratio) from per-node,
per-time-chunk link/system state. Intended to eventually back the "mlof" RPL
objective function's parent-selection decision (see
rpl/contiki-ng/os/net/routing/rpl-lite/rpl-mlof.c) — wiring the trained model
into that C code is a separate, later task; this only trains and saves it.

Usage as CLI:
    python models/train.py --runs-dir scripts/runs --save-path pdr_model.txt

Usage as module:
    from models.train import train_model

    model = train_model(runs_dir="scripts/runs", save_path="pdr_model.txt")
"""

import argparse
import os
import sys
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.data import FEATURE_COLUMNS, LABEL_COLUMN, gather_training_data


def train_model(
    runs_dir: str = "runs",
    save_path: str | None = None,
    test_size: float = 0.2,
    **lgbm_params,
) -> lgb.LGBMRegressor:
    """Gather training data, fit an LGBMRegressor on it, and optionally save it."""
    df = gather_training_data(Path(runs_dir))
    if df.empty:
        raise ValueError(f"No training data found under {runs_dir}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    params = {"n_estimators": 200, "learning_rate": 0.05, "verbosity": -1, **lgbm_params}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="l2",
        callbacks=[lgb.log_evaluation(period=50)],
    )

    val_pred = model.predict(X_val)
    print(f"Validation MAE: {mean_absolute_error(y_val, val_pred):.4f}")
    print(
        "Feature importances: "
        + ", ".join(
            f"{name}={imp}"
            for name, imp in zip(FEATURE_COLUMNS, model.feature_importances_)
        )
    )

    if save_path:
        model.booster_.save_model(save_path)
        print(f"Model saved to {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a LightGBM PDR predictor from simulation run data."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory of simulation run subdirs (as written by simulate.sh)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the trained model (LightGBM text format)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)

    args = parser.parse_args()
    train_model(
        runs_dir=args.runs_dir,
        save_path=args.save_path,
        test_size=args.test_size,
    )
