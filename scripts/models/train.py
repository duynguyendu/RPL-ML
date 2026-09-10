#!/usr/bin/env python3

import json
import os
import statistics
import sys
from itertools import combinations, product

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.data import FEATURE_COLUMNS, LABEL_COLUMN


def fit_one(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    test_size: float,
    callbacks: list | None = None,
    **lgbm_params,
) -> tuple[lgb.LGBMRegressor, float]:
    """Split (X, y) with random_state=seed, fit one LGBMRegressor (seed also
    drives its feature_fraction/bagging_fraction sampling), and return
    (model, validation MAE)."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "verbosity": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": seed,
        **lgbm_params,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="l2",
        callbacks=callbacks or [],
    )
    val_pred = model.predict(X_val)
    return model, mean_absolute_error(y_val, val_pred)


def grid_search(
    train_csv: str,
    features_to_test: list[str],
    param_grid: dict[str, list],
    seeds: list[int] = (0, 1, 2, 3, 4),
    test_size: float = 0.2,
    output_path: str | None = None,
) -> list[dict]:
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=[LABEL_COLUMN])
    if df.empty:
        raise ValueError(f"No rows with a labeled {LABEL_COLUMN} in {train_csv}")
    y = df[LABEL_COLUMN]

    param_names = list(param_grid.keys())
    param_combos = list(product(*param_grid.values())) if param_names else [()]

    rows = []
    for k in range(0, len(features_to_test) + 1):
        for combo in combinations(features_to_test, k):
            feature_columns = tuple(c for c in FEATURE_COLUMNS if c not in combo)
            if not feature_columns:
                continue
            X = df[list(feature_columns)]

            for param_values in param_combos:
                params = dict(zip(param_names, param_values))
                maes = []
                importances = []
                for seed in seeds:
                    model, mae = fit_one(X, y, seed, test_size, **params)
                    maes.append(mae)
                    importances.append(model.feature_importances_)
                avg_mae = statistics.mean(maes)
                avg_importance = {
                    name: float(value)
                    for name, value in zip(feature_columns, np.mean(importances, axis=0))
                }
                row = {
                    "features": list(feature_columns),
                    **params,
                    "avg_mae": avg_mae,
                    "feature_importance": avg_importance,
                }
                rows.append(row)
                print(
                    f"include {list(feature_columns)} {params}: "
                    f"avg MAE over {len(seeds)} seeds = {avg_mae:.4f}"
                )

    rows.sort(key=lambda row: row["avg_mae"])

    print("\nTop 5 smallest avg MAE:")
    for row in rows[:5]:
        print(f"  {row}")

    if output_path:
        out_df = pd.DataFrame(
            [
                {
                    **row,
                    "features": ",".join(row["features"]),
                    "feature_importance": json.dumps(row["feature_importance"]),
                }
                for row in rows
            ]
        )
        out_df = out_df[["features", *param_names, "avg_mae", "feature_importance"]]
        out_df.to_csv(output_path, index=False)
        print(f"\nSaved grid search results to {output_path}")

    return rows


if __name__ == "__main__":
    import train_config
    from pathlib import Path

    train_csv = Path(train_config.data_dir) / "train.csv"
    grid_search(
        str(train_csv),
        features_to_test=train_config.features_to_test,
        param_grid=train_config.param_grid,
        seeds=train_config.seeds,
    )
