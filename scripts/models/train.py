#!/usr/bin/env python3
"""
LightGBM regressor predicting PDR (packet delivery ratio) from per-node,
per-time-chunk link/system state. Intended to eventually back the "mlof" RPL
objective function's parent-selection decision (see
rpl/contiki-ng/os/net/routing/rpl-lite/rpl-mlof.c) — wiring the trained model
into that C code is a separate, later task; this only trains and saves it.

Usage as CLI (reads train_config.py, see that file):
    python models/train.py

Usage as module:
    from models.train import train_model

    models = train_model(train_csv="scripts/runs/train.csv", save_path="pdr_model.txt")

Pass multiple seeds (train_config.seeds / seeds=[...] as a module) to train
one model per seed, each with its own train/validation split and its own
LightGBM sampling randomness (feature_fraction/bagging_fraction, enabled
below so the seed actually changes anything) -- comparing validation MAE
across seeds is a quick check for overfitting to one particular
split/sampling.
"""

import os
import statistics
import sys
from itertools import combinations

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.data import FEATURE_COLUMNS, LABEL_COLUMN


def _fit_one(
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


def train_model(
    train_csv: str = "train.csv",
    save_path: str | None = None,
    test_size: float = 0.2,
    seeds: list[int] = (0,),
    feature_columns: list[str] = FEATURE_COLUMNS,
    **lgbm_params,
) -> list[lgb.LGBMRegressor]:
    """Load train.csv (see models/data.py) and fit one LGBMRegressor per seed
    in seeds, using feature_columns as the model's inputs (defaults to every
    feature in FEATURE_COLUMNS) -- each seed drives both the train/validation
    split and LightGBM's own sampling randomness (feature_fraction/
    bagging_fraction), so different seeds produce genuinely different
    models. Prints each model's validation MAE and feature importances;
    saves each to save_path (suffixed with the seed when there's more than
    one)."""
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=[LABEL_COLUMN])
    if df.empty:
        raise ValueError(f"No rows with a labeled {LABEL_COLUMN} in {train_csv}")

    X = df[feature_columns]
    y = df[LABEL_COLUMN]

    models = []
    for seed in seeds:
        print(f"\n=== seed={seed} ===")
        model, mae = _fit_one(
            X, y, seed, test_size, callbacks=[lgb.log_evaluation(period=50)], **lgbm_params
        )
        print(f"Validation MAE: {mae:.4f}")

        print(
            "Feature importances: "
            + ", ".join(
                f"{name}={imp}"
                for name, imp in zip(feature_columns, model.feature_importances_)
            )
        )

        if save_path:
            seed_save_path = save_path if len(seeds) == 1 else f"{save_path}.seed{seed}"
            model.booster_.save_model(seed_save_path)
            print(f"Model saved to {seed_save_path}")

        models.append(model)

    return models


# Ablation set: one model with every feature, plus one per feature in
# ABLATION_FEATURES with that single feature dropped -- e.g. to check
# whether a feature with low feature_importances_ actually matters.
ABLATION_FEATURES = ["drop_rate", "is_new", "nbr_count"]


def train_ablation_models(
    train_csv: str,
    seeds: list[int] = (0,),
) -> dict[str, list[lgb.LGBMRegressor]]:
    """train_model() once with every feature (feature_sets["full"]), plus
    once per feature in ABLATION_FEATURES with that single feature dropped
    (feature_sets["no_<feature>"]) -- see ABLATION_FEATURES."""
    feature_sets = {"full": FEATURE_COLUMNS}
    for feature in ABLATION_FEATURES:
        feature_sets[f"no_{feature}"] = [c for c in FEATURE_COLUMNS if c != feature]

    results = {}
    for name, feature_columns in feature_sets.items():
        print(f"\n########## {name} ##########")
        results[name] = train_model(
            train_csv=train_csv,
            seeds=seeds,
            feature_columns=feature_columns,
        )
    return results


def test_feature_importance(
    train_csv: str,
    features_to_test: list[str],
    seeds: list[int] = (0, 1, 2, 3, 4),
    test_size: float = 0.2,
    output_path: str | None = None,
) -> dict[tuple[str, ...], float]:
    """For every combination of features_to_test, including the empty one
    (i.e. every feature included, nothing dropped) -- one at a time, two at
    a time, ..., up to removing all of them at once -- train with that
    combination dropped from FEATURE_COLUMNS across seeds, and report only
    the average validation MAE across those seeds (not each seed's own
    result). Returns {included_features: average_mae} (the features that
    remained after removal). A combination that would leave zero columns to
    train on (features_to_test covers every feature) is skipped rather than
    raising a LightGBMError. If output_path is given, saves every
    combination's result (sorted ascending by avg_mae) as a CSV with columns
    included_features (comma-joined) and avg_mae."""
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=[LABEL_COLUMN])
    if df.empty:
        raise ValueError(f"No rows with a labeled {LABEL_COLUMN} in {train_csv}")
    y = df[LABEL_COLUMN]

    results = {}
    for k in range(0, len(features_to_test) + 1):
        for combo in combinations(features_to_test, k):
            feature_columns = tuple(c for c in FEATURE_COLUMNS if c not in combo)
            if not feature_columns:
                print("include []: skipped (no features left to train on)")
                continue
            X = df[list(feature_columns)]
            maes = [_fit_one(X, y, seed, test_size)[1] for seed in seeds]
            avg_mae = statistics.mean(maes)
            results[feature_columns] = avg_mae
            print(
                f"include {list(feature_columns)}: avg MAE over {len(seeds)} seeds = {avg_mae:.4f}"
            )

    ranked = sorted(results.items(), key=lambda item: item[1])

    print("\nTop 5 smallest avg MAE:")
    for feature_columns, avg_mae in ranked[:5]:
        print(f"  include {list(feature_columns)}: avg MAE = {avg_mae:.4f}")

    if output_path:
        out_df = pd.DataFrame(
            [
                {"included_features": ",".join(feature_columns), "avg_mae": avg_mae}
                for feature_columns, avg_mae in ranked
            ]
        )
        out_df.to_csv(output_path, index=False)
        print(f"\nSaved feature importance results to {output_path}")

    return results


if __name__ == "__main__":
    import train_config
    from pathlib import Path

    train_csv = Path(train_config.data_dir) / "train.csv"
    train_ablation_models(str(train_csv), seeds=train_config.seeds)
