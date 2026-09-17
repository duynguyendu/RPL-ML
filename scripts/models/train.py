#!/usr/bin/env python3

import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.data import LABEL_COLUMN

NON_PORTABLE_MODELS = set()


def _fit_one(
    df: pd.DataFrame,
    y: pd.Series,
    cv: ShuffleSplit,
    included_features: list[str],
    model_name: str,
    is_pipeline: bool,
    estimator,
    param_grid: dict,
    scaler_choice,
) -> dict:
    X = df[included_features]
    if is_pipeline:
        estimator = clone(estimator).set_params(scaler=scaler_choice)
        search_param_grid = {f"model__{key}": values for key, values in param_grid.items()}
    else:
        search_param_grid = param_grid
    search = GridSearchCV(
        estimator,
        search_param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,
    )
    search.fit(X, y)

    best_params = (
        {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
        if is_pipeline
        else search.best_params_
    )
    if is_pipeline:
        best_params["scaler"] = "none" if scaler_choice == "passthrough" else "standard"

    best_estimator = search.best_estimator_
    if is_pipeline:
        if scaler_choice == "passthrough":
            best_estimator = best_estimator.named_steps["model"]
        else:
            scaler = best_estimator.named_steps["scaler"]
            inner = best_estimator.named_steps["model"]
            if hasattr(inner, "coef_"):
                folded = clone(inner)
                folded.coef_ = inner.coef_ / scaler.scale_
                folded.intercept_ = inner.intercept_ - np.sum(
                    inner.coef_ * scaler.mean_ / scaler.scale_
                )
                folded.n_features_in_ = inner.n_features_in_
                folded.feature_names_in_ = np.array(included_features, dtype=object)
                best_estimator = folded

    avg_mae = -search.best_score_
    importances = getattr(best_estimator, "feature_importances_", None)
    if importances is None:
        importances = getattr(best_estimator, "coef_", None)
    avg_importance = (
        dict(zip(included_features, importances.tolist()))
        if importances is not None
        else None
    )

    return {
        "model": model_name,
        "features": included_features,
        **best_params,
        "avg_mae": avg_mae,
        "feature_importance": avg_importance,
        "_model": best_estimator,
    }


def grid_search() -> list[dict]:
    import train_config

    train_csv = Path(train_config.data_dir) / "train.csv"
    models_dir = Path(train_config.data_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "grid_search.csv"
    test_size = 0.2

    df = pd.read_csv(train_csv)
    all_rows = df
    df = df.dropna(subset=[LABEL_COLUMN])
    if train_config.filter_unknown:
        for col, sentinel in train_config.UNKNOWN_SENTINELS.items():
            if col in df.columns:
                df = df[df[col] != sentinel]
    dropped_rows = all_rows.loc[all_rows.index.difference(df.index)]
    reason = "missing label or unknown feature" if train_config.filter_unknown else "missing label"
    print(f"Dropped {len(dropped_rows)} of {len(all_rows)} rows ({reason})")
    if not dropped_rows.empty:
        print(dropped_rows.head(100).to_string())
    if df.empty:
        raise ValueError(f"No fully-labeled rows in {train_csv}")
    y = df[LABEL_COLUMN]

    cv = ShuffleSplit(
        n_splits=len(train_config.seeds), test_size=test_size, random_state=0
    )

    model_configs = [
        (
            "lgbm",
            False,
            lgb.LGBMRegressor(
                verbosity=-1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                random_state=0,
                n_jobs=train_config.lgbm_n_jobs,
            ),
            train_config.lgbm_param_grid,
        ),
        (
            "ridge",
            True,
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            train_config.ridge_param_grid,
        ),
        (
            "dtree",
            False,
            DecisionTreeRegressor(random_state=0),
            train_config.dtree_param_grid,
        ),
        (
            "svr",
            True,
            Pipeline([("scaler", StandardScaler()), ("model", LinearSVR(random_state=0))]),
            train_config.svr_param_grid,
        ),
    ]
    hyperparam_names = sorted({name for _, _, _, grid in model_configs for name in grid} | {"scaler"})

    jobs = [
        (train_config.FIXED_FEATURES + list(dynamic_subset), model_name, is_pipeline, estimator, param_grid, scaler_choice)
        for num_dynamic in range(train_config.min_dynamic_features, train_config.max_dynamic_features + 1)
        for dynamic_subset in combinations(train_config.DYNAMIC_FEATURES, num_dynamic)
        for model_name, is_pipeline, estimator, param_grid in model_configs
        for scaler_choice in ((StandardScaler(), "passthrough") if is_pipeline else (None,))
    ]

    start_time = time.monotonic()
    results = Parallel(n_jobs=train_config.n_jobs)(
        delayed(_fit_one)(df, y, cv, included_features, model_name, is_pipeline, estimator, param_grid, scaler_choice)
        for included_features, model_name, is_pipeline, estimator, param_grid, scaler_choice in jobs
    )
    print(f"\nTrained {len(jobs)} models in {time.monotonic() - start_time:.1f}s")

    results.sort(key=lambda result: result["avg_mae"])

    top_by_group = {}
    for result in results:
        key = (result["model"], result.get("scaler"))
        top_by_group.setdefault(key, []).append(result)

    for (model_name, scaler), group_results in top_by_group.items():
        top_n = group_results[: train_config.top_n_per_model]
        label = model_name if scaler is None else f"{model_name} ({scaler} scaler)"
        print(f"\nTop {len(top_n)} {label} models:")
        for result in top_n:
            printable = {k: v for k, v in result.items() if k not in ("model", "_model")}
            print(f"  {printable}")

    best_by_num_features = {}
    for result in results:
        best_by_num_features.setdefault(len(result["features"]), result)

    print("\nBest combination per number of features:")
    for num_features in sorted(best_by_num_features):
        result = best_by_num_features[num_features]
        printable = {k: v for k, v in result.items() if k != "_model"}
        print(f"  {num_features} features: {printable}")

    out_df = pd.DataFrame(
        [
            {
                **result,
                "features": ",".join(result["features"]),
                "feature_importance": json.dumps(result["feature_importance"]),
            }
            for result in results
        ]
    )
    out_df = out_df[["model", "features", *hyperparam_names, "avg_mae", "feature_importance"]]
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved grid search results to {output_path}")

    return results


if __name__ == "__main__":
    grid_search()
