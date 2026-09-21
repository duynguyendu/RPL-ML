#!/usr/bin/env python3

import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
) -> dict:
    X = df[included_features]
    if is_pipeline:
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
        best_params["scaler"] = "minmax+standard"

    best_estimator = search.best_estimator_
    inner_estimator = best_estimator.named_steps["model"] if is_pipeline else best_estimator

    avg_mae = -search.best_score_
    importances = getattr(inner_estimator, "feature_importances_", None)
    if importances is None:
        importances = getattr(inner_estimator, "coef_", None)
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


def plot_feature_importance(top_by_group: dict, models_dir: Path) -> None:
    """Plot the best model's feature importance for each (model, scaler) group."""
    for (model_name, scaler), group_results in top_by_group.items():
        importance = group_results[0].get("feature_importance")
        if not importance:
            continue
        label = model_name if scaler is None else f"{model_name} ({scaler} scaler)"
        features, values = zip(
            *sorted(importance.items(), key=lambda kv: abs(kv[1]))
        )

        fig, ax = plt.subplots(figsize=(6, 0.4 * len(features) + 1))
        ax.barh(features, values, color="#4363d8")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature importance - best {label} model")
        fig.tight_layout()

        out_path = models_dir / f"feature_importance_{model_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


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
                learning_rate=0.1,
                random_state=0,
                n_jobs=train_config.lgbm_n_jobs,
            ),
            train_config.lgbm_param_grid,
        ),
        (
            "ridge",
            True,
            Pipeline(
                [
                    ("minmax", MinMaxScaler(feature_range=(0, 65535))),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
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
            Pipeline(
                [
                    ("minmax", MinMaxScaler(feature_range=(0, 65535))),
                    ("scaler", StandardScaler()),
                    ("model", LinearSVR(random_state=0)),
                ]
            ),
            train_config.svr_param_grid,
        ),
        (
            "xgboost",
            False,
            xgb.XGBRegressor(learning_rate=0.1, random_state=0, n_jobs=1, verbosity=0),
            train_config.xgb_param_grid,
        ),
        (
            "catboost",
            False,
            CatBoostRegressor(learning_rate=0.1, random_state=0, thread_count=1, verbose=False),
            train_config.catboost_param_grid,
        ),
        (
            "rf",
            False,
            RandomForestRegressor(random_state=0, n_jobs=1),
            train_config.rf_param_grid,
        ),
    ]
    hyperparam_names = sorted({name for _, _, _, grid in model_configs for name in grid} | {"scaler"})

    jobs = [
        (train_config.FIXED_FEATURES + list(dynamic_subset), model_name, is_pipeline, estimator, param_grid)
        for dynamic_subset in combinations(train_config.DYNAMIC_FEATURES, train_config.min_dynamic_features)
        for model_name, is_pipeline, estimator, param_grid in model_configs
    ]
    if not jobs:
        raise ValueError(
            "No feature combinations to search: "
            f"min_dynamic_features={train_config.min_dynamic_features}, "
            f"FIXED_FEATURES={train_config.FIXED_FEATURES}, "
            f"DYNAMIC_FEATURES={train_config.DYNAMIC_FEATURES} "
            "-- check exclude_features/min_dynamic_features in train_config."
        )

    start_time = time.monotonic()
    results = Parallel(n_jobs=train_config.n_jobs)(
        delayed(_fit_one)(df, y, cv, included_features, model_name, is_pipeline, estimator, param_grid)
        for included_features, model_name, is_pipeline, estimator, param_grid in jobs
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

    plot_feature_importance(top_by_group, models_dir)

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
