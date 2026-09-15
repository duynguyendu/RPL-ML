#!/usr/bin/env python3

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.data import FEATURE_COLUMNS, LABEL_COLUMN

NON_PORTABLE_MODELS = {"gp"}


def grid_search() -> list[dict]:
    import train_config

    train_csv = Path(train_config.data_dir) / "train.csv"
    output_path = Path(train_config.data_dir) / "grid_search.csv"
    test_size = 0.2

    df = pd.read_csv(train_csv)
    total_rows = len(df)
    df = df.dropna(subset=[LABEL_COLUMN])
    for col, sentinel in train_config.UNKNOWN_SENTINELS.items():
        if col in df.columns:
            df = df[df[col] != sentinel]
    print(f"Dropped {total_rows - len(df)} of {total_rows} rows (missing label or unknown feature)")
    if df.empty:
        raise ValueError(f"No fully-labeled rows in {train_csv}")
    y = df[LABEL_COLUMN]

    cv = ShuffleSplit(
        n_splits=len(train_config.seeds), test_size=test_size, random_state=0
    )

    model_configs = [
        (
            "lgbm",
            lgb.LGBMRegressor(
                verbosity=-1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                random_state=0,
                n_jobs=train_config.lgbm_n_jobs,
            ),
            train_config.param_grid,
        ),
        ("ridge", Ridge(), train_config.ridge_param_grid),
        ("dtree", DecisionTreeRegressor(random_state=0), train_config.dtree_param_grid),
        ("svr", LinearSVR(random_state=0), train_config.svr_param_grid),
        # ("gp", GaussianProcessRegressor(random_state=0), train_config.gp_param_grid),
    ]
    hyperparam_names = sorted({name for _, _, grid in model_configs for name in grid})

    results = []
    for num_included in range(train_config.min_features, len(FEATURE_COLUMNS) + 1):
        for included_features in combinations(FEATURE_COLUMNS, num_included):
            included_features = list(included_features)
            X = df[included_features]

            for model_name, estimator, param_grid in model_configs:
                search = GridSearchCV(
                    estimator,
                    param_grid,
                    scoring="neg_mean_absolute_error",
                    cv=cv,
                    n_jobs=train_config.n_jobs,
                )
                search.fit(X, y)

                avg_mae = -search.best_score_
                importances = getattr(search.best_estimator_, "feature_importances_", None)
                if importances is None:
                    importances = getattr(search.best_estimator_, "coef_", None)
                avg_importance = (
                    dict(zip(included_features, importances.tolist()))
                    if importances is not None
                    else None
                )

                results.append(
                    {
                        "model": model_name,
                        "features": included_features,
                        **search.best_params_,
                        "avg_mae": avg_mae,
                        "feature_importance": avg_importance,
                        "_model": search.best_estimator_,
                    }
                )
                print(
                    f"[{model_name}] include {included_features} {search.best_params_}: "
                    f"avg MAE over {len(train_config.seeds)} splits = {avg_mae:.4f}, "
                    f"feature_importance={avg_importance}"
                )

    results.sort(key=lambda result: result["avg_mae"])

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
