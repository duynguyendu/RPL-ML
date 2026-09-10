#!/usr/bin/env python3

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV, ShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.data import FEATURE_COLUMNS, LABEL_COLUMN


def grid_search() -> list[dict]:
    import train_config

    train_csv = Path(train_config.data_dir) / "train.csv"
    output_path = Path(train_config.data_dir) / "grid_search.csv"
    test_size = 0.2

    df = pd.read_csv(train_csv)
    df = df.dropna(subset=[LABEL_COLUMN])
    if df.empty:
        raise ValueError(f"No rows with a labeled {LABEL_COLUMN} in {train_csv}")
    y = df[LABEL_COLUMN]

    cv = ShuffleSplit(n_splits=len(train_config.seeds), test_size=test_size, random_state=0)
    hyperparam_names = list(train_config.param_grid.keys())

    results = []
    for num_included in range(1, len(FEATURE_COLUMNS) + 1):
        for included_features in combinations(FEATURE_COLUMNS, num_included):
            included_features = list(included_features)
            X = df[included_features]

            search = GridSearchCV(
                lgb.LGBMRegressor(
                    verbosity=-1,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=1,
                    random_state=0,
                ),
                train_config.param_grid,
                scoring="neg_mean_absolute_error",
                cv=cv,
            )
            search.fit(X, y)

            avg_mae = -search.best_score_
            avg_importance = dict(
                zip(included_features, search.best_estimator_.feature_importances_.tolist())
            )
            results.append(
                {
                    "features": included_features,
                    **search.best_params_,
                    "avg_mae": avg_mae,
                    "feature_importance": avg_importance,
                    "_model": search.best_estimator_,
                }
            )
            print(
                f"include {included_features} {search.best_params_}: "
                f"avg MAE over {len(train_config.seeds)} splits = {avg_mae:.4f}"
            )

    results.sort(key=lambda result: result["avg_mae"])

    print("\nTop 5 smallest avg MAE:")
    for result in results[:5]:
        print(f"  { {k: v for k, v in result.items() if k != '_model'} }")

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
    out_df = out_df[["features", *hyperparam_names, "avg_mae", "feature_importance"]]
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved grid search results to {output_path}")

    return results


if __name__ == "__main__":
    grid_search()
