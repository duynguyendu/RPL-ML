#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, ParameterGrid, ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.data import LABEL_COLUMN, MAXUINT16
from models.to_c import compiled_size, convert_to_c_fixed, convert_to_c_linear

NON_PORTABLE_MODELS = set()


def _feature_importance(inner_estimator, features: list[str]) -> dict | None:
    importances = getattr(inner_estimator, "feature_importances_", None)
    if importances is None:
        importances = getattr(inner_estimator, "coef_", None)
    return dict(zip(features, importances.tolist())) if importances is not None else None


def _with_params(estimator, is_pipeline: bool, params: dict):
    if is_pipeline:
        params = {f"model__{key}": value for key, value in params.items()}
    return clone(estimator).set_params(**params)


def _cv_mae(
    df: pd.DataFrame,
    y: pd.Series,
    cv: ShuffleSplit,
    features: list[str],
    data_seed: int | None,
    model_name: str,
    is_pipeline: bool,
    estimator,
    params: dict,
) -> dict:
    """Cross-validated MAE of one hyperparameter set on one data seed's sample."""
    scores = cross_val_score(
        _with_params(estimator, is_pipeline, params),
        df[features],
        y,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,
    )
    return {"model": model_name, "params": params, "data_seed": data_seed, "avg_mae": -scores.mean()}


def _ported_size(model) -> dict[str, int] | None:
    """Port a fitted model to C (fixed-point, else linear) and compile it for the mote."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.joblib"
        joblib.dump(model, model_path)
        converted = convert_to_c_fixed(model_path, tmp) or convert_to_c_linear(model_path, tmp)
        if converted is None:
            return None
        return compiled_size(converted[0])


def _refit_and_size(
    df: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    data_seed: int | None,
    model_name: str,
    is_pipeline: bool,
    estimator,
    params: dict,
    avg_mae: float,
) -> dict:
    """Refit one hyperparameter set on its best data seed and measure its compiled size."""
    model = _with_params(estimator, is_pipeline, params)
    model.fit(df[features], y)
    inner_estimator = model.named_steps["model"] if is_pipeline else model
    size = _ported_size(model) or {}

    return {
        "model": model_name,
        "features": features,
        "data_seed": data_seed,
        **params,
        **({"scaler": "minmax+standard"} if is_pipeline else {}),
        "avg_mae": avg_mae,
        "avg_maep": avg_mae / MAXUINT16 * 100,
        "text": size.get("text"),
        "data": size.get("data"),
        "bss": size.get("bss"),
        "flash": size.get("flash"),
        "feature_importance": _feature_importance(inner_estimator, features),
        "_model": model,
    }


def _fit_one(
    df: pd.DataFrame,
    y: pd.Series,
    cv: ShuffleSplit,
    included_features: list[str],
    data_seed: int,
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
    avg_importance = _feature_importance(inner_estimator, included_features)

    return {
        "model": model_name,
        "features": included_features,
        "data_seed": data_seed,
        **best_params,
        "avg_mae": avg_mae,
        # MAE as a percentage of the 0-65535 PDR scale
        "avg_maep": avg_mae / MAXUINT16 * 100,
        "feature_importance": avg_importance,
        "_model": best_estimator,
    }


def print_pdr_ppm_summary(df: pd.DataFrame) -> None:
    """Print the PDR-category breakdown, and avg ppm per hop_count within each.

    PDR categories are percentages of the label's 0-65535 scale:
    low = 0-30%, medium = >30-65%, high = >65%.
    """
    pdr_pct = df[LABEL_COLUMN] / 65535 * 100
    category = pd.cut(
        pdr_pct, bins=[-0.01, 30, 65, 100.01], labels=["low", "medium", "high"]
    )

    print("\nPDR category breakdown (low=0-30%, medium=30-65%, high=>65%):")
    counts = category.value_counts(normalize=True).mul(100)
    for cat in ["low", "medium", "high"]:
        print(f"  {cat}: {counts.get(cat, 0.0):.2f}%")

    print("\nAverage ppm by hop_count within each PDR category:")
    avg_ppm_by_hop = (
        df.assign(pdr_category=category)
        .groupby(["pdr_category", "hop_count"], observed=True)["ppm"]
        .mean()
    )
    for (cat, hop_count), avg_ppm in avg_ppm_by_hop.items():
        print(f"  {cat}, hop_count={hop_count}: avg ppm={avg_ppm:.2f}")


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


def _load_training_data(train_config) -> pd.DataFrame:
    """Load train.csv, dropping unlabeled rows and (if configured) unknown-sentinel rows."""
    train_csv = Path(train_config.data_dir) / "train.csv"
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
    # if not dropped_rows.empty:
    #     print(dropped_rows.head(100).to_string())
    if df.empty:
        raise ValueError(f"No fully-labeled rows in {train_csv}")
    return df


def analyse_data() -> None:
    import train_config

    df = _load_training_data(train_config)
    # print_pdr_ppm_summary(df)


def get_model_configs(train_config) -> list[tuple]:
    """(model_name, is_pipeline, unfitted estimator, param_grid) for each model type."""
    return [
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
        # (
        #     "ridge",
        #     True,
        #     Pipeline(
        #         [
        #             ("minmax", MinMaxScaler(feature_range=(0, 65535))),
        #             ("scaler", StandardScaler()),
        #             ("model", Ridge()),
        #         ]
        #     ),
        #     train_config.ridge_param_grid,
        # ),
        (
            "dtree",
            False,
            DecisionTreeRegressor(random_state=0),
            train_config.dtree_param_grid,
        ),
        # (
        #     "svr",
        #     True,
        #     Pipeline(
        #         [
        #             ("minmax", MinMaxScaler(feature_range=(0, 65535))),
        #             ("scaler", StandardScaler()),
        #             ("model", LinearSVR(random_state=0)),
        #         ]
        #     ),
        #     train_config.svr_param_grid,
        # ),
    ]


def grid_search() -> list[dict]:
    import train_config

    models_dir = Path(train_config.data_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "grid_search.csv"
    test_size = 0.2

    df = _load_training_data(train_config)
    if len(df) > train_config.max_train_rows:
        data_by_seed = {}
        for data_seed in train_config.training_data_seeds:
            sample = df.sample(n=train_config.max_train_rows, random_state=data_seed)
            data_by_seed[data_seed] = (sample, sample[LABEL_COLUMN])
        print(
            f"Training on {train_config.max_train_rows} of {len(df)} rows per data seed "
            f"(seeds={train_config.training_data_seeds})"
        )
    else:
        # everything fits under the cap -- no sampling, so data seeds are moot
        data_by_seed = {None: (df, df[LABEL_COLUMN])}
        print(f"Training on all {len(df)} rows (no data-seed sampling)")

    cv = ShuffleSplit(
        n_splits=len(train_config.seeds), test_size=test_size, random_state=0
    )

    model_configs = get_model_configs(train_config)
    hyperparam_names = sorted({name for _, _, _, grid in model_configs for name in grid})
    if any(is_pipeline for _, is_pipeline, _, _ in model_configs):
        hyperparam_names.append("scaler")
        hyperparam_names.sort()

    jobs = [
        (train_config.FIXED_FEATURES + list(dynamic_subset), data_seed, model_name, is_pipeline, estimator, param_grid)
        for dynamic_subset in combinations(train_config.DYNAMIC_FEATURES, train_config.min_dynamic_features)
        for data_seed in data_by_seed
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

    # stage 1: pick the best feature set per model (best hyperparameters and
    # data seed for each feature set)
    start_time = time.monotonic()
    feature_results = Parallel(n_jobs=train_config.n_jobs)(
        delayed(_fit_one)(
            *data_by_seed[data_seed], cv, included_features, data_seed, model_name, is_pipeline, estimator, param_grid
        )
        for included_features, data_seed, model_name, is_pipeline, estimator, param_grid in jobs
    )
    print(f"\nFeature search: trained {len(jobs)} models in {time.monotonic() - start_time:.1f}s")
    feature_results.sort(key=lambda result: result["avg_mae"])
    _save_results(
        feature_results,
        models_dir / "feature_search.csv",
        ["model", "features", "data_seed", *hyperparam_names, "avg_mae", "avg_maep", "feature_importance"],
    )

    best_features = {}
    for result in feature_results:
        best_features.setdefault(result["model"], result["features"])
    print("\nBest feature set per model:")
    for model_name, features in best_features.items():
        print(f"  {model_name}: {features}")

    # stage 2: every hyperparameter set on that feature set, scored on each
    # data seed; keep the best seed, refit on it and measure the ported C size
    cv_jobs = [
        (best_features[model_name], data_seed, model_name, is_pipeline, estimator, params)
        for model_name, is_pipeline, estimator, param_grid in model_configs
        for params in ParameterGrid(param_grid)
        for data_seed in data_by_seed
    ]
    start_time = time.monotonic()
    cv_results = Parallel(n_jobs=train_config.n_jobs)(
        delayed(_cv_mae)(*data_by_seed[data_seed], cv, features, data_seed, model_name, is_pipeline, estimator, params)
        for features, data_seed, model_name, is_pipeline, estimator, params in cv_jobs
    )

    best_seed_by_config = {}
    for cv_result in cv_results:
        key = (cv_result["model"], tuple(sorted(cv_result["params"].items())))
        best = best_seed_by_config.get(key)
        if best is None or cv_result["avg_mae"] < best["avg_mae"]:
            best_seed_by_config[key] = cv_result

    configs_by_name = {model_name: (is_pipeline, estimator) for model_name, is_pipeline, estimator, _ in model_configs}
    results = Parallel(n_jobs=train_config.n_jobs)(
        delayed(_refit_and_size)(
            *data_by_seed[best["data_seed"]],
            best_features[best["model"]],
            best["data_seed"],
            best["model"],
            *configs_by_name[best["model"]],
            best["params"],
            best["avg_mae"],
        )
        for best in best_seed_by_config.values()
    )
    print(
        f"\nHyperparameter search: {len(cv_jobs)} CV runs, {len(results)} models ported "
        f"and compiled in {time.monotonic() - start_time:.1f}s"
    )

    results.sort(key=lambda result: result["avg_mae"])

    top_by_group = {}
    for result in results:
        key = (result["model"], result.get("scaler"))
        top_by_group.setdefault(key, []).append(result)

    plot_feature_importance(top_by_group, models_dir)

    _save_results(
        results,
        output_path,
        [
            "model", "features", "data_seed", *hyperparam_names,
            "avg_mae", "avg_maep", "text", "data", "bss", "flash", "feature_importance",
        ],
    )

    best_maep = results[0]["avg_maep"]
    near_best = [r for r in results if r["avg_maep"] <= best_maep + train_config.maep_tolerance]
    summary = pd.DataFrame(
        [
            {
                "model": r["model"],
                "features": ",".join(r["features"]),
                "data_seed": r["data_seed"],
                "params": ", ".join(f"{name}={r[name]}" for name in hyperparam_names if name in r),
                "MAE": round(r["avg_mae"], 1),
                "MAEP (%)": round(r["avg_maep"], 3),
                "flash (B)": r["flash"],
            }
            for r in near_best
        ]
    )
    print(
        f"\n{len(near_best)} models with MAEP within {train_config.maep_tolerance} "
        f"percentage point(s) of the best ({best_maep:.3f}%):"
    )
    print(summary.to_string(index=False))

    return results


def select_for_porting(results: list[dict], maep_tolerance: float) -> dict[str, dict]:
    """Per model type, the smallest-flash model within ``maep_tolerance``
    percentage points of that type's best MAEP (ties -> lower MAE).

    Falls back to the lowest-MAE model when no size was measured (e.g.
    msp430-gcc isn't installed).
    """
    by_model = {}
    for result in results:
        by_model.setdefault(result["model"], []).append(result)

    selected = {}
    for model_name, model_results in by_model.items():
        best_maep = min(r["avg_maep"] for r in model_results)
        candidates = [
            r for r in model_results
            if r["avg_maep"] <= best_maep + maep_tolerance and r["flash"] is not None
        ]
        if candidates:
            selected[model_name] = min(candidates, key=lambda r: (r["flash"], r["avg_mae"]))
        else:
            selected[model_name] = min(model_results, key=lambda r: r["avg_mae"])
    return selected


def _save_results(results: list[dict], output_path: Path, desired_columns: list[str]) -> None:
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
    out_df = out_df[[col for col in desired_columns if col in out_df.columns]]
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(out_df)} results to {output_path}")

if __name__ == "__main__":
    grid_search()
