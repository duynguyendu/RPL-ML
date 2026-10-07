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
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.data import LABEL_COLUMN, MAXUINT16, display_name
from models.to_c import compiled_size, convert_to_c_fixed, convert_to_c_linear, predict_fixed

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


def mae_to_maep(mae: float) -> float:
    """MAE as a percentage of the 0-65535 PDR scale."""
    return mae / MAXUINT16 * 100


def _cv_maes(X, y, splits: list, feature_idx: list[int], is_pipeline: bool, estimator, params_list: list[dict]) -> list[float]:
    """Cross-validated MAE of each hyperparameter set on one feature set / data seed.

    X and y are plain numpy arrays so joblib memory-maps them once and shares
    them between workers, instead of pickling a DataFrame for every job.
    """
    X = X[:, feature_idx]
    maes = []
    for params in params_list:
        fold_maes = []
        for train_idx, test_idx in splits:
            model = _with_params(estimator, is_pipeline, params)
            model.fit(X[train_idx], y[train_idx])
            fold_maes.append(np.abs(model.predict(X[test_idx]) - y[test_idx]).mean())
        maes.append(float(np.mean(fold_maes)))
    return maes


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
    model.fit(df[features], df[LABEL_COLUMN])
    inner_estimator = model.named_steps["model"] if is_pipeline else model
    size = _ported_size(model) or {}

    return {
        "model": model_name,
        "features": features,
        "data_seed": data_seed,
        **params,
        **({"scaler": "minmax+standard"} if is_pipeline else {}),
        "avg_mae": avg_mae,
        "avg_maep": mae_to_maep(avg_mae),
        "text": size.get("text"),
        "data": size.get("data"),
        "bss": size.get("bss"),
        "flash": size.get("flash"),
        "feature_importance": _feature_importance(inner_estimator, features),
        "_model": model,
    }


def print_pdr_ppm_summary(df: pd.DataFrame) -> None:
    """Print the PDR-category breakdown, and avg ppm per hop_count within each.

    PDR categories are percentages of the label's 0-65535 scale:
    low = 0-30%, medium = >30-65%, high = >65%.
    """
    pdr_pct = df[LABEL_COLUMN] / MAXUINT16 * 100
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


def plot_feature_importance(results: list[dict], models_dir: Path) -> None:
    """Plot the best model's feature importance for each model type.

    ``results`` must be sorted best-first.
    """
    best_by_model = {}
    for result in results:
        best_by_model.setdefault(result["model"], result)

    for model_name, result in best_by_model.items():
        importance = result.get("feature_importance")
        if not importance:
            continue
        scaler = result.get("scaler")
        label = model_name if scaler is None else f"{model_name} ({scaler} scaler)"
        features, values = zip(
            *sorted(importance.items(), key=lambda kv: abs(kv[1]))
        )

        fig, ax = plt.subplots(figsize=(6, 0.4 * len(features) + 1))
        ax.barh([display_name(f) for f in features], values, color="#4363d8")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature importance - best {label} model")
        fig.tight_layout()

        out_path = models_dir / f"feature_importance_{model_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_predicted_vs_actual(models: dict[str, dict], eval_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Predicted vs. actual PDR, one panel per model on shared axes, with a y = x line.

    ``models`` maps a model type to its result row (needs ``_model`` and ``features``).
    """
    if not models:
        return
    n_cols = 2
    n_rows = -(-len(models) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    actual = eval_df[LABEL_COLUMN].to_numpy() / MAXUINT16 * 100
    predicted_by_model = {
        model_name: result["_model"].predict(eval_df[result["features"]]) / MAXUINT16 * 100
        for model_name, result in models.items()
    }
    low = min(0.0, actual.min(), *(p.min() for p in predicted_by_model.values()))
    high = max(100.0, actual.max(), *(p.max() for p in predicted_by_model.values()))

    for ax, (model_name, predicted) in zip(axes.flat, predicted_by_model.items()):
        ax.scatter(actual, predicted, s=4, alpha=0.15, color="#4363d8", linewidths=0, rasterized=True)
        ax.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=1, label="y = x")
        maep = np.abs(predicted - actual).mean()
        ax.text(0.04, 0.96, f"{model_name}\nMAEP = {maep:.2f}%", transform=ax.transAxes, va="top")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.grid(color="#dddddd", linewidth=0.5)
    for ax in axes.flat[len(predicted_by_model):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Actual PDR (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted PDR (%)")
    axes.flat[0].legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
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
                    ("model", LinearSVR()),
                ]
            ),
            train_config.svr_param_grid,
        ),
    ]


def _stratified_samples(df: pd.DataFrame, train_config) -> dict:
    """One sample per training-data seed, mixing near-overloading and other rows.

    ``near_overloading_fraction`` of each sample comes from rows whose node is
    within ``near_overloading_hops`` radio hops of an overloading client; the
    rest comes from the other rows. The sample size is capped by
    ``max_train_rows`` and shrunk if either pool is too small to hold the
    ratio, so every seed's sample has the same length (required by the shared
    CV splits).
    """
    frac = train_config.near_overloading_fraction
    is_near = df["overloading_hops"] <= train_config.near_overloading_hops
    near, far = df[is_near], df[~is_near]

    n_total = min(train_config.max_train_rows, len(df))
    n_total = min(n_total, int(len(near) / frac))
    if frac < 1.0:
        n_total = min(n_total, int(len(far) / (1.0 - frac)))
    n_near = round(n_total * frac)
    n_far = n_total - n_near
    if n_total == 0:
        raise ValueError(
            f"Cannot sample {frac:.0%} near-overloading rows: {len(near)} near rows "
            f"(<= {train_config.near_overloading_hops} hops), {len(far)} other rows"
        )

    print(
        f"Training on {n_total} of {len(df)} rows per data seed "
        f"(seeds={train_config.training_data_seeds}): {n_near} of {len(near)} rows within "
        f"{train_config.near_overloading_hops} hop(s) of an overloading client, "
        f"{n_far} of {len(far)} other rows"
    )
    return {
        data_seed: pd.concat(
            [
                near.sample(n=n_near, random_state=data_seed),
                far.sample(n=n_far, random_state=data_seed),
            ]
        ).sample(frac=1.0, random_state=data_seed)
        for data_seed in train_config.training_data_seeds
    }


def _hold_out_last_topo_seed(df: pd.DataFrame, train_config) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split into (train+validation, test) rows, testing on the highest topo_seed.

    Returns ``(df, None)`` when disabled or when there is only one topo seed.
    """
    if not train_config.hold_out_last_topo_seed:
        return df, None
    seeds = sorted(df["topo_seed"].unique().tolist())
    if len(seeds) < 2:
        print(f"Warning: only topo seed(s) {seeds} -- no unseen-topology test")
        return df, None
    is_test = df["topo_seed"] == seeds[-1]
    print(
        f"Unseen-topology split: training/validating on topo seeds {seeds[:-1]} "
        f"({(~is_test).sum()} rows), testing on topo seed {seeds[-1]} ({is_test.sum()} rows)"
    )
    return df[~is_test], df[is_test]


def _test_mae(model, test_df: pd.DataFrame, features: list[str]) -> float:
    return float(np.abs(model.predict(test_df[features]) - test_df[LABEL_COLUMN].to_numpy()).mean())


def grid_search() -> list[dict]:
    import train_config

    models_dir = Path(train_config.data_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "grid_search.csv"
    test_size = 0.2

    df = _load_training_data(train_config)
    df, test_df = _hold_out_last_topo_seed(df, train_config)
    if train_config.near_overloading_fraction > 0.0 and "overloading_hops" not in df:
        print(
            "Warning: train.csv has no overloading_hops column (re-run process_data) "
            "-- falling back to plain random sampling"
        )
    if 0.0 < train_config.near_overloading_fraction <= 1.0 and "overloading_hops" in df:
        data_by_seed = _stratified_samples(df, train_config)
    elif len(df) > train_config.max_train_rows:
        data_by_seed = {
            data_seed: df.sample(n=train_config.max_train_rows, random_state=data_seed)
            for data_seed in train_config.training_data_seeds
        }
        print(
            f"Training on {train_config.max_train_rows} of {len(df)} rows per data seed "
            f"(seeds={train_config.training_data_seeds})"
        )
    else:
        # everything fits under the cap -- no sampling, so data seeds are moot
        data_by_seed = {None: df}
        print(f"Training on all {len(df)} rows (no data-seed sampling)")

    all_features = train_config.FEATURE_COLUMNS
    arrays_by_seed = {
        data_seed: (sample[all_features].to_numpy(), sample[LABEL_COLUMN].to_numpy())
        for data_seed, sample in data_by_seed.items()
    }

    cv = ShuffleSplit(
        n_splits=len(train_config.seeds), test_size=test_size, random_state=0
    )

    model_configs = get_model_configs(train_config)
    configs_by_name = {model_name: (is_pipeline, estimator) for model_name, is_pipeline, estimator, _ in model_configs}
    hyperparam_names = sorted({name for _, _, _, grid in model_configs for name in grid})
    if any(is_pipeline for _, is_pipeline, _, _ in model_configs):
        hyperparam_names.append("scaler")
        hyperparam_names.sort()

    feature_sets = [
        train_config.FIXED_FEATURES + list(dynamic_subset)
        for dynamic_subset in combinations(train_config.DYNAMIC_FEATURES, train_config.min_dynamic_features)
    ]
    if not feature_sets:
        raise ValueError(
            "No feature combinations to search: "
            f"min_dynamic_features={train_config.min_dynamic_features}, "
            f"FIXED_FEATURES={train_config.FIXED_FEATURES}, "
            f"DYNAMIC_FEATURES={train_config.DYNAMIC_FEATURES} "
            "-- check exclude_features/min_dynamic_features in train_config."
        )

    # one CV run per (feature set, data seed, model, hyperparameter set); every
    # later stage reuses these scores instead of cross-validating again
    cv_runs = [
        {"model": model_name, "features": features, "data_seed": data_seed, "params": params}
        for features in feature_sets
        for data_seed in data_by_seed
        for model_name, _, _, param_grid in model_configs
        for params in ParameterGrid(param_grid)
    ]
    # the same splits for every run (and every data seed: all samples have
    # the same length), so all hyperparameter sets are compared fairly
    splits = list(cv.split(next(iter(arrays_by_seed.values()))[0]))

    # batch runs sharing (feature set, data seed, model) into chunks: small
    # enough to keep every worker busy, big enough to amortise dispatch cost
    n_workers = joblib.cpu_count() if train_config.n_jobs < 0 else train_config.n_jobs
    chunk_size = max(1, len(cv_runs) // (n_workers * 4))
    chunks = []
    for run in cv_runs:
        key = (run["model"], tuple(run["features"]), run["data_seed"])
        if not chunks or chunks[-1][0] != key or len(chunks[-1][1]) >= chunk_size:
            chunks.append((key, []))
        chunks[-1][1].append(run)

    start_time = time.monotonic()
    chunk_maes = Parallel(n_jobs=train_config.n_jobs)(
        delayed(_cv_maes)(
            *arrays_by_seed[runs[0]["data_seed"]],
            splits,
            [all_features.index(f) for f in runs[0]["features"]],
            *configs_by_name[runs[0]["model"]],
            [run["params"] for run in runs],
        )
        for _, runs in chunks
    )
    for (_, runs), maes in zip(chunks, chunk_maes):
        for run, mae in zip(runs, maes):
            run["avg_mae"] = mae
    print(f"\nCross-validated {len(cv_runs)} configurations in {time.monotonic() - start_time:.1f}s")

    # stage 1: best feature set per model; per (feature set, data seed, model)
    # keep the best hyperparameters for the report
    feature_results = {}
    for run in cv_runs:
        key = (run["model"], tuple(run["features"]), run["data_seed"])
        if key not in feature_results or run["avg_mae"] < feature_results[key]["avg_mae"]:
            feature_results[key] = run
    feature_results = sorted(feature_results.values(), key=lambda run: run["avg_mae"])
    _save_results(
        [{**run, **run["params"], "avg_maep": mae_to_maep(run["avg_mae"])} for run in feature_results],
        models_dir / "feature_search.csv",
        ["model", "features", "data_seed", *hyperparam_names, "avg_mae", "avg_maep"],
    )

    best_features = {}
    for run in feature_results:
        best_features.setdefault(run["model"], run["features"])
    print("\nBest feature set per model:")
    for model_name, features in best_features.items():
        print(f"  {model_name}: {features}")

    # stage 2: every hyperparameter set on that feature set, with its best
    # data seed; refit on that seed and measure the ported C size
    best_seed_by_config = {}
    for run in cv_runs:
        if run["features"] != best_features[run["model"]]:
            continue
        key = (run["model"], tuple(sorted(run["params"].items())))
        if key not in best_seed_by_config or run["avg_mae"] < best_seed_by_config[key]["avg_mae"]:
            best_seed_by_config[key] = run

    start_time = time.monotonic()
    results = Parallel(n_jobs=train_config.n_jobs)(
        delayed(_refit_and_size)(
            data_by_seed[run["data_seed"]],
            run["features"],
            run["data_seed"],
            run["model"],
            *configs_by_name[run["model"]],
            run["params"],
            run["avg_mae"],
        )
        for run in best_seed_by_config.values()
    )
    print(f"\nRefit, ported and compiled {len(results)} models in {time.monotonic() - start_time:.1f}s")

    results.sort(key=lambda result: result["avg_mae"])
    if test_df is not None:
        # scored only after CV-based selection, so the held-out topology never
        # influences which model is picked
        for result in results:
            test_mae = _test_mae(result["_model"], test_df, result["features"])
            result["test_mae"] = test_mae
            result["test_maep"] = mae_to_maep(test_mae)
    plot_feature_importance(results, models_dir)

    _save_results(
        results,
        output_path,
        [
            "model", "features", "data_seed", *hyperparam_names,
            "avg_mae", "avg_maep", "test_mae", "test_maep",
            "text", "data", "bss", "flash", "feature_importance",
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
                **({"test MAEP (%)": round(r["test_maep"], 3)} if "test_maep" in r else {}),
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


def print_best_models(best_by_model: dict[str, dict], train_df: pd.DataFrame, test_df: pd.DataFrame | None) -> None:
    """Per model type: hyperparameters, and MAE/MAEP on the training data, on the
    test data, and on the test data using the quantised (ported C) model.
    """
    import train_config

    param_names = {model_name: list(grid) for model_name, _, _, grid in get_model_configs(train_config)}
    evaluations = [("train", train_df, False)]
    if test_df is not None:
        evaluations += [("test", test_df, False), ("test quantised", test_df, True)]

    rows = []
    for model_name, result in best_by_model.items():
        model = result["_model"]
        row = {
            "model": model_name,
            "params": ", ".join(f"{name}={result[name]}" for name in param_names[model_name]),
        }
        for label, eval_df, quantised in evaluations:
            predicted = predict_fixed(model, eval_df) if quantised else model.predict(eval_df[result["features"]])
            if predicted is None:
                row[f"{label} MAE"] = row[f"{label} MAEP (%)"] = None
                continue
            mae = float(np.abs(predicted - eval_df[LABEL_COLUMN].to_numpy()).mean())
            row[f"{label} MAE"] = round(mae, 1)
            row[f"{label} MAEP (%)"] = round(mae_to_maep(mae), 3)
        rows.append(row)

    print("\nBest model per type (training data = all non-held-out rows, in-sample):")
    print(pd.DataFrame(rows).to_string(index=False))
    if test_df is None:
        print("  (no held-out test set -- hold_out_last_topo_seed is off or there is one topo seed)")


def _save_results(results: list[dict], output_path: Path, desired_columns: list[str]) -> None:
    out_df = pd.DataFrame(
        [
            {
                **result,
                "features": ",".join(result["features"]),
                "feature_importance": json.dumps(result.get("feature_importance")),
            }
            for result in results
        ]
    )
    out_df = out_df[[col for col in desired_columns if col in out_df.columns]]
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(out_df)} results to {output_path}")


if __name__ == "__main__":
    grid_search()
