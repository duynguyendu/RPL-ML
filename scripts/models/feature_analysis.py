#!/usr/bin/env python3
"""Correlation, permutation importance and SHAP analysis of every training feature.

analyse_correlation() needs no model: Spearman correlation between features
(to find redundant ones, which also split credit in permutation importance)
and each feature's Spearman correlation / mutual information with PDR.

For analyse_features(), for each model type, the best hyperparameters from grid_search.csv are refit
on ALL of train_config.FEATURE_COLUMNS (not just the feature set the grid
search picked), then analysed on a held-out split:

- permutation importance: how much the held-out MAE grows when one feature's
  values are shuffled -- model-agnostic, so comparable across model types
  (unlike dtree's impurity importance or lgbm's split counts);
- SHAP: per-row contribution of each feature to the predicted PDR, showing
  both how much a feature matters and in which direction it pushes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.data import LABEL_COLUMN, MAXUINT16
from models.train import _load_training_data, _with_params, get_model_configs


def _to_maep(mae):
    return mae / MAXUINT16 * 100


def _best_params(grid_search_df: pd.DataFrame, model_name: str, param_grid: dict) -> tuple[dict, int | None]:
    """Best row's hyperparameters and data seed for ``model_name`` in grid_search.csv."""
    rows = grid_search_df[grid_search_df["model"] == model_name]
    if rows.empty:
        raise ValueError(f"No {model_name} rows in grid_search.csv -- run the grid search first")
    best = rows.loc[rows["avg_mae"].idxmin()]

    params = {}
    for name in param_grid:
        value = best[name]
        if isinstance(value, np.generic):
            value = value.item()
        # CSV round-trips ints as floats when other models leave the column empty
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        params[name] = value
    data_seed = None if pd.isna(best.get("data_seed")) else int(best["data_seed"])
    return params, data_seed


def _plot_permutation(model_name: str, features: list[str], importances: np.ndarray, out_path: Path) -> None:
    order = np.argsort(importances.mean(axis=1))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(features) + 1.5))
    ax.boxplot(
        [_to_maep(importances[i]) for i in order],
        orientation="horizontal",
        tick_labels=[features[i] for i in order],
    )
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Increase in held-out MAEP when shuffled (percentage points)")
    ax.set_title(f"Permutation importance - {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _save_shap_plot(plot_fn, explanation, title: str, out_path: Path) -> None:
    plt.figure()
    plot_fn(explanation, max_display=explanation.shape[1], show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel(f"{fig.axes[0].get_xlabel()} [PDR, 0-{MAXUINT16} scale]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_correlation(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(0.7 * len(corr) + 2, 0.6 * len(corr) + 1.5))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iat[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(value) > 0.6 else "black",
            )
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Feature correlation (Spearman)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def analyse_correlation() -> pd.DataFrame:
    """Spearman correlation between features, their clusters, and relation to PDR."""
    import train_config

    models_dir = Path(train_config.data_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    features = train_config.FEATURE_COLUMNS

    df = _load_training_data(train_config)
    if len(df) > train_config.max_train_rows:
        df = df.sample(n=train_config.max_train_rows, random_state=0)

    # a constant column has no rank correlation (NaN); treat it as uncorrelated
    corr = df[[*features, LABEL_COLUMN]].corr(method="spearman").fillna(0.0)
    for name in corr.columns:
        corr.loc[name, name] = 1.0
    corr.to_csv(models_dir / "feature_correlation.csv")
    _plot_correlation(corr, models_dir / "feature_correlation.png")

    feature_corr = corr.loc[features, features]
    pairs = [
        (a, b, feature_corr.at[a, b])
        for i, a in enumerate(features)
        for b in features[i + 1 :]
        if abs(feature_corr.at[a, b]) >= train_config.corr_threshold
    ]
    pairs.sort(key=lambda pair: -abs(pair[2]))
    print(f"\nFeature pairs with |Spearman| >= {train_config.corr_threshold}:")
    for a, b, rho in pairs:
        print(f"  {a} ~ {b}: {rho:+.3f}")
    if not pairs:
        print("  (none)")

    # cluster on distance 1 - |rho|; "complete" linkage cuts at the threshold
    # so every pair inside a cluster is at least that correlated
    distance = squareform(1 - feature_corr.abs().to_numpy(), checks=False)
    linkage = hierarchy.linkage(distance, method="complete")
    cluster_ids = hierarchy.fcluster(linkage, t=1 - train_config.corr_threshold, criterion="distance")
    clusters = {}
    for feature, cluster_id in zip(features, cluster_ids):
        clusters.setdefault(cluster_id, []).append(feature)
    print(f"\nFeature clusters (all members |Spearman| >= {train_config.corr_threshold}):")
    for members in clusters.values():
        if len(members) > 1:
            print(f"  {members}")

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(features) + 1.5))
    hierarchy.dendrogram(
        linkage, labels=features, orientation="left", color_threshold=1 - train_config.corr_threshold, ax=ax
    )
    ax.axvline(1 - train_config.corr_threshold, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("1 - |Spearman correlation| (complete linkage)")
    ax.set_title("Feature clustering")
    fig.tight_layout()
    out_path = models_dir / "feature_dendrogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    mutual_info = mutual_info_regression(df[features], df[LABEL_COLUMN], random_state=0)
    relation = pd.DataFrame(
        {
            "feature": features,
            "spearman_with_pdr": corr.loc[features, LABEL_COLUMN].to_numpy(),
            "mutual_info_with_pdr": mutual_info,
            "cluster": cluster_ids,
        }
    ).sort_values("mutual_info_with_pdr", ascending=False)
    print(f"\nRelation of each feature to {LABEL_COLUMN} (mutual information in nats):")
    print(relation.round(3).to_string(index=False))
    relation.to_csv(models_dir / "feature_relation.csv", index=False)
    return relation


def analyse_features() -> pd.DataFrame:
    import train_config

    models_dir = Path(train_config.data_dir) / "models"
    grid_search_df = pd.read_csv(models_dir / "grid_search.csv")
    features = train_config.FEATURE_COLUMNS

    df = _load_training_data(train_config)
    rows = []
    for model_name, is_pipeline, estimator, param_grid in get_model_configs(train_config):
        params, data_seed = _best_params(grid_search_df, model_name, param_grid)

        # same row cap as the grid search, sampled with the best row's data seed
        sample = df
        if len(df) > train_config.max_train_rows:
            sample = df.sample(n=train_config.max_train_rows, random_state=data_seed or 0)
        X_train, X_test, y_train, y_test = train_test_split(
            sample[features], sample[LABEL_COLUMN], test_size=0.2, random_state=0
        )

        model = _with_params(estimator, is_pipeline, params)
        model.fit(X_train, y_train)
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        print(
            f"\n=== {model_name} on all {len(features)} features, params={params}, "
            f"data_seed={data_seed}: held-out MAE={test_mae:.1f} (MAEP={_to_maep(test_mae):.3f}%) ==="
        )

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=train_config.permutation_repeats,
            random_state=0,
            n_jobs=train_config.n_jobs,
        )
        _plot_permutation(
            model_name, features, perm.importances, models_dir / f"permutation_importance_{model_name}.png"
        )

        X_shap = X_test.sample(n=min(len(X_test), train_config.shap_sample_rows), random_state=0)
        if is_pipeline:
            explainer = shap.Explainer(model.predict, shap.sample(X_train, 100, random_state=0))
        else:
            explainer = shap.TreeExplainer(model)
        explanation = explainer(X_shap)
        _save_shap_plot(
            shap.plots.beeswarm, explanation, f"SHAP values - {model_name}", models_dir / f"shap_beeswarm_{model_name}.png"
        )
        _save_shap_plot(
            shap.plots.bar, explanation, f"Mean |SHAP| - {model_name}", models_dir / f"shap_bar_{model_name}.png"
        )
        mean_abs_shap = np.abs(explanation.values).mean(axis=0)

        for i, feature in enumerate(features):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "perm_mae_increase": perm.importances_mean[i],
                    "perm_mae_increase_std": perm.importances_std[i],
                    "perm_maep_increase": _to_maep(perm.importances_mean[i]),
                    "mean_abs_shap": mean_abs_shap[i],
                    "mean_abs_shap_maep": _to_maep(mean_abs_shap[i]),
                }
            )

    result = pd.DataFrame(rows).sort_values(["model", "perm_mae_increase"], ascending=[True, False])
    for model_name, group in result.groupby("model", sort=False):
        print(f"\nFeature importance - {model_name} (MAEP points):")
        print(
            group[["feature", "perm_maep_increase", "mean_abs_shap_maep"]]
            .rename(columns={"perm_maep_increase": "permutation", "mean_abs_shap_maep": "mean |SHAP|"})
            .round(3)
            .to_string(index=False)
        )

    output_path = models_dir / "feature_analysis.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved feature analysis to {output_path}")
    return result


if __name__ == "__main__":
    analyse_correlation()
    analyse_features()
