#!/usr/bin/env python3

import os
import sys
import tempfile
from pathlib import Path

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_config
from models.data import process_data
from models.feature_analysis import analyse_correlation, analyse_features
from models.to_c import convert_to_c_fixed, measure_size, verify_fixed
from models.train import (
    NON_PORTABLE_MODELS,
    _hold_out_last_topo_seed,
    _load_training_data,
    analyse_data,
    grid_search,
    plot_predicted_vs_actual,
    select_for_porting,
)

# model types ported to rpl-lite's mlof-<name>.{c,h}
EXPORTED_MODELS = ("dtree", "lgbm")


def export_model(model_name: str, model_path: Path) -> None:
    """Port a fitted tree model to rpl-lite's mlof-<model_name>.{c,h}; nothing else is written."""
    func_name = f"mlof_predict_pdr_{model_name}"
    with tempfile.TemporaryDirectory() as tmp:
        c_path, h_path = convert_to_c_fixed(str(model_path), tmp, func_name)
        verify_fixed(str(model_path), str(train_config.data_dir))
        measure_size(c_path)

        train_config.rpl_lite_dir.mkdir(parents=True, exist_ok=True)
        dest_c = train_config.rpl_lite_dir / f"mlof-{model_name}.c"
        dest_h = train_config.rpl_lite_dir / f"mlof-{model_name}.h"
        dest_c.write_text(c_path.read_text().replace(f'"{func_name}.h"', f'"mlof-{model_name}.h"'))
        dest_h.write_text(h_path.read_text())
    print(f"Exported to {dest_c} and {dest_h}")


def run_pipeline() -> None:
    models_dir = Path(train_config.data_dir) / "models"

    if train_config.is_processing_data:
        process_data(train_config.data_dir)

    if train_config.is_analyse_data:
        analyse_data()

    best_by_model = {}
    if train_config.is_training_model:
        rows = grid_search()
        portable_rows = [row for row in rows if row["model"] not in NON_PORTABLE_MODELS]
        best_by_model = select_for_porting(portable_rows, train_config.maep_tolerance)
        print(
            f"\nSelected (smallest flash within {train_config.maep_tolerance} "
            "percentage point(s) of each model type's best MAEP):"
        )
        for model_name, row in best_by_model.items():
            print(
                f"  {model_name}: MAEP={row['avg_maep']:.3f}%, flash={row['flash']} B, "
                f"features={row['features']}"
            )

        models_dir.mkdir(parents=True, exist_ok=True)
        train_df, test_df = _hold_out_last_topo_seed(_load_training_data(train_config), train_config)
        if test_df is not None:
            eval_df, title = test_df, "Predicted vs. actual PDR (unseen-topology test set)"
        else:
            eval_df, title = train_df, "Predicted vs. actual PDR (training data, in-sample)"
        plot_predicted_vs_actual(best_by_model, eval_df, title, models_dir / "predicted_vs_actual.png")

        for model_name in EXPORTED_MODELS:
            if model_name in best_by_model:
                joblib.dump(best_by_model[model_name]["_model"], models_dir / f"best_model_{model_name}.joblib")

    if train_config.is_feature_analysis:
        analyse_correlation()
        analyse_features()

    # Deliberately NOT nested inside `is_training_model`: with
    # is_training_model=False (skip the expensive grid search) and
    # is_porting=True, this re-ports the best_model_*.joblib files a previous
    # training run already left in models_dir, instead of silently doing
    # nothing.
    if train_config.is_porting:
        for model_name in EXPORTED_MODELS:
            best_model_path = models_dir / f"best_model_{model_name}.joblib"
            selected = best_by_model.get(model_name)
            if selected is not None:
                print(
                    f"\n=== selected {model_name} -> mlof-{model_name} "
                    f"(avg_mae={selected['avg_mae']:.4f}, avg_maep={selected['avg_maep']:.2f}%, "
                    f"flash={selected['flash']} B) ==="
                )
            elif best_model_path.exists():
                print(f"\n=== {model_name} -> mlof-{model_name} (re-porting previously trained model) ===")
            else:
                print(f"\n=== {model_name}: no trained model available -- skipped ===")
                continue
            export_model(model_name, best_model_path)


if __name__ == "__main__":
    run_pipeline()
