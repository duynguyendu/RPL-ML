#!/usr/bin/env python3
"""
Runs three independent steps, each gated by its own train_config.py flag
(the same way config.py controls pipeline.py), as direct function calls in
this process (not subprocesses):
  - is_processing_data: models/data.py's process_data() builds train.csv
    from COOJA.testlog files under train_config.data_dir.
  - is_training_model: models/train.py's grid_search() sweeps every
    combination of FEATURE_COLUMNS crossed with every combination of each
    model's hyperparameter grid, for LGBMRegressor, Ridge,
    DecisionTreeRegressor, SVR, and GaussianProcessRegressor (see
    models/train.py), then saves the top 5% of (feature, hyperparameter,
    model) combinations by avg MAE (to
    train_config.data_dir/pdr_model_<rank>.joblib, rank 0 = best).
    GaussianProcessRegressor is excluded from this top 5% (see
    models/train.py's NON_PORTABLE_MODELS) since m2cgen can't export it to C
    -- it still appears in grid_search.csv for comparison.
  - is_porting: converts every train_config.data_dir/pdr_model_*.joblib to C
    (see models/to_c.py) and prints its msp430 size -- reads whatever
    models are already saved there, so this can run on its own against
    models saved by an earlier invocation.

Usage:
    python models/train_pipeline.py
    python models/train_pipeline.py --data_dir=runs/my_run --seeds=[0,1,2]
"""

import os
import sys
from pathlib import Path

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_config
from models.data import process_data
from models.to_c import convert_to_c, measure_size
from models.train import NON_PORTABLE_MODELS, grid_search


def run_pipeline() -> None:
    if train_config.is_processing_data:
        process_data(train_config.data_dir)

    if train_config.is_training_model:
        rows = grid_search()
        portable_rows = [row for row in rows if row["model"] not in NON_PORTABLE_MODELS]

        param_grids = {
            "lgbm": train_config.param_grid,
            "ridge": train_config.ridge_param_grid,
            "dtree": train_config.dtree_param_grid,
            "svr": train_config.svr_param_grid,
            "gp": train_config.gp_param_grid,
        }
        combo_counts = {}
        for grid_name, param_grid in param_grids.items():
            count = 1
            for values in param_grid.values():
                count *= len(values)
            combo_counts[grid_name] = count
        total_combinations = sum(combo_counts[row["model"]] for row in portable_rows)

        top_n = min(max(1, round(total_combinations * 0.05)), len(portable_rows))
        print(f"\nTop {top_n} of {total_combinations} combinations (top 5%, portable models only):")
        for rank, row in enumerate(portable_rows[:top_n]):
            model_path = Path(train_config.data_dir) / f"pdr_model_{rank}.joblib"
            joblib.dump(row["_model"], model_path)
            print(
                f"  [{rank}] model={row['model']} avg_mae={row['avg_mae']:.4f} "
                f"features={row['features']} -> {model_path}"
            )

    if train_config.is_porting:
        model_paths = sorted(
            Path(train_config.data_dir).glob("pdr_model_*.joblib"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        for model_path in model_paths:
            func_name = f"mlof_predict_pdr_{model_path.stem.rsplit('_', 1)[-1]}"
            c_path, h_path = convert_to_c(str(model_path), str(train_config.data_dir), func_name)
            print(f"\n=== {model_path.name} ===")
            measure_size(c_path)


if __name__ == "__main__":
    run_pipeline()
