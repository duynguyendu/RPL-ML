#!/usr/bin/env python3
"""
Runs three independent steps, each gated by its own train_config.py flag
(the same way config.py controls pipeline.py), as direct function calls in
this process (not subprocesses):
  - is_processing_data: models/data.py's process_data() builds train.csv
    from COOJA.testlog files under train_config.data_dir.
  - is_training_model: models/train.py's grid_search() sweeps every
    combination of FEATURE_COLUMNS crossed with every combination of
    train_config.param_grid (see models/train.py), then saves the top 5%
    of (feature, hyperparameter) combinations by avg MAE (to
    train_config.data_dir/pdr_model_<rank>.txt, rank 0 = best).
  - is_porting: converts every train_config.data_dir/pdr_model_*.txt to C
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_config
from models.data import process_data
from models.to_c import convert_to_c, measure_size
from models.train import grid_search


def run_pipeline() -> None:
    if train_config.is_processing_data:
        process_data(train_config.data_dir)

    if train_config.is_training_model:
        rows = grid_search()

        hyperparam_combo_count = 1
        for values in train_config.param_grid.values():
            hyperparam_combo_count *= len(values)
        total_combinations = len(rows) * hyperparam_combo_count

        top_n = min(max(1, round(total_combinations * 0.05)), len(rows))
        print(f"\nTop {top_n} of {total_combinations} combinations (top 5%):")
        for rank, row in enumerate(rows[:top_n]):
            model_path = Path(train_config.data_dir) / f"pdr_model_{rank}.txt"
            row["_model"].booster_.save_model(str(model_path))
            print(
                f"  [{rank}] avg_mae={row['avg_mae']:.4f} features={row['features']} "
                f"-> {model_path}"
            )

    if train_config.is_porting:
        model_paths = sorted(
            Path(train_config.data_dir).glob("pdr_model_*.txt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        for model_path in model_paths:
            func_name = f"mlof_predict_pdr_{model_path.stem.rsplit('_', 1)[-1]}"
            c_path, h_path = convert_to_c(str(model_path), str(train_config.data_dir), func_name)
            print(f"\n=== {model_path.name} ===")
            measure_size(c_path)


if __name__ == "__main__":
    run_pipeline()
