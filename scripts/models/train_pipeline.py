#!/usr/bin/env python3
"""
Runs three independent steps, each gated by its own train_config.py flag
(the same way config.py controls pipeline.py), as direct function calls in
this process (not subprocesses):
  - is_processing_data: models/data.py's process_data() builds train.csv
    from COOJA.testlog files under train_config.data_dir.
  - is_training_model: models/train.py's grid_search() sweeps every
    combination of FEATURE_COLUMNS crossed with every combination of
    train_config.param_grid (see models/train.py), then retrains and saves
    (to train_config.data_dir/pdr_model.txt) a model on whichever
    combination had the lowest avg MAE.
  - is_porting: converts train_config.data_dir/pdr_model.txt to C (see
    models/to_c.py) -- reads whatever model is already saved there, so this
    can run on its own against a model saved by an earlier invocation.

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

    model_path = Path(train_config.data_dir) / "pdr_model.txt"

    if train_config.is_training_model:
        rows = grid_search()

        best = rows[0]
        print(
            f"\nBest combination for porting: { {k: v for k, v in best.items() if k != '_model'} }"
        )

        best["_model"].booster_.save_model(str(model_path))
        print(f"Model saved to {model_path}")

    if train_config.is_porting:
        c_path, h_path = convert_to_c(str(model_path), str(train_config.data_dir))
        measure_size(c_path)


if __name__ == "__main__":
    run_pipeline()
