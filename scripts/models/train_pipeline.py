#!/usr/bin/env python3
"""
Runs three independent steps, each gated by its own train_config.py flag
(the same way config.py controls pipeline.py), as direct function calls in
this process (not subprocesses):
  - is_processing_data: models/data.py's process_data() builds train.csv
    from COOJA.testlog files under train_config.data_dir.
  - is_training_model: models/train.py's test_feature_importance() sweeps
    every combination of train_config.features_to_test (see models/train.py),
    then retrains and saves (to train_config.data_dir/pdr_model.txt) a model
    on whichever combination had the lowest avg MAE.
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
from models.to_c import convert_to_c
from models.train import test_feature_importance, train_model


def run_pipeline() -> None:
    if train_config.is_processing_data:
        process_data(train_config.data_dir)

    model_path = Path(train_config.data_dir) / "pdr_model.txt"

    if train_config.is_training_model:
        train_csv = Path(train_config.data_dir) / "train.csv"
        output_path = Path(train_config.data_dir) / "feature_importance.csv"
        results = test_feature_importance(
            str(train_csv),
            features_to_test=train_config.features_to_test,
            seeds=train_config.seeds,
            output_path=str(output_path),
        )

        best_features, best_mae = min(results.items(), key=lambda item: item[1])
        print(f"\nBest combination for porting: {list(best_features)} (avg MAE = {best_mae:.4f})")

        train_model(
            train_csv=str(train_csv),
            save_path=str(model_path),
            seeds=[train_config.seeds[0]],
            feature_columns=list(best_features),
        )

    if train_config.is_porting:
        # TODO: to_c.py's FEATURE_SCALES only covers the old feature set
        # (etx, hop_count, ppm, cpu_util, num_neighbours) -- most of the
        # current FEATURE_COLUMNS (is_new, cpu, p_cpu, drop_rate,
        # nbr_count) have no configured scale, and rssi is negative while
        # the generated C assumes every feature is uint16_t (unsigned).
        # convert_to_c() will likely raise KeyError below until
        # FEATURE_SCALES (and the signed-rssi handling) are fixed.
        convert_to_c(str(model_path), str(train_config.data_dir))


if __name__ == "__main__":
    run_pipeline()
