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
    DecisionTreeRegressor, and SVR (see models/train.py), then saves the
    top train_config.top_n_per_model (feature, hyperparameter) combinations
    by avg MAE independently within each (model type, scaler) group -- so
    Ridge/SVR each get their own top N with a scaler and top N without one
    -- to train_config.data_dir/models/pdr_model_<rank>_<model>.joblib (all
    model-related output, including grid_search.csv, lives under that
    models/ subdirectory -- train.csv itself stays directly in data_dir).
  - is_porting: converts every
    train_config.data_dir/models/pdr_model_*.joblib to C with both m2cgen
    and, when the model type is supported, emlearn (see models/to_c.py),
    printing the msp430 size of each so the two backends can be compared
    -- reads whatever models are already saved there, so this can run on
    its own against models saved by an earlier invocation.

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
from models.to_c import (
    convert_to_c,
    convert_to_c_emlearn,
    convert_to_c_fixed,
    measure_size,
    verify_fixed,
)
from models.train import NON_PORTABLE_MODELS, grid_search


def run_pipeline() -> None:
    models_dir = Path(train_config.data_dir) / "models"

    if train_config.is_processing_data:
        process_data(train_config.data_dir)

    if train_config.is_training_model:
        rows = grid_search()
        portable_rows = [row for row in rows if row["model"] not in NON_PORTABLE_MODELS]

        rows_by_group = {}
        for row in portable_rows:
            key = (row["model"], row.get("scaler"))
            rows_by_group.setdefault(key, []).append(row)

        models_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in models_dir.glob("pdr_model_*.joblib"):
            stale_path.unlink()

        rank = 0
        for (model_name, scaler), group_rows in rows_by_group.items():
            for row in group_rows[: train_config.top_n_per_model]:
                model_path = models_dir / f"pdr_model_{rank}_{row['model']}.joblib"
                joblib.dump(row["_model"], model_path)
                rank += 1

    if train_config.is_porting:
        model_paths = sorted(
            models_dir.glob("pdr_model_*.joblib"),
            key=lambda p: int(p.stem.split("_")[2]),
        )
        for model_path in model_paths:
            _, _, rank, model_type = model_path.stem.split("_", 3)
            print(f"\n=== {model_path.name} (model={model_type}) ===")

            func_name = f"mlof_predict_pdr_{rank}_{model_type}"
            try:
                c_path, _ = convert_to_c(str(model_path), str(models_dir), func_name)
                print("[m2cgen]")
                measure_size(c_path)
            except Exception as e:
                print(f"[m2cgen] FAILED to convert: {e!r} -- skipped")

            emlearn_func_name = f"mlof_predict_pdr_emlearn_{rank}_{model_type}"
            try:
                emlearn_result = convert_to_c_emlearn(
                    str(model_path), str(models_dir), emlearn_func_name
                )
            except Exception as e:
                print(f"[emlearn] FAILED to convert: {e!r} -- skipped")
                continue
            if emlearn_result is None:
                print("[emlearn] model type not supported by emlearn -- skipped")
            else:
                emlearn_c_path, _ = emlearn_result
                print("[emlearn]")
                measure_size(emlearn_c_path)

            fixed_func_name = f"mlof_predict_pdr_fixed_{rank}_{model_type}"
            try:
                fixed_result = convert_to_c_fixed(
                    str(model_path), str(models_dir), fixed_func_name
                )
            except Exception as e:
                print(f"[fixed] FAILED to convert: {e!r} -- skipped")
                continue
            if fixed_result is None:
                print("[fixed] model type not supported by the fixed-point exporter -- skipped")
            else:
                fixed_c_path, _ = fixed_result
                print("[fixed]")
                verify_fixed(str(model_path), str(train_config.data_dir))
                measure_size(fixed_c_path)


if __name__ == "__main__":
    run_pipeline()
