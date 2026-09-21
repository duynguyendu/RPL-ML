#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_config
from models.data import process_data
from models.to_c import (
    convert_to_c_fixed,
    convert_to_c_linear,
    measure_size,
    verify_fixed,
    verify_linear,
)
from models.train import NON_PORTABLE_MODELS, grid_search


def run_pipeline() -> None:
    models_dir = Path(train_config.data_dir) / "models"

    if train_config.is_processing_data:
        process_data(train_config.data_dir)

    best_by_model = {}
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
        for stale_path in models_dir.glob("best_model_*.joblib"):
            stale_path.unlink()

        rank = 0
        for (model_name, scaler), group_rows in rows_by_group.items():
            for row in group_rows[: train_config.top_n_per_model]:
                model_path = models_dir / f"pdr_model_{rank}_{row['model']}.joblib"
                joblib.dump(row["_model"], model_path)
                rank += 1

        for row in portable_rows:
            best_by_model.setdefault(row["model"], row)

    # Deliberately NOT nested inside `is_training_model`: with
    # is_training_model=False (skip the expensive grid search) and
    # is_porting=True, this re-ports whatever best_model_*.joblib files a
    # previous training run already left in models_dir, instead of silently
    # doing nothing.
    if train_config.is_porting:
        for model_name, type_name, converter in (
            ("dtree", "dtree", convert_to_c_fixed),
            ("lgbm", "lgbm", convert_to_c_fixed),
            ("ridge", "linear", convert_to_c_linear),
            ("svr", "svm", convert_to_c_linear),
        ):
            best_model_path = models_dir / f"best_model_{type_name}.joblib"
            if model_name in best_by_model:
                joblib.dump(best_by_model[model_name]["_model"], best_model_path)
                avg_mae = best_by_model[model_name]["avg_mae"]
                print(
                    f"\n=== best {model_name} -> mlof-{type_name} (avg_mae={avg_mae:.4f}) ==="
                )
            elif best_model_path.exists():
                print(
                    f"\n=== best {model_name} -> mlof-{type_name} "
                    "(re-porting previously trained model) ==="
                )
            else:
                print(f"\n=== {model_name}: no trained model available -- skipped ===")
                continue

            func_name = f"mlof_predict_pdr_{type_name}"
            try:
                c_path, h_path = converter(
                    str(best_model_path), str(models_dir), func_name
                )
            except Exception as e:
                print(f"FAILED to convert: {e!r} -- skipped")
                continue
            if converter is convert_to_c_linear:
                verify_linear(str(best_model_path), str(train_config.data_dir))
            elif converter is convert_to_c_fixed:
                verify_fixed(str(best_model_path), str(train_config.data_dir))
            measure_size(c_path)

            train_config.rpl_lite_dir.mkdir(parents=True, exist_ok=True)
            dest_c = train_config.rpl_lite_dir / f"mlof-{type_name}.c"
            dest_h = train_config.rpl_lite_dir / f"mlof-{type_name}.h"
            dest_c.write_text(
                c_path.read_text().replace(
                    f'"{func_name}.h"', f'"mlof-{type_name}.h"'
                )
            )
            dest_h.write_text(h_path.read_text())
            print(f"Exported to {dest_c} and {dest_h}")

    if train_config.is_porting:
        model_paths = sorted(
            models_dir.glob("pdr_model_*.joblib"),
            key=lambda p: int(p.stem.split("_")[2]),
        )
        for model_path in model_paths:
            _, _, rank, model_type = model_path.stem.split("_", 3)
            print(f"\n=== {model_path.name} (model={model_type}) ===")

            fixed_func_name = f"mlof_predict_pdr_fixed_{rank}_{model_type}"
            try:
                fixed_result = convert_to_c_fixed(
                    str(model_path), str(models_dir), fixed_func_name
                )
            except Exception as e:
                print(f"[fixed] FAILED to convert: {e!r} -- skipped")
                fixed_result = None
            if fixed_result is None:
                print(
                    "[fixed] model type not supported by the fixed-point exporter -- skipped"
                )
            else:
                fixed_c_path, _ = fixed_result
                print("[fixed]")
                verify_fixed(str(model_path), str(train_config.data_dir))
                measure_size(fixed_c_path)

            linear_func_name = f"mlof_predict_pdr_linear_{rank}_{model_type}"
            try:
                linear_result = convert_to_c_linear(
                    str(model_path), str(models_dir), linear_func_name
                )
            except Exception as e:
                print(f"[linear] FAILED to convert: {e!r} -- skipped")
                continue
            if linear_result is None:
                print(
                    "[linear] model type not supported by the standardised-linear exporter -- skipped"
                )
            else:
                linear_c_path, _ = linear_result
                print("[linear]")
                verify_linear(str(model_path), str(train_config.data_dir))
                measure_size(linear_c_path)


if __name__ == "__main__":
    run_pipeline()
