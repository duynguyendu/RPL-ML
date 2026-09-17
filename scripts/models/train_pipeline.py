#!/usr/bin/env python3

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
    convert_to_c_linear,
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
        for stale_path in models_dir.glob("best_model_*.joblib"):
            stale_path.unlink()

        rank = 0
        for (model_name, scaler), group_rows in rows_by_group.items():
            for row in group_rows[: train_config.top_n_per_model]:
                model_path = models_dir / f"pdr_model_{rank}_{row['model']}.joblib"
                joblib.dump(row["_model"], model_path)
                rank += 1

        if train_config.is_porting:
            best_by_model = {}
            for row in portable_rows:
                best_by_model.setdefault(row["model"], row)

            for model_name, type_name, converter in (
                ("dtree", "dtree", convert_to_c_fixed),
                ("ridge", "linear", convert_to_c_linear),
                ("svr", "svm", convert_to_c_linear),
            ):
                if model_name not in best_by_model:
                    continue
                row = best_by_model[model_name]

                best_model_path = models_dir / f"best_model_{type_name}.joblib"
                joblib.dump(row["_model"], best_model_path)

                func_name = f"mlof_predict_pdr_{type_name}"
                print(
                    f"\n=== best {model_name} -> mlof-{type_name} (avg_mae={row['avg_mae']:.4f}) ==="
                )
                try:
                    c_path, h_path = converter(
                        str(best_model_path), str(models_dir), func_name
                    )
                except Exception as e:
                    print(f"FAILED to convert: {e!r} -- skipped")
                    continue
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
                print(
                    "[fixed] model type not supported by the fixed-point exporter -- skipped"
                )
            else:
                fixed_c_path, _ = fixed_result
                print("[fixed]")
                verify_fixed(str(model_path), str(train_config.data_dir))
                measure_size(fixed_c_path)


if __name__ == "__main__":
    run_pipeline()
