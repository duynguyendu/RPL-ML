from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_config import UNKNOWN_SENTINELS

FIXED_SUPPORTED_ESTIMATORS = {
    "DecisionTreeRegressor",
    "LGBMRegressor",
    "RandomForestRegressor",
    "XGBRegressor",
    "CatBoostRegressor",
}
LINEAR_SUPPORTED_ESTIMATORS = {"Ridge", "LinearSVR"}
_TREE_LEAF = -1  # see sklearn/tree/_tree.pyx

# native types from rpl-mlof.c's MLOF metrics struct
FEATURE_C_TYPES = {
    "is_new": "uint8_t",
    "cpu": "uint8_t",
    "p_cpu": "uint8_t",
    "etx": "uint16_t",
    "rssi": "int16_t",
    "ppm": "uint16_t",
    "drop_rate": "uint8_t",
    "parent_ppm": "uint16_t",
    "parent_drop_rate": "uint8_t",
    "hop_count": "uint8_t",
    "nbr_count": "uint8_t",
}


def _c_params(feature_names: list[str]) -> str:
    return ", ".join(f"{FEATURE_C_TYPES.get(name, 'uint16_t')} {name}" for name in feature_names)


def _fixed_feature_names(model) -> list[str]:
    # CatBoostRegressor doesn't set the sklearn-standard feature_names_in_.
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = model.feature_names_
    return [str(name) for name in names]


def _int_threshold(threshold: float) -> int:
    """Integer T such that ``raw <= T`` matches a real ``raw <= threshold``

    for integer raw (sklearn/LightGBM's split convention)."""
    return math.floor(threshold + 1e-6)


def _ceil_threshold(threshold: float) -> int:
    """Integer T such that ``raw < T`` matches a real ``raw < threshold``

    for integer raw (XGBoost's split convention, which is strict-less-than
    rather than sklearn/LightGBM's less-equal)."""
    return math.ceil(threshold - 1e-6)


def _dtree_body(model, feature_names: list[str], func_name: str) -> str:
    tree = model.tree_
    lines = [f"uint16_t {func_name}({_c_params(feature_names)}) {{"]

    def walk(node_id: int, indent: str) -> None:
        if tree.children_left[node_id] == _TREE_LEAF:
            leaf_value = max(0, min(65535, round(tree.value[node_id][0][0])))
            lines.append(f"{indent}return {leaf_value};")
            return
        feature_idx = tree.feature[node_id]
        threshold = _int_threshold(tree.threshold[node_id])
        lines.append(f"{indent}if ({feature_names[feature_idx]} <= {threshold}) {{")
        walk(tree.children_left[node_id], indent + "    ")
        lines.append(f"{indent}}} else {{")
        walk(tree.children_right[node_id], indent + "    ")
        lines.append(f"{indent}}}")

    walk(0, "    ")
    lines.append("}")
    return "\n".join(lines)


def _dtree_predict_fixed(model, rows: np.ndarray) -> np.ndarray:
    tree = model.tree_
    preds = []
    for row in rows:
        node_id = 0
        while tree.children_left[node_id] != _TREE_LEAF:
            threshold = _int_threshold(tree.threshold[node_id])
            if row[tree.feature[node_id]] <= threshold:
                node_id = tree.children_left[node_id]
            else:
                node_id = tree.children_right[node_id]
        preds.append(max(0, min(65535, round(tree.value[node_id][0][0]))))
    return np.array(preds)


def _lgbm_tree_body(tree_node: dict, feature_names: list[str], tree_func_name: str) -> str:
    lines = [f"static int32_t {tree_func_name}({_c_params(feature_names)}) {{"]

    def walk(node: dict, indent: str) -> None:
        if "leaf_value" in node:
            lines.append(f"{indent}return {round(node['leaf_value'])};")
            return
        feature_idx = node["split_feature"]
        threshold = _int_threshold(node["threshold"])
        lines.append(f"{indent}if ({feature_names[feature_idx]} <= {threshold}) {{")
        walk(node["left_child"], indent + "    ")
        lines.append(f"{indent}}} else {{")
        walk(node["right_child"], indent + "    ")
        lines.append(f"{indent}}}")

    walk(tree_node, "    ")
    lines.append("}")
    return "\n".join(lines)


def _lgbm_tree_predict(tree_node: dict, row: np.ndarray) -> int:
    node = tree_node
    while "leaf_value" not in node:
        threshold = _int_threshold(node["threshold"])
        node = node["left_child"] if row[node["split_feature"]] <= threshold else node["right_child"]
    return round(node["leaf_value"])


def _lgbm_body(model, feature_names: list[str], func_name: str) -> str:
    trees = [t["tree_structure"] for t in model.booster_.dump_model()["tree_info"]]
    args = ", ".join(feature_names)

    parts = []
    tree_func_names = []
    for i, tree_node in enumerate(trees):
        tree_func_name = f"{func_name}_tree{i}"
        tree_func_names.append(tree_func_name)
        parts.append(_lgbm_tree_body(tree_node, feature_names, tree_func_name))

    main_lines = [
        f"uint16_t {func_name}({_c_params(feature_names)}) {{",
        "    int32_t sum = 0;",
        *(f"    sum += {name}({args});" for name in tree_func_names),
        "    if (sum < 0) sum = 0;",
        "    if (sum > 65535) sum = 65535;",
        "    return (uint16_t)sum;",
        "}",
    ]
    parts.append("\n".join(main_lines))
    return "\n\n".join(parts)


def _lgbm_predict_fixed(model, rows: np.ndarray) -> np.ndarray:
    trees = [t["tree_structure"] for t in model.booster_.dump_model()["tree_info"]]
    preds = []
    for row in rows:
        total = sum(_lgbm_tree_predict(t, row) for t in trees)
        preds.append(max(0, min(65535, total)))
    return np.array(preds)


# ---------------------------------------------------------------------------
# RandomForestRegressor -- an ensemble of plain sklearn trees (same .tree_
# shape as DecisionTreeRegressor); the prediction is their *average*, not
# their sum, so each per-tree leaf is left unclamped/unrounded-to-range and
# only the final averaged result is clamped to [0, 65535].
# ---------------------------------------------------------------------------


def _rf_tree_body(tree, feature_names: list[str], tree_func_name: str) -> str:
    lines = [f"static int32_t {tree_func_name}({_c_params(feature_names)}) {{"]

    def walk(node_id: int, indent: str) -> None:
        if tree.children_left[node_id] == _TREE_LEAF:
            lines.append(f"{indent}return {round(tree.value[node_id][0][0])};")
            return
        feature_idx = tree.feature[node_id]
        threshold = _int_threshold(tree.threshold[node_id])
        lines.append(f"{indent}if ({feature_names[feature_idx]} <= {threshold}) {{")
        walk(tree.children_left[node_id], indent + "    ")
        lines.append(f"{indent}}} else {{")
        walk(tree.children_right[node_id], indent + "    ")
        lines.append(f"{indent}}}")

    walk(0, "    ")
    lines.append("}")
    return "\n".join(lines)


def _rf_tree_predict(tree, row: np.ndarray) -> int:
    node_id = 0
    while tree.children_left[node_id] != _TREE_LEAF:
        threshold = _int_threshold(tree.threshold[node_id])
        if row[tree.feature[node_id]] <= threshold:
            node_id = tree.children_left[node_id]
        else:
            node_id = tree.children_right[node_id]
    return round(tree.value[node_id][0][0])


def _rf_body(model, feature_names: list[str], func_name: str) -> str:
    n_estimators = len(model.estimators_)
    args = ", ".join(feature_names)

    parts = []
    tree_func_names = []
    for i, estimator in enumerate(model.estimators_):
        tree_func_name = f"{func_name}_tree{i}"
        tree_func_names.append(tree_func_name)
        parts.append(_rf_tree_body(estimator.tree_, feature_names, tree_func_name))

    main_lines = [
        f"uint16_t {func_name}({_c_params(feature_names)}) {{",
        "    int32_t sum = 0;",
        *(f"    sum += {name}({args});" for name in tree_func_names),
        f"    int32_t avg = sum / {n_estimators};",
        "    if (avg < 0) avg = 0;",
        "    if (avg > 65535) avg = 65535;",
        "    return (uint16_t)avg;",
        "}",
    ]
    parts.append("\n".join(main_lines))
    return "\n\n".join(parts)


def _rf_predict_fixed(model, rows: np.ndarray) -> np.ndarray:
    n_estimators = len(model.estimators_)
    preds = []
    for row in rows:
        total = sum(_rf_tree_predict(est.tree_, row) for est in model.estimators_)
        # RF leaves are always >= 0 (they're literal label averages), so
        # plain truncating division matches C's here -- no sign-safe divide
        # needed the way the boosting exporters below require.
        preds.append(max(0, min(65535, total // n_estimators)))
    return np.array(preds)


# ---------------------------------------------------------------------------
# XGBRegressor and CatBoostRegressor -- gradient-boosted ensembles whose
# leaves are small signed residual corrections around a base_score/bias,
# rather than label-scale values. Rounding each leaf straight to an integer
# (like LightGBM's leaves, which already sit at label scale) would throw
# away most of their precision, so instead every leaf and the bias share one
# adaptively-picked power-of-10 fixed-point scale (same idea as the linear
# exporter's std_scale/coef_scale), with a single division at the very end.
# ---------------------------------------------------------------------------


def _pick_ensemble_scale(bias: float, leaf_values: list[float]) -> int:
    worst_case = abs(bias) + sum(abs(float(v)) for v in leaf_values)
    if worst_case <= 0:
        return 10000
    return _largest_power_of_10(_INT32_BUDGET / worst_case)


def _xgb_trees_and_base_score(model) -> tuple[list[dict], float]:
    booster = model.get_booster()
    trees = [json.loads(d) for d in booster.get_dump(dump_format="json")]
    config = json.loads(booster.save_config())
    base_score = float(config["learner"]["learner_model_param"]["base_score"].strip("[]"))
    return trees, base_score


def _xgb_node_map(node: dict, mapping: dict) -> None:
    mapping[node["nodeid"]] = node
    for child in node.get("children", ()):
        _xgb_node_map(child, mapping)


def _xgb_tree_body(tree_root: dict, feature_names: list[str], tree_func_name: str, scale: int) -> str:
    nodes: dict = {}
    _xgb_node_map(tree_root, nodes)
    lines = [f"static int32_t {tree_func_name}({_c_params(feature_names)}) {{"]

    def walk(node: dict, indent: str) -> None:
        if "leaf" in node:
            lines.append(f"{indent}return {round(node['leaf'] * scale)};")
            return
        threshold = _ceil_threshold(node["split_condition"])
        lines.append(f"{indent}if ({node['split']} < {threshold}) {{")
        walk(nodes[node["yes"]], indent + "    ")
        lines.append(f"{indent}}} else {{")
        walk(nodes[node["no"]], indent + "    ")
        lines.append(f"{indent}}}")

    walk(tree_root, "    ")
    lines.append("}")
    return "\n".join(lines)


def _xgb_tree_predict(tree_root: dict, row: np.ndarray, feature_index: dict[str, int]) -> float:
    nodes: dict = {}
    _xgb_node_map(tree_root, nodes)
    node = tree_root
    while "leaf" not in node:
        threshold = _ceil_threshold(node["split_condition"])
        node_id = node["yes"] if row[feature_index[node["split"]]] < threshold else node["no"]
        node = nodes[node_id]
    return node["leaf"]


def _xgb_body(model, feature_names: list[str], func_name: str) -> str:
    trees, base_score = _xgb_trees_and_base_score(model)
    all_leaves = [node["leaf"] for tree in trees for node in _flatten_leaves(tree)]
    scale = _pick_ensemble_scale(base_score, all_leaves)
    bias_fixed = round(base_score * scale)
    args = ", ".join(feature_names)

    parts = []
    tree_func_names = []
    for i, tree_root in enumerate(trees):
        tree_func_name = f"{func_name}_tree{i}"
        tree_func_names.append(tree_func_name)
        parts.append(_xgb_tree_body(tree_root, feature_names, tree_func_name, scale))

    main_lines = [
        f"uint16_t {func_name}({_c_params(feature_names)}) {{",
        f"    int32_t sum = {bias_fixed};",
        *(f"    sum += {name}({args});" for name in tree_func_names),
        f"    int32_t result = sum / {scale};",
        "    if (result < 0) result = 0;",
        "    if (result > 65535) result = 65535;",
        "    return (uint16_t)result;",
        "}",
    ]
    parts.append("\n".join(main_lines))
    return "\n\n".join(parts)


def _flatten_leaves(node: dict) -> list[dict]:
    if "leaf" in node:
        return [node]
    leaves = []
    for child in node.get("children", ()):
        leaves.extend(_flatten_leaves(child))
    return leaves


def _xgb_predict_fixed(model, rows: np.ndarray, feature_names: list[str]) -> np.ndarray:
    trees, base_score = _xgb_trees_and_base_score(model)
    all_leaves = [node["leaf"] for tree in trees for node in _flatten_leaves(tree)]
    scale = _pick_ensemble_scale(base_score, all_leaves)
    bias_fixed = round(base_score * scale)
    feature_index = {name: i for i, name in enumerate(feature_names)}

    preds = []
    for row in rows:
        total = bias_fixed + sum(
            round(_xgb_tree_predict(tree, row, feature_index) * scale) for tree in trees
        )
        result = total // scale if total >= 0 else -(-total // scale)
        preds.append(max(0, min(65535, result)))
    return np.array(preds)


def _catboost_dump(model) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        model.save_model(str(path), format="json")
        return json.loads(path.read_text())


def _catboost_feature_index_map(dump: dict) -> dict[int, str]:
    return {
        ff["feature_index"]: ff["feature_id"]
        for ff in dump["features_info"]["float_features"]
    }


def _catboost_tree_body(
    tree: dict,
    idx_to_name: dict[int, str],
    feature_names: list[str],
    tree_func_name: str,
    scale: int,
    cat_scale: float,
) -> str:
    splits = tree["splits"]  # bit i (LSB-first) of the leaf index comes from splits[i]
    leaves_fixed = [round(v * cat_scale * scale) for v in tree["leaf_values"]]

    lines = [
        f"static int32_t {tree_func_name}({_c_params(feature_names)}) {{",
        f"    static const int32_t leaves[{len(leaves_fixed)}] = "
        f"{{{', '.join(str(v) for v in leaves_fixed)}}};",
        "    int idx = 0;",
    ]
    for i, split in enumerate(splits):
        name = idx_to_name[split["float_feature_index"]]
        threshold = _int_threshold(split["border"])
        lines.append(f"    if ({name} > {threshold}) idx |= {1 << i};")
    lines.append("    return leaves[idx];")
    lines.append("}")
    return "\n".join(lines)


def _catboost_body(model, feature_names: list[str], func_name: str) -> str:
    dump = _catboost_dump(model)
    idx_to_name = _catboost_feature_index_map(dump)
    cat_scale, cat_bias = dump["scale_and_bias"][0], dump["scale_and_bias"][1][0]
    trees = dump["oblivious_trees"]

    all_leaves = [v * cat_scale for tree in trees for v in tree["leaf_values"]]
    scale = _pick_ensemble_scale(cat_bias * cat_scale, all_leaves)
    bias_fixed = round(cat_bias * cat_scale * scale)
    args = ", ".join(feature_names)

    parts = []
    tree_func_names = []
    for i, tree in enumerate(trees):
        tree_func_name = f"{func_name}_tree{i}"
        tree_func_names.append(tree_func_name)
        parts.append(
            _catboost_tree_body(tree, idx_to_name, feature_names, tree_func_name, scale, cat_scale)
        )

    main_lines = [
        f"uint16_t {func_name}({_c_params(feature_names)}) {{",
        f"    int32_t sum = {bias_fixed};",
        *(f"    sum += {name}({args});" for name in tree_func_names),
        f"    int32_t result = sum / {scale};",
        "    if (result < 0) result = 0;",
        "    if (result > 65535) result = 65535;",
        "    return (uint16_t)result;",
        "}",
    ]
    parts.append("\n".join(main_lines))
    return "\n\n".join(parts)


def _catboost_predict_fixed(model, rows: np.ndarray, feature_names: list[str]) -> np.ndarray:
    dump = _catboost_dump(model)
    idx_to_name = _catboost_feature_index_map(dump)
    cat_scale, cat_bias = dump["scale_and_bias"][0], dump["scale_and_bias"][1][0]
    trees = dump["oblivious_trees"]

    all_leaves = [v * cat_scale for tree in trees for v in tree["leaf_values"]]
    fixed_scale = _pick_ensemble_scale(cat_bias * cat_scale, all_leaves)
    bias_fixed = round(cat_bias * cat_scale * fixed_scale)
    feature_index = {name: i for i, name in enumerate(feature_names)}

    preds = []
    for row in rows:
        total = bias_fixed
        for tree in trees:
            idx = 0
            for i, split in enumerate(tree["splits"]):
                threshold = _int_threshold(split["border"])
                name = idx_to_name[split["float_feature_index"]]
                if row[feature_index[name]] > threshold:
                    idx |= 1 << i
            total += round(tree["leaf_values"][idx] * cat_scale * fixed_scale)
        result = total // fixed_scale if total >= 0 else -(-total // fixed_scale)
        preds.append(max(0, min(65535, result)))
    return np.array(preds)


def convert_to_c_fixed(
    model_path: str | Path, out_dir: str | Path, func_name: str = "mlof_predict_pdr"
) -> tuple[Path, Path] | None:
    model = joblib.load(model_path)
    model_type = type(model).__name__
    if model_type not in FIXED_SUPPORTED_ESTIMATORS:
        return None

    feature_names = _fixed_feature_names(model)
    if model_type == "DecisionTreeRegressor":
        source_body = _dtree_body(model, feature_names, func_name)
    elif model_type == "LGBMRegressor":
        source_body = _lgbm_body(model, feature_names, func_name)
    elif model_type == "RandomForestRegressor":
        source_body = _rf_body(model, feature_names, func_name)
    elif model_type == "XGBRegressor":
        source_body = _xgb_body(model, feature_names, func_name)
    else:
        source_body = _catboost_body(model, feature_names, func_name)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = out_dir / f"{func_name}.c"
    h_path = out_dir / f"{func_name}.h"

    header_guard = f"{func_name.upper()}_H_"
    header = f"""\
/* Auto-generated by scripts/models/to_c.py (fixed-point, no float/double)
 * from a trained model. DO NOT EDIT BY HAND -- retrain and re-run the
 * converter instead.
 */
#ifndef {header_guard}
#define {header_guard}
#include <stdint.h>

uint16_t {func_name}({_c_params(feature_names)});

#endif /* {header_guard} */
"""
    source = (
        "/* Auto-generated by scripts/models/to_c.py (fixed-point, no "
        "float/double) from a trained model.\n"
        " * DO NOT EDIT BY HAND -- retrain and re-run the converter "
        "instead. */\n"
        f'#include "{func_name}.h"\n\n' + source_body + "\n"
    )

    c_path.write_text(source)
    h_path.write_text(header)
    return c_path, h_path


def _linear_model_type(model) -> str:
    inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
    return type(inner).__name__


def _scaler_offset_divisor(scaler) -> tuple[np.ndarray, np.ndarray]:
    """(offset, divisor) such that ``(raw - offset) / divisor`` matches the

    scaler's own transform -- StandardScaler and MinMaxScaler parameterise
    that affine map differently (mean/std vs data_min_/scale_, and
    MinMaxScaler's ``scale_`` is already a multiplier, not a divisor), so
    this is the one place that difference needs to be known.
    """
    scaler_type = type(scaler).__name__
    if scaler_type == "StandardScaler":
        return np.asarray(scaler.mean_, dtype=float), np.asarray(scaler.scale_, dtype=float)
    if scaler_type == "MinMaxScaler":
        return np.asarray(scaler.data_min_, dtype=float), 1.0 / np.asarray(scaler.scale_, dtype=float)
    raise ValueError(f"unsupported scaler: {scaler_type}")


def _compose_affine(
    offset1: np.ndarray, divisor1: np.ndarray, offset2: np.ndarray, divisor2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two chained ``(raw - offset) / divisor`` transforms into one.

    If step 1 maps raw -> u and step 2 maps u -> z, then
    z = ((raw - offset1) / divisor1 - offset2) / divisor2
      = (raw - (offset1 + offset2 * divisor1)) / (divisor1 * divisor2)
    """
    return offset1 + offset2 * divisor1, divisor1 * divisor2


def _linear_params(
    model,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, float]:
    """(feature_names, mean, scale, coef, intercept) for a Ridge/LinearSVR model.

    ``model`` is expected to be a fitted ``Pipeline`` ending in the estimator,
    with one or more preceding preprocessing steps (``MinMaxScaler`` and/or
    ``StandardScaler``, chained in fit order) -- their combined effect is
    folded into a single (mean, scale) pair, so predict() can apply that one
    affine transform to raw features before coef_/intercept_ (which were
    themselves fit on the fully scaled features).
    """
    feature_names = [str(name) for name in model.feature_names_in_]
    if hasattr(model, "named_steps"):
        inner = model.named_steps["model"]
        mean = np.zeros(len(feature_names))
        scale = np.ones(len(feature_names))
        for step_name, step in model.named_steps.items():
            if step_name == "model":
                continue
            step_offset, step_divisor = _scaler_offset_divisor(step)
            mean, scale = _compose_affine(mean, scale, step_offset, step_divisor)
    else:
        inner = model
        mean = np.zeros(len(feature_names))
        scale = np.ones(len(feature_names))

    coef = np.asarray(inner.coef_, dtype=float).ravel()
    intercept = inner.intercept_
    intercept = float(intercept if np.isscalar(intercept) else intercept[0])
    return feature_names, mean, scale, coef, intercept


_TYPE_RANGE = {"uint8_t": (0, 255), "uint16_t": (0, 65535), "int16_t": (-32768, 32767)}
_INT32_BUDGET = 0.9 * 2**31  # headroom under int32_t's ~2.147e9 ceiling

# msp430-gcc's 64-bit multiply expansion (msp430_expand_mul) crashes with an
# internal compiler error on a >32-bit constant multiplicand, but that's a
# secondary concern here: msp430f2617 has no native wide multiplier for
# int64_t at all, so every 64-bit multiply/divide is emulated in pure
# software (__muldi3/__divdi3), while int32_t multiplies can use the chip's
# hardware 16x16 multiplier (__mulsi3 backed by __MPY/__OP2/...). Every
# constant and every intermediate value below is therefore kept within
# int32_t -- both ``std_scale`` (bounding the standardise step's *product*,
# not just its multiplier) and ``coef_scale`` are picked against
# _INT32_BUDGET rather than a 64-bit budget.


def _largest_power_of_10(budget: float) -> int:
    scale = 1
    while scale * 10 <= budget:
        scale *= 10
    return scale


def _raw_bound(feature_name: str, mean_fixed: int) -> int:
    lo, hi = _TYPE_RANGE.get(FEATURE_C_TYPES.get(feature_name, "uint16_t"), (0, 65535))
    return max(abs(lo - mean_fixed), abs(hi - mean_fixed))


def _standardise_fixed_params(
    feature_names: list[str], scale: np.ndarray, mean: np.ndarray, combined_budget: float
) -> tuple[list[int], list[int], int]:
    """(mean_fixed, inv_scale_mult, std_scale) so that, with no division,

        z_fixed = (raw - mean_fixed) * inv_scale_mult

    approximates the standardised value scaled up by ``std_scale`` -- e.g. a
    real ratio of 0.2 becomes an integer multiply (``* 2``), with the ``/ 10``
    folded into the final divisor rather than done here. ``std_scale`` is
    picked as large as possible while keeping every ``z_fixed`` itself --
    not just its multiplier -- within int32_t, since ``z_fixed`` is stored
    in an int32_t and used again as a multiplicand in predict(); it is also
    capped by its share of ``combined_budget`` (see ``_linear_fixed_params``)
    so that, even at coef_scale's floor of 1, the final accumulator can't
    overflow int32_t.
    """
    mean_fixed = [round(float(m)) for m in mean]
    per_feature_cap = [
        _INT32_BUDGET * float(s) / _raw_bound(name, mf) if _raw_bound(name, mf) > 0 else float("inf")
        for name, mf, s in zip(feature_names, mean_fixed, scale)
    ]
    std_scale = _largest_power_of_10(min([*per_feature_cap, combined_budget], default=1.0))
    inv_scale_mult = [max(1, round(std_scale / float(s))) for s in scale]
    return mean_fixed, inv_scale_mult, std_scale


def _pick_coef_scale(coef, std_scale: int, combined_budget: float) -> int:
    max_coef = max((abs(float(c)) for c in coef), default=0.0)

    # each coef_fixed must itself fit in int32_t, and std_scale already used
    # up its share of combined_budget -- whatever's left is coef_scale's.
    max_scale_coef = _INT32_BUDGET / max_coef if max_coef > 0 else float("inf")
    max_scale_budget = combined_budget / std_scale
    return _largest_power_of_10(min(max_scale_coef, max_scale_budget))


def _linear_fixed_params(model):
    feature_names, mean, scale, coef, intercept = _linear_params(model)

    # K = the worst-case |intercept| + sum(|coef_i| * raw_bound_i / scale_i),
    # i.e. the accumulator's worst-case magnitude with std_scale=coef_scale=1.
    # int32_t needs std_scale * coef_scale * K <= _INT32_BUDGET; since either
    # scale can shrink to a floor of 1 but no lower, that product is capped
    # by combined_budget = _INT32_BUDGET / K, split between the two stages.
    k = abs(intercept) + sum(
        abs(float(c)) * _raw_bound(name, round(float(m))) / float(s)
        for name, m, s, c in zip(feature_names, mean, scale, coef)
    )
    combined_budget = _INT32_BUDGET / k if k > 0 else float("inf")

    mean_fixed, inv_scale_mult, std_scale = _standardise_fixed_params(
        feature_names, scale, mean, combined_budget
    )
    coef_scale = _pick_coef_scale(coef, std_scale, combined_budget)
    coef_fixed = [round(float(c) * coef_scale) for c in coef]
    intercept_fixed = round(intercept * std_scale * coef_scale)
    total_scale = std_scale * coef_scale
    return feature_names, mean_fixed, inv_scale_mult, coef_fixed, intercept_fixed, total_scale


def convert_to_c_linear(
    model_path: str | Path, out_dir: str | Path, func_name: str = "mlof_predict_pdr"
) -> tuple[Path, Path] | None:
    model = joblib.load(model_path)
    if _linear_model_type(model) not in LINEAR_SUPPORTED_ESTIMATORS:
        return None

    feature_names, mean_fixed, inv_scale_mult, coef_fixed, intercept_fixed, total_scale = (
        _linear_fixed_params(model)
    )
    std_func_name = f"{func_name}_standardise"

    body_lines = [
        f"static void {std_func_name}({_c_params(feature_names)}, int32_t *out) {{",
        *(
            f"    out[{i}] = ((int32_t){name} - {mf}L) * {ism}L;"
            for i, (name, mf, ism) in enumerate(zip(feature_names, mean_fixed, inv_scale_mult))
        ),
        "}",
        "",
        f"uint16_t {func_name}({_c_params(feature_names)}) {{",
        f"    int32_t z[{len(feature_names)}];",
        f"    {std_func_name}({', '.join(feature_names)}, z);",
        f"    int32_t sum = {intercept_fixed}L;",
        *(f"    sum += {cf}L * z[{i}];" for i, cf in enumerate(coef_fixed)),
        # total_scale is only ever a divisor (never a multiply operand), so
        # it's safe even if it needs the full 64 bits -- but keep it in a
        # plain `long` when it fits, so the common case stays a cheap 32-bit
        # divide instead of pulling in __divdi3 for no reason.
        f"    int32_t result = sum / {total_scale}{'L' if total_scale <= _INT32_BUDGET else 'LL'};",
        "    if (result < 0) result = 0;",
        "    if (result > 65535) result = 65535;",
        "    return (uint16_t)result;",
        "}",
    ]
    source_body = "\n".join(body_lines)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = out_dir / f"{func_name}.c"
    h_path = out_dir / f"{func_name}.h"

    header_guard = f"{func_name.upper()}_H_"
    header = f"""\
/* Auto-generated by scripts/models/to_c.py (fixed-point, no float/double)
 * from a trained model. DO NOT EDIT BY HAND -- retrain and re-run the
 * converter instead.
 *
 * {func_name}() standardises its inputs (subtract mean, scale by the
 * inverse std, both as integers) before applying the linear model, matching
 * how the model was trained.
 */
#ifndef {header_guard}
#define {header_guard}
#include <stdint.h>

uint16_t {func_name}({_c_params(feature_names)});

#endif /* {header_guard} */
"""
    source = (
        "/* Auto-generated by scripts/models/to_c.py (fixed-point, no "
        "float/double) from a trained model.\n"
        " * DO NOT EDIT BY HAND -- retrain and re-run the converter "
        "instead. */\n"
        f'#include "{func_name}.h"\n\n' + source_body + "\n"
    )

    c_path.write_text(source)
    h_path.write_text(header)
    return c_path, h_path


def _drop_unknown_rows(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Drop rows carrying an "unknown" sentinel, matching train.py's grid_search()

    filter_unknown behaviour. Without this, verify_* would score the model on
    inputs it was never trained on (e.g. etx=32767) -- a linear model in
    particular extrapolates wildly there, drowning out the real quantization
    error with a difference that has nothing to do with fixed-point rounding.
    """
    for col, sentinel in UNKNOWN_SENTINELS.items():
        if col in feature_names and col in df.columns:
            df = df[df[col] != sentinel]
    return df


def _linear_predict_fixed(
    mean_fixed: list[int],
    inv_scale_mult: list[int],
    coef_fixed: list[int],
    intercept_fixed: int,
    total_scale: int,
    rows: np.ndarray,
) -> np.ndarray:
    """Emulate the generated C exactly, int32_t wraparound included.

    Computed once in int64 (the "true" value) and once in int32 (matching
    the MCU); a mismatch means the scale-picker let something overflow and
    is a real bug, not just quantization noise.
    """
    mean_arr = np.array(mean_fixed, dtype=np.int64)
    mult_arr = np.array(inv_scale_mult, dtype=np.int64)
    z64 = (rows.astype(np.int64) - mean_arr) * mult_arr
    z32 = z64.astype(np.int32)
    if not np.array_equal(z64, z32.astype(np.int64)):
        print("[linear] WARNING: standardised value overflowed int32_t -- scale-picker bug")

    coef_arr = np.array(coef_fixed, dtype=np.int64)
    sums64 = intercept_fixed + z32.astype(np.int64) @ coef_arr
    sums32 = sums64.astype(np.int32)
    if not np.array_equal(sums64, sums32.astype(np.int64)):
        print("[linear] WARNING: accumulator overflowed int32_t -- scale-picker bug")

    sums = sums32.astype(np.int64)
    results = np.where(sums >= 0, sums // total_scale, -(-sums // total_scale))
    return np.clip(results, 0, 65535)


def verify_linear(model_path: str | Path, data_dir: str | Path) -> float | None:
    model = joblib.load(model_path)
    if _linear_model_type(model) not in LINEAR_SUPPORTED_ESTIMATORS:
        return None

    feature_names, mean_fixed, inv_scale_mult, coef_fixed, intercept_fixed, total_scale = (
        _linear_fixed_params(model)
    )
    df = pd.read_csv(Path(data_dir) / "train.csv")
    df = df.dropna(subset=feature_names)
    df = _drop_unknown_rows(df, feature_names)
    rows = df[feature_names].to_numpy()

    float_preds = model.predict(df[feature_names])
    fixed_preds = _linear_predict_fixed(
        mean_fixed, inv_scale_mult, coef_fixed, intercept_fixed, total_scale, rows
    )

    mae = float(np.abs(fixed_preds - float_preds).mean())
    print(
        f"[linear] quantization error vs float model: MAE={mae:.2f} "
        f"(pdr scale 0-65535, total_scale={total_scale}, over {len(df)} rows)"
    )
    return mae


def verify_fixed(model_path: str | Path, data_dir: str | Path) -> float | None:
    model = joblib.load(model_path)
    model_type = type(model).__name__
    if model_type not in FIXED_SUPPORTED_ESTIMATORS:
        return None

    feature_names = _fixed_feature_names(model)
    df = pd.read_csv(Path(data_dir) / "train.csv")
    df = df.dropna(subset=feature_names)
    df = _drop_unknown_rows(df, feature_names)
    rows = df[feature_names].to_numpy()

    float_preds = model.predict(df[feature_names])
    if model_type == "DecisionTreeRegressor":
        fixed_preds = _dtree_predict_fixed(model, rows)
    elif model_type == "LGBMRegressor":
        fixed_preds = _lgbm_predict_fixed(model, rows)
    elif model_type == "RandomForestRegressor":
        fixed_preds = _rf_predict_fixed(model, rows)
    elif model_type == "XGBRegressor":
        fixed_preds = _xgb_predict_fixed(model, rows, feature_names)
    else:
        fixed_preds = _catboost_predict_fixed(model, rows, feature_names)

    mae = float(np.abs(fixed_preds - float_preds).mean())
    print(
        f"[fixed] quantization error vs float model: MAE={mae:.2f} "
        f"(pdr scale 0-65535, over {len(df)} rows)"
    )
    return mae


def measure_size(c_path: str | Path, mcu: str = "msp430f2617") -> str | None:
    gcc = shutil.which("msp430-gcc")
    size_tool = shutil.which("msp430-size")
    if gcc is None or size_tool is None:
        print("msp430-gcc/msp430-size not found -- skipping size measurement.")
        return None

    c_path = Path(c_path)
    with tempfile.TemporaryDirectory() as tmp:
        o_path = Path(tmp) / (c_path.stem + ".o")
        result = subprocess.run(
            [
                gcc,
                f"-mmcu={mcu}",
                "-Os",
                "-I",
                str(c_path.parent),
                "-c",
                str(c_path),
                "-o",
                str(o_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("msp430-gcc FAILED to compile:\n" + result.stderr)
            return None

        size_result = subprocess.run(
            [size_tool, str(o_path)], capture_output=True, text=True
        )
        print(size_result.stdout)
        return size_result.stdout
