"""Feature/label schema and training-data gathering for the PDR-predicting
LightGBM model used by the "mlof" RPL objective function (see
rpl/contiki-ng/os/net/routing/rpl-lite/rpl-mlof.c).

Each training row is one (node, time chunk) within one run: the node's
link/system state averaged over that chunk, paired with the PDR it achieved
during it.

Chunking (per node, per run): a chunk starts at every parent switch (each
dodag.csv row) and is split into consecutive <=MAX_CHUNK_SECONDS sub-chunks;
the last interval in a run runs to config["duration"] + config["ramp_up_duration"].
The ramp-up period (config["ramp_up_duration"]) is excluded, since network
formation then isn't representative of steady-state behavior.

# TODO: missing/invalid data (NaN) is left unhandled -- see _own_metrics(),
# _node_chunk_features() and _chunk_pdr() for where/why NaNs occur. Decide
# drop vs. impute vs. something else before training on this.
# TODO: metrics.c leaves etx/rssi at raw sentinel values (etx ~4.29e9, rssi
# -1) and packet counters at 0 when a node has no preferred parent --
# confirmed on 46% of metrics.csv rows in one real run, corrupting every
# own-node feature (and _window_delta() packet counts) whenever it lands in
# a chunk's window. Left unresolved.
# TODO: parent comparison should account for the load switching would add to
# the potential parent's CPU.
# TODO: track how long the current parent has been held.
# TODO: cpu_util is not yet computed for add_metrics()/train.csv (unlike
# gather_training_data(), which has _cpu_util_delta()) -- see add_metrics().
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV_NAMES = ["dodag.csv", "latency.csv", "metrics.csv"]

FEATURE_COLUMNS = [
    "etx",
    "hop_count",
    "ppm",
    "cpu_util",
    "parent_cpu_util",
    "rssi",
    "tx_packets",
    "acked_packets",
    "dropped_packets",
]
LABEL_COLUMN = "pdr"
MAX_CHUNK_SECONDS = 60.0


def _chunk_bounds(
    start: float, end: float, max_len: float
) -> list[tuple[float, float]]:
    """Split [start, end) into consecutive max_len-wide windows (a shorter
    final window covers the remainder)."""
    bounds = []
    t = start
    while t < end:
        t2 = min(t + max_len, end)
        bounds.append((t, t2))
        t = t2
    return bounds


def _window_delta(rows: pd.DataFrame, col: str) -> float:
    """Change in rows[col] over the window (last row minus first row), rather
    than its raw cumulative-since-boot magnitude. NaN if fewer than two rows
    fall in the window."""
    if len(rows) < 2:
        return float("nan")
    return float(rows[col].iloc[-1] - rows[col].iloc[0])


def _cpu_util_delta(rows: pd.DataFrame) -> float:
    """CPU utilization (%) during the rows' span: delta(cpu_ticks) /
    delta(total_ticks) * 100, matching rpl-mlof.c's on-mote
    cpu_usage_percent(). NaN if fewer than two rows fall in the window, or
    ticks don't advance."""
    delta_cpu = _window_delta(rows, "cpu_ticks")
    delta_total = _window_delta(rows, "total_ticks")
    if pd.isna(delta_total) or delta_total <= 0:
        return float("nan")
    return delta_cpu / delta_total * 100


def _window(
    df: pd.DataFrame, start: float, end: float, col: str = "time_s"
) -> pd.DataFrame:
    """Rows of df (already filtered to one node_id) with df[col] in [start, end)."""
    return df[(df[col] >= start) & (df[col] < end)]


def _own_metrics(own_rows: pd.DataFrame) -> dict:
    """etx/hop_count/rssi (mean) and tx_packets/acked_packets/dropped_packets
    (delta over the window, see _window_delta()) from one node's own
    metrics.csv rows, already filtered to one chunk window. NaN wherever the
    node has no metrics.csv rows in that window; does not include cpu_util
    (see _node_chunk_features() / add_metrics())."""

    def _mean(col):
        return float(own_rows[col].mean()) if len(own_rows) else float("nan")

    return {
        "etx": _mean("etx"),
        "hop_count": _mean("hop_count"),
        "rssi": _mean("rssi"),
        # Cumulative-since-boot, per-preferred-parent link_stats counters
        # (see rpl/motes/client/metrics.c), delta'd to this chunk's own
        # contribution -- safe since callers scope own_rows to one parent
        # interval (see build_chunks()), so the delta never crosses a switch.
        "tx_packets": _window_delta(own_rows, "tx_packets"),
        "acked_packets": _window_delta(own_rows, "acked_packets"),
        "dropped_packets": _window_delta(own_rows, "dropped_packets"),
    }


def _node_chunk_features(
    metrics_by_node: dict, node_id: int, parent_id: int, start: float, end: float
) -> dict:
    """_own_metrics() for node_id, plus cpu_util/parent_cpu_util (delta'd
    over [start, end), see _cpu_util_delta())."""
    own = metrics_by_node.get(node_id)
    own_rows = _window(own, start, end) if own is not None else pd.DataFrame()
    parent = metrics_by_node.get(parent_id)
    parent_rows = _window(parent, start, end) if parent is not None else pd.DataFrame()

    return {
        **_own_metrics(own_rows),
        "cpu_util": _cpu_util_delta(own_rows),
        "parent_cpu_util": _cpu_util_delta(parent_rows),
    }


def _chunk_pdr(latency_node: pd.DataFrame, start: float, end: float) -> float:
    """PDR over [start, end): received / sent, bucketed by client_send_time
    (mirrors plot_metrics.py's compute_pdr(), applied per chunk). NaN if
    nothing was sent in the window (0/0)."""
    rows = _window(latency_node, start, end, col="client_send_time")
    sent = len(rows)
    if sent == 0:
        return float("nan")
    received = int((rows["server_receive_time"] != 0).sum())
    return received / sent


def build_chunks(run_dir: Path) -> pd.DataFrame:
    """Build the (node, chunk) boundary table for one run: parent held plus
    chunk start/end/duration (see the module docstring for the chunking
    rule), from dodag.csv + config.json only -- no etx/cpu_util/rssi/pdr yet
    (see _gather_run(), which layers those on top).

    Rows carry no run_id; callers tracing a chunk back to its run (e.g.
    gather_chunks()) should attach one themselves (e.g. run_dir.name).
    """
    with open(run_dir / "config.json") as fh:
        cfg = json.load(fh)
    ramp_up_end = cfg["ramp_up_duration"]
    run_end = cfg["duration"] + ramp_up_end

    dodag = pd.read_csv(run_dir / "dodag.csv")

    rows = []
    for node_id, switches in dodag.groupby("node_id"):
        switches = switches.sort_values("time_s")
        starts = switches["time_s"].tolist()
        parents = switches["parent_id"].tolist()

        for i, (interval_start, parent_id) in enumerate(zip(starts, parents)):
            interval_end = starts[i + 1] if i + 1 < len(starts) else run_end
            interval_start = max(interval_start, ramp_up_end)
            if interval_end <= interval_start:
                continue
            for chunk_start, chunk_end in _chunk_bounds(
                interval_start, interval_end, MAX_CHUNK_SECONDS
            ):
                rows.append(
                    {
                        "node_id": node_id,
                        "parent_id": parent_id,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "chunk_duration": chunk_end - chunk_start,
                    }
                )

    return pd.DataFrame(rows)


def add_metrics(chunks: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Merge each chunk's own-node metrics.csv stats onto chunks (as
    returned by build_chunks()): see _own_metrics() for how etx/hop_count/
    rssi/tx_packets/acked_packets/dropped_packets are computed.

    # TODO: cpu_util is not added here -- decide whether to reuse
    # gather_training_data()'s _cpu_util_delta() (and, if so, whether to
    # also add parent_cpu_util, see _node_chunk_features()).
    """
    metric_cols = ["etx", "hop_count", "rssi", "tx_packets", "acked_packets", "dropped_packets"]
    if chunks.empty:
        return chunks.assign(**{col: pd.Series(dtype=float) for col in metric_cols})

    metrics = pd.read_csv(run_dir / "metrics.csv")
    metrics_by_node = {
        node_id: grp.sort_values("time_s") for node_id, grp in metrics.groupby("node_id")
    }

    def _row_metrics(row) -> pd.Series:
        own = metrics_by_node.get(row.node_id)
        own_rows = _window(own, row.chunk_start, row.chunk_end) if own is not None else pd.DataFrame()
        return pd.Series(_own_metrics(own_rows))

    return pd.concat(
        [chunks.reset_index(drop=True), chunks.apply(_row_metrics, axis=1)], axis=1
    )


def find_dirs_with_data(data_dir: Path):
    """Immediate subdirectories of data_dir with dodag.csv, config.json and
    metrics.csv -- everything build_chunks() + add_metrics() need."""
    needed = ["dodag.csv", "config.json", "metrics.csv"]
    for sub_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if all((sub_dir / name).exists() for name in needed):
            yield sub_dir


def gather_chunks(data_dir: Path) -> pd.DataFrame:
    """build_chunks() + add_metrics() across every run subdirectory found
    under data_dir (one level deep, see find_dirs_with_data())."""
    data_dir = Path(data_dir)
    frames = [
        add_metrics(build_chunks(run_dir), run_dir)
        for run_dir in find_dirs_with_data(data_dir)
    ]
    cols = [
        "node_id",
        "parent_id",
        "chunk_start",
        "chunk_end",
        "chunk_duration",
        "etx",
        "hop_count",
        "rssi",
        "tx_packets",
        "acked_packets",
        "dropped_packets",
    ]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def _iter_run_dirs(runs_dir: Path):
    """Run directories under runs_dir with config.json, metrics.csv,
    latency.csv and dodag.csv (mirrors plot_comparison.py's
    collect_runs())."""
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        needed = ["config.json", "metrics.csv", "latency.csv", "dodag.csv"]
        if all((run_dir / name).exists() for name in needed):
            yield run_dir


def _gather_run(run_dir: Path) -> pd.DataFrame:
    with open(run_dir / "config.json") as fh:
        cfg = json.load(fh)
    ppm = cfg["ppm"]

    metrics = pd.read_csv(run_dir / "metrics.csv")
    latency = pd.read_csv(run_dir / "latency.csv")

    metrics_by_node = {
        node_id: grp.sort_values("time_s")
        for node_id, grp in metrics.groupby("node_id")
    }
    latency_by_node = {
        node_id: grp.sort_values("client_send_time")
        for node_id, grp in latency.groupby("node_id")
    }

    rows = []
    for chunk in build_chunks(run_dir).itertuples():
        latency_node = latency_by_node.get(
            chunk.node_id, pd.DataFrame(columns=latency.columns)
        )
        features = _node_chunk_features(
            metrics_by_node,
            chunk.node_id,
            chunk.parent_id,
            chunk.chunk_start,
            chunk.chunk_end,
        )
        rows.append(
            {
                "run_id": run_dir.name,
                "node_id": chunk.node_id,
                "parent_id": chunk.parent_id,
                "chunk_start": chunk.chunk_start,
                "chunk_end": chunk.chunk_end,
                "ppm": ppm,
                **features,
                "pdr": _chunk_pdr(latency_node, chunk.chunk_start, chunk.chunk_end),
            }
        )

    return pd.DataFrame(rows)


def gather_training_data(runs_dir: Path) -> pd.DataFrame:
    """Build a (node, time chunk) feature/label table across every run under
    runs_dir (see module docstring for the chunking rule). Includes
    identifying columns (run_id, node_id, parent_id, chunk_start, chunk_end)
    alongside FEATURE_COLUMNS + LABEL_COLUMN; select FEATURE_COLUMNS
    explicitly for model inputs (as models/train.py does).
    """
    runs_dir = Path(runs_dir)
    run_frames = [_gather_run(run_dir) for run_dir in _iter_run_dirs(runs_dir)]
    if not run_frames:
        print(f"No usable run directories found under {runs_dir} -- returning no rows.")
        cols = (
            ["run_id", "node_id", "parent_id", "chunk_start", "chunk_end"]
            + FEATURE_COLUMNS
            + [LABEL_COLUMN]
        )
        return pd.DataFrame(columns=cols)
    return pd.concat(run_frames, ignore_index=True)


def make_dummy_training_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic stand-in for gather_training_data(), so training/eval code
    can be exercised end-to-end before real training data exists. Not a
    substitute for real data — values are random within plausible ranges and
    have no relationship to real network behavior.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "etx": rng.uniform(1.0, 4.0, n),
            "hop_count": rng.integers(1, 6, n),
            "ppm": rng.choice([15, 20, 30, 40, 60], n),
            "cpu_util": rng.uniform(0.0, 100.0, n),
            "parent_cpu_util": rng.uniform(0.0, 100.0, n),
            "rssi": rng.uniform(-95.0, -40.0, n),
            "tx_packets": rng.integers(0, 5000, n),
            "acked_packets": rng.integers(0, 5000, n),
            "dropped_packets": rng.integers(0, 100, n),
        }
    )
    df[LABEL_COLUMN] = np.clip(
        1.0 - 0.15 * df["etx"] - 0.05 * df["hop_count"] + rng.normal(0, 0.05, n),
        0.0,
        1.0,
    )
    return df


def _find_raw_run_dirs(data_dir: Path):
    for sub_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if all((sub_dir / name).exists() for name in RAW_CSV_NAMES):
            yield sub_dir


def load_raw_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    rows_by_name: dict[str, list[pd.DataFrame]] = {name: [] for name in RAW_CSV_NAMES}
    for run_dir in _find_raw_run_dirs(data_dir):
        for name in RAW_CSV_NAMES:
            print(name)
            df = pd.read_csv(run_dir / name)
            df.insert(0, "run_id", run_dir.name)
            rows_by_name[name].append(df)

    return {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in rows_by_name.items()
    }


def save_raw_csvs(combined: dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Write each combined CSV (as returned by load_raw_csvs()) back into
    data_dir, under its original filename (e.g. data_dir/dodag.csv)."""
    data_dir = Path(data_dir)
    for name, df in combined.items():
        out_path = data_dir / name
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build train.csv: the (node_id, parent_id, chunk_start, "
            "chunk_end, chunk_duration) chunk-boundary table (see "
            "build_chunks()) plus each chunk's own-node etx/hop_count/rssi/"
            "tx_packets/acked_packets/dropped_packets from metrics.csv (see "
            "add_metrics() -- cpu_util is a TODO, not included yet), across "
            "every run subdirectory found under --data-dir that has "
            "dodag.csv + config.json + metrics.csv (one level deep). PDR is "
            "a separate, later step."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory whose immediate subdirectories each hold one run's "
        "dodag.csv + config.json + metrics.csv",
    )
    args = parser.parse_args()

    run_dirs = list(find_dirs_with_data(args.data_dir))
    print(f"Found {len(run_dirs)} run dir(s) under {args.data_dir}:")
    for run_dir in run_dirs:
        print(f"  {run_dir.name}")

    chunks = gather_chunks(args.data_dir)
    train_path = args.data_dir / "train.csv"
    chunks.to_csv(train_path, index=False)
    print(f"  Saved {train_path} ({len(chunks)} rows)")
