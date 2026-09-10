from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

LINE_RE = re.compile(r"^(\d+):(\d+):(.+)$")
RE_MLOF_METRICS = re.compile(
    r"^\[PRI : RPL       \] MLOF metrics: is_new=(\d+) parent_id=(\d+) cpu=(\d+) "
    r"p_cpu=(\d+) etx=(\d+) rssi=(-?\d+) ppm=(\d+) drop_rate=(\d+) hop_count=(\d+) "
    r"nbr_count=(\d+)"
)
RE_CLIENT_SEND = re.compile(r"Sending request '(\d+)' to")
RE_CLIENT_SKIP = re.compile(
    r"Skipping request '(\d+)': root not (registered|reachable)"
)
RE_CLIENT_RECEIVE = re.compile(r"HOP_COUNT=(\d+) Received response '(\d+)' from")

FEATURE_COLUMNS = [
    "is_new",
    "cpu",
    "p_cpu",
    "etx",
    "rssi",
    "ppm",
    "drop_rate",
    "hop_count",
    "nbr_count",
]
LABEL_COLUMN = "pdr"

MIN_CHUNK_SECONDS = 15.0

# TODO: missing data
#   - etx/rssi/ppm: 32767 means unknown (no parent yet, the parent's own
#     value is itself unknown, or -- for ppm -- the 30s traffic window
#     hasn't elapsed since the last parent change/counter wrap). etx=32767
#     in ~1% of MLOF log lines in that run.


def _parse_log(log_path: Path):
    rows_mlof = []
    rows_send = []
    rows_recv = []

    with open(log_path) as fh:
        for raw in fh:
            m = LINE_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            time_s = int(m.group(1)) / 1_000_000.0
            node_id = int(m.group(2))
            content = m.group(3)

            mlof = RE_MLOF_METRICS.match(content)
            if mlof:
                rows_mlof.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "is_new": int(mlof.group(1)),
                        "parent_id": int(mlof.group(2)),
                        "cpu": int(mlof.group(3)),
                        "p_cpu": int(mlof.group(4)),
                        "etx": int(mlof.group(5)),
                        "rssi": int(mlof.group(6)),
                        "ppm": int(mlof.group(7)),
                        "drop_rate": int(mlof.group(8)),
                        "hop_count": int(mlof.group(9)),
                        "nbr_count": int(mlof.group(10)),
                    }
                )
                continue

            send = RE_CLIENT_SEND.match(content)
            if send:
                rows_send.append({"time_s": time_s, "node_id": node_id})
                continue

            skip = RE_CLIENT_SKIP.match(content)
            if skip:
                rows_send.append({"time_s": time_s, "node_id": node_id})
                continue

            recv = RE_CLIENT_RECEIVE.match(content)
            if recv:
                rows_recv.append({"time_s": time_s, "node_id": node_id})
                continue

    return pd.DataFrame(rows_mlof), pd.DataFrame(rows_send), pd.DataFrame(rows_recv)


def _window(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["time_s"] >= start) & (df["time_s"] < end)]


def _chunk_pdr(
    send_node: pd.DataFrame, recv_node: pd.DataFrame, start: float, end: float
) -> float:
    sent = len(_window(send_node, start, end))
    if sent == 0:
        return float("nan")
    received = len(_window(recv_node, start, end))
    return received / sent * 100


_OUTPUT_COLUMNS = [
    "node_id",
    "parent_id",
    "chunk_start",
    "chunk_end",
    "chunk_duration",
    *FEATURE_COLUMNS,
    LABEL_COLUMN,
]


def build_chunks(run_dir: Path) -> pd.DataFrame:
    with open(run_dir / "config.json") as fh:
        cfg = json.load(fh)
    ramp_up_end = cfg["ramp_up_duration"]
    run_end = cfg["duration"] + ramp_up_end

    mlof, send, recv = _parse_log(run_dir / "COOJA.testlog")
    if mlof.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    rows = []
    for node_id, node_rows in mlof.groupby("node_id"):
        node_rows = node_rows.sort_values("time_s").reset_index(drop=True)
        send_node = send[send["node_id"] == node_id] if not send.empty else send
        recv_node = recv[recv["node_id"] == node_id] if not recv.empty else recv

        for i in range(len(node_rows)):
            row = node_rows.iloc[i]
            chunk_start = row["time_s"]
            if i + 1 < len(node_rows):
                chunk_end = node_rows.iloc[i + 1]["time_s"]
            else:
                chunk_end = run_end

            chunk_start = max(chunk_start, ramp_up_end)
            if chunk_end - chunk_start < MIN_CHUNK_SECONDS:
                continue

            rows.append(
                {
                    "node_id": node_id,
                    "parent_id": row["parent_id"],
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "chunk_duration": chunk_end - chunk_start,
                    **{col: row[col] for col in FEATURE_COLUMNS},
                    "pdr": _chunk_pdr(send_node, recv_node, chunk_start, chunk_end),
                }
            )

    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)


def find_dirs_with_data(data_dir: Path):
    needed = ["COOJA.testlog", "config.json"]
    for sub_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if all((sub_dir / name).exists() for name in needed):
            yield sub_dir


def gather_chunks(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = [build_chunks(run_dir) for run_dir in find_dirs_with_data(data_dir)]
    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def gather_training_data(runs_dir: Path) -> pd.DataFrame:
    return gather_chunks(runs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory whose immediate subdirectories each hold one run's "
        "COOJA.testlog + config.json",
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
