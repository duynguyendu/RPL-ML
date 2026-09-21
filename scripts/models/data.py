from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_config import FEATURE_COLUMNS, LABEL_COLUMN

LINE_RE = re.compile(r"^(\d+):(\d+):(.+)$")
RE_MLOF_METRICS = re.compile(
    r"^\[PRI : RPL       \] MLOF metrics: is_new=(\d+) parent_id=(\d+) cpu=(\d+) "
    r"p_cpu=(\d+) etx=(\d+) rssi=(-?\d+) ppm=(\d+) drop_rate=(\d+) "
    r"parent_ppm=(\d+) parent_drop_rate=(\d+) hop_count=(\d+) nbr_count=(\d+)"
)
RE_CLIENT_SEND = re.compile(r"Sending request '(\d+)' to")
RE_CLIENT_SKIP = re.compile(
    r"Skipping request '(\d+)': root not (registered|reachable)"
)
RE_SERVER_RECEIVE = re.compile(r"Sending response '(\d+)' to ([0-9a-f:]+)")

MIN_CHUNK_SECONDS = 15.0
MAXUINT16 = 65535


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
                        "parent_ppm": int(mlof.group(9)),
                        "parent_drop_rate": int(mlof.group(10)),
                        "hop_count": int(mlof.group(11)),
                        "nbr_count": int(mlof.group(12)),
                    }
                )
                continue

            send = RE_CLIENT_SEND.match(content)
            if send:
                rows_send.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "seqno": int(send.group(1)),
                    }
                )
                continue

            skip = RE_CLIENT_SKIP.match(content)
            if skip:
                rows_send.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "seqno": int(skip.group(1)),
                    }
                )
                continue

            recv = RE_SERVER_RECEIVE.match(content)
            if recv:
                client_id = int(recv.group(2).split(":")[5], 16)
                rows_recv.append(
                    {
                        "time_s": time_s,
                        "node_id": client_id,
                        "seqno": int(recv.group(1)),
                    }
                )
                continue

    return pd.DataFrame(rows_mlof), pd.DataFrame(rows_send), pd.DataFrame(rows_recv)


def _window(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["time_s"] >= start) & (df["time_s"] < end)]


def _chunk_pdr(
    send_node: pd.DataFrame, received_seqnos: set, start: float, end: float
) -> float:
    window = _window(send_node, start, end)
    sent = len(window)
    if sent == 0:
        return float("nan")
    received = window["seqno"].isin(received_seqnos).sum()
    return received / sent * MAXUINT16


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
        received_seqnos = set(recv_node["seqno"]) if not recv_node.empty else set()

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
                    "pdr": _chunk_pdr(
                        send_node, received_seqnos, chunk_start, chunk_end
                    ),
                }
            )

    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)


def _is_mlof_run(run_dir: Path) -> bool:
    with open(run_dir / "config.json") as fh:
        cfg = json.load(fh)
    return cfg.get("rpl_of", "").startswith("mlof")


def find_dirs_with_data(data_dir: Path):
    needed = ["COOJA.testlog", "config.json"]
    for sub_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if all((sub_dir / name).exists() for name in needed) and _is_mlof_run(sub_dir):
            yield sub_dir


def gather_chunks(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = [build_chunks(run_dir) for run_dir in find_dirs_with_data(data_dir)]
    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def gather_training_data(runs_dir: Path) -> pd.DataFrame:
    return gather_chunks(runs_dir)


def process_data(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    run_dirs = list(find_dirs_with_data(data_dir))
    print(f"Found {len(run_dirs)} run dir(s) under {data_dir}:")
    for run_dir in run_dirs:
        print(f"  {run_dir.name}")

    chunks = gather_chunks(data_dir)
    train_path = data_dir / "train.csv"
    chunks.to_csv(train_path, index=False)
    print(f"  Saved {train_path} ({len(chunks)} rows)")
    return chunks


if __name__ == "__main__":
    import train_config

    process_data(train_config.data_dir)
