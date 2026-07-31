import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

LINE_RE = re.compile(r"^(\d+):(\d+):(.+)$")

RE_ETX_LOG = re.compile(r"^ETX:\s+(\d+)\.(\d+)")
RE_ETX_NOT_FOUND = re.compile(r"^ETX:\s+no preferred parent")

RE_DODAG_LOG = re.compile(
    r"^DODAG:\s+instance=(\d+)\s+version=(\d+)\s+rank=(\d+)\s+"
    r"grounded=(\d+)\s+role=(\w+)\s+dag_id=([0-9a-f:]+)\s+"
    r"preferred_parent=([0-9a-f:]+|none)"
)
RE_DODAG_NOT_JOIN = re.compile(r"^DODAG:\s+not joined")

RE_ENERGEST_LOG = re.compile(
    r"^ENERGEST: CPU=(\d+) LPM=(\d+) DEEP_LPM=(\d+) LISTEN=(\d+) TRANSMIT=(\d+) OFF=(\d+) TOTAL=(\d+)"
)

# TODO: hop count is being evaluated
RE_HOP_COUNT = re.compile(r"HOP_COUNT: (\d+)")

RE_LATENCY_LOG = re.compile(r"LATENCY: seqno=(\d+) rtt_tick=(\d+) rtt_ms=(\d+)")

# RE_TXRX = re.compile(r"Tx/Rx/MissedTx:\s+(\d+)/(\d+)/(\d+)")
# RE_NOT_REACHABLE = re.compile(r"^Not reachable yet$")
# RE_RECEIVED = re.compile(r"\[INFO:\s+App\s+\]\s+Received request 'hello (\d+)' from")

METRIC_PERIOD = 10


def normalise_time(time: float) -> int:
    return int((time + METRIC_PERIOD / 2) / METRIC_PERIOD) * METRIC_PERIOD


def process_log(log_path):
    rows_etx, rows_dodag, rows_energest = [], [], []
    rows_latency, rows_hop_count = [], []

    with open(log_path) as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            m = LINE_RE.match(raw)
            if not m:
                continue
            time_us = int(m.group(1))
            node_id = int(m.group(2))
            content = m.group(3)
            time_s = normalise_time(time_us / 1_000_000.0)

            em = RE_ETX_LOG.match(content)
            if em:
                etx_val = int(em.group(1)) + int(em.group(2)) / 10.0
                rows_etx.append({"time_s": time_s, "node_id": node_id, "etx": etx_val})
                continue
            if RE_ETX_NOT_FOUND.match(content):
                rows_etx.append({"time_s": time_s, "node_id": node_id, "etx": np.nan})
                continue

            dm = RE_DODAG_LOG.match(content)
            if dm:
                rows_dodag.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "instance": int(dm.group(1)),
                        "version": int(dm.group(2)),
                        "rank": int(dm.group(3)),
                        "grounded": int(dm.group(4)),
                        "role": dm.group(5),
                        "dag_id": dm.group(6),
                        "preferred_parent": dm.group(7),
                    }
                )
                continue
            if RE_DODAG_NOT_JOIN.match(content):
                rows_dodag.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "instance": np.nan,
                        "version": np.nan,
                        "rank": 65535,
                        "grounded": np.nan,
                        "role": np.nan,
                        "dag_id": np.nan,
                        "preferred_parent": np.nan,
                    }
                )
                continue

            eg = RE_ENERGEST_LOG.match(content)
            if eg:
                rows_energest.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "cpu_ticks": int(eg.group(1)),
                        "lpm_ticks": int(eg.group(2)),
                        "deep_lpm_ticks": int(eg.group(3)),
                        "tx_ticks": int(eg.group(4)),
                        "rx_ticks": int(eg.group(5)),
                        "off_ticks": int(eg.group(6)),
                        "total_ticks": int(eg.group(7)),
                    }
                )
                continue

            latency = RE_LATENCY_LOG.match(content)
            if latency:
                rows_latency.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "seqno": int(latency.group(1)),
                        "rtt_tick": int(latency.group(2)),
                        "rtt_ms": int(latency.group(3)),
                    }
                )

    def _df(rows):
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return {
        "etx": _df(rows_etx),
        "dodag": _df(rows_dodag),
        "energest": _df(rows_energest),
        "latency": _df(rows_latency),
    }


def save_data(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, df in data.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")


def parse_log(log_dir: str | None = None, output_dir: str | None = None) -> None:
    log_name = "COOJA.testlog"
    if log_dir is None:
        log_dir = os.getcwd()
    if output_dir is None:
        output_dir = os.getcwd()
    log_file = Path(log_dir).resolve() / log_name

    if not os.path.exists(log_file):
        print(f"Error: log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)
    print(f"Parsing {log_file} ...")
    data = process_log(log_file)
    print(f"Saving parsed DataFrames to {output_dir}/ ...")
    save_data(data, output_dir)

    print("\n--- Data Summary ---")
    for name, df in data.items():
        if df.empty:
            print(f"  {name:12s}: empty")
        else:
            n = df["node_id"].nunique() if "node_id" in df.columns else "?"
            print(f"  {name:12s}: {len(df)} rows, {n} nodes")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse COOJA simulation logs.")
    parser.add_argument("--log-dir", default=None, help="Path to COOJA.testlog")
    parser.add_argument(
        "--output-dir",
        help="Where to save parsed CSVs (default: parsed_data)",
    )
    args = parser.parse_args()
    parse_log(log_dir=args.log_dir, output_dir=args.output_dir)
