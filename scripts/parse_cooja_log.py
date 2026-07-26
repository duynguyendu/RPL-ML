import argparse
import os
import re
import sys
import numpy as np
import pandas as pd

LINE_RE = re.compile(r"^(\d+):(\d+):(.+)$")

RE_ETX_NUM = re.compile(r"^ETX:\s+(\d+)\.(\d+)")
RE_ETX_NONE = re.compile(r"^ETX:\s+no preferred parent")

RE_DODAG_JOINED = re.compile(
    r"^DODAG:\s+instance=(\d+)\s+version=(\d+)\s+rank=(\d+)\s+"
    r"grounded=(\d+)\s+role=(\w+)\s+dag_id=([0-9a-f:]+)\s+"
    r"preferred_parent=([0-9a-f:]+|none)"
)
RE_DODAG_NOT = re.compile(r"^DODAG:\s+not joined")

RE_ENERGEST = re.compile(
    r"^ENERGEST: CPU=(\d+)s LPM=(\d+)s DEEP_LPM=(\d+)s LISTEN=(\d+)s TRANSMIT=(\d+)s OFF=(\d+)s TOTAL=(\d+)s"
)

RE_CPU_UTIL = re.compile(r"^CPU_UTIL:\s+([\d.]+)%")
RE_CPU_NA = re.compile(r"^CPU_UTIL:\s+n/a")

RE_TX_POWER = re.compile(r"^TX_POWER:\s+(-?\d+)")

RE_TXRX = re.compile(r"Tx/Rx/MissedTx:\s+(\d+)/(\d+)/(\d+)")

RE_NOT_REACHABLE = re.compile(r"^Not reachable yet$")

RE_RECEIVED = re.compile(r"\[INFO:\s+App\s+\]\s+Received request 'hello (\d+)' from")


def parse_log(log_path):
    rows_etx, rows_dodag, rows_energest = [], [], []
    rows_cpu, rows_txpower, rows_app = [], [], []

    with open(log_path) as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            m = LINE_RE.match(raw)
            if not m:
                continue
            time_us = int(m.group(1))
            node_id = int(m.group(2))
            content = m.group(3)
            time_s = time_us / 1_000_000.0

            em = RE_ETX_NUM.match(content)
            if em:
                etx_val = int(em.group(1)) + int(em.group(2)) / 10.0
                rows_etx.append({"time_s": time_s, "node_id": node_id, "etx": etx_val})
                continue
            if RE_ETX_NONE.match(content):
                rows_etx.append({"time_s": time_s, "node_id": node_id, "etx": np.nan})
                continue

            dm = RE_DODAG_JOINED.match(content)
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
            if RE_DODAG_NOT.match(content):
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

            eg = RE_ENERGEST.match(content)
            if eg:
                rows_energest.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "cpu_ticks": int(eg.group(1)),
                        "lpm_ticks": int(eg.group(2)),
                        "tx_ticks": int(eg.group(3)),
                        "rx_ticks": int(eg.group(4)),
                        "ticks_per_sec": int(eg.group(5)),
                    }
                )
                continue

            cm = RE_CPU_UTIL.match(content)
            if cm:
                rows_cpu.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "cpu_pct": float(cm.group(1)),
                    }
                )
                continue
            if RE_CPU_NA.match(content):
                rows_cpu.append(
                    {"time_s": time_s, "node_id": node_id, "cpu_pct": np.nan}
                )
                continue

            tm = RE_TX_POWER.match(content)
            if tm:
                rows_txpower.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "tx_power_dbm": int(tm.group(1)),
                    }
                )
                continue

            if node_id == 1:
                rm = RE_RECEIVED.search(content)
                if rm:
                    rows_app.append(
                        {
                            "time_s": time_s,
                            "node_id": node_id,
                            "event": "received_request",
                            "tx_count": np.nan,
                            "rx_count": np.nan,
                            "missed_count": np.nan,
                        }
                    )
                continue

            txrx = RE_TXRX.search(content)
            if txrx:
                rows_app.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "event": "tx_rx_stats",
                        "tx_count": int(txrx.group(1)),
                        "rx_count": int(txrx.group(2)),
                        "missed_count": int(txrx.group(3)),
                    }
                )
                continue

            if RE_NOT_REACHABLE.match(content):
                rows_app.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "event": "not_reachable",
                        "tx_count": np.nan,
                        "rx_count": np.nan,
                        "missed_count": np.nan,
                    }
                )
                continue

    def _df(rows):
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return {
        "etx": _df(rows_etx),
        "dodag": _df(rows_dodag),
        "energest": _df(rows_energest),
        "cpu_util": _df(rows_cpu),
        "tx_power": _df(rows_txpower),
        "app_events": _df(rows_app),
    }


def save_data(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, df in data.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse COOJA simulation logs.")
    parser.add_argument(
        "--log", default="simulation_logs/COOJA.testlog", help="Path to COOJA.testlog"
    )
    parser.add_argument(
        "--output-dir",
        default="parsed_data",
        help="Where to save parsed CSVs (default: parsed_data)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Error: log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)
    print(f"Parsing {args.log} ...")
    data = parse_log(args.log)
    print(f"Saving parsed DataFrames to {args.output_dir}/ ...")
    save_data(data, args.output_dir)

    print("\n--- Data Summary ---")
    for name, df in data.items():
        if df.empty:
            print(f"  {name:12s}: empty")
        else:
            n = df["node_id"].nunique() if "node_id" in df.columns else "?"
            print(f"  {name:12s}: {len(df)} rows, {n} nodes")

    print("\nDone.")


if __name__ == "__main__":
    main()
