#!/usr/bin/env python3
"""
plot_metrics.py — Parse COOJA simulation logs and generate metric plots.

Prerequisites:
    pip install pandas matplotlib

Usage:
    # Parse log + plot
    python3 plot_metrics.py

    # Skip parsing, reuse existing CSVs
    python3 plot_metrics.py --df parsed_data/

    # Custom paths
    python3 plot_metrics.py --log simulation_logs/COOJA.testlog \
                            --outdir plots/ --parseddir parsed_data/ --dpi 200
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    r"^ENERGEST:\s+CPU\s+(\d+)\s+LPM\s+(\d+)\s+TX\s+(\d+)\s+"
    r"RX\s+(\d+)\s+\(ticks,\s+(\d+)\s+ticks/sec\)"
)

RE_CPU_UTIL = re.compile(r"^CPU_UTIL:\s+([\d.]+)%")
RE_CPU_NA = re.compile(r"^CPU_UTIL:\s+n/a")

RE_TX_POWER = re.compile(r"^TX_POWER:\s+(-?\d+)")

RE_TXRX = re.compile(r"Tx/Rx/MissedTx:\s+(\d+)/(\d+)/(\d+)")

RE_NOT_REACHABLE = re.compile(r"^Not reachable yet$")

RE_RECEIVED = re.compile(
    r"\[INFO:\s+App\s+\]\s+Received request 'hello (\d+)' from"
)


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
                rows_dodag.append({
                    "time_s": time_s, "node_id": node_id,
                    "instance": int(dm.group(1)), "version": int(dm.group(2)),
                    "rank": int(dm.group(3)), "grounded": int(dm.group(4)),
                    "role": dm.group(5), "dag_id": dm.group(6),
                    "preferred_parent": dm.group(7),
                })
                continue
            if RE_DODAG_NOT.match(content):
                rows_dodag.append({
                    "time_s": time_s, "node_id": node_id,
                    "instance": np.nan, "version": np.nan,
                    "rank": 65535, "grounded": np.nan,
                    "role": np.nan, "dag_id": np.nan, "preferred_parent": np.nan,
                })
                continue

            eg = RE_ENERGEST.match(content)
            if eg:
                rows_energest.append({
                    "time_s": time_s, "node_id": node_id,
                    "cpu_ticks": int(eg.group(1)), "lpm_ticks": int(eg.group(2)),
                    "tx_ticks": int(eg.group(3)), "rx_ticks": int(eg.group(4)),
                    "ticks_per_sec": int(eg.group(5)),
                })
                continue

            cm = RE_CPU_UTIL.match(content)
            if cm:
                rows_cpu.append({"time_s": time_s, "node_id": node_id, "cpu_pct": float(cm.group(1))})
                continue
            if RE_CPU_NA.match(content):
                rows_cpu.append({"time_s": time_s, "node_id": node_id, "cpu_pct": np.nan})
                continue

            tm = RE_TX_POWER.match(content)
            if tm:
                rows_txpower.append({"time_s": time_s, "node_id": node_id, "tx_power_dbm": int(tm.group(1))})
                continue

            if node_id == 1:
                rm = RE_RECEIVED.search(content)
                if rm:
                    rows_app.append({
                        "time_s": time_s, "node_id": node_id,
                        "event": "received_request",
                        "tx_count": np.nan, "rx_count": np.nan, "missed_count": np.nan,
                    })
                continue

            txrx = RE_TXRX.search(content)
            if txrx:
                rows_app.append({
                    "time_s": time_s, "node_id": node_id, "event": "tx_rx_stats",
                    "tx_count": int(txrx.group(1)), "rx_count": int(txrx.group(2)),
                    "missed_count": int(txrx.group(3)),
                })
                continue

            if RE_NOT_REACHABLE.match(content):
                rows_app.append({
                    "time_s": time_s, "node_id": node_id, "event": "not_reachable",
                    "tx_count": np.nan, "rx_count": np.nan, "missed_count": np.nan,
                })
                continue

    def _df(rows):
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return {
        "etx": _df(rows_etx), "dodag": _df(rows_dodag),
        "energest": _df(rows_energest), "cpu_util": _df(rows_cpu),
        "tx_power": _df(rows_txpower), "app_events": _df(rows_app),
    }


def save_data(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, df in data.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")


def load_data(in_dir):
    data = {}
    for name in ("etx", "dodag", "energest", "cpu_util", "tx_power", "app_events"):
        path = os.path.join(in_dir, f"{name}.csv")
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            print(f"  Warning: {path} not found, skipping {name}")
            data[name] = pd.DataFrame()
    return data


NODE_COLORS = {
    2: "#e6194b", 3: "#3cb44b", 4: "#4363d8", 5: "#f58231",
    6: "#911eb4", 7: "#42d4f4", 8: "#f032e6", 9: "#bfef45",
    10: "#fabed4", 11: "#469990",
}


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


def plot_etx(data, out_dir, dpi):
    df = data["etx"]
    if df.empty:
        print("  Skipping ETX plot (no data)")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for nid in sorted(df["node_id"].unique()):
        sub = df[df["node_id"] == nid].dropna(subset=["etx"])
        if sub.empty:
            continue
        ax.plot(sub["time_s"], sub["etx"], "o-", label=f"Node {nid}",
                color=NODE_COLORS.get(nid), markersize=5)
    _style_ax(ax, "ETX to Preferred RPL Parent", "Simulated Time (s)", "ETX")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "etx_per_node.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved etx_per_node.png")


def plot_rank(data, out_dir, dpi):
    df = data["dodag"]
    if df.empty:
        print("  Skipping rank plot (no data)")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for nid in sorted(df["node_id"].unique()):
        sub = df[df["node_id"] == nid]
        connected = sub["rank"] < 65535
        disconnected = ~connected
        if connected.any():
            ax.plot(sub.loc[connected, "time_s"], sub.loc[connected, "rank"],
                    "o-", label=f"Node {nid}",
                    color=NODE_COLORS.get(nid), markersize=5)
        if disconnected.any():
            ax.plot(sub.loc[disconnected, "time_s"],
                    [np.nan] * disconnected.sum(),
                    "x--", color=NODE_COLORS.get(nid), markersize=5, alpha=0.4)
    _style_ax(ax, "DODAG Rank per Node", "Simulated Time (s)", "Rank")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rank_per_node.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved rank_per_node.png")


def plot_energest(data, out_dir, dpi):
    df = data["energest"]
    if df.empty:
        print("  Skipping energest plot (no data)")
        return
    components = ["cpu_ticks", "lpm_ticks", "tx_ticks", "rx_ticks"]
    comp_labels = ["CPU", "LPM", "TX", "RX"]
    comp_colors = ["#e6194b", "#4363d8", "#f58231", "#3cb44b"]
    cycles = sorted(df["time_s"].unique())
    nodes = sorted(df["node_id"].unique())
    n_nodes = len(nodes)

    fig, axes = plt.subplots(1, len(cycles), figsize=(6 * len(cycles), 5),
                             sharey=True, squeeze=False)
    for ci, cycle_t in enumerate(cycles):
        ax = axes[0][ci]
        cycle_df = df[df["time_s"] == cycle_t]
        x = np.arange(n_nodes)
        bottoms = np.zeros(n_nodes)
        for comp, label, color in zip(components, comp_labels, comp_colors):
            vals = []
            for nid in nodes:
                row = cycle_df[cycle_df["node_id"] == nid]
                vals.append(row[comp].values[0] if len(row) else 0)
            vals = np.array(vals, dtype=float)
            ax.bar(x, vals, 0.7, bottom=bottoms, label=label, color=color,
                   edgecolor="white", linewidth=0.3)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in nodes], fontsize=7)
        ax.set_xlabel("Node ID")
        ax.set_title(f"t \u2248 {cycle_t:.0f}s", fontsize=9)
        ax.tick_params(labelsize=8)
        if ci == 0:
            ax.set_ylabel("Ticks")
            ax.legend(fontsize=7)
    fig.suptitle("Energest Breakdown per Node per Cycle", fontsize=11,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "energest_per_node.png"), dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved energest_per_node.png")


def plot_cpu_util(data, out_dir, dpi):
    df = data["cpu_util"]
    if df.empty:
        print("  Skipping CPU util plot (no data)")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for nid in sorted(df["node_id"].unique()):
        sub = df[df["node_id"] == nid].dropna(subset=["cpu_pct"])
        if sub.empty:
            continue
        overflow = sub["cpu_pct"] > 100
        normal = ~overflow
        if normal.any():
            ax.plot(sub.loc[normal, "time_s"], sub.loc[normal, "cpu_pct"],
                    "o-", label=f"Node {nid}",
                    color=NODE_COLORS.get(nid), markersize=5)
        if overflow.any():
            ax.plot(sub.loc[overflow, "time_s"], sub.loc[overflow, "cpu_pct"],
                    "x", color=NODE_COLORS.get(nid), markersize=8, markeredgewidth=2)
            for _, row in sub[overflow].iterrows():
                ax.annotate("overflow", (row["time_s"], row["cpu_pct"]),
                            fontsize=6, color="red", ha="left",
                            textcoords="offset points", xytext=(4, 0))
    _style_ax(ax, "CPU Utilization per Node", "Simulated Time (s)", "CPU Utilization (%)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cpu_util_per_node.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved cpu_util_per_node.png")


def plot_tx_power(data, out_dir, dpi):
    df = data["tx_power"]
    if df.empty:
        print("  Skipping TX power plot (no data)")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    last = df.groupby("node_id").last().reset_index()
    bars = ax.bar([str(n) for n in last["node_id"]], last["tx_power_dbm"],
                  color=[NODE_COLORS.get(n, "#888") for n in last["node_id"]])
    _style_ax(ax, "TX Power per Node (last reading)", "Node ID", "TX Power (dBm)")
    for bar, val in zip(bars, last["tx_power_dbm"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(int(val)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tx_power.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved tx_power.png")


def plot_packet_delivery(data, out_dir, dpi):
    df = data["app_events"]
    if df.empty:
        print("  Skipping packet delivery plot (no data)")
        return
    stats = df[df["event"] == "tx_rx_stats"].copy()
    if stats.empty:
        print("  Skipping packet delivery plot (no Tx/Rx/MissedTx stats)")
        return
    last = stats.groupby("node_id").last().reset_index()
    last = last[last["node_id"] != 1]
    if last.empty:
        print("  Skipping packet delivery plot (no client stats)")
        return
    nodes = last["node_id"].values
    x = np.arange(len(nodes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, last["tx_count"], width, label="TX", color="#4363d8")
    ax.bar(x, last["rx_count"], width, label="RX", color="#3cb44b")
    ax.bar(x + width, last["missed_count"], width, label="Missed TX", color="#e6194b")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nodes])
    _style_ax(ax, "Packet Delivery (cumulative counters)", "Node ID", "Count")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "packet_delivery.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved packet_delivery.png")


def plot_connectivity(data, out_dir, dpi):
    df = data["app_events"]
    if df.empty:
        print("  Skipping connectivity plot (no data)")
        return
    clients = df[df["node_id"] != 1]["node_id"].unique()
    if len(clients) == 0:
        print("  Skipping connectivity plot (no client events)")
        return
    nodes = sorted(clients)
    bin_size = 5.0
    t_min, t_max = df["time_s"].min(), df["time_s"].max()
    bins = np.arange(t_min, t_max + bin_size, bin_size)
    n_bins = len(bins) - 1
    matrix = np.full((len(nodes), n_bins), np.nan)

    for i, nid in enumerate(nodes):
        node_df = df[df["node_id"] == nid]
        for bi in range(n_bins):
            in_bin = node_df[(node_df["time_s"] >= bins[bi]) &
                             (node_df["time_s"] < bins[bi + 1])]
            if in_bin.empty:
                continue
            has_txrx = (in_bin["event"] == "tx_rx_stats").any()
            matrix[i, bi] = 1 if has_txrx else 0

    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = matplotlib.colors.ListedColormap(["#e6194b", "#3cb44b"])
    cmap.set_bad(color="#dddddd")
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest",
              extent=[t_min, t_max, len(nodes) - 0.5, -0.5])
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels([str(n) for n in nodes])
    _style_ax(ax, "Node Connectivity over Time", "Simulated Time (s)", "Node ID")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#3cb44b", label="Reachable"),
        Patch(facecolor="#e6194b", label="Not reachable"),
        Patch(facecolor="#dddddd", label="No data"),
    ], fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "connectivity.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved connectivity.png")


def main():
    parser = argparse.ArgumentParser(
        description="Parse COOJA simulation logs and generate metric plots.")
    parser.add_argument("--log", default="simulation_logs/COOJA.testlog",
                        help="Path to COOJA.testlog")
    parser.add_argument("--df", default=None,
                        help="Path to directory of pre-parsed CSVs (skip parsing)")
    parser.add_argument("--outdir", default="plots",
                        help="Output directory for PNG plots (default: plots/)")
    parser.add_argument("--parseddir", default="parsed_data",
                        help="Where to save parsed CSVs (default: parsed_data)")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI")
    args = parser.parse_args()

    if args.df:
        print(f"Loading pre-parsed data from {args.df}/ ...")
        data = load_data(args.df)
    else:
        if not os.path.exists(args.log):
            print(f"Error: log file not found: {args.log}", file=sys.stderr)
            sys.exit(1)
        print(f"Parsing {args.log} ...")
        data = parse_log(args.log)
        print(f"Saving parsed DataFrames to {args.parseddir}/ ...")
        save_data(data, args.parseddir)

    print("\n--- Data Summary ---")
    for name, df in data.items():
        if df.empty:
            print(f"  {name:12s}: empty")
        else:
            n = df["node_id"].nunique() if "node_id" in df.columns else "?"
            print(f"  {name:12s}: {len(df)} rows, {n} nodes")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nGenerating plots in {args.outdir}/ ...")
    plot_etx(data, args.outdir, args.dpi)
    plot_rank(data, args.outdir, args.dpi)
    plot_energest(data, args.outdir, args.dpi)
    plot_cpu_util(data, args.outdir, args.dpi)
    plot_tx_power(data, args.outdir, args.dpi)
    plot_packet_delivery(data, args.outdir, args.dpi)
    plot_connectivity(data, args.outdir, args.dpi)
    print("\nDone.")


if __name__ == "__main__":
    main()
