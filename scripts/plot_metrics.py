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
                            --output_dir plots/ --parseddir parsed_data/ --dpi 200
"""

import argparse
import os

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(in_dir):
    data = {}
    for name in ("etx", "dodag", "energest", "cpu_util", "app_events"):
        path = os.path.join(in_dir, f"{name}.csv")
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            print(f"  Warning: {path} not found, skipping {name}")
            data[name] = pd.DataFrame()
    return data


NODE_COLORS = {
    2: "#e6194b",
    3: "#3cb44b",
    4: "#4363d8",
    5: "#f58231",
    6: "#911eb4",
    7: "#42d4f4",
    8: "#f032e6",
    9: "#bfef45",
    10: "#fabed4",
    11: "#469990",
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
    for node_id in sorted(df["node_id"].unique()):
        node_etx = df[df["node_id"] == node_id].dropna(subset=["etx"])
        if node_etx.empty:
            print(f"  Skipping ETX plot for node {node_id}")
            continue

        ax.plot(
            node_etx["time_s"],
            node_etx["etx"],
            "o-",
            label=f"Node {node_id}",
            color=NODE_COLORS.get(node_id),
            markersize=2,
        )
    _style_ax(ax, "ETX to Preferred RPL Parent", "Simulated Time (s)", "ETX")
    ax.legend(fontsize=7, ncol=2, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "etx_per_node.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved etx_per_node.png")


def plot_dodag(data, out_dir, dpi):
    # TODO: plot the graph and rank as the same time
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
            ax.plot(
                sub.loc[connected, "time_s"],
                sub.loc[connected, "rank"],
                "o-",
                label=f"Node {nid}",
                color=NODE_COLORS.get(nid),
                markersize=5,
            )
        if disconnected.any():
            ax.plot(
                sub.loc[disconnected, "time_s"],
                [np.nan] * disconnected.sum(),
                "x--",
                color=NODE_COLORS.get(nid),
                markersize=5,
                alpha=0.4,
            )
    _style_ax(ax, "DODAG Rank per Node", "Simulated Time (s)", "Rank")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rank_per_node.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved rank_per_node.png")


def plot_cpu_util(data, out_dir, dpi):
    df = data["energest"]
    if df.empty:
        print("  Skipping energest plot (no data)")
        return

    nodes = sorted(df["node_id"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for ni, node_id in enumerate(nodes):
        node_df = df[df["node_id"] == node_id]

        prev_total_tick = 0
        prev_comp_total_tick = 0
        comp_percentage = []
        for _, row in node_df.iterrows():
            total_tick_delta = row["total_ticks"] - prev_total_tick
            comp_tick_delta = row["cpu_ticks"] - prev_comp_total_tick
            comp_percentage.append(round(comp_tick_delta / total_tick_delta, 2))

            # Update prev_ticks
            prev_comp_total_tick = row["cpu_ticks"]
            prev_total_tick = row["total_ticks"]
        comp_percentage = np.array(comp_percentage)

        ax.plot(
            node_df["time_s"].values,
            comp_percentage,
            "o-",
            label=f"Node {node_id}",
            color=NODE_COLORS.get(node_id),
            markersize=5,
        )

    _style_ax(
        ax, "CPU Utilization per Node", "Simulated Time (s)", "CPU Utilization (%)"
    )
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cpu_util_per_node.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved cpu_util_per_node.png")


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
            in_bin = node_df[
                (node_df["time_s"] >= bins[bi]) & (node_df["time_s"] < bins[bi + 1])
            ]
            if in_bin.empty:
                continue
            has_txrx = (in_bin["event"] == "tx_rx_stats").any()
            matrix[i, bi] = 1 if has_txrx else 0

    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = matplotlib.colors.ListedColormap(["#e6194b", "#3cb44b"])
    cmap.set_bad(color="#dddddd")
    ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=[t_min, t_max, len(nodes) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels([str(n) for n in nodes])
    _style_ax(ax, "Node Connectivity over Time", "Simulated Time (s)", "Node ID")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#3cb44b", label="Reachable"),
            Patch(facecolor="#e6194b", label="Not reachable"),
            Patch(facecolor="#dddddd", label="No data"),
        ],
        fontsize=8,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "connectivity.png"), dpi=dpi)
    plt.close(fig)
    print("  Saved connectivity.png")


def plot_metrics(df_dir: str, output_dir: str, dpi: int = 150):
    data = load_data(df_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}/ ...")
    plot_etx(data, output_dir, dpi)
    # plot_dodag(data, output_dir, dpi)
    plot_cpu_util(data, output_dir, dpi)
    # plot_packet_delivery(data, output_dir, dpi)
    # plot_connectivity(data, output_dir, dpi)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse COOJA simulation logs and generate metric plots."
    )

    parser.add_argument(
        "--df-dir",
        default=None,
        help="Path to directory of pre-parsed CSVs (skip parsing)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Output directory for PNG plots (default: plots/)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI")
    args = parser.parse_args()

    plot_metrics(args.df_dir, args.output_dir, args.dpi)
