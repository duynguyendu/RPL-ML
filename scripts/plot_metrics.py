#!/usr/bin/env python3

import argparse
import os

import duckdb
import pandas as pd


def load_data(in_dir):
    data = {}
    for name in ("metrics", "latency"):
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


def plot_and_save(df, x, y, kind, title, xlabel, ylabel, path, **plot_kwargs):
    ax = df.plot(x=x, y=y, kind=kind, legend=True, **plot_kwargs)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)

    ax.get_figure().savefig(path, dpi=150, bbox_inches="tight")


def plot_etx(data, out_dir, dpi):
    df = data["metrics"]
    df = duckdb.sql(
        """
        SELECT time_s, AVG(etx) as avg_etx
        FROM df
        WHERE etx <> 65535.0
        GROUP BY time_s
        ORDER BY time_s
        """
    ).df()

    plot_and_save(
        df,
        x="time_s",
        y="avg_etx",
        kind="line",
        title="Average ETX by simulated time",
        xlabel="Simulated time (s)",
        ylabel="Average ETX",
        path=f"{out_dir}/average_etx.png",
        figsize=(10, 5),
    )
    print("  Saved average_etx.png")


def plot_energy_usage(data, out_dir, dpi):
    df = data["metrics"]
    result_df = duckdb.sql(
        """
        SELECT df.node_id, energy_comp
        FROM df
            JOIN (SELECT node_id, MAX(time_s) as latest_time FROM df GROUP BY node_id) as nodes
            ON df.node_id = nodes.node_id
        WHERE df.time_s = nodes.latest_time
        ORDER BY df.node_id
        """
    ).df()

    plot_and_save(
        result_df,
        x="node_id",
        y="energy_comp",
        kind="bar",
        title="Energy Usage after simulation",
        xlabel="Node ID",
        ylabel="Energy Usage (mAh)",
        path=f"{out_dir}/energy_comp.png",
        figsize=(10, 5),
    )
    print("  Saved energy_comp.png")

    result_df = duckdb.sql(
        """
        SELECT df.node_id, (cpu_ticks / total_ticks) * 100 as cpu_usage
        FROM df
            JOIN (SELECT node_id, MAX(time_s) as latest_time FROM df GROUP BY node_id) as nodes
            ON df.node_id = nodes.node_id
        WHERE df.time_s = nodes.latest_time
        ORDER BY df.node_id
        """
    ).df()

    plot_and_save(
        result_df,
        x="node_id",
        y="cpu_usage",
        kind="bar",
        title="CPU usage through the simulation",
        xlabel="Node ID",
        ylabel="CPU Usage (%)",
        path=f"{out_dir}/cpu_usage.png",
        figsize=(10, 5),
    )
    print("  Saved cpu_usage.png")


def plot_packet_delivery(data, out_dir, dpi):
    df = data["latency"]
    df = duckdb.sql(
        """
        SELECT node_id, rx/tx as pdr, 1 - rx/tx as plr
        FROM (
            SELECT node_id, COUNT(*) as tx, COUNT(*) FILTER (WHERE server_receive_time <> 0) as rx
            FROM df
            GROUP BY node_id
            ORDER BY node_id
            )
        """
    ).df()

    plot_and_save(
        df,
        x="node_id",
        y="pdr",
        kind="bar",
        title="Packet Delivery Ratio",
        xlabel="Node ID",
        ylabel="Delivery Ratio",
        path=f"{out_dir}/packet_delivery.png",
        figsize=(10, 5),
    )
    print("  Saved packet_delivery.png")

    plot_and_save(
        df,
        x="node_id",
        y="plr",
        kind="bar",
        title="Packet Loss Ratio",
        xlabel="Node ID",
        ylabel="Loss Ratio",
        path=f"{out_dir}/packet_loss.png",
        figsize=(10, 5),
    )
    print("  Saved packet_loss.png")


def plot_metrics(df_dir: str, output_dir: str, dpi: int = 150):
    data = load_data(df_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}/ ...")
    plot_etx(data, output_dir, dpi)
    # plot average energy usage by hop_count
    # plot average energy usage by children count
    plot_energy_usage(data, output_dir, dpi)
    # Plot packet delivery ratio by hop_count
    plot_packet_delivery(data, output_dir, dpi)
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
