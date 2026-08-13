#!/usr/bin/env python3

import argparse
import json
import math
import os

import duckdb
import pandas as pd
import plotly.graph_objects as go

from topology_utils import build_connectivity_graph

backend = "plotly"
pd.options.plotting.backend = backend
extension = "html" if backend == "plotly" else "png"


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


def compute_etx(data):
    df = data["metrics"]
    if df.empty:
        return pd.DataFrame(columns=["time_s", "avg_etx"])
    return duckdb.sql(
        """
        SELECT time_s, AVG(etx) as avg_etx
        FROM df
        WHERE etx <> 65535.0
        GROUP BY time_s
        ORDER BY time_s
        """
    ).df()


def compute_energy(data):
    df = data["metrics"]
    if df.empty:
        energy = pd.DataFrame(columns=["node_id", "energy_comp"])
        cpu = pd.DataFrame(columns=["node_id", "cpu_usage"])
    else:
        energy = duckdb.sql(
            """
            SELECT df.node_id, energy_comp
            FROM df
                JOIN (SELECT node_id, MAX(time_s) as latest_time FROM df GROUP BY node_id) as nodes
                ON df.node_id = nodes.node_id
            WHERE df.time_s = nodes.latest_time
            ORDER BY df.node_id
            """
        ).df()
        cpu = duckdb.sql(
            """
            SELECT df.node_id, (cpu_ticks / total_ticks) * 100 as cpu_usage
            FROM df
                JOIN (SELECT node_id, MAX(time_s) as latest_time FROM df GROUP BY node_id) as nodes
                ON df.node_id = nodes.node_id
            WHERE df.time_s = nodes.latest_time
            ORDER BY df.node_id
            """
        ).df()
    return {"energy": energy, "cpu": cpu}


def compute_pdr(data):
    df = data["latency"]
    if df.empty:
        return pd.DataFrame(columns=["node_id", "pdr", "plr"])
    return duckdb.sql(
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


def compute_energy_by_hop(data):
    df = data["metrics"]
    if df.empty:
        return pd.DataFrame(columns=["time_s", "hop_count", "avg_energy"])
    return duckdb.sql(
        """
        WITH deltas AS (
            SELECT node_id, time_s, hop_count,
                   energy_comp - LAG(energy_comp)
                       OVER (PARTITION BY node_id ORDER BY time_s) AS energy_delta
            FROM df
        )
        SELECT time_s, hop_count, AVG(energy_delta) AS avg_energy
        FROM deltas
        WHERE energy_delta IS NOT NULL
        GROUP BY time_s, hop_count
        ORDER BY time_s, hop_count
        """
    ).df()


def compute_energy_usage_by_hop(data):
    df = data["metrics"]
    if df.empty:
        return pd.DataFrame(columns=["node_id", "hop_count", "energy_comp"])
    return duckdb.sql(
        """
        WITH latest AS (
            SELECT node_id, MAX(time_s) AS t FROM df GROUP BY node_id
        )
        SELECT df.node_id, df.hop_count, df.energy_comp
        FROM df JOIN latest ON df.node_id = latest.node_id AND df.time_s = latest.t
        ORDER BY df.hop_count, df.node_id
        """
    ).df()


def compute_cpu_usage_by_hop(data):
    df = data["metrics"]
    if df.empty:
        return pd.DataFrame(columns=["node_id", "hop_count", "cpu_usage"])
    return duckdb.sql(
        """
        WITH latest AS (
            SELECT node_id, MAX(time_s) AS t FROM df GROUP BY node_id
        )
        SELECT df.node_id, df.hop_count,
               (df.cpu_ticks / df.total_ticks) * 100 AS cpu_usage
        FROM df JOIN latest ON df.node_id = latest.node_id AND df.time_s = latest.t
        ORDER BY df.hop_count, df.node_id
        """
    ).df()


def compute_latency_by_hop(data):
    lat = data["latency"]
    m = data["metrics"]
    if lat.empty or m.empty:
        return pd.DataFrame(columns=["node_id", "hop_count", "avg_latency"])
    return duckdb.sql(
        """
        WITH delivered AS (
            SELECT node_id, client_send_time AS send_time,
                   server_receive_time - client_send_time AS one_way_latency
            FROM lat
            WHERE server_receive_time <> 0
        )
        SELECT d.node_id, m.hop_count, AVG(d.one_way_latency) AS avg_latency
        FROM delivered d
        ASOF JOIN m
            ON d.node_id = m.node_id AND m.time_s >= d.send_time
        GROUP BY d.node_id, m.hop_count
        ORDER BY m.hop_count, d.node_id
        """
    ).df()


def compute_latency_by_node(data):
    df = data["latency"]
    if df.empty:
        return pd.DataFrame(columns=["node_id", "avg_latency"])
    return duckdb.sql(
        """
        SELECT node_id,
               AVG(server_receive_time - client_send_time) AS avg_latency
        FROM df
        WHERE server_receive_time <> 0
        GROUP BY node_id
        ORDER BY node_id
        """
    ).df()


def compute_metrics(data):
    return {
        "etx": compute_etx(data),
        "energy": compute_energy(data),
        "pdr": compute_pdr(data),
        "energy_by_hop": compute_energy_by_hop(data),
        # TODO: could optimise this by aggr on energy_by_hop instead
        "energy_usage_by_hop": compute_energy_usage_by_hop(data),
        "cpu_usage_by_hop": compute_cpu_usage_by_hop(data),
        "latency_by_hop": compute_latency_by_hop(data),
        "latency_by_node": compute_latency_by_node(data),
    }


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

GRAPH_COLORS = {
    "etx": "#636efa",
    "energy": "#e6194b",
    "cpu": "#3cb44b",
    "pdr": "#4363d8",
    "plr": "#f58231",
}

PDR_COLORSCALE = [[0.0, "red"], [1.0, "green"]]
CPU_COLORSCALE = [[0.0, "darkblue"], [1.0, "red"]]


def get_fig(df, x, y, kind, title, xlabel, ylabel, color=None, width=None, height=None, category_x=False, **plot_kwargs):
    fig = df.plot(x=x, y=y, kind=kind)
    if color is not None:
        for trace in fig.data:
            if trace.type == "bar":
                trace.marker.color = color
            elif trace.type == "box":
                trace.marker.color = color
                trace.line.color = color
            else:
                trace.line.color = color
    if category_x:
        fig.update_xaxes(type="category")
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Arial", weight="bold")),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",  # alpha=0.3 equivalent
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
        ),
        width=width,
        height=height,
    )
    return fig


def plot_etx(metrics, out_dir, dpi):
    df = metrics["etx"]
    figs = []

    print("  Add etx plots")
    figs.append(
        get_fig(
            df,
            x="time_s",
            y="avg_etx",
            kind="line",
            title="Average ETX by simulated time",
            xlabel="Simulated time (s)",
            ylabel="Average ETX",
            color=GRAPH_COLORS["etx"],
        )
    )
    return figs


def plot_energy_usage(metrics, out_dir, dpi):
    energy = metrics["energy"]
    figs = []

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=energy["energy"]["node_id"],
            y=energy["energy"]["energy_comp"],
            name="Energy Usage (mAh)",
            marker_color="#8ecae6",
            hovertemplate="Node %{x}<br>Energy: %{y:.3f} mAh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=energy["cpu"]["node_id"],
            y=energy["cpu"]["cpu_usage"],
            name="CPU Usage (%)",
            marker_color=GRAPH_COLORS["cpu"],
            hovertemplate="Node %{x}<br>CPU: %{y:.1f}%<extra></extra>",
            visible=False,
        )
    )
    fig.update_layout(
        title=dict(text="Energy Usage after simulation", font=dict(size=14, family="Arial", weight="bold")),
        xaxis=dict(
            type="category",
            title=dict(text="Node ID", font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
        ),
        yaxis=dict(
            title=dict(text="Energy Usage (mAh)", font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.5,
                y=1.18,
                xanchor="center",
                yanchor="top",
                buttons=[
                    dict(
                        label="Energy Usage (mAh)",
                        method="update",
                        args=[
                            {"visible": [True, False]},
                            {
                                "yaxis": {"title": {"text": "Energy Usage (mAh)"}},
                                "title": {"text": "Energy Usage after simulation"},
                            },
                        ],
                    ),
                    dict(
                        label="CPU Usage (%)",
                        method="update",
                        args=[
                            {"visible": [False, True]},
                            {
                                "yaxis": {"title": {"text": "CPU Usage (%)"}},
                                "title": {"text": "CPU usage through the simulation"},
                            },
                        ],
                    ),
                ],
            )
        ],
    )
    figs.append(fig)
    print("  Add energy_comp / cpu_usage")
    return figs


def plot_energy_by_hop(metrics, out_dir, dpi):
    df = metrics["energy_by_hop"]
    figs = []
    if df.empty:
        return figs
    pivoted = df.pivot(index="time_s", columns="hop_count", values="avg_energy").reset_index()
    cols = [c for c in pivoted.columns if c != "time_s"]
    figs.append(
        get_fig(
            pivoted,
            x="time_s",
            y=cols,
            kind="line",
            title="Average Energy Usage by Hop Count",
            xlabel="Simulated time (s)",
            ylabel="Average Energy per Interval (mAh)",
        )
    )
    print("  Add energy_by_hop")
    return figs


def plot_by_hop(metrics, out_dir, dpi):
    figs = []
    specs = [
        ("energy_usage_by_hop", "energy_comp", "Energy Usage (mAh)", "#b8860b"),
        ("cpu_usage_by_hop", "cpu_usage", "CPU Usage (%)", "#e6194b"),
        ("latency_by_hop", "avg_latency", "One-way Latency (s)", "#636efa"),
    ]
    traces = []
    labels = []
    for key, ycol, label, color in specs:
        df = metrics[key]
        if df.empty:
            continue
        labels.append(label)
        traces.append(
            go.Box(
                x=df["hop_count"].astype(str),
                y=df[ycol],
                name=label,
                marker_color=color,
                line_color=color,
                boxmean=True,
                visible=False,
            )
        )
    if not traces:
        return figs

    latency_df = metrics["latency_by_hop"]
    outlier_rows = (
        _box_outliers(latency_df, "hop_count", "avg_latency")
        if not latency_df.empty
        else pd.DataFrame()
    )
    if not outlier_rows.empty:
        traces.append(
            go.Scatter(
                x=[str(row.hop_count) for row in outlier_rows.itertuples()],
                y=[row.avg_latency for row in outlier_rows.itertuples()],
                mode="text",
                text=[str(int(row.node_id)) for row in outlier_rows.itertuples()],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                hoverinfo="skip",
                visible=False,
            )
        )

    n_box = len(labels)
    n_traces = len(traces)
    traces[0].visible = True
    fig = go.Figure(data=traces)
    buttons = []
    for i, label in enumerate(labels):
        vis = [False] * n_traces
        vis[i] = True
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": vis},
                    {"yaxis": {"title": {"text": label}}},
                ],
            )
        )
    fig.update_layout(
        title=dict(text="Metrics by Hop Count", font=dict(size=14, family="Arial", weight="bold")),
        xaxis=dict(
            type="category",
            title=dict(text="Hop Count", font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
        ),
        yaxis=dict(
            title=dict(text=labels[0], font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
        ),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.5,
                y=1.18,
                xanchor="center",
                yanchor="top",
                buttons=buttons,
            )
        ],
    )
    figs.append(fig)
    print("  Add metrics_by_hop")
    return figs


def _box_outliers(df, group, value):
    outliers = []
    for _, grp in df.groupby(group):
        q1 = grp[value].quantile(0.25)
        q3 = grp[value].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers.append(grp[(grp[value] < lo) | (grp[value] > hi)])
    if not outliers:
        return pd.DataFrame(columns=[group, value])
    return pd.concat(outliers)


def plot_packet_delivery(metrics, out_dir, dpi):
    df = metrics["pdr"]
    figs = []
    if df.empty:
        return figs

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["node_id"],
            y=df["pdr"],
            name="PDR",
            marker_color=GRAPH_COLORS["pdr"],
            hovertemplate="Node %{x}<br>PDR: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["node_id"],
            y=df["plr"],
            name="PLR",
            marker_color=GRAPH_COLORS["plr"],
            hovertemplate="Node %{x}<br>PLR: %{y:.2f}<extra></extra>",
            visible=False,
        )
    )
    fig.update_layout(
        title=dict(text="Packet Delivery / Loss Ratio", font=dict(size=14, family="Arial", weight="bold")),
        xaxis=dict(
            type="category",
            title=dict(text="Node ID", font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
        ),
        yaxis=dict(
            title=dict(text="Ratio", font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.5,
                y=1.18,
                xanchor="center",
                yanchor="top",
                buttons=[
                    dict(
                        label="PDR",
                        method="restyle",
                        args=[{"visible": [True, False]}],
                    ),
                    dict(
                        label="PLR",
                        method="restyle",
                        args=[{"visible": [False, True]}],
                    ),
                ],
            )
        ],
    )
    figs.append(fig)
    print("  Saved packet_delivery / packet_loss")
    return figs


def plot_topology(df_dir, out_dir, metrics):
    path = os.path.join(df_dir, "topology.json")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping topology plot")
        return []
    with open(path) as fh:
        topo = json.load(fh)

    pdr_df = metrics["pdr"]
    has_pdr = not pdr_df.empty
    pdr_by_node = {
        int(row.node_id): float(row.pdr) for row in pdr_df.itertuples()
    }
    latency_by_node = {
        int(row.node_id): float(row.avg_latency)
        for row in metrics["latency_by_node"].itertuples()
    }

    radio = topo.get("radio", {})
    tx_range = radio.get("tx_range", 0)
    interference_range = radio.get("interference_range", 0)
    motes = topo.get("motes", [])

    server = [m for m in motes if str(m.get("role", "")).lower() == "server"]
    clients = [m for m in motes if str(m.get("role", "")).lower() != "server"]
    if not server:
        print(f"  Warning: no server in {path}, skipping topology plot")
        return []

    sx, sy = float(server[0]["x"]), float(server[0]["y"])

    pad = max(tx_range, interference_range) * 1.05
    all_x = [float(m["x"]) for m in motes]
    all_y = [float(m["y"]) for m in motes]
    cx_axis, cy_axis = (min(all_x) + max(all_x)) / 2, (min(all_y) + max(all_y)) / 2
    half_span = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2 + pad
    x_axis = [cx_axis - half_span, cx_axis + half_span]
    y_axis = [cy_axis - half_span, cy_axis + half_span]

    adjacency = build_connectivity_graph(motes, tx_range)
    positions = {int(m["id"]): (float(m["x"]), float(m["y"])) for m in motes}
    nodes_data = {
        str(mid): {
            "x": positions[mid][0],
            "y": positions[mid][1],
            "n": sorted(adjacency.get(mid, [])),
        }
        for mid in positions
    }

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=GRAPH_COLORS["plr"], width=2.5),
            hoverinfo="skip",
            visible=False,
            showlegend=False,
            name="Neighbor Links",
        )
    )

    angles = [2 * math.pi * i / 48 for i in range(49)]
    circ_x = [math.cos(a) for a in angles]
    circ_y = [math.sin(a) for a in angles]
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=GRAPH_COLORS["cpu"], width=1.5),
            hoverinfo="skip",
            visible=False,
            showlegend=False,
            name="TX Range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=GRAPH_COLORS["plr"], width=1.5, dash="dash"),
            hoverinfo="skip",
            visible=False,
            showlegend=False,
            name="Interference Range",
        )
    )

    node_hover = (
        "Node %{customdata[0]} (%{customdata[1]})<br>"
        "Position: (%{x:.1f}, %{y:.1f})<br>"
        f"TX range: {tx_range} m<br>"
        f"Interference range: {interference_range} m"
        "<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=[sx],
            y=[sy],
            mode="markers+text",
            marker=dict(symbol="star", size=22, color=GRAPH_COLORS["energy"]),
            text=[f"{int(server[0]['id'])}"],
            textposition="top center",
            customdata=[[int(server[0]["id"]), "server"]],
            hovertemplate=node_hover,
            name="Server",
        )
    )
    energy = metrics["energy"]
    energy_by_node = {
        int(row.node_id): float(row.energy_comp)
        for row in energy["energy"].itertuples()
    }
    cpu_by_node = {
        int(row.node_id): float(row.cpu_usage) for row in energy["cpu"].itertuples()
    }
    clients_pdr = [pdr_by_node.get(int(m["id"]), float("nan")) for m in clients]
    clients_energy = [
        energy_by_node.get(int(m["id"]), float("nan")) for m in clients
    ]
    clients_cpu = [cpu_by_node.get(int(m["id"]), float("nan")) for m in clients]
    clients_latency = [
        latency_by_node.get(int(m["id"]), float("nan")) for m in clients
    ]
    clients_custom = [
        [int(m["id"]), m.get("role", "client"), pdr, eng, cpu, lat]
        for m, pdr, eng, cpu, lat in zip(
            clients, clients_pdr, clients_energy, clients_cpu, clients_latency
        )
    ]
    clients_hover = (
        "Node %{customdata[0]} (%{customdata[1]})<br>"
        "Position: (%{x:.1f}, %{y:.1f})<br>"
        f"TX range: {tx_range} m<br>"
        f"Interference range: {interference_range} m<br>"
        "PDR: %{customdata[2]:.2f}"
        "<extra></extra>"
    )
    energy_hover = (
        "Node %{customdata[0]} (%{customdata[1]})<br>"
        "Position: (%{x:.1f}, %{y:.1f})<br>"
        f"TX range: {tx_range} m<br>"
        f"Interference range: {interference_range} m<br>"
        "Energy: %{customdata[3]:.3f} mAh"
        "<extra></extra>"
    )
    cpu_hover = (
        "Node %{customdata[0]} (%{customdata[1]})<br>"
        "Position: (%{x:.1f}, %{y:.1f})<br>"
        f"TX range: {tx_range} m<br>"
        f"Interference range: {interference_range} m<br>"
        "CPU: %{customdata[4]:.1f}%"
        "<extra></extra>"
    )
    latency_hover = (
        "Node %{customdata[0]} (%{customdata[1]})<br>"
        "Position: (%{x:.1f}, %{y:.1f})<br>"
        f"TX range: {tx_range} m<br>"
        f"Interference range: {interference_range} m<br>"
        "Latency: %{customdata[5]:.3f} s"
        "<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=[float(m["x"]) for m in clients],
            y=[float(m["y"]) for m in clients],
            mode="markers+text",
            marker=dict(
                symbol="circle",
                size=14,
                color=clients_pdr if has_pdr else GRAPH_COLORS["etx"],
                colorscale=PDR_COLORSCALE,
                cmin=0.0,
                cmax=1.0,
                line=dict(width=1, color="black"),
                showscale=has_pdr,
                colorbar=dict(
                    title=dict(text="PDR", side="right"),
                    thickness=14,
                    len=0.7,
                ),
            ),
            text=[f"{int(m['id'])}" for m in clients],
            textposition="top center",
            customdata=clients_custom,
            hovertemplate=clients_hover if has_pdr else node_hover,
            name="Clients",
        )
    )

    edge_script = (
        "(function() {\n"
        "  var gd = document.getElementById('topology_plot');\n"
        "  if (!gd) return;\n"
        "  var nodes = %s;\n"
        "  var TX_RANGE = %s, INT_RANGE = %s;\n"
        "  var CIRC_X = %s, CIRC_Y = %s;\n"
        "  var EDGE_IDX = 0, TX_CIRCLE_IDX = 1, INT_CIRCLE_IDX = 2;\n"
        "  var SERVER_IDX = 3, CLIENTS_IDX = 4;\n"
        "  var current = null;\n"
        "  var busy = false;\n"
        "  var pending = null;\n"
        "  var pendingHide = false;\n"
        "  gd.style.position = 'relative';\n"
        "  var tip = document.createElement('div');\n"
        "  tip.style.cssText = 'position:absolute;top:8px;display:none;' +\n"
        "    'background:rgba(255,255,255,0.97);border:1px solid #b0b7c3;' +\n"
        "    'border-radius:4px;padding:6px 10px;font-family:Arial;font-size:11px;' +\n"
        "    'color:#2a3f5f;box-shadow:0 2px 6px rgba(0,0,0,0.25);' +\n"
        "    'pointer-events:none;z-index:10;white-space:pre-line;max-width:280px;';\n"
        "  gd.appendChild(tip);\n"
        "  var st = document.createElement('style');\n"
        "  st.textContent = '#topology_plot .hoverlayer { visibility: hidden; }';\n"
        "  document.head.appendChild(st);\n"
        "  function positionTip() {\n"
        "    if (!gd._fullLayout || !gd._fullLayout._size) return;\n"
        "    var sz = gd._fullLayout._size;\n"
        "    var w = tip.offsetWidth || 120;\n"
        "    tip.style.left = (sz.l + sz.w - w - 8) + 'px';\n"
        "    tip.style.top = (sz.t + 8) + 'px';\n"
        "  }\n"
        "  function showTip(html) {\n"
        "    tip.innerHTML = html;\n"
        "    tip.style.display = 'block';\n"
        "    positionTip();\n"
        "  }\n"
        "  function hideTip() { tip.style.display = 'none'; }\n"
        "  function segs(id) {\n"
        "    var node = nodes[id];\n"
        "    if (!node) return {x: [], y: []};\n"
        "    var xs = [], ys = [];\n"
        "    for (var i = 0; i < node.n.length; i++) {\n"
        "      var nb = nodes[node.n[i]];\n"
        "      xs.push(node.x, nb.x, null);\n"
        "      ys.push(node.y, nb.y, null);\n"
        "    }\n"
        "    return {x: xs, y: ys};\n"
        "  }\n"
        "  function circleXY(cx, cy, r) {\n"
        "    var xs = [], ys = [];\n"
        "    for (var i = 0; i < CIRC_X.length; i++) {\n"
        "      xs.push(cx + CIRC_X[i] * r);\n"
        "      ys.push(cy + CIRC_Y[i] * r);\n"
        "    }\n"
        "    return {x: xs, y: ys};\n"
        "  }\n"
        "  function showEdges(id) {\n"
        "    var node = nodes[id];\n"
        "    var s = segs(id);\n"
        "    var tx = circleXY(node.x, node.y, TX_RANGE);\n"
        "    var it = circleXY(node.x, node.y, INT_RANGE);\n"
        "    busy = true;\n"
        "    Plotly.restyle(gd, {\n"
        "      x: [s.x, tx.x, it.x],\n"
        "      y: [s.y, tx.y, it.y],\n"
        "      visible: [true, true, true]\n"
        "    }, [EDGE_IDX, TX_CIRCLE_IDX, INT_CIRCLE_IDX]).then(done, done);\n"
        "  }\n"
        "  function hideEdges() {\n"
        "    busy = true;\n"
        "    Plotly.restyle(gd, {\n"
        "      visible: [false, false, false]\n"
        "    }, [EDGE_IDX, TX_CIRCLE_IDX, INT_CIRCLE_IDX]).then(done, done);\n"
        "  }\n"
        "  function done() {\n"
        "    busy = false;\n"
        "    if (pending !== null) {\n"
        "      var id = pending;\n"
        "      pending = null;\n"
        "      pendingHide = false;\n"
        "      if (id === current) return;\n"
        "      current = id;\n"
        "      showEdges(id);\n"
        "    } else if (pendingHide) {\n"
        "      pendingHide = false;\n"
        "      if (current === null) return;\n"
        "      current = null;\n"
        "      hideEdges();\n"
        "    }\n"
        "  }\n"
        "  gd.on('plotly_hover', function(e) {\n"
        "    var pt = e.points[0];\n"
        "    if (!pt) return;\n"
        "    if (pt.curveNumber === SERVER_IDX || pt.curveNumber === CLIENTS_IDX) {\n"
        "      var id = String(pt.customdata[0]);\n"
        "      var metricIdx = 2;\n"
        "      var cbTitle = gd.data[CLIENTS_IDX].marker.colorbar.title.text;\n"
        "      if (cbTitle === 'Energy (mAh)') metricIdx = 3;\n"
        "      else if (cbTitle === 'CPU Usage (%%)') metricIdx = 4;\n"
        "      else if (cbTitle === 'Avg Latency (s)') metricIdx = 5;\n"
        "      var tipHtml = 'Node <b>' + id + '</b> (' + pt.customdata[1] + ')<br>' +\n"
        "        'Position: (' + pt.x.toFixed(1) + ', ' + pt.y.toFixed(1) + ')<br>' +\n"
        "        'TX range: ' + TX_RANGE + ' m<br>' +\n"
        "        'Interference range: ' + INT_RANGE + ' m';\n"
        "      if (pt.customdata[metricIdx] !== undefined && pt.customdata[metricIdx] !== null && !isNaN(pt.customdata[metricIdx])) {\n"
        "        if (metricIdx === 3) {\n"
        "          tipHtml += '<br>Energy: ' + pt.customdata[3].toFixed(3) + ' mAh';\n"
        "        } else if (metricIdx === 4) {\n"
        "          tipHtml += '<br>CPU: ' + pt.customdata[4].toFixed(1) + '%%';\n"
        "        } else if (metricIdx === 5) {\n"
        "          tipHtml += '<br>Latency: ' + pt.customdata[5].toFixed(3) + ' s';\n"
        "        } else {\n"
        "          tipHtml += '<br>PDR: ' + (pt.customdata[2] * 100).toFixed(1) + '%%';\n"
        "        }\n"
        "      }\n"
        "      showTip(tipHtml);\n"
        "      if (busy) {\n"
        "        pending = id;\n"
        "        pendingHide = false;\n"
        "        return;\n"
        "      }\n"
        "      if (id === current) return;\n"
        "      current = id;\n"
        "      showEdges(id);\n"
        "    }\n"
        "  });\n"
        "  gd.on('plotly_unhover', function() {\n"
        "    hideTip();\n"
        "    if (busy) {\n"
        "      pending = null;\n"
        "      pendingHide = true;\n"
        "      return;\n"
        "    }\n"
        "    if (current === null) return;\n"
        "    current = null;\n"
        "    hideEdges();\n"
        "  });\n"
        "})();"
    ) % (
        json.dumps(nodes_data),
        tx_range,
        interference_range,
        json.dumps(circ_x),
        json.dumps(circ_y),
    )

    def _finite_bounds(values, default_min=0.0, default_max=1.0):
        finite = [v for v in values if not math.isnan(v)]
        if not finite:
            return default_min, default_max
        return min(finite), max(finite)

    energy_cmax = _finite_bounds(clients_energy)[1]
    cpu_cmin, cpu_cmax = _finite_bounds(clients_cpu, 0.0, 10.0)
    real_latency = [v for v in clients_latency if not math.isnan(v)]
    latency_cmax = max(real_latency) if real_latency else 1.0
    latency_cmin_real = min(real_latency) if real_latency else 0.0
    has_missing_latency = len(real_latency) < len(clients_latency)
    if has_missing_latency or latency_cmin_real == 0.0:
        clients_latency_color = [
            0.0 if math.isnan(v) else v for v in clients_latency
        ]
        latency_cmin = 0.0
        p0 = latency_cmin_real / latency_cmax if latency_cmax > 0 else 1.0
        latency_colorscale = [
            [0.0, "red"],
            [max(p0, 1e-6), "rgb(189,215,231)"],
            [1.0, "rgb(8,48,107)"],
        ]
    else:
        clients_latency_color = clients_latency
        latency_cmin = latency_cmin_real
        latency_colorscale = "Blues"
    layout: dict = dict(
        title=dict(text="Network Topology", font=dict(size=14, family="Arial", weight="bold")),
        dragmode="pan",
        xaxis=dict(
            title=dict(text="X (m)", font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
            range=x_axis,
        ),
        yaxis=dict(
            title=dict(text="Y (m)", font=dict(size=12)),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.3)",
            range=y_axis,
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    if has_pdr:
        layout["updatemenus"] = [
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.5,
                y=1.18,
                xanchor="center",
                yanchor="top",
                buttons=[
                    dict(
                        label="PDR",
                        method="restyle",
                        args=[
                            {
                                "marker.color": [clients_pdr],
                                "marker.cmin": [0.0],
                                "marker.cmax": [1.0],
                                "marker.colorscale": [PDR_COLORSCALE],
                                "marker.colorbar.title.text": ["PDR"],
                                "hovertemplate": [clients_hover],
                            },
                            [4],
                        ],
                    ),
                    dict(
                        label="Energy Usage",
                        method="restyle",
                        args=[
                            {
                                "marker.color": [clients_energy],
                                "marker.cmin": [0.0],
                                "marker.cmax": [energy_cmax],
                                "marker.colorscale": ["Viridis"],
                                "marker.colorbar.title.text": ["Energy (mAh)"],
                                "hovertemplate": [energy_hover],
                            },
                            [4],
                        ],
                    ),
                    dict(
                        label="CPU Util",
                        method="restyle",
                        args=[
                            {
                                "marker.color": [clients_cpu],
                                "marker.cmin": [cpu_cmin],
                                "marker.cmax": [cpu_cmax],
                                "marker.colorscale": [CPU_COLORSCALE],
                                "marker.colorbar.title.text": ["CPU Usage (%)"],
                                "hovertemplate": [cpu_hover],
                            },
                            [4],
                        ],
                    ),
                    dict(
                        label="Avg Latency",
                        method="restyle",
                        args=[
                            {
                                "marker.color": [clients_latency_color],
                                "marker.cmin": [latency_cmin],
                                "marker.cmax": [latency_cmax],
                                "marker.colorscale": [latency_colorscale],
                                "marker.colorbar.title.text": ["Avg Latency (s)"],
                                "hovertemplate": [latency_hover],
                            },
                            [4],
                        ],
                    ),
                ],
            )
        ]
    fig.update_layout(layout)
    fig._topology_post_script = edge_script
    print("  Add topology")
    return [fig]


def plot_metrics(df_dir: str, output_dir: str, dpi: int = 150):
    data = load_data(df_dir)
    metrics = compute_metrics(data)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}/ ...")
    figs = []
    figs.extend(plot_topology(df_dir, output_dir, metrics))
    figs.extend(plot_etx(metrics, output_dir, dpi))
    figs.extend(plot_energy_usage(metrics, output_dir, dpi))
    figs.extend(plot_energy_by_hop(metrics, output_dir, dpi))
    figs.extend(plot_by_hop(metrics, output_dir, dpi))
    # plot average energy usage by children count (including all children of children)
    # Plot packet delivery ratio by hop_count
    figs.extend(plot_packet_delivery(metrics, output_dir, dpi))

    with open(f"{output_dir}/dashboard.html", "w") as f:
        f.write(
            figs[0].to_html(
                full_html=True,
                include_plotlyjs="cdn",
                div_id="topology_plot",
                post_script=getattr(figs[0], "_topology_post_script", None),
            )
        )
        for fig in figs[1:]:
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
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
