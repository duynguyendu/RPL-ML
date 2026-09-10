#!/usr/bin/env python3

import argparse
import json
import math
import os

import duckdb
import pandas as pd
import plotly.graph_objects as go

from plotly_utils import plotly_src
from topology_utils import build_connectivity_graph

pd.options.plotting.backend = "plotly"

INVALID = 65535  # sentinel for "no value" in etx / hop_count columns


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


def _query(sql, df, columns):
    """Run a DuckDB query that refers to the bound frame as ``df``.

    Returns an empty frame with ``columns`` when ``df`` has no rows so callers
    never have to special-case a missing input CSV.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    return duckdb.sql(sql).df()


def compute_etx(data):
    return _query(
        """
        SELECT time_s, AVG(etx) AS avg_etx
        FROM df
        WHERE etx <> 65535.0
        GROUP BY time_s
        ORDER BY time_s
        """,
        data["metrics"],
        ["time_s", "avg_etx"],
    )


def compute_latest(data):
    """State of every node at the end of the simulation (its last metrics row)."""
    return _query(
        """
        SELECT node_id, hop_count,
               (cpu_ticks / total_ticks) * 100 AS cpu_usage
        FROM df
        QUALIFY row_number() OVER (PARTITION BY node_id ORDER BY time_s DESC) = 1
        ORDER BY node_id
        """,
        data["metrics"],
        ["node_id", "hop_count", "cpu_usage"],
    )


def compute_cpu(latest):
    return latest[["node_id", "cpu_usage"]].reset_index(drop=True)


def _latest_by_hop(latest, value):
    df = latest[latest["hop_count"] != INVALID]
    return (
        df[["node_id", "hop_count", value]]
        .sort_values(["hop_count", "node_id"])
        .reset_index(drop=True)
    )


def compute_cpu_usage_by_hop(latest):
    return _latest_by_hop(latest, "cpu_usage")


def compute_pdr(data):
    return _query(
        """
        SELECT node_id, rx / tx AS pdr, 1 - rx / tx AS plr
        FROM (
            SELECT node_id,
                   COUNT(*) AS tx,
                   COUNT(*) FILTER (WHERE server_receive_time <> 0) AS rx
            FROM df
            GROUP BY node_id
        )
        ORDER BY node_id
        """,
        data["latency"],
        ["node_id", "pdr", "plr"],
    )


def compute_latency(data):
    return _query(
        """
        SELECT node_id, hop_count, (server_receive_time - client_send_time) AS latency
        FROM df
        WHERE hop_count <> 65535
        """,
        data["latency"],
        ["node_id", "hop_count", "latency"],
    )


def compute_metrics(data):
    latest = compute_latest(data)
    latency = compute_latency(data)
    return {
        "etx": compute_etx(data),
        "cpu": compute_cpu(latest),
        "pdr": compute_pdr(data),
        "latency": latency,
        "latency_by_hop": (
            latency.groupby(["node_id", "hop_count"])
            .agg(avg_latency=("latency", "mean"))
            .sort_values(by="hop_count")
            .reset_index()
        ),
        "latency_by_node": latency.groupby("node_id")["latency"].mean(),
        "cpu_usage_by_hop": compute_cpu_usage_by_hop(latest),
    }


GRAPH_COLORS = {
    "etx": "#636efa",
    "server": "#e6194b",
    "cpu": "#3cb44b",
    "pdr": "#4363d8",
    "plr": "#f58231",
}

PDR_COLORSCALE = [[0.0, "red"], [1.0, "green"]]
CPU_COLORSCALE = [[0.0, "darkblue"], [1.0, "red"]]
HOP_COLORSCALE = "Plasma"


def _title(text):
    return dict(text=text, font=dict(size=14, family="Arial", weight="bold"))


def _axis(label, **extra):
    return dict(
        title=dict(text=label, font=dict(size=12)),
        tickfont=dict(size=11),
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128,128,128,0.3)",
        **extra,
    )


def _button_menu(buttons):
    return [
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
    ]


def get_fig(df, x, y, kind, title, xlabel, ylabel, color=None):
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
    fig.update_layout(title=_title(title), xaxis=_axis(xlabel), yaxis=_axis(ylabel))
    return fig


def plot_etx(metrics):
    print("  Add etx plots")
    return [
        get_fig(
            metrics["etx"],
            x="time_s",
            y="avg_etx",
            kind="line",
            title="Average ETX by simulated time",
            xlabel="Simulated time (s)",
            ylabel="Average ETX",
            color=GRAPH_COLORS["etx"],
        )
    ]


def plot_cpu_usage(metrics):
    cpu = metrics["cpu"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cpu["node_id"],
            y=cpu["cpu_usage"],
            name="CPU Usage (%)",
            marker_color=GRAPH_COLORS["cpu"],
            hovertemplate="Node %{x}<br>CPU: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=_title("CPU usage through the simulation"),
        xaxis=_axis("Node ID", type="category"),
        yaxis=_axis("CPU Usage (%)"),
    )
    print("  Add cpu_usage")
    return [fig]


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


def plot_by_hop(metrics):
    specs = [
        ("cpu_usage_by_hop", "cpu_usage", "CPU Usage (%)", "#e6194b"),
        ("latency_by_hop", "avg_latency", "One-way Latency (s)", "#636efa"),
    ]
    traces, labels = [], []
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
        return []

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
                args=[{"visible": vis}, {"yaxis": {"title": {"text": label}}}],
            )
        )
    fig.update_layout(
        title=_title("Metrics by Hop Count"),
        xaxis=_axis("Hop Count", type="category"),
        yaxis=_axis(labels[0]),
        showlegend=False,
        updatemenus=_button_menu(buttons),
    )
    print("  Add metrics_by_hop")
    return [fig]


def plot_packet_delivery(metrics):
    df = metrics["pdr"]
    if df.empty:
        return []

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
        title=_title("Packet Delivery / Loss Ratio"),
        xaxis=_axis("Node ID", type="category"),
        yaxis=_axis("Ratio", range=[0, 1]),
        updatemenus=_button_menu(
            [
                dict(label="PDR", method="restyle", args=[{"visible": [True, False]}]),
                dict(label="PLR", method="restyle", args=[{"visible": [False, True]}]),
            ]
        ),
    )
    print("  Saved packet_delivery / packet_loss")
    return [fig]


# config keys that make up the run_id (see pipeline.py); "etx" is derived
RUN_ID_CONFIG_KEYS = [
    "num_of_nodes",
    "ppm",
    "packet_size",
    "buffer_size",
    "duration",
    "rpl_of",
    "interference_range",
    "topo_type",
    "platform",
    "seed",
]


def load_run_config(df_dir):
    """Return the subset of config.json that is encoded in the run_id."""
    path = os.path.join(df_dir, "config.json")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping run config panel")
        return {}
    with open(path) as fh:
        cfg = json.load(fh)
    run_cfg = {k: cfg[k] for k in RUN_ID_CONFIG_KEYS if k in cfg}
    try:
        run_cfg["etx"] = round(1 / (cfg["success_tx"] * cfg["success_rx"]), 2)
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    return run_cfg


def _render_strip(title, items, bg="#eef2f7"):
    """Render one labelled key/value strip for the dashboard header."""
    body = "".join(
        f'<span style="margin-right:18px;white-space:nowrap;"><b>{k}</b>: {v}</span>'
        for k, v in items
    )
    return (
        '<div style="font-family:Arial;font-size:13px;color:#2a3f5f;'
        f"padding:10px 16px;background:{bg};border-bottom:1px solid #e0e0e0;"
        'display:flex;flex-wrap:wrap;align-items:center;">'
        f'<b style="margin-right:18px;">{title}</b>'
        f"{body}</div>"
    )


def render_run_config_html(run_config):
    """Render the run config as the top header strip of the dashboard."""
    if not run_config:
        return ""
    return _render_strip("Run config", list(run_config.items()), bg="#f6f8fa")


def _parent_switches_by_node(df_dir):
    """Parent switches per node as a Series indexed by node_id.

    Each node's first ``parent switch`` log line is its initial DODAG join and
    is not counted as a switch.
    """
    path = os.path.join(df_dir, "dodag.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    dodag = pd.read_csv(path)
    if dodag.empty or "node_id" not in dodag:
        return pd.Series(dtype=float)
    return (dodag.groupby("node_id").size() - 1).clip(lower=0)


def _isnan(v):
    return isinstance(v, float) and math.isnan(v)


def _packet_totals(data):
    """(sent, received by root, lost: not-joined, root-unreachable, in-network).

    ``not_joined`` / ``root_unreachable`` are requests the client dropped before
    transmitting (root not registered / not reachable, tagged via the ``status``
    column). ``in_network`` is everything else that never reached the root:
    total - not_joined - root_unreachable - received_by_root.
    """
    nan = float("nan")
    lat = data.get("latency") if data else None
    if lat is None or lat.empty:
        return nan, nan, nan, nan, nan
    sent = len(lat)
    status = lat["status"] if "status" in lat else pd.Series(dtype=object)
    not_joined = int((status == "not_joined").sum())
    unreachable = int((status == "unreachable").sum())
    if "server_receive_time" not in lat:
        return sent, nan, not_joined, unreachable, nan
    recv = int((lat["server_receive_time"] != 0).sum())
    in_network = sent - not_joined - unreachable - recv
    return sent, recv, not_joined, unreachable, in_network


# per-node metric key -> (strip label, value formatter)
_SUMMARY_METRICS = {
    "pdr": ("PDR", lambda v: f"{v * 100:.1f}%"),
    "latency": ("latency", lambda v: f"{v:.3f} s"),
    "cpu_util": ("CPU util", lambda v: f"{v:.1f}%"),
    "parent_switch": (
        "parent switch",
        lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}",
    ),
}

# aggregation name -> reducer over a per-node Series. avg/max/min/p95 are shown
# in the dashboard strips; q1/median/q3 are extra box-plot stats kept only in
# aggregate.json (consumed by plot_comparison.py's candle charts).
_AGGREGATIONS = {
    "min": lambda s: s.min(),
    "q1": lambda s: s.quantile(0.25),
    "median": lambda s: s.quantile(0.5),
    "avg": lambda s: s.mean(),
    "q3": lambda s: s.quantile(0.75),
    "p95": lambda s: s.quantile(0.95),
    "max": lambda s: s.max(),
}

# subset of _AGGREGATIONS rendered as dashboard header strips, in display order
_STRIP_AGGREGATIONS = ["avg", "max", "min", "p95"]


def _summary_node_series(metrics, df_dir):
    """Per-node Series for each summary metric, keyed as in ``_SUMMARY_METRICS``."""

    def _col(df, col):
        if df is None or getattr(df, "empty", True) or col not in df:
            return pd.Series(dtype=float)
        return df.set_index("node_id")[col]

    lat = metrics.get("latency_by_node")
    if lat is None or len(lat) == 0:
        lat = pd.Series(dtype=float)
    return {
        "pdr": _col(metrics.get("pdr"), "pdr"),
        "latency": lat,
        "cpu_util": _col(metrics.get("cpu"), "cpu_usage"),
        "parent_switch": _parent_switches_by_node(df_dir),
    }


def compute_aggregate(metrics, data, df_dir):
    """Per-run summary as a plain (JSON-able) dict.

    ``total`` holds packet / parent-switch counts; the remaining keys
    (min, q1, median, avg, q3, p95, max) each map every summary metric to that
    statistic taken across nodes. Metric values are raw numbers -- pdr as a 0..1
    fraction, latency in seconds, cpu_util in percent, parent_switch as a count
    -- and missing values are ``None``.
    """
    series_map = _summary_node_series(metrics, df_dir)

    def _agg(agg_fn):
        out = {}
        for key in _SUMMARY_METRICS:
            s = series_map[key].dropna()
            out[key] = None if s.empty else float(agg_fn(s))
        return out

    switch = series_map["parent_switch"]
    sent, recv, not_joined, unreachable, in_network = _packet_totals(data)
    result = {
        "total": {
            "parent_switch": int(switch.sum()) if len(switch) else None,
            "packets_sent": None if _isnan(sent) else int(sent),
            "packets_received_by_root": None if _isnan(recv) else int(recv),
            "packets_lost_not_joined": None if _isnan(not_joined) else int(not_joined),
            "packets_lost_root_unreachable": (
                None if _isnan(unreachable) else int(unreachable)
            ),
            "packets_lost_in_network": None if _isnan(in_network) else int(in_network),
        }
    }
    for name, fn in _AGGREGATIONS.items():
        result[name] = _agg(fn)
    return result


def render_summary_html(aggregate):
    """Render Total / Avg / Max / Min / P95 header strips from ``compute_aggregate``."""

    def _fmt_row(prefix, values):
        items = []
        for key, (label, fmt) in _SUMMARY_METRICS.items():
            v = values.get(key)
            items.append((f"{prefix} {label}", "n/a" if v is None else fmt(v)))
        return items

    total = aggregate.get("total", {})

    def _n(key):
        v = total.get(key)
        return "n/a" if v is None else str(v)

    total_items = [
        ("Total parent switches", _n("parent_switch")),
        ("Packets sent", _n("packets_sent")),
        ("Packets received by root", _n("packets_received_by_root")),
        ("Packets lost (not joined)", _n("packets_lost_not_joined")),
        ("Packets lost (root unreachable)", _n("packets_lost_root_unreachable")),
        ("Packets lost (in network)", _n("packets_lost_in_network")),
    ]

    strips = [_render_strip("Total", total_items)]
    for name in _STRIP_AGGREGATIONS:
        title = name.capitalize()
        strips.append(_render_strip(title, _fmt_row(title, aggregate.get(name, {}))))
    return "".join(strips)


def plot_topology(df_dir, metrics):
    path = os.path.join(df_dir, "topology.json")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping topology plot")
        return []
    with open(path) as fh:
        topo = json.load(fh)

    dodag_path = os.path.join(df_dir, "dodag.csv")
    parent_of = {}
    parent_steps = []
    if os.path.exists(dodag_path):
        dodag = pd.read_csv(dodag_path)
        if not dodag.empty and {"time_s", "node_id", "parent_id"}.issubset(
            dodag.columns
        ):
            dodag = dodag.sort_values("time_s")
            parent_of = {
                int(row.node_id): int(row.parent_id)
                for row in dodag.drop_duplicates("node_id", keep="last").itertuples()
            }
            steps = []
            for t, grp in dodag.groupby("time_s", sort=True):
                steps.append(
                    [
                        float(t),
                        [[int(r.node_id), int(r.parent_id)] for r in grp.itertuples()],
                    ]
                )
            parent_steps = [[0.0, []]] + steps

    pdr_df = metrics["pdr"]
    has_pdr = not pdr_df.empty
    pdr_by_node = {int(row.node_id): float(row.pdr) for row in pdr_df.itertuples()}
    latency_by_node = metrics["latency_by_node"]

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
            marker=dict(symbol="star", size=22, color=GRAPH_COLORS["server"]),
            text=[f"{int(server[0]['id'])}"],
            textposition="top center",
            customdata=[[int(server[0]["id"]), "server"]],
            hovertemplate=node_hover,
            name="Server",
        )
    )
    cpu_by_node = {
        int(row.node_id): float(row.cpu_usage)
        for row in metrics["cpu"].itertuples()
    }
    def _per_client(by_node):
        return [by_node.get(int(m["id"]), float("nan")) for m in clients]

    def _finite_bounds(values, default_min=0.0, default_max=1.0):
        finite = [v for v in values if not math.isnan(v)]
        if not finite:
            return default_min, default_max
        return min(finite), max(finite)

    clients_pdr = _per_client(pdr_by_node)
    pdr_cmin, pdr_cmax = _finite_bounds(clients_pdr, 0.0, 10.0)
    pdr_cmin = min(pdr_cmin, 0.75)
    clients_cpu = _per_client(cpu_by_node)
    clients_latency = _per_client(latency_by_node)
    hop_df = metrics.get("cpu_usage_by_hop")
    hop_by_node = (
        {int(row.node_id): int(row.hop_count) for row in hop_df.itertuples()}
        if hop_df is not None and not hop_df.empty
        else {}
    )
    clients_hop = _per_client(hop_by_node)
    clients_custom = [
        [int(m["id"]), m.get("role", "client"), pdr, cpu, lat, hop]
        for m, pdr, cpu, lat, hop in zip(
            clients,
            clients_pdr,
            clients_cpu,
            clients_latency,
            clients_hop,
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
                cmin=pdr_cmin,
                cmax=pdr_cmax,
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

    for mid, pid in parent_of.items():
        if mid not in positions or pid not in positions:
            continue
        cx, cy = positions[mid]
        px, py = positions[pid]
        fig.add_annotation(
            x=px,
            y=py,
            ax=cx,
            ay=cy,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="rgba(60,60,60,0.7)",
            standoff=9,
            startstandoff=9,
        )

    edge_script = (
        "(function() {\n"
        "  var gd = document.getElementById('topology_plot');\n"
        "  if (!gd) return;\n"
        "  var nodes = %s;\n"
        "  var TX_RANGE = %s, INT_RANGE = %s;\n"
        "  var CIRC_X = %s, CIRC_Y = %s;\n"
        "  var PARENT_STEPS = %s;\n"
        "  var EDGE_IDX = 0, TX_CIRCLE_IDX = 1, INT_CIRCLE_IDX = 2;\n"
        "  var SERVER_IDX = 3, CLIENTS_IDX = 4;\n"
        "  var current = null;\n"
        "  gd.style.position = 'relative';\n"
        "  var st = document.createElement('style');\n"
        "  st.textContent = 'html,body{height:100%%;margin:0;}' +\n"
        "    '#topology_plot .hoverlayer{visibility:hidden;}' +\n"
        "    '#topology_plot{height:calc(100%% - 48px) !important;}';\n"
        "  document.head.appendChild(st);\n"
        "  var tip = document.createElement('div');\n"
        "  tip.style.cssText = 'position:absolute;top:8px;display:none;' +\n"
        "    'background:rgba(255,255,255,0.97);border:1px solid #b0b7c3;' +\n"
        "    'border-radius:4px;padding:6px 10px;font-family:Arial;font-size:11px;' +\n"
        "    'color:#2a3f5f;box-shadow:0 2px 6px rgba(0,0,0,0.25);' +\n"
        "    'pointer-events:none;z-index:10;white-space:pre-line;max-width:280px;';\n"
        "  gd.appendChild(tip);\n"
        "  function showTip(html) {\n"
        "    tip.innerHTML = html;\n"
        "    tip.style.display = 'block';\n"
        "    if (!gd._fullLayout || !gd._fullLayout._size) return;\n"
        "    var sz = gd._fullLayout._size;\n"
        "    tip.style.right = (gd._fullLayout.width - sz.l - sz.w + 8) + 'px';\n"
        "    tip.style.top = (sz.t + 8) + 'px';\n"
        "  }\n"
        "  function hideTip() { tip.style.display = 'none'; }\n"
        "  var PRE = {};\n"
        "  for (var id in nodes) {\n"
        "    var node = nodes[id];\n"
        "    var segX = [], segY = [];\n"
        "    for (var i = 0; i < node.n.length; i++) {\n"
        "      var nb = nodes[node.n[i]];\n"
        "      if (!nb) continue;\n"
        "      segX.push(node.x, nb.x, null);\n"
        "      segY.push(node.y, nb.y, null);\n"
        "    }\n"
        "    var txX = [], txY = [], itX = [], itY = [];\n"
        "    for (var k = 0; k < CIRC_X.length; k++) {\n"
        "      txX.push(node.x + CIRC_X[k] * TX_RANGE);\n"
        "      txY.push(node.y + CIRC_Y[k] * TX_RANGE);\n"
        "      itX.push(node.x + CIRC_X[k] * INT_RANGE);\n"
        "      itY.push(node.y + CIRC_Y[k] * INT_RANGE);\n"
        "    }\n"
        "    PRE[id] = {seg:{x:segX,y:segY}, tx:{x:txX,y:txY}, it:{x:itX,y:itY}};\n"
        "  }\n"
        "  var rafId = null, pendingId = null;\n"
        "  function requestEdges(id) {\n"
        "    pendingId = id;\n"
        "    if (rafId) return;\n"
        "    rafId = requestAnimationFrame(function() {\n"
        "      rafId = null;\n"
        "      if (pendingId === null) return;\n"
        "      var nid = pendingId; pendingId = null;\n"
        "      var p = PRE[nid];\n"
        "      if (!p) return;\n"
        "      Plotly.restyle(gd, {\n"
        "        x: [p.seg.x, p.tx.x, p.it.x],\n"
        "        y: [p.seg.y, p.tx.y, p.it.y],\n"
        "        visible: [true, true, true]\n"
        "      }, [EDGE_IDX, TX_CIRCLE_IDX, INT_CIRCLE_IDX]);\n"
        "    });\n"
        "  }\n"
        "  function clearEdges() {\n"
        "    pendingId = null;\n"
        "    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }\n"
        "    Plotly.restyle(gd, {visible: [false, false, false]}, [EDGE_IDX, TX_CIRCLE_IDX, INT_CIRCLE_IDX]);\n"
        "  }\n"
        "  gd.on('plotly_hover', function(e) {\n"
        "    var pt = e.points[0];\n"
        "    if (!pt) return;\n"
        "    if (pt.curveNumber !== SERVER_IDX && pt.curveNumber !== CLIENTS_IDX) return;\n"
        "    var id = String(pt.customdata[0]);\n"
        "    if (id === current) return;\n"
        "    current = id;\n"
        "    var metricIdx = 2;\n"
        "    var cb = gd.data[CLIENTS_IDX].marker.colorbar;\n"
        "    var cbTitle = cb ? (cb.title ? cb.title.text : '') : '';\n"
        "    if (cbTitle === 'CPU Usage (%%)') metricIdx = 3;\n"
        "    else if (cbTitle === 'Avg Latency (s)') metricIdx = 4;\n"
        "    else if (cbTitle === 'Hop Count') metricIdx = 5;\n"
        "    var tipHtml = 'Node <b>' + id + '</b> (' + pt.customdata[1] + ')<br>' +\n"
        "      'Position: (' + pt.x.toFixed(1) + ', ' + pt.y.toFixed(1) + ')<br>' +\n"
        "      'TX range: ' + TX_RANGE + ' m<br>' +\n"
        "      'Interference range: ' + INT_RANGE + ' m';\n"
        "    var v = pt.customdata[metricIdx];\n"
        "    if (v !== undefined && v !== null && !isNaN(v)) {\n"
        "      if (metricIdx === 3) tipHtml += '<br>CPU: ' + v.toFixed(1) + '%%';\n"
        "      else if (metricIdx === 4) tipHtml += '<br>Latency: ' + v.toFixed(3) + ' s';\n"
        "      else if (metricIdx === 5) tipHtml += '<br>Hop count: ' + v + ' hops';\n"
        "      else tipHtml += '<br>PDR: ' + (v * 100).toFixed(1) + '%%';\n"
        "    }\n"
        "    showTip(tipHtml);\n"
        "    requestEdges(id);\n"
        "  });\n"
        "  gd.on('plotly_unhover', function() {\n"
        "    hideTip();\n"
        "    current = null;\n"
        "    clearEdges();\n"
        "  });\n"
        "  if (PARENT_STEPS.length > 1) {\n"
        "    var arrowState = {}, stepIdx = 0;\n"
        "    var last = PARENT_STEPS.length - 1;\n"
        "    function fmtTime(i) {\n"
        "      var t = PARENT_STEPS[i][0];\n"
        "      return (Math.round(t * 10) / 10) + ' s';\n"
        "    }\n"
        "    var wrap = document.createElement('div');\n"
        "    wrap.style.cssText = 'box-sizing:border-box;height:48px;padding:6px 16px 8px;' +\n"
        "      'background:#fff;border-top:1px solid #e0e0e0;font-family:Arial;';\n"
        "    wrap.innerHTML = '<div style=\"font-size:12px;color:#2a3f5f;margin-bottom:2px;\">' +\n"
        "      'DODAG parent at t = <b id=\"topo_time_label\">' + fmtTime(last) + '</b></div>' +\n"
        '      \'<input id="topo_time_slider" type="range" min="0" max="\' + last +\n'
        '      \'" value="\' + last + \'" step="1" style="width:100%%;margin:0;">\';\n'
        "    gd.insertAdjacentElement('afterend', wrap);\n"
        "    var range = document.getElementById('topo_time_slider');\n"
        "    var label = document.getElementById('topo_time_label');\n"
        "    function rebuild(idx) {\n"
        "      var fwd = idx >= stepIdx;\n"
        "      var st = fwd ? arrowState : {};\n"
        "      for (var k = fwd ? stepIdx + 1 : 1; k <= idx; k++) {\n"
        "        var d = PARENT_STEPS[k][1];\n"
        "        for (var j = 0; j < d.length; j++) st[d[j][0]] = d[j][1];\n"
        "      }\n"
        "      arrowState = st;\n"
        "      stepIdx = idx;\n"
        "      var anns = [];\n"
        "      for (var nid in arrowState) {\n"
        "        var a = nodes[nid], b = nodes[arrowState[nid]];\n"
        "        if (!a || !b) continue;\n"
        "        anns.push({x:b.x, y:b.y, ax:a.x, ay:a.y, xref:'x', yref:'y',\n"
        "          axref:'x', ayref:'y', showarrow:true, arrowhead:2, arrowsize:1.2,\n"
        "          arrowwidth:1.5, arrowcolor:'rgba(60,60,60,0.7)',\n"
        "          standoff:9, startstandoff:9});\n"
        "      }\n"
        "      Plotly.relayout(gd, {annotations: anns});\n"
        "    }\n"
        "    var sRaf = null, sTarget = null;\n"
        "    range.addEventListener('input', function() {\n"
        "      sTarget = parseInt(range.value, 10);\n"
        "      if (sRaf) return;\n"
        "      sRaf = requestAnimationFrame(function() {\n"
        "        sRaf = null;\n"
        "        if (sTarget === null) return;\n"
        "        var t = sTarget; sTarget = null;\n"
        "        if (t === stepIdx) return;\n"
        "        rebuild(t);\n"
        "        label.textContent = fmtTime(t);\n"
        "      });\n"
        "    });\n"
        "  }\n"
        "})();"
    ) % (
        json.dumps(nodes_data),
        tx_range,
        interference_range,
        json.dumps(circ_x),
        json.dumps(circ_y),
        json.dumps(parent_steps),
    )

    cpu_cmin, cpu_cmax = _finite_bounds(clients_cpu, 0.0, 10.0)
    real_latency = [v for v in clients_latency if not math.isnan(v)]
    latency_cmin_real, latency_cmax = _finite_bounds(clients_latency, 0.0, 1.0)
    has_missing_latency = len(real_latency) < len(clients_latency)
    if has_missing_latency or latency_cmin_real == 0.0:
        clients_latency_color = [0.0 if math.isnan(v) else v for v in clients_latency]
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
    real_hop = [v for v in clients_hop if not math.isnan(v)]
    has_hop = len(real_hop) == len(clients_hop) and len(real_hop) > 0
    hop_cmin, hop_cmax = _finite_bounds(clients_hop, 0, 1)
    layout: dict = dict(
        title=_title("Network Topology"),
        dragmode="pan",
        xaxis=_axis("X (m)", range=x_axis),
        yaxis=_axis("Y (m)", range=y_axis, scaleanchor="x", scaleratio=1),
    )

    def _color_button(label, color, cmin, cmax, scale, bar_title):
        return dict(
            label=label,
            method="restyle",
            args=[
                {
                    "marker.color": [color],
                    "marker.cmin": [cmin],
                    "marker.cmax": [cmax],
                    "marker.colorscale": [scale],
                    "marker.colorbar.title.text": [bar_title],
                },
                [4],
            ],
        )

    buttons = []
    if has_pdr:
        buttons = [
            _color_button("PDR", clients_pdr, pdr_cmin, pdr_cmax, PDR_COLORSCALE, "PDR"),
            _color_button(
                "CPU Util", clients_cpu, cpu_cmin, cpu_cmax, CPU_COLORSCALE, "CPU Usage (%)"
            ),
            _color_button(
                "Avg Latency",
                clients_latency_color,
                latency_cmin,
                latency_cmax,
                latency_colorscale,
                "Avg Latency (s)",
            ),
        ]
    if has_hop:
        buttons.append(
            _color_button(
                "Hop Count", clients_hop, hop_cmin, hop_cmax, HOP_COLORSCALE, "Hop Count"
            )
        )
    if buttons:
        layout["updatemenus"] = _button_menu(buttons)
    fig.update_layout(layout)
    fig._topology_post_script = edge_script
    print("  Add topology")
    return [fig]


def save_metrics_csv(metrics, out_dir):
    out_dir = os.path.join(out_dir, "aggregate_metrics")
    os.makedirs(out_dir, exist_ok=True)
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, df in value.items():
                df.to_csv(os.path.join(out_dir, f"{subkey}.csv"), index=False)
                print(f"  Saved {subkey}.csv")
        elif isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(out_dir, f"{key}.csv"), index=False)
            print(f"  Saved {key}.csv")


def plot_metrics(df_dir: str, output_dir: str):
    data = load_data(df_dir)
    metrics = compute_metrics(data)
    os.makedirs(output_dir, exist_ok=True)
    save_metrics_csv(metrics, output_dir)

    print(f"\nGenerating plots in {output_dir}/ ...")
    figs = [
        *plot_topology(df_dir, metrics),
        # *plot_etx(metrics),
        *plot_cpu_usage(metrics),
        *plot_by_hop(metrics),
        *plot_packet_delivery(metrics),
    ]

    html_config = {"toImageButtonOptions": {"format": "png", "scale": 8}}

    for fig in figs:
        fig.update_layout(template=None)
        fig.update_layout(plot_bgcolor="#E5ECF6", paper_bgcolor="white")

    aggregate = compute_aggregate(metrics, data, df_dir)
    with open(os.path.join(output_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"  Wrote {os.path.join(output_dir, 'aggregate.json')}")

    header_html = render_run_config_html(load_run_config(df_dir)) + render_summary_html(
        aggregate
    )

    with open(f"{output_dir}/dashboard.html", "w") as f:
        topo_html = figs[0].to_html(
            full_html=True,
            include_plotlyjs=plotly_src(output_dir),
            div_id="topology_plot",
            post_script=getattr(figs[0], "_topology_post_script", None),
            config=html_config,
        )
        if header_html:
            topo_html = topo_html.replace("<body>", f"<body>\n{header_html}", 1)
        f.write(topo_html)
        for fig in figs[1:]:
            f.write(
                fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config=html_config,
                )
            )
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
        help="Output directory for the dashboard (default: plots/)",
    )
    args = parser.parse_args()

    plot_metrics(args.df_dir, args.output_dir)
