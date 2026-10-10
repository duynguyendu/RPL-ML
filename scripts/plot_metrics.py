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


def compute_etx_by_node(data):
    """Each node's link ETX to its preferred parent, averaged over its reports."""
    return _query(
        """
        SELECT node_id, AVG(etx) AS etx
        FROM df
        WHERE etx <> 65535.0
        GROUP BY node_id
        """,
        data["metrics"],
        ["node_id", "etx"],
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
    # server_receive_time <> 0 is the correct "was this packet actually
    # measurable" filter -- it only needs the request to have reached the
    # server, not a reply to come back to the client. hop_count is a
    # separate field (only ever set alongside client_receive_time, i.e.
    # only when a reply *did* come back) and must not gate latency itself,
    # or every row goes missing the moment replies stop, even though
    # server_receive_time - client_send_time is still perfectly computable.
    return _query(
        """
        SELECT node_id, hop_count, (server_receive_time - client_send_time) AS latency
        FROM df
        WHERE server_receive_time <> 0
        """,
        data["latency"],
        ["node_id", "hop_count", "latency"],
    )


def compute_dio_rate(data):
    """Average DIOs sent per minute per node.

    ``dio_sent`` is a cumulative counter since boot, so each node's last
    reading divided by its elapsed simulated time gives the mean rate.
    """
    df = data["metrics"]
    columns = ["node_id", "dio_per_min"]
    if df is None or df.empty or "dio_sent" not in df:
        return pd.DataFrame(columns=columns)
    return _query(
        """
        SELECT node_id,
               arg_max(dio_sent, time_s) * 60.0 / max(time_s) AS dio_per_min
        FROM df
        WHERE dio_sent IS NOT NULL
        GROUP BY node_id
        HAVING max(time_s) > 0
        ORDER BY node_id
        """,
        df,
        columns,
    )


def compute_metrics(data):
    latest = compute_latest(data)
    latency = compute_latency(data)
    latency_known_hop = latency[latency["hop_count"] != INVALID]
    return {
        "etx": compute_etx(data),
        "etx_by_node": compute_etx_by_node(data),
        "cpu": compute_cpu(latest),
        "pdr": compute_pdr(data),
        "dio_rate": compute_dio_rate(data),
        "latency": latency,
        "latency_by_hop": (
            latency_known_hop.groupby(["node_id", "hop_count"])
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
    "dio": "#911eb4",
}

# Overloading clients (higher send rate) are drawn as a spiky star instead of a circle
OVERLOADING_SYMBOL = "hexagram"


def _is_overloading(mote):
    return str(mote.get("role", "")).lower() == "overloading_client"


PDR_COLORSCALE = [[0.0, "red"], [1.0, "green"]]
CPU_COLORSCALE = [[0.0, "darkblue"], [1.0, "red"]]
LOAD_COLORSCALE = "YlOrRd"
LOAD_TITLE = "Load (bps)"


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


def plot_dio_rate(metrics):
    df = metrics["dio_rate"]
    if df.empty:
        return []

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["node_id"],
            y=df["dio_per_min"],
            name="DIO / min",
            marker_color=GRAPH_COLORS["dio"],
            hovertemplate="Node %{x}<br>DIO: %{y:.2f} /min<extra></extra>",
        )
    )
    fig.update_layout(
        title=_title("Average DIO sent per minute"),
        xaxis=_axis("Node ID", type="category"),
        yaxis=_axis("DIO sent per minute"),
    )
    print("  Add dio_rate")
    return [fig]


# config keys that make up the run_id (see pipeline.py); "etx" is derived
RUN_ID_CONFIG_KEYS = [
    "num_of_nodes",
    "bps",
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


def _retransmissions_by_node(data, df_dir):
    """Retransmissions per node as a Series indexed by node_id.

    The firmware logs TX (every transmission attempt, retries included) and
    ACKED (frames acknowledged) as cumulative counters of the link to the
    current preferred parent, so TX - ACKED counts unacknowledged attempts.
    """
    return _parent_link_growth_by_node(
        data, df_dir, ["tx_packets", "acked_packets"],
        lambda m: m["tx_packets"] - m["acked_packets"],
    )


def _queue_drops_by_node(data, df_dir):
    """Queue drops per node as a Series indexed by node_id.

    The firmware logs DROPPED, the link's num_queue_drops: frames to the
    current preferred parent discarded because the MAC queue was full
    (MAC_TX_QUEUE_FULL), as a cumulative counter like TX / ACKED.
    """
    return _parent_link_growth_by_node(
        data, df_dir, ["dropped_packets"], lambda m: m["dropped_packets"]
    )


def _parent_link_growth_by_node(data, df_dir, columns, counter):
    """Per-node growth of a cumulative preferred-parent link counter.

    ``counter(frame)`` derives the counter from ``columns`` of metrics.csv.
    Each report is matched to the parent the node had then (dodag.csv); per
    (node, parent) link the growth over the run is its last value minus its
    value at the node's first report (0 for links first used later), and a
    node's total sums its links.
    """
    df = data.get("metrics")
    path = os.path.join(df_dir, "dodag.csv")
    if (
        df is None
        or df.empty
        or any(c not in df for c in columns)
        or not os.path.exists(path)
    ):
        return pd.Series(dtype=float)
    dodag = pd.read_csv(path)
    if dodag.empty:
        return pd.Series(dtype=float)
    # rows without a preferred parent log ETX=65535 and zero counters
    m = df.loc[df["etx"] != INVALID, ["time_s", "node_id", *columns]]
    m = pd.merge_asof(
        m.astype({"time_s": float}).sort_values("time_s"),
        dodag[["time_s", "node_id", "parent_id"]].sort_values("time_s"),
        on="time_s",
        by="node_id",
    ).dropna(subset=["parent_id"])
    if m.empty:
        return pd.Series(dtype=float)
    m["count"] = counter(m)
    m = m.sort_values("time_s")
    first_t = m.groupby("node_id")["time_s"].transform("min")
    links = m.groupby(["node_id", "parent_id"])
    last = links["count"].last()
    base = (
        m[m["time_s"] == first_t].groupby(["node_id", "parent_id"])["count"].first()
    )
    growth = (last - base.reindex(last.index, fill_value=0)).clip(lower=0)
    return growth.groupby(level="node_id").sum().astype(float)


LOAD_INTERVAL_S = 60  # simulated seconds between load-model snapshots


def _load_inputs(df_dir):
    """(config, role_of, rate_of, server_id) for the load model, or None.

    ``rate_of`` is each node's own send rate r(v) in bps: ``bps`` for a client,
    ``overloading_client_bps`` for an overloading client and 0 for the server.
    """
    paths = {
        name: os.path.join(df_dir, f"{name}.json") for name in ("config", "topology")
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    with open(paths["config"]) as fh:
        cfg = json.load(fh)
    with open(paths["topology"]) as fh:
        motes = json.load(fh).get("motes", [])
    if "bps" not in cfg:
        return None
    role_of = {int(m["id"]): str(m.get("role", "client")).lower() for m in motes}
    server_id = next((i for i, r in role_of.items() if r == "server"), None)
    if server_id is None:
        return None
    rate_of = {
        nid: 0.0
        if role == "server"
        else float(cfg.get("overloading_client_bps", cfg["bps"]))
        if role == "overloading_client"
        else float(cfg["bps"])
        for nid, role in role_of.items()
    }
    return cfg, role_of, rate_of, server_id


def _tree_load(parent, rate_of, server_id):
    """(load, hop, children) per node for one DODAG given as {node: parent}.

    L(v) = r(v) + sum of L(children); a node without a parent has not joined
    and offers nothing. ``hop`` only holds nodes whose chain reaches the server.
    """
    load = {nid: 0.0 for nid in rate_of}
    hop = {server_id: 0}
    children = {nid: 0 for nid in rate_of}
    for nid in rate_of:
        if nid in parent:
            children[parent[nid]] = children.get(parent[nid], 0) + 1
        # walk up the parent chain once per node: add its rate to itself and
        # every ancestor, and record its depth if the chain hits the server
        if nid == server_id or nid not in parent:
            continue
        r = rate_of[nid]
        load[nid] += r
        seen, u, depth = {nid}, parent[nid], 1
        while u not in seen:
            load[u] = load.get(u, 0.0) + r
            if u == server_id:
                hop[nid] = depth
                break
            seen.add(u)
            if u not in parent:
                break
            u, depth = parent[u], depth + 1
    return load, hop, children


def compute_load(df_dir):
    """Offered load per node, sampled every ``LOAD_INTERVAL_S`` simulated seconds.

    The DODAG at each snapshot is the latest parent of every node from
    dodag.csv. A joined client offers its send rate r(v) (``bps``, or
    ``overloading_client_bps`` for overloading clients); a node's load is its
    own rate plus the load of all its children, L(v) = r(v) + sum L(children),
    so the server's load is everything that reaches it. Nodes not yet joined
    offer nothing. ``hop`` is the depth along parent links, NaN if the chain
    never reaches the server (detached or looping).

    Returns columns time_s, node_id, role, parent_id, hop, children, rate, load.
    """
    columns = [
        "time_s", "node_id", "role", "parent_id", "hop", "children", "rate", "load",
    ]
    empty = pd.DataFrame(columns=columns)
    inputs = _load_inputs(df_dir)
    dodag_path = os.path.join(df_dir, "dodag.csv")
    if inputs is None or not os.path.exists(dodag_path):
        return empty
    cfg, role_of, rate_of, server_id = inputs
    dodag = pd.read_csv(dodag_path)
    if dodag.empty:
        return empty
    end = cfg.get("duration", 0) + cfg.get("ramp_up_duration", 0)
    if end <= 0:
        end = float(dodag["time_s"].max())

    events = dodag.sort_values("time_s")[["time_s", "node_id", "parent_id"]]
    events = list(events.itertuples(index=False))
    parent, ev = {}, 0
    rows = []
    for t in range(LOAD_INTERVAL_S, int(end) + 1, LOAD_INTERVAL_S):
        while ev < len(events) and events[ev].time_s <= t:
            parent[int(events[ev].node_id)] = int(events[ev].parent_id)
            ev += 1

        load, hop, children = _tree_load(parent, rate_of, server_id)
        for nid, role in role_of.items():
            rows.append(
                (
                    t,
                    nid,
                    role,
                    parent.get(nid, float("nan")),
                    hop.get(nid, float("nan")),
                    children.get(nid, 0),
                    rate_of[nid],
                    load[nid],
                )
            )
    return pd.DataFrame(rows, columns=columns)


def plot_load(load):
    """Heatmap of every client's load over time, plus the server's children's
    loads stacked to show how traffic is split across the DODAG branches."""
    if load.empty:
        return []
    figs = []
    unit = "Offered load (bps)"

    clients = load[load["role"] != "server"].copy()
    ol_ids = set(clients.loc[clients["role"] == "overloading_client", "node_id"])
    label = {
        nid: f"{nid} ★" if nid in ol_ids else str(nid)
        for nid in sorted(clients["node_id"].unique())
    }
    times = sorted(clients["time_s"].unique())

    def _grid(col):
        return clients.pivot(index="node_id", columns="time_s", values=col).loc[
            list(label)
        ]

    z, par, hop, kids = (_grid(c) for c in ("load", "parent_id", "hop", "children"))
    custom = [
        [[p, h, k] for p, h, k in zip(pr, hr, kr)]
        for pr, hr, kr in zip(par.values, hop.values, kids.values)
    ]
    heat = go.Figure(
        go.Heatmap(
            x=times,
            y=list(label.values()),
            z=z.values,
            customdata=custom,
            colorscale="YlOrRd",
            colorbar=dict(title=dict(text=unit, side="right"), thickness=14),
            hovertemplate=(
                "Node %{y}<br>t = %{x} s<br>Load: %{z:.0f} bps<br>"
                "Parent: %{customdata[0]}<br>Hop: %{customdata[1]}<br>"
                "Children: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    heat.update_layout(
        title=_title(
            f"Load per node every {LOAD_INTERVAL_S} s "
            "(L = own rate + children's load, ★ = overloading client)"
        ),
        xaxis=_axis("Simulated time (s)"),
        yaxis=_axis("Node ID", type="category", autorange="reversed"),
        height=max(450, 14 * len(label) + 150),
    )
    figs.append(heat)

    server_ids = set(load.loc[load["role"] == "server", "node_id"])
    root_kids = load[load["parent_id"].isin(server_ids)]
    if not root_kids.empty:
        # overloading clients carried by each first-hop branch at each snapshot
        ol_rows = load[load["node_id"].isin(ol_ids)]
        ol_branch = {}
        parent_at = {
            t: dict(zip(g["node_id"], g["parent_id"])) for t, g in load.groupby("time_s")
        }
        for row in ol_rows.itertuples():
            par_t, u, seen = parent_at[row.time_s], row.node_id, set()
            while u in par_t and not _isnan(par_t[u]) and u not in seen:
                seen.add(u)
                if par_t[u] in server_ids:
                    key = (row.time_s, u)
                    ol_branch[key] = ol_branch.get(key, 0) + 1
                    break
                u = int(par_t[u])

        stack = go.Figure()
        for nid in sorted(root_kids["node_id"].unique()):
            g = root_kids[root_kids["node_id"] == nid].set_index("time_s")
            y = [g["load"].get(t, 0.0) for t in times]
            n_ol = [ol_branch.get((t, nid), 0) for t in times]
            stack.add_trace(
                go.Scatter(
                    x=times,
                    y=y,
                    customdata=n_ol,
                    mode="lines",
                    stackgroup="branches",
                    line=dict(width=0.5),
                    name=label.get(nid, str(nid)),
                    hovertemplate=(
                        f"Branch via node {nid}<br>t = %{{x}} s<br>"
                        "Load: %{y:.0f} bps<br>"
                        "Overloading clients: %{customdata}<extra></extra>"
                    ),
                )
            )
        stack.update_layout(
            title=_title("Load reaching the server, split by first-hop branch"),
            xaxis=_axis("Simulated time (s)"),
            yaxis=_axis(unit),
            hovermode="x unified",
        )
        figs.append(stack)
    print("  Add load model")
    return figs


def _tree_layout(kids, roots):
    """{node: (x, depth)} for a tidy top-down forest: leaves get consecutive x
    slots in DFS order, each parent sits centred over its children and every
    root after the first starts one slot to the right of the previous tree."""
    pos, next_x = {}, [0]

    def place(u, depth):
        cs = kids.get(u, [])
        if not cs:
            pos[u] = (next_x[0], depth)
            next_x[0] += 1
            return
        for c in cs:
            place(c, depth + 1)
        pos[u] = ((pos[cs[0]][0] + pos[cs[-1]][0]) / 2, depth)

    for i, r in enumerate(roots):
        if i:
            next_x[0] += 1
        place(r, 0)
    return pos


def _forest(parent, node_ids, server_id):
    """(kids, roots) drawing every node: the server's DODAG first, then one
    sub-tree per detached group. A detached group is rooted at a node without
    a parent or, for a parent loop, at the loop's lowest id (its parent edge
    is dropped so the loop becomes a tree)."""
    roots, cut = [server_id], set()
    for nid in node_ids:
        if nid == server_id:
            continue
        path, u = [], nid
        while u in parent and u != server_id and u not in path:
            path.append(u)
            u = parent[u]
        if u == server_id:
            continue
        if u in path:  # loop: cut it at its lowest id
            r = min(path[path.index(u):])
            cut.add(r)
        else:  # chain ends at a node with no parent
            r = u
        if r not in roots:
            roots.append(r)
    kids = {}
    for nid in sorted(node_ids):
        if nid != server_id and nid in parent and nid not in cut:
            kids.setdefault(parent[nid], []).append(nid)
    return kids, [roots[0], *sorted(roots[1:])]


def _avg_children(kids, nodes):
    """Mean child count over the parent (non-leaf) nodes among ``nodes``,
    NaN if none has a child."""
    counts = [len(kids[n]) for n in nodes if kids.get(n)]
    return sum(counts) / len(counts) if counts else float("nan")


def plot_dodag_tree(load):
    """The DODAG drawn as a tree (server on top, one row per hop) at every
    load snapshot, with a slider over time; the title reports the average
    number of children per parent (non-leaf) node of the server's DODAG.

    Nodes whose parent chain never reaches the server (no parent yet, or a
    parent loop -- dodag.csv logs only switches to a new parent, so a lost
    parent leaves a stale pointer) are drawn faded as separate sub-trees."""
    if load.empty:
        return []
    server_ids = load.loc[load["role"] == "server", "node_id"].unique()
    if len(server_ids) == 0:
        return []
    server_id = int(server_ids[0])

    snaps, traces, max_depth = [], [], 0
    for t, g in load.groupby("time_s"):
        # only nodes whose parent chain reaches the server are in the DODAG;
        # the rest are drawn as detached sub-trees to its right
        attached = g[g["hop"].notna()]
        parent = {
            int(r.node_id): int(r.parent_id)
            for r in g.itertuples()
            if not _isnan(r.parent_id)
        }
        kids, roots = _forest(parent, g["node_id"].astype(int).tolist(), server_id)
        pos = _tree_layout(kids, roots)
        in_dodag = set(attached["node_id"].astype(int))
        info = g.set_index("node_id")
        ex, ey = [], []
        for p, cs in kids.items():
            for c in cs:
                ex += [pos[p][0], pos[c][0], None]
                ey += [pos[p][1], pos[c][1], None]
        ids = sorted(pos)
        traces.append(
            go.Scatter(
                x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
                line=dict(color="rgba(60,60,60,0.5)", width=1), visible=False,
            )
        )
        traces.append(
            go.Scatter(
                x=[pos[n][0] for n in ids],
                y=[pos[n][1] for n in ids],
                mode="markers+text",
                text=[str(n) for n in ids],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    symbol=[
                        "star" if n == server_id
                        else OVERLOADING_SYMBOL
                        if info.at[n, "role"] == "overloading_client"
                        else "circle"
                        for n in ids
                    ],
                    size=[22 if n == server_id else 16 for n in ids],
                    color=[len(kids.get(n, [])) for n in ids],
                    colorscale="Viridis",
                    cmin=0,
                    cmax=max(1, int(load["children"].max())),
                    opacity=[1.0 if n in in_dodag else 0.45 for n in ids],
                    line=dict(width=1, color="black"),
                    showscale=True,
                    colorbar=dict(title=dict(text="Children", side="right"), thickness=14),
                ),
                customdata=[
                    [n, info.at[n, "role"], len(kids.get(n, [])),
                     pos[n][1] if n in in_dodag else "detached", info.at[n, "load"]]
                    for n in ids
                ],
                hovertemplate=(
                    "Node %{customdata[0]} (%{customdata[1]})<br>"
                    "Hop: %{customdata[3]}<br>Children: %{customdata[2]}<br>"
                    "Load: %{customdata[4]:.0f} bps<extra></extra>"
                ),
                showlegend=False,
                visible=False,
            )
        )
        snaps.append((t, _avg_children(kids, in_dodag), len(g) - len(in_dodag)))
        max_depth = max(max_depth, max(d for _, d in pos.values()))

    # mean over the snapshots (skipping ones where the DODAG has no edge yet)
    run_avg = float(pd.Series([a for _, a, _ in snaps]).mean())

    def _snap_title(t, avg, detached):
        avg_txt = "n/a" if math.isnan(avg) else f"{avg:.2f}"
        run_txt = "n/a" if math.isnan(run_avg) else f"{run_avg:.2f}"
        extra = f"<br>{detached} node(s) detached (faded, right)" if detached else ""
        return _title(
            f"DODAG tree at t = {t} s — avg children per parent node: {avg_txt} "
            f"(run avg {run_txt}){extra}"
        )

    n = len(traces)
    last = len(snaps) - 1
    traces[2 * last].visible = traces[2 * last + 1].visible = True
    steps = []
    for i, (t, avg, detached) in enumerate(snaps):
        vis = [False] * n
        vis[2 * i] = vis[2 * i + 1] = True
        steps.append(
            dict(
                label=str(t),
                method="update",
                args=[{"visible": vis}, {"title": _snap_title(t, avg, detached)}],
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=_snap_title(*snaps[last]),
        xaxis=dict(visible=False),
        yaxis=_axis("Depth (hop in DODAG)", autorange="reversed", dtick=1, zeroline=False),
        height=max(500, 70 * (max_depth + 1) + 220),
        sliders=[
            dict(
                active=last,
                currentvalue=dict(prefix="Snapshot t (s) = "),
                pad=dict(t=30),
                steps=steps,
            )
        ],
    )
    print("  Add dodag_tree")
    return [fig]


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


def _timer_totals(data, us_col, count_col):
    """(total run time in us, total calls) summed over nodes for one firmware
    timer, e.g. predict_us / predict_count for MLOF predict_pdr().

    The firmware logs both as cumulative counters since boot, so each node's
    last metrics row holds its totals. Returns (None, None) for logs without
    those fields.
    """
    df = data.get("metrics")
    if df is None or df.empty or count_col not in df:
        return None, None
    totals = _query(
        f"""
        SELECT SUM({us_col}) AS us, SUM({count_col}) AS count
        FROM (
            SELECT {us_col}, {count_col}
            FROM df
            WHERE {count_col} IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY node_id ORDER BY time_s DESC) = 1
        )
        """,
        df,
        ["us", "count"],
    )
    us, count = totals["us"].iloc[0], totals["count"].iloc[0]
    if pd.isna(count):
        return None, None
    return int(us), int(count)


# per-node metric key -> (strip label, value formatter)
_SUMMARY_METRICS = {
    "pdr": ("PDR", lambda v: f"{v * 100:.1f}%"),
    "latency": ("latency", lambda v: f"{v:.3f} s"),
    "cpu_util": ("CPU util", lambda v: f"{v:.1f}%"),
    "parent_switch": (
        "parent switch",
        lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}",
    ),
    "dio_per_min": ("DIO/min", lambda v: f"{v:.2f}"),
    "retransmissions": (
        "retransmissions",
        lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}",
    ),
    "queue_drops": (
        "queue drops",
        lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}",
    ),
    "load": ("load", lambda v: f"{v:.0f} bps"),
    "hop_count": ("hop count", lambda v: f"{v:.2f}"),
    "etx": ("ETX", lambda v: f"{v:.2f}"),
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
    "jain": lambda s: _jain_index(s),
}

# subset of _AGGREGATIONS rendered as dashboard header strips, in display order
_STRIP_AGGREGATIONS = ["avg", "max", "min", "p95"]


def _jain_index(s):
    """Jain's fairness index (sum x)^2 / (n * sum x^2) over nodes, in [1/n, 1].

    1 means every node has the same value; all-zero values count as equal.
    """
    sq = (s**2).sum()
    return 1.0 if sq == 0 else s.sum() ** 2 / (len(s) * sq)


def _summary_node_series(metrics, data, df_dir):
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
        "dio_per_min": _col(metrics.get("dio_rate"), "dio_per_min"),
        "retransmissions": _retransmissions_by_node(data, df_dir),
        "queue_drops": _queue_drops_by_node(data, df_dir),
        "load": metrics.get("load_by_node", pd.Series(dtype=float)),
        "hop_count": metrics.get("hop_by_node", pd.Series(dtype=float)),
        "etx": _col(metrics.get("etx_by_node"), "etx"),
    }


def compute_aggregate(metrics, data, df_dir):
    """Per-run summary as a plain (JSON-able) dict.

    ``total`` holds packet / parent-switch counts; the remaining keys
    (min, q1, median, avg, q3, p95, max, jain) each map every summary metric to
    that statistic taken across nodes (jain = Jain's fairness index). Metric
    values are raw numbers -- pdr as a 0..1 fraction, latency in seconds, cpu_util in percent, parent_switch as a count,
    dio_per_min in DIOs per minute, retransmissions as a count of
    unacknowledged link-layer TX attempts to the preferred parent,
    queue_drops as a count of frames to the preferred parent dropped on a
    full MAC queue, load in bps (each client's offered load
    averaged over the 60 s snapshots), hop_count in hops to the root (each
    client's depth averaged over the snapshots it was attached), etx as each client's link ETX
    to its preferred parent averaged over its reports -- and missing
    values are ``None``.
    """
    series_map = _summary_node_series(metrics, data, df_dir)

    def _agg(agg_fn):
        out = {}
        for key in _SUMMARY_METRICS:
            s = series_map[key].dropna()
            out[key] = None if s.empty else float(agg_fn(s))
        return out

    switch = series_map["parent_switch"]
    retx = series_map["retransmissions"]
    drops = series_map["queue_drops"]
    sent, recv, not_joined, unreachable, in_network = _packet_totals(data)
    predict_us, predict_count = _timer_totals(data, "predict_us", "predict_count")
    dag_update_us, dag_update_count = _timer_totals(
        data, "dag_update_us", "dag_update_count"
    )
    result = {
        "total": {
            "parent_switch": int(switch.sum()) if len(switch) else None,
            "retransmissions": int(retx.sum()) if len(retx) else None,
            "queue_drops": int(drops.sum()) if len(drops) else None,
            "packets_sent": None if _isnan(sent) else int(sent),
            "packets_received_by_root": None if _isnan(recv) else int(recv),
            "packets_lost_not_joined": None if _isnan(not_joined) else int(not_joined),
            "packets_lost_root_unreachable": (
                None if _isnan(unreachable) else int(unreachable)
            ),
            "packets_lost_in_network": None if _isnan(in_network) else int(in_network),
            "predict_us": predict_us,
            "predict_count": predict_count,
            "avg_predict_us": (
                predict_us / predict_count if predict_count else None
            ),
            "dag_update_us": dag_update_us,
            "dag_update_count": dag_update_count,
            "avg_dag_update_us": (
                dag_update_us / dag_update_count if dag_update_count else None
            ),
        }
    }
    for name, fn in _AGGREGATIONS.items():
        result[name] = _agg(fn)
    return result


def render_summary_html(aggregate):
    """Render Total / Avg / Max / Min / P95 / Jain fairness header strips from ``compute_aggregate``."""

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
        ("Total retransmissions", _n("retransmissions")),
        ("Total queue drops", _n("queue_drops")),
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
    # unitless 0..1, so not the metric's own formatter
    jain = aggregate.get("jain", {})
    strips.append(
        _render_strip(
            "Jain fairness",
            [
                (label, "n/a" if jain.get(key) is None else f"{jain[key]:.3f}")
                for key, (label, _) in _SUMMARY_METRICS.items()
            ],
        )
    )
    return "".join(strips)


def render_predict_time_html(aggregate):
    """Render the average MLOF predict_pdr() and DODAG update run times as a
    header strip."""
    total = aggregate.get("total", {})
    avg = total.get("avg_predict_us")
    count = total.get("predict_count")
    dag_avg = total.get("avg_dag_update_us")
    dag_count = total.get("dag_update_count")
    items = [
        ("Avg predict_pdr() time", "n/a" if avg is None else f"{avg:.1f} us"),
        ("predict_pdr() calls", "n/a" if count is None else str(count)),
    ]
    # DODAG update timing is only in newer logs; leave it out when missing
    if dag_count:
        items += [
            ("Avg DODAG update time", f"{dag_avg:.1f} us"),
            ("DODAG update calls", str(dag_count)),
        ]
    return _render_strip("Computation time", items)


def plot_topology(df_dir, metrics, load=None):
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
            # one [time, [[node, parent], ...]] step per distinct timestamp; plain
            # lists instead of a pandas groupby, which is slow with ~10k groups
            steps = []
            for t, nid, pid in zip(
                dodag["time_s"].tolist(),
                dodag["node_id"].astype(int).tolist(),
                dodag["parent_id"].astype(int).tolist(),
            ):
                if not steps or steps[-1][0] != t:
                    steps.append([float(t), []])
                steps[-1][1].append([nid, pid])
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
    # load of each client under the final DODAG (the slider's default position);
    # the page script recomputes it for whichever DODAG the slider shows
    load_inputs = _load_inputs(df_dir)
    has_load = load_inputs is not None and bool(parent_of)
    rate_of, server_id = ({}, None) if load_inputs is None else load_inputs[2:]
    final_load = _tree_load(parent_of, rate_of, server_id)[0] if has_load else {}
    clients_load = _per_client(final_load)
    # fixed colour-scale top so colours stay comparable while the slider moves:
    # the highest client load over the 60 s snapshots (and the final DODAG).
    # Brief spikes between snapshots are clipped to the top colour.
    snap_max = (
        float(load.loc[load["role"] != "server", "load"].max())
        if load is not None and not load.empty
        else 0.0
    )
    load_cmax = max(
        [snap_max, *(v for v in clients_load if not math.isnan(v))], default=0.0
    ) or 1.0
    clients_custom = [
        [int(m["id"]), m.get("role", "client"), pdr, cpu, lat, load]
        for m, pdr, cpu, lat, load in zip(
            clients,
            clients_pdr,
            clients_cpu,
            clients_latency,
            clients_load,
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
                symbol=[
                    OVERLOADING_SYMBOL if _is_overloading(m) else "circle"
                    for m in clients
                ],
                size=[18 if _is_overloading(m) else 14 for m in clients],
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

    # set every parent arrow in one update: add_annotation per arrow re-validates
    # the whole annotation list each call
    fig.update_layout(
        annotations=[
            dict(
                x=positions[pid][0],
                y=positions[pid][1],
                ax=positions[mid][0],
                ay=positions[mid][1],
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
            for mid, pid in parent_of.items()
            if mid in positions and pid in positions
        ]
    )

    edge_script = (
        "(function() {\n"
        "  var gd = document.getElementById('topology_plot');\n"
        "  if (!gd) return;\n"
        "  var nodes = %s;\n"
        "  var TX_RANGE = %s, INT_RANGE = %s;\n"
        "  var CIRC_X = %s, CIRC_Y = %s;\n"
        "  var PARENT_STEPS = %s;\n"
        "  var RATES = %s, SERVER_ID = %s, FINAL_PARENTS = %s, LOAD_MAX = %s;\n"
        "  var LOAD_TITLE = %s;\n"
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
        "  // load model: L(v) = own rate + sum of children's load, for a DODAG\n"
        "  // given as {node: parent}; nodes without a parent offer nothing\n"
        "  function treeLoad(par) {\n"
        "    var L = {};\n"
        "    for (var id in RATES) L[id] = 0;\n"
        "    for (var id in RATES) {\n"
        "      if (id === SERVER_ID || !(id in par)) continue;\n"
        "      var r = RATES[id], seen = {}, u = String(par[id]);\n"
        "      L[id] += r; seen[id] = true;\n"
        "      while (!seen[u]) {\n"
        "        L[u] = (L[u] || 0) + r;\n"
        "        if (u === SERVER_ID || !(u in par)) break;\n"
        "        seen[u] = true;\n"
        "        u = String(par[u]);\n"
        "      }\n"
        "    }\n"
        "    return L;\n"
        "  }\n"
        "  var curParents = FINAL_PARENTS, curLoad = treeLoad(FINAL_PARENTS);\n"

        "  function colorTitle() {\n"
        "    var cb = gd.data[CLIENTS_IDX].marker.colorbar;\n"
        "    return cb && cb.title ? cb.title.text : '';\n"
        "  }\n"
        "  function applyLoadColors() {\n"
        "    if (colorTitle() !== LOAD_TITLE) return;\n"
        "    var ids = gd.data[CLIENTS_IDX].customdata.map(function(c) { return String(c[0]); });\n"
        "    Plotly.restyle(gd, {\n"
        "      'marker.color': [ids.map(function(id) { return curLoad[id] || 0; })],\n"
        "      'marker.cmin': [0], 'marker.cmax': [LOAD_MAX]\n"
        "    }, [CLIENTS_IDX]);\n"
        "  }\n"
        "  gd.on('plotly_buttonclicked', function() { setTimeout(applyLoadColors, 0); });\n"
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
        "    else if (cbTitle === LOAD_TITLE) metricIdx = 5;\n"
        "    var tipHtml = 'Node <b>' + id + '</b> (' + pt.customdata[1] + ')<br>' +\n"
        "      'Position: (' + pt.x.toFixed(1) + ', ' + pt.y.toFixed(1) + ')<br>' +\n"
        "      'TX range: ' + TX_RANGE + ' m<br>' +\n"
        "      'Interference range: ' + INT_RANGE + ' m';\n"
        "    var v = metricIdx === 5 ? curLoad[id] : pt.customdata[metricIdx];\n"
        "    if (v !== undefined && v !== null && !isNaN(v)) {\n"
        "      if (metricIdx === 3) tipHtml += '<br>CPU: ' + v.toFixed(1) + '%%';\n"
        "      else if (metricIdx === 4) tipHtml += '<br>Latency: ' + v.toFixed(3) + ' s';\n"
        "      else if (metricIdx === 5) tipHtml += '<br>Load: ' + v.toFixed(0) + ' bps';\n"
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
        "      curParents = arrowState;\n"
        "      curLoad = treeLoad(curParents);\n"
        "      applyLoadColors();\n"
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
        json.dumps({str(k): v for k, v in rate_of.items()}),
        json.dumps(str(server_id)),
        json.dumps({str(k): v for k, v in parent_of.items()}),
        json.dumps(load_cmax),
        json.dumps(LOAD_TITLE),
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
    if has_load:
        buttons.append(
            _color_button(
                "Load",
                [0.0 if math.isnan(v) else v for v in clients_load],
                0.0,
                load_cmax,
                LOAD_COLORSCALE,
                LOAD_TITLE,
            )
        )
    if buttons:
        layout["updatemenus"] = _button_menu(buttons)
    fig.update_layout(layout)
    fig._topology_post_script = edge_script
    print("  Add topology")
    return [fig]


def plot_metrics(df_dir: str, output_dir: str, runs_dir: str | None = None):
    if runs_dir is None:
        runs_dir = os.path.dirname(os.path.abspath(output_dir)) or output_dir
    data = load_data(df_dir)
    metrics = compute_metrics(data)
    os.makedirs(output_dir, exist_ok=True)

    load = compute_load(df_dir)
    if not load.empty:
        load.to_csv(os.path.join(output_dir, "load.csv"), index=False)
        print(f"  Wrote {os.path.join(output_dir, 'load.csv')}")
        # each client's load averaged over every snapshot of the run
        clients = load[load["role"] != "server"]
        metrics["load_by_node"] = clients.groupby("node_id")["load"].mean()
        # each client's hop depth averaged over the snapshots it was attached
        metrics["hop_by_node"] = clients.groupby("node_id")["hop"].mean()

    print(f"\nGenerating plots in {output_dir}/ ...")
    figs = [
        *plot_topology(df_dir, metrics, load),
        *plot_load(load),
        *plot_dodag_tree(load),
        # *plot_etx(metrics),
        *plot_cpu_usage(metrics),
        *plot_by_hop(metrics),
        *plot_packet_delivery(metrics),
        *plot_dio_rate(metrics),
    ]

    html_config = {"toImageButtonOptions": {"format": "png", "scale": 8}}

    for fig in figs:
        fig.update_layout(template=None)
        fig.update_layout(plot_bgcolor="#E5ECF6", paper_bgcolor="white")

    aggregate = compute_aggregate(metrics, data, df_dir)
    with open(os.path.join(output_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"  Wrote {os.path.join(output_dir, 'aggregate.json')}")

    header_html = (
        render_run_config_html(load_run_config(df_dir))
        + render_predict_time_html(aggregate)
        + render_summary_html(aggregate)
    )

    with open(f"{output_dir}/dashboard.html", "w") as f:
        topo_html = figs[0].to_html(
            full_html=True,
            include_plotlyjs=plotly_src(output_dir, runs_dir),
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
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Shared runs directory to copy plotly.min.js into (default: output-dir's parent)",
    )
    args = parser.parse_args()

    plot_metrics(args.df_dir, args.output_dir, args.runs_dir)
