#!/usr/bin/env python3
"""
Topology JSON generator for grid and random topologies.

Notes:
- Ensures initial connectivity under given radio.tx_range (raises on failure).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import config

from topology_utils import assert_connected


@dataclass
class RadioConf:
    tx_range: float = 50.0
    interference_range: float = 100.0
    success_tx: float = 1.0
    success_rx: float = 1.0


def make_base(
    topology_id: str,
    topo_type: str,
    platform: str,
    seed: int,
    radio: RadioConf,
    duration_s: int,
) -> dict:
    return {
        "version": 1,
        "topology_id": topology_id,
        "type": topo_type,
        "platform": platform,
        "seed": seed,
        "radio": {
            "tx_range": radio.tx_range,
            "interference_range": radio.interference_range,
            "success_tx": radio.success_tx,
            "success_rx": radio.success_rx,
        },
        "timing": {"duration_s": duration_s},
        "motes": [],
    }


def grid_positions(num_clients: int) -> List[Tuple[float, float]]:
    # Approximate square grid
    cols = math.ceil(math.sqrt(num_clients))
    rows = math.ceil(num_clients / cols)
    pts = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= num_clients:
                break
            pts.append((c * config.spacing, r * config.spacing))
            idx += 1
    return pts


def random_positions(
    num_clients: int,
    seed: int | None = None,
    tx_range: float = 50.0,
    max_attempts: int = 6000,
) -> List[Tuple[float, float]]:
    if seed is not None:
        random.seed(seed)

    upper_x, upper_y = tx_range, tx_range
    lower_x, lower_y = -tx_range, -tx_range
    positions = [(0, 0)]

    min_spacing_sqr = (tx_range * 0.4) ** 2
    max_spacing_sqr = (tx_range * 0.80) ** 2
    range_mul = 0.8
    for i in range(num_clients):
        # the first client only has the root to connect to; later ones need 2 neighbours
        min_neighbours = min(2, len(positions))
        for _ in range(max_attempts):
            x, y = (random.uniform(lower_x, upper_x), random.uniform(lower_y, upper_y))
            if (
                all(
                    (x - x0) ** 2 + (y - y0) ** 2 >= min_spacing_sqr
                    for (x0, y0) in positions
                )
                and sum(
                    (x - x0) ** 2 + (y - y0) ** 2 <= max_spacing_sqr
                    for (x0, y0) in positions
                )
                >= min_neighbours
            ):
                positions.append((x, y))
                upper_x = max(upper_x, x + tx_range * range_mul)
                lower_x = min(lower_x, x - tx_range * range_mul)
                upper_y = max(upper_y, y + tx_range * range_mul)
                lower_y = min(lower_y, y - tx_range * range_mul)
                break
        else:
            raise RuntimeError(
                f"Failed to generate connected random topology after {max_attempts} attempts"
            )

    if _is_connected(positions, tx_range=tx_range):
        return positions[1:]


def _is_connected(positions: List[Tuple[float, float]], tx_range: float) -> bool:
    """
    Check if the graph formed by connecting nodes within tx_range is connected.
    Includes the server at (0,0) in the connectivity check.

    Args:
        positions: List of (x, y) positions for client nodes
        tx_range: Maximum distance for connectivity

    Returns:
        True if graph is connected, False otherwise
    """
    # Add server at (0,0) to the positions
    server_position = (0.0, 0.0)
    all_positions = [server_position] + positions

    if len(all_positions) <= 1:
        return True

    # Build adjacency list
    n = len(all_positions)
    adj = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = all_positions[i]
            x2, y2 = all_positions[j]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance <= tx_range:
                adj[i].append(j)
                adj[j].append(i)

    # BFS to check connectivity starting from server (index 0)
    visited = [False] * n
    queue = [0]  # Start from server node
    visited[0] = True
    connected_count = 1

    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                connected_count += 1

    return connected_count == n


def build_topology(
    topo_type: str,
    n: int,
    spacing: float,
    radio: RadioConf,
    duration_s: int,
    platform: str,
    seed: int,
    add_overloading_client: bool = True,
    overloading_client_ratio: float = 0.05,
    overloading_client_seed: int = 999,
) -> dict:
    assert n >= 2, "Need at least server + 1 client"
    num_clients = n - 1
    motes = [{"id": 1, "role": "server", "x": 0.0, "y": 0.0}]

    if topo_type == "grid":
        pts = grid_positions(num_clients, spacing)
    elif topo_type == "random":
        pts = random_positions(
            num_clients,
            seed=seed,
            max_attempts=6000,
            tx_range=radio.tx_range,
        )
    else:
        raise ValueError(f"Unknown topology type: {topo_type}")

    for i, (x, y) in enumerate(pts, start=2):
        motes.append({"id": i, "role": "client", "x": float(x), "y": float(y)})

    # Randomly turn some clients into overloading clients that send at a higher
    # rate (config.overloading_client_bps). Uses its own RNG so node positions
    # for a given seed are unchanged by this option. Overloading clients are
    # kept at least 2 * tx_range apart so their neighbourhoods don't overlap.
    if add_overloading_client:
        rng = random.Random(overloading_client_seed)
        clients = motes[1:]
        num_overloading = math.ceil(round(len(clients) * overloading_client_ratio, 6))
        picked = []
        for m in rng.sample(clients, len(clients)):
            if len(picked) == num_overloading:
                break
            if all(
                math.hypot(m["x"] - p["x"], m["y"] - p["y"]) >= 2 * radio.tx_range
                for p in picked
            ):
                picked.append(m)
        if len(picked) < num_overloading:
            print(
                f"Warning: only placed {len(picked)}/{num_overloading} overloading "
                "clients at least 2 * tx_range apart"
            )
        for m in picked:
            m["role"] = "overloading_client"

    topology_id = f"{topo_type}_n{n}_s{int(spacing)}_seed{seed}"
    topo = make_base(topology_id, topo_type, platform, seed, radio, duration_s)
    topo["motes"] = motes

    # Connectivity check
    assert_connected(topo["motes"], radio.tx_range)
    return topo


def generate_topology(
    topo_type: str,
    num_of_nodes: int,
    spacing: float,
    out_json: str,
    platform: str = "sky",
    seed: int = 123456,
    tx_range: float = 50.0,
    interference_range: float = 100.0,
    success_tx: float = 1.0,
    success_rx: float = 1.0,
    duration: int = 180,
    add_overloading_client: bool = True,
    overloading_client_ratio: float = 0.05,
    overloading_client_seed: int = 999,
):
    random.seed(seed)

    radio = RadioConf(
        tx_range=tx_range,
        interference_range=interference_range,
        success_tx=success_tx,
        success_rx=success_rx,
    )

    topo = build_topology(
        topo_type=topo_type,
        n=num_of_nodes,
        spacing=spacing,
        radio=radio,
        duration_s=duration,
        platform=platform,
        seed=seed,
        add_overloading_client=add_overloading_client,
        overloading_client_ratio=overloading_client_ratio,
        overloading_client_seed=overloading_client_seed,
    )

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(topo, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate topology JSON")
    p.add_argument("type", choices=["grid", "random"])
    p.add_argument(
        "-n", "--num", type=int, required=True, help="Total motes incl. server"
    )
    p.add_argument(
        "-s",
        "--spacing",
        type=float,
        required=True,
        help="Base spacing between neighbors",
    )
    p.add_argument("-o", "--out", type=str, required=True, help="Output JSON path")
    p.add_argument("--platform", default="sky")
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--tx-range", type=float, default=50.0)
    p.add_argument("--interference-range", type=float, default=100.0)
    p.add_argument("--success-tx", type=float, default=1.0)
    p.add_argument("--success-rx", type=float, default=1.0)
    p.add_argument("--duration", type=int, default=180)
    p.add_argument(
        "--overloading-client-ratio",
        type=float,
        default=0.05,
        help="Fraction of clients randomly turned into overloading_client (at least one)",
    )
    p.add_argument(
        "--overloading-client-seed",
        type=int,
        default=999,
        help="Seed for the random choice of overloading clients",
    )
    args = p.parse_args()
    generate_topology(
        topo_type=args.type,
        num_of_nodes=args.num,
        spacing=args.spacing,
        out_json=args.out,
        platform=args.platform,
        seed=args.seed,
        tx_range=args.tx_range,
        interference_range=args.interference_range,
        success_tx=args.success_tx,
        success_rx=args.success_rx,
        duration=args.duration,
        add_overloading_client=True,
        overloading_client_ratio=args.overloading_client_ratio,
        overloading_client_seed=args.overloading_client_seed,
    )
