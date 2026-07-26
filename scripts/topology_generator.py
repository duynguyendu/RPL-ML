#!/usr/bin/env python3
"""
Topology JSON generator for ring, star, grid, tree, line, mesh, and sparse_grid.

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


def ring_positions(
    num_clients: int, spacing: float, tx_range: float
) -> List[Tuple[float, float]]:
    # Chord length between adjacent clients should be <= spacing and <= tx_range
    # radius from chord: c = 2 r sin(pi/N) -> r = c / (2 sin(pi/N))
    N = max(3, num_clients)
    r_from_spacing = spacing / (2.0 * math.sin(math.pi / N))
    # Ensure server at center is connected: enforce r <= tx_range
    r = min(r_from_spacing, tx_range * 0.9)
    pts = []
    for k in range(num_clients):
        theta = 2.0 * math.pi * k / num_clients
        pts.append((r * math.cos(theta), r * math.sin(theta)))
    return pts


def star_positions(num_clients: int, radius: float) -> List[Tuple[float, float]]:
    pts = []
    for k in range(num_clients):
        theta = 2.0 * math.pi * k / max(1, num_clients)
        pts.append((radius * math.cos(theta), radius * math.sin(theta)))
    return pts


def grid_positions(num_clients: int, spacing: float) -> List[Tuple[float, float]]:
    # Approximate square grid
    cols = math.ceil(math.sqrt(num_clients))
    rows = math.ceil(num_clients / cols)
    pts = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= num_clients:
                break
            pts.append((c * spacing, r * spacing))
            idx += 1
    return pts


def line_positions(num_clients: int, spacing: float) -> List[Tuple[float, float]]:
    return [(i * spacing, 0.0) for i in range(num_clients)]


def tree_positions(
    num_clients: int, spacing: float, branching: int
) -> List[Tuple[float, float]]:
    # Simple layered layout: server at (0,0), children at y=spacing, etc.
    # Horizontal spread per level to reduce overlap.
    pts = []
    # We place clients only (server is separate at 0,0)
    level = 1
    placed = 0
    parent_count = 1  # server
    while placed < num_clients:
        nodes_this_level = min(num_clients - placed, parent_count * branching)
        width = max(1, nodes_this_level)
        # Center around x=0
        start_x = -0.5 * (width - 1) * spacing
        for i in range(nodes_this_level):
            pts.append((start_x + i * spacing, level * spacing))
        placed += nodes_this_level
        parent_count = nodes_this_level
        level += 1
    return pts


def mesh_positions(
    num_clients: int, spacing: float, jitter: float, seed: int
) -> List[Tuple[float, float]]:
    # Start from grid, add small jitter to avoid perfect regularity while staying connected.
    random.seed(seed)
    base = grid_positions(num_clients, spacing)
    pts = []
    for x, y in base:
        jx = x + random.uniform(-jitter, jitter)
        jy = y + random.uniform(-jitter, jitter)
        pts.append((jx, jy))
    return pts


def sparse_grid_positions(
    num_clients: int,
    grid_spacing: float = 40.0,
    seed: int | None = None,
    max_attempts: int = 20,
    tx_range: float = 50.0,
) -> List[Tuple[float, float]]:
    """
    Generate sparse grid positions with connectivity check.

    Args:
        num_clients: Number of client nodes (excluding server)
        grid_spacing: Spacing between grid points
        seed: Random seed for reproducibility

    Returns:
        List of (x, y) positions for client nodes

    Raises:
        RuntimeError: If unable to generate connected topology after 10 attempts
    """
    if seed is not None:
        random.seed(seed)

    # Try multiple attempts because random removal/jitter may disconnect the graph
    for attempt in range(max_attempts):
        # Sample R and C uniformly from [ceil(sqrt(N)), ceil(1.5 * sqrt(N))]
        sqrt_n = math.sqrt(num_clients)
        min_dim = math.ceil(sqrt_n)
        max_dim = math.ceil(1.5 * sqrt_n)

        R = random.randint(min_dim, max_dim)
        C = random.randint(min_dim, max_dim)

        # print(f"Iteration/Loop {attempt}: R: {R}, C: {C}")
        # Generate all grid positions starting at (0,0)
        all_positions = []
        for r in range(R):
            for c in range(C):
                if r == 0 and c == 0:
                    continue
                all_positions.append((c * grid_spacing, r * grid_spacing))

        # Randomly remove excess nodes to get exactly N nodes
        if len(all_positions) > num_clients:
            selected_positions = random.sample(all_positions, num_clients)
            # print(f"Iteration/Loop {attempt}: Selected positions: {selected_positions}")
        else:
            selected_positions = all_positions

        # Add jitter to all positions (except root at (0,0) which is handled separately)
        jittered_positions = []
        for x, y in selected_positions:
            # Add jitter: uniform random delta in [-grid_spacing/2, grid_spacing/2]
            delta_x = random.uniform(-grid_spacing / 2, grid_spacing / 2)
            delta_y = random.uniform(-grid_spacing / 2, grid_spacing / 2)
            jittered_positions.append((x + delta_x, y + delta_y))

        # Connectivity check under provided tx_range
        if _is_connected(jittered_positions, tx_range=tx_range):
            # print(f"success after {attempt}/{max_attempts} attempts")
            return jittered_positions

    # If we get here, all attempts failed
    raise RuntimeError(
        f"Failed to generate connected sparse grid topology after {max_attempts} attempts"
    )


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
    branching: int = 2,
    mesh_jitter: float = 0.2,
    sparse_max_attempts: int = 10,
) -> dict:
    assert n >= 2, "Need at least server + 1 client"
    num_clients = n - 1
    motes = [{"id": 1, "role": "server", "x": 0.0, "y": 0.0}]

    if topo_type == "ring":
        pts = ring_positions(num_clients, spacing, radio.tx_range)
    elif topo_type == "star":
        # Ensure radius within tx_range so direct connectivity to server
        radius = min(spacing, radio.tx_range * 0.9)
        pts = star_positions(num_clients, radius)
    elif topo_type == "grid":
        pts = grid_positions(num_clients, spacing)
    elif topo_type == "tree":
        # Keep spacing conservative for edge length <= tx_range
        if spacing > radio.tx_range:
            spacing = radio.tx_range * 0.5
        pts = tree_positions(num_clients, spacing, branching=branching)
    elif topo_type == "line":
        pts = line_positions(num_clients, spacing)
    elif topo_type == "mesh":
        pts = mesh_positions(num_clients, spacing, jitter=mesh_jitter, seed=seed)
    elif topo_type == "sparse_grid":
        pts = sparse_grid_positions(
            num_clients,
            grid_spacing=spacing,
            seed=seed,
            max_attempts=sparse_max_attempts,
            tx_range=radio.tx_range,
        )
    else:
        raise ValueError(f"Unknown topology type: {topo_type}")

    for i, (x, y) in enumerate(pts, start=2):
        motes.append({"id": i, "role": "client", "x": float(x), "y": float(y)})

    topology_id = f"{topo_type}_n{n}_s{int(spacing)}_seed{seed}"
    topo = make_base(topology_id, topo_type, platform, seed, radio, duration_s)
    topo["motes"] = motes

    # Connectivity check
    assert_connected(
        topo["motes"], radio.tx_range, sparse_max_attempts=sparse_max_attempts
    )
    return topo


def main():
    p = argparse.ArgumentParser(description="Generate topology JSON")
    p.add_argument(
        "type", choices=["ring", "star", "grid", "tree", "line", "mesh", "sparse_grid"]
    )
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
    p.add_argument("--platform", default="z1")
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--tx-range", type=float, default=50.0)
    p.add_argument("--interference-range", type=float, default=100.0)
    p.add_argument("--success-tx", type=float, default=1.0)
    p.add_argument("--success-rx", type=float, default=1.0)
    p.add_argument("--duration", type=int, default=180)
    p.add_argument("--branching", type=int, default=2, help="Tree branching factor")
    p.add_argument(
        "--mesh-jitter",
        type=float,
        default=0.2,
        help="Mesh jitter amplitude (in spacing units)",
    )
    p.add_argument(
        "--sparse-max-attempts",
        type=int,
        default=20,
        help="Max attempts for sparse_grid connectivity",
    )
    args = p.parse_args()

    # Seed global RNG for reproducibility where used
    random.seed(args.seed)

    radio = RadioConf(
        tx_range=args.tx_range,
        interference_range=args.interference_range,
        success_tx=args.success_tx,
        success_rx=args.success_rx,
    )

    topo = build_topology(
        topo_type=args.type,
        n=args.num,
        spacing=args.spacing,
        radio=radio,
        duration_s=args.duration,
        platform=args.platform,
        seed=args.seed,
        branching=args.branching,
        mesh_jitter=args.mesh_jitter * args.spacing,
        sparse_max_attempts=args.sparse_max_attempts,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(topo, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
