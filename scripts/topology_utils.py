"""
Topology utilities for headless Cooja pipeline.

Functions:
- build_connectivity_graph(motes, tx_range)
- find_server_id(motes)
- is_connected_to_server(motes, tx_range)
- assert_connected(motes, tx_range)

Mote schema expected (from topology JSON):
  {"id": int, "role": "server"|"client", "x": float, "y": float}
"""

from typing import Dict, List, Set, Tuple
import math


def build_connectivity_graph(motes: List[dict], tx_range: float) -> Dict[int, Set[int]]:
    """Return undirected adjacency based on Euclidean distance and tx_range.

    Nodes i and j are adjacent if distance(i, j) <= tx_range.
    """
    positions = {int(m["id"]): (float(m["x"]), float(m["y"])) for m in motes}
    node_ids = list(positions.keys())
    adjacency: Dict[int, Set[int]] = {i: set() for i in node_ids}
    for idx, i in enumerate(node_ids):
        xi, yi = positions[i]
        for j in node_ids[idx + 1 :]:
            xj, yj = positions[j]
            if math.hypot(xi - xj, yi - yj) <= tx_range:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def find_server_id(motes: List[dict]) -> int:
    """Return the unique server id. Raises ValueError if not exactly one server."""
    servers = [
        int(m["id"]) for m in motes if str(m.get("role", "")).lower() == "server"
    ]
    if len(servers) != 1:
        raise ValueError(f"Expected exactly one server, found {len(servers)}")
    return servers[0]


def is_connected_to_server(
    motes: List[dict], tx_range: float
) -> Tuple[bool, List[int]]:
    """Check connectivity of all motes to the server via geometric graph.

    Returns (connected, disconnected_ids_sorted).
    """
    if not motes:
        return True, []
    adjacency = build_connectivity_graph(motes, tx_range)
    server_id = find_server_id(motes)

    # BFS from server
    seen: Set[int] = set()
    queue: List[int] = [server_id]
    seen.add(server_id)
    while queue:
        u = queue.pop(0)
        for v in adjacency.get(u, ()):  # type: ignore
            if v not in seen:
                seen.add(v)
                queue.append(v)

    all_ids = {int(m["id"]) for m in motes}
    disconnected = sorted(all_ids - seen)
    return len(disconnected) == 0, disconnected


def assert_connected(
    motes: List[dict], tx_range: float, sparse_max_attempts: int = 20
) -> None:
    """Assert that all motes are connected to the server under tx_range.

    Raises ValueError with the list of disconnected ids when not connected.
    """
    ok, missing = is_connected_to_server(motes, tx_range)
    if not ok:
        raise ValueError(
            f"Disconnected motes (by id): {missing} (max attempts: {sparse_max_attempts})"
        )
