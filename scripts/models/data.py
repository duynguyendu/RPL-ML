"""Feature/label schema and training-data gathering for the PDR-predicting
LightGBM model used by the "mlof" RPL objective function (see
rpl/contiki-ng/os/net/routing/rpl-lite/rpl-mlof.c).

Each training row is one (node, time chunk) within one simulation run: the
node's link/system state averaged over that chunk, paired with the PDR it
achieved during that same chunk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# TODO: maybe add parent num of children
# TODO: include current CPU util and parent CPU util
# TODO: if comparing between current parent and potential parent should take it's current load affecting parent's CPU into account
# TODO: number of successful transmit per dropped packet, should be weighted when changing parents
# TODO: let's include RSSI
# TODO: how to process missing data
# TODO: parent switch since when
FEATURE_COLUMNS = ["etx", "hop_count", "ppm", "cpu_util", "num_neighbours"]
LABEL_COLUMN = "pdr"

# Width of the aggregation window used to bucket metrics.csv / latency.csv rows
# before averaging features and computing PDR within each bucket. TODO: tune.
TIME_CHUNK_SECONDS = 300

# No per-node neighbour count is tracked anywhere yet (on-mote or in the log
# parser), so every row gets this fixed placeholder for now.
NUM_NEIGHBOURS_PLACEHOLDER = 123


def gather_training_data(runs_dir: Path) -> pd.DataFrame:
    """Build a (node, time chunk) feature/label table across every run under
    ``runs_dir``.

    TODO: not implemented yet. Real algorithm, per run directory that has
    config.json + metrics.csv + latency.csv (mirror the run-scanning pattern
    in scripts/plot_comparison.py's collect_runs()):
      1. Bucket metrics.csv's `time_s` into TIME_CHUNK_SECONDS-wide windows
         per node_id. Within each window: average `etx`, take `hop_count`,
         and compute `cpu_util = cpu_ticks / total_ticks * 100` averaged over
         the window (same formula as compute_latest() in
         scripts/plot_metrics.py, but aggregated over the window instead of
         "latest row only").
      2. Bucket latency.csv's `client_send_time` into the same windows per
         node_id. PDR per window = (rows with server_receive_time != 0) /
         (all rows) in that window (same idea as compute_pdr() in
         scripts/plot_metrics.py, applied per window instead of over the
         whole run).
      3. Join features + label on (node_id, time_chunk) for that run. `ppm`
         comes from the run's config.json (constant for the whole run).
         `num_neighbours` is filled with NUM_NEIGHBOURS_PLACEHOLDER.
      4. Concatenate across all runs under runs_dir.

    For now this just returns an empty, correctly-columned DataFrame so the
    rest of the training pipeline is runnable end-to-end before real data
    exists.
    """
    print(f"TODO: gather_training_data({runs_dir}) is not implemented yet — "
          "returning no rows. See the docstring for the intended algorithm.")
    return pd.DataFrame(columns=FEATURE_COLUMNS + [LABEL_COLUMN])


def make_dummy_training_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic stand-in for gather_training_data(), so training/eval code
    can be exercised end-to-end before real training data exists. Not a
    substitute for real data — values are random within plausible ranges and
    have no relationship to real network behavior.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "etx": rng.uniform(1.0, 4.0, n),
            "hop_count": rng.integers(1, 6, n),
            "ppm": rng.choice([15, 20, 30, 40, 60], n),
            "cpu_util": rng.uniform(0.0, 100.0, n),
            "num_neighbours": np.full(n, NUM_NEIGHBOURS_PLACEHOLDER),
        }
    )
    df[LABEL_COLUMN] = np.clip(
        1.0 - 0.15 * df["etx"] - 0.05 * df["hop_count"] + rng.normal(0, 0.05, n),
        0.0,
        1.0,
    )
    return df
