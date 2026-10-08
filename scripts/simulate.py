#!/usr/bin/env python3
"""Run pipeline.py for a range of node counts / send rates / objective
functions. Local replacement for simulate.sh's parallel-dispatch loop, with
optional sharding across machines (see --shard-index/--num-shards).

Every (num_nodes, bps, rpl_of, seed, overloading_client_seed, mlof_path_w_pdr)
combination is one job (non-MLOF OFs ignore mlof_path_w_pdr, so they run once
per remaining combination). Up to
--max-active-jobs run concurrently, each pinned to its own reusable
--build_dir_name slot (named with this machine's hostname, since multiple
machines may share this same filesystem -- see ansible/simulate_seeds.yml).

Usage:
    python3 simulate.py [--label=run1] [--max-active-jobs=8] [--topo-type=grid]

Override the swept grid (each defaults to the hardcoded *_LIST constant
below; comma/space-separated):
    python3 simulate.py --node-list=30,60 --bps-list=128,256 --seed-list=111,222 --overloading-client-seed-list=999,998
    python3 simulate.py --mlof-path-w-pdr-list=0,2,4,6,8,10,12,14,16

Sharding across machines (they must share this filesystem -- job identities
never collide across shards, so every machine can safely write into the
same --run-dir with no merge step needed afterward):
    python3 simulate.py --run-dir=runs/sim_shared --shard-index=0 --num-shards=3
    python3 simulate.py --run-dir=runs/sim_shared --shard-index=1 --num-shards=3
    python3 simulate.py --run-dir=runs/sim_shared --shard-index=2 --num-shards=3
"""

import argparse
import itertools
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from cooja_simulation import ensure_template_built

BASE_COOJA = "../rpl/contiki-ng/tools/cooja/"

PLATFORM = "z1"
TOPO_TYPE = "random"
# topology types handled by topology_generator.generate_topology
TOPO_TYPES = ["grid", "random"]
NODE_LIST = [30, 60]
# offered load per client in bit/s (send interval = PACKET_SIZE * 8 / bps)
BPS_LIST = [512, 384, 256, 128]
# OF_LIST = ["of0", "mhrof", "mlof_dtree", "mlof_lgbm"]
OF_LIST = ["of0", "mlof_dtree"]
SEED_LIST = [12756, 826352, 927106, 538256, 389271]
OVERLOADING_CLIENT_SEED_LIST = [999, 998]
BUFFER_SIZE = 8
DURATION = 1800
PACKET_SIZE = 64
# 1 = MLOF firmware logs training-data lines (MLOF_CONF_LOG_TRAINING_DATA)
MLOF_LOG_TRAINING_DATA = 1
# MLOF path-cost weight of predicted PDR vs ETX, out of 16 (MLOF_CONF_PATH_W_PDR)
MLOF_PATH_W_PDR_LIST = [12]


def fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def parse_list(value: str | None, default: list, cast=str) -> list:
    if not value:
        return default
    return [cast(v) for v in value.replace(",", " ").split()]


def build_jobs(
    node_list: list[int],
    bps_list: list[int],
    of_list: list[str],
    seed_list: list[int],
    overloading_seed_list: list[int],
    w_pdr_list: list[int],
) -> list[tuple[int, int, str, int, int, int]]:
    jobs = []
    for num_nodes, bps, rpl_of, seed, ol_seed in itertools.product(
        node_list, bps_list, of_list, seed_list, overloading_seed_list
    ):
        # only MLOF reads the weight; sweeping it for other OFs would just
        # rerun the identical simulation
        ws = w_pdr_list if rpl_of.startswith("mlof") else w_pdr_list[:1]
        jobs.extend((num_nodes, bps, rpl_of, seed, ol_seed, w) for w in ws)
    return jobs


def start_job(
    slot: str,
    run_dir: Path,
    num_nodes: int,
    bps: int,
    rpl_of: str,
    seed: int,
    overloading_seed: int,
    duration: int,
    topo_type: str,
    mlof_log_training_data: int,
    w_pdr: int,
):
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{slot}.log"
    log_f = open(log_path, "ab")
    proc = subprocess.Popen(
        [
            sys.executable,
            "pipeline.py",
            f"--duration={duration}",
            "--is_simulate=True",
            "--is_generate_topology=True",
            "--is_plot_metrics=True",
            f"--packet_size={PACKET_SIZE}",
            f"--buffer_size={BUFFER_SIZE}",
            f"--rpl_of={rpl_of}",
            f"--num_of_nodes={num_nodes}",
            f"--bps={bps}",
            f"--platform={PLATFORM}",
            f"--topo_type={topo_type}",
            f"--seed={seed}",
            "--add_overloading_client=True",
            f"--overloading_client_seed={overloading_seed}",
            f"--base_output_dir={run_dir}",
            f"--build_dir_name={slot}",
            f"--mlof_log_training_data={mlof_log_training_data}",
            f"--mlof_path_w_pdr={w_pdr}",
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group, like bash's setsid
    )
    log_f.close()  # the child keeps its own fd via dup(); safe to close ours
    return proc


def job_desc(
    job_id: int,
    shard_total: int,
    slot: str,
    num_nodes: int,
    bps: int,
    rpl_of: str,
    seed: int,
    overloading_seed: int,
    w_pdr: int,
) -> str:
    w_part = f" w_pdr={w_pdr}" if rpl_of.startswith("mlof") else ""
    return f"({job_id}/{shard_total}) {slot} of={rpl_of}{w_part} nodes={num_nodes} bps={bps} seed={seed} overloading_seed={overloading_seed}"


def terminate_all(active: dict) -> None:
    for proc, _, _ in active.values():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for proc, _, _ in active.values():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--label",
        default="",
        help="Folded into the run directory name (ignored if --run-dir is given)",
    )
    parser.add_argument("--max-active-jobs", type=int, default=8)
    parser.add_argument(
        "--duration",
        type=int,
        default=DURATION,
        help=f"Simulated seconds per run, forwarded to pipeline.py (default: {DURATION})",
    )
    parser.add_argument(
        "--topo-type",
        default=TOPO_TYPE,
        choices=TOPO_TYPES,
        help=f"Topology type for every job, forwarded to pipeline.py (default: {TOPO_TYPE})",
    )
    parser.add_argument(
        "--mlof-log-training-data",
        type=int,
        default=MLOF_LOG_TRAINING_DATA,
        choices=[0, 1],
        help="Build firmware with MLOF_CONF_LOG_TRAINING_DATA=0/1, forwarded to "
        f"pipeline.py (default: {MLOF_LOG_TRAINING_DATA})",
    )
    parser.add_argument(
        "--node-list",
        default=None,
        help=f"Comma/space-separated node counts to sweep (default: {NODE_LIST})",
    )
    parser.add_argument(
        "--bps-list",
        default=None,
        help=f"Comma/space-separated offered loads in bit/s to sweep (default: {BPS_LIST})",
    )
    parser.add_argument(
        "--of-list",
        default=None,
        help=f"Comma/space-separated objective functions to sweep (default: {OF_LIST})",
    )
    parser.add_argument(
        "--seed-list",
        default=None,
        help=f"Comma/space-separated seeds to sweep (default: {SEED_LIST})",
    )
    parser.add_argument(
        "--overloading-client-seed-list",
        default=None,
        help="Comma/space-separated seeds for the random choice of overloading "
        f"clients to sweep (default: {OVERLOADING_CLIENT_SEED_LIST})",
    )
    parser.add_argument(
        "--mlof-path-w-pdr-list",
        default=None,
        help="Comma/space-separated MLOF_CONF_PATH_W_PDR values (0-16) to sweep "
        f"for MLOF OFs (default: {MLOF_PATH_W_PDR_LIST})",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This machine's shard, 0-based (default: 0)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards/machines (default: 1)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Reuse an existing run directory -- required when sharding across "
        "machines that share this filesystem, so every shard writes into the "
        "same place with no merge step needed. Auto-generated (timestamped) "
        "if omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print this shard's job list and exit"
    )
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must be in [0, num_shards)")

    node_list = parse_list(args.node_list, NODE_LIST, int)
    bps_list = parse_list(args.bps_list, BPS_LIST, int)
    of_list = parse_list(args.of_list, OF_LIST, str)
    seed_list = parse_list(args.seed_list, SEED_LIST, int)
    if args.topo_type != "random" and len(seed_list) > 1:
        # only the random topology sweeps seeds; every other type runs once
        # per configuration, using the first seed (still Cooja's randomseed)
        print(
            f"=== topo_type={args.topo_type} is not random: using only seed "
            f"{seed_list[0]} (ignoring {seed_list[1:]}) ==="
        )
        seed_list = seed_list[:1]

    overloading_seed_list = parse_list(
        args.overloading_client_seed_list, OVERLOADING_CLIENT_SEED_LIST, int
    )

    w_pdr_list = parse_list(args.mlof_path_w_pdr_list, MLOF_PATH_W_PDR_LIST, int)
    if not all(0 <= w <= 16 for w in w_pdr_list):
        parser.error("--mlof-path-w-pdr-list values must be in [0, 16]")

    all_jobs = build_jobs(
        node_list, bps_list, of_list, seed_list, overloading_seed_list, w_pdr_list
    )
    total_global = len(all_jobs)
    my_jobs = all_jobs[args.shard_index :: args.num_shards]

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        max_nodes = max(node_list)
        min_rate = min(bps_list)
        label_part = f"_{args.label}" if args.label else ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            Path("runs")
            / f"sim_{PLATFORM}_{args.topo_type}_n{max_nodes}_bps{min_rate}_buffer{BUFFER_SIZE}{label_part}_{ts}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Writing runs to {run_dir} "
        f"(shard {args.shard_index}/{args.num_shards}, {len(my_jobs)}/{total_global} jobs, "
        f"topo_type={args.topo_type}, mlof_log_training_data={args.mlof_log_training_data}, "
        f"mlof_path_w_pdr={w_pdr_list}) ==="
    )

    if args.dry_run:
        for i, (num_nodes, bps, rpl_of, seed, ol_seed, w_pdr) in enumerate(my_jobs, start=1):
            print(
                f"  {job_desc(i, len(my_jobs), '(dry-run)', num_nodes, bps, rpl_of, seed, ol_seed, w_pdr)}"
            )
        return

    ensure_template_built(BASE_COOJA)

    hostname = socket.gethostname()
    free_slots = [
        f"build_{PLATFORM}_{hostname}_w{s}" for s in range(1, args.max_active_jobs + 1)
    ]
    active: dict[str, tuple[subprocess.Popen, float, tuple]] = {}
    pending = list(enumerate(my_jobs, start=1))
    start_all = time.monotonic()

    def handle_signal(signum, frame):
        print(
            f"\n=== Interrupted: terminating {len(active)} in-flight job(s) ===",
            flush=True,
        )
        terminate_all(active)
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while pending or active:
        while pending and free_slots:
            job_id, (num_nodes, bps, rpl_of, seed, ol_seed, w_pdr) = pending.pop(0)
            slot = free_slots.pop()
            desc = job_desc(
                job_id, len(my_jobs), slot, num_nodes, bps, rpl_of, seed, ol_seed, w_pdr
            )
            print(f"=== [{time.strftime('%H:%M:%S')}] START {desc} ===", flush=True)
            proc = start_job(
                slot, run_dir, num_nodes, bps, rpl_of, seed, ol_seed, args.duration,
                args.topo_type, args.mlof_log_training_data, w_pdr,
            )
            active[slot] = (
                proc,
                time.monotonic(),
                (job_id, num_nodes, bps, rpl_of, seed, ol_seed, w_pdr),
            )

        finished = [
            slot for slot, (proc, _, _) in active.items() if proc.poll() is not None
        ]
        for slot in finished:
            proc, start, (job_id, num_nodes, bps, rpl_of, seed, ol_seed, w_pdr) = active.pop(slot)
            desc = job_desc(
                job_id, len(my_jobs), slot, num_nodes, bps, rpl_of, seed, ol_seed, w_pdr
            )
            status = (
                "DONE" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            )
            dur = fmt_dur(time.monotonic() - start)
            print(
                f"=== [{time.strftime('%H:%M:%S')}] {status} {desc} (took {dur}) ===",
                flush=True,
            )
            free_slots.append(slot)

        if not finished and (pending or active):
            time.sleep(1)

    print(
        f"=== All {len(my_jobs)} job(s) in this shard finished in {fmt_dur(time.monotonic() - start_all)} ==="
    )

    if args.num_shards == 1:
        print("=== Building cross-run comparison dashboard ===")
        subprocess.run(
            [sys.executable, "plot_comparison.py", "--runs-dir", str(run_dir)]
        )
        print(f"=== Results in {run_dir} ===")
    else:
        print(
            f"=== Shard {args.shard_index}/{args.num_shards} done -- other shards write "
            f"into this same {run_dir} via the shared filesystem, no merge needed. "
            "Rebuild the dashboard once every shard has finished: "
            f"python3 plot_comparison.py --runs-dir {run_dir} ==="
        )


if __name__ == "__main__":
    main()
