# AGENTS.md

## What This Is

Contiki-NG / Cooja simulation of an RPL (IPv6 low-power routing) network with UDP. The project adds a custom **metrics collection module** (`rpl-motes/rpl-udp/metrics.{h,c}`) on top of the stock `rpl-udp` example. 1 DAG root (server) + 10 DAG nodes (clients) on Sky motes.

## Architecture

- `rpl-motes/rpl-udp/` — the only custom code you edit. Everything else is framework.
  - `udp-server.c` — DAG root, UDP echo server (port 5678). Unmodified from upstream.
  - `udp-client.c` — DAG node, UDP client. Modified: imports `metrics.h` and calls `metrics_start()`.
  - `metrics.h` / `metrics.c` — the project's core contribution. Periodic process prints ETX, DODAG, Energest, CPU util, TX power every `METRICS_PERIOD` (30s). Per-packet helpers: latency, hop count, compute-time stopwatch.
  - `Makefile` — builds both binaries. Add new `.c` files to `PROJECT_SOURCEFILES`.
- `RPL.csc` — Cooja simulation config (XML). Defines motes, positions, radio medium (UDGM, 60% TX / 50% RX success).
- `simulations_script.js` — Cooja test harness. Times out at 300s wall clock; passes after 120s simulated time.
- `run_sim.sh` — launches Cooja headless. Must be run from repo root.
- `contiki-ng/` — Contiki-NG framework (large, don't edit directly).
- `report/` — LaTeX report (VUW project).
- `simulation_logs/` — output directory (gitignored).

## Build & Run

```bash
# Build motes (from rpl-motes/rpl-udp/)
cd rpl-motes/rpl-udp && bear -- make TARGET=sky

# Run simulation headless (from repo root)
./run_sim.sh
```

`run_sim.sh` calls `./gradlew run --args='--no-gui ../../../RPL.csc ...'` from inside `contiki-ng/tools/cooja/`. Requires Gradle and Java. Logs go to `simulation_logs/`.

## Key Gotchas

- **RPL Lite only**: `metrics.c` uses `net/routing/rpl-lite/rpl.h` APIs. Switching to RPL Classic requires changing the ETX/DODAG accessors.
- **Metrics process is client-only**: `metrics_start()` is only called from `udp-client.c`. The server does not run metrics.
- **Per-packet metrics are not automatic**: latency (`metrics_log_latency`), hop count (`metrics_print_hop_count`), and compute timing must be called manually from send/receive callbacks. The periodic process only covers ETX/energy/CPU/TX power.
- **`metrics_time_start/end` is not reentrant**: one timer at a time; no nesting.
- **Target is always `sky`**: the `.csc` file builds with `TARGET=sky` (MSP430-based TelosB motes). Do not use a different target unless you also update `RPL.csc`.

## Modifying Metrics

If adding a new metric:
1. Declare in `metrics.h`, implement in `metrics.c`.
2. If it should be periodic, add it to the `metrics_process` loop and update `METRICS_PERIOD` if needed (default 30s, override via `project-conf.h`).
3. If it's per-packet, call the new function from `udp-client.c`'s send/rx callbacks.

## Simulation Notes

- Fixed random seed (`123456`) in `RPL.csc` for reproducibility.
- Radio medium: UDGM with 50m TX range, 100m interference, 60% TX success, 50% RX success.
- 1 server mote (ID 1) + 10 client motes (IDs 2-11) at fixed positions.
- All metric output is grep-parsable (tagged: `ETX:`, `DODAG:`, `ENERGEST:`, `CPU_UTIL:`, `TX_POWER:`, `LATENCY:`, `HOP_COUNT:`, `COMPUTE_TIME:`).
