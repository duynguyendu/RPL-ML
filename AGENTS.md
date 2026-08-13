# AGENTS.md

Guidance for AI agents working in this repository.

## Project overview

Research project combining RPL (IPv6 Routing Protocol for Low-power and Lossy
Networks) simulation with TinyML. The pipeline generates a wireless sensor
network topology, runs it in the Cooja simulator (Contiki-NG), parses the
simulation log into metrics (ETX, energy, CPU, latency, PDR), plots them, and
optionally trains a small MLP model.

## Repository structure

```
AGENTS.md              This file
ansible/               Ansible playbook to deploy & run the pipeline remotely
models/                PyTorch MLP model, config, train/inference scripts
rpl/
  contiki-ng/          Contiki-NG checkout (git submodule)
  motes/               RPL firmware: server/, client/, overload-client.c
scripts/               Main Python pipeline (see below)
thesis_report/         LaTeX report
venv/                  Python 3.14 virtualenv (not committed)
requirements.txt       Root deps (matplotlib, numpy, pandas, torch)
```

## Pipeline (scripts/)

Run from `scripts/` so `config.py` resolves paths. Pipeline order:

1. `topology_generator.py` — generate `topology.json` (ring, star, grid, tree,
   line, mesh, sparse_grid, random, scatter)
2. `csc_generator.py` — `topology.json` → Cooja `.csc` file
3. `cooja_simulation.py` — run Cooja headless via `./gradlew run` (slow)
4. `parse_cooja_log.py` — parse `COOJA.testlog` into `metrics.csv`, `latency.csv`
5. `plot_metrics.py` — aggregate with duckdb SQL, render Plotly plots + `dashboard.html`

`pipeline.py` orchestrates all steps; each step can also run standalone via CLI.

## Common commands

```bash
# Full pipeline (all steps enabled)
venv/bin/python pipeline.py           # from scripts/

# Pipeline step as CLI
venv/bin/python topology_generator.py random -n 60 -s 40 -o runs/x/topology.json
venv/bin/python csc_generator.py runs/x/topology.json runs/x/topology.csc --base-dir ../rpl/motes
venv/bin/python cooja_simulation.py runs/x/topology.csc --cooja-base ../rpl/contiki-ng/tools/cooja/ --output-dir runs/x
venv/bin/python parse_cooja_log.py --log-dir runs/x --output-dir runs/x
venv/bin/python plot_metrics.py --df-dir runs/x --output-dir runs/x

# Train MLP
venv/bin/python ../models/train.py --input-dim 5 --hidden-dims 64 32 --epochs 50
```

## Configuration (scripts/config.py)

- All pipeline knobs live in `scripts/config.py` (topology params, radio,
  platform, send rate, duration, output dirs).
- Scripts read config via `import config` — the module is a global singleton.
- CLI overrides: `python pipeline.py path/to/cfg.py` (exec a config file) or
  `python pipeline.py --num_of_nodes=100` (must match the existing type).
- `run_id` (scripts/pipeline.py) encodes the experiment params, e.g.
  `node60_sendrate30_buffer8_duration3600_etx1.23_int_range60_random`.

## Platforms (scripts/platforms.py)

- Supported: `z1`, `cooja`, `sky`, `wismote` via `PlatformSpec` dataclasses
  that map to Cooja mote types, firmware paths, and Makefile build commands.
- New platforms must add an entry to `PLATFORMS`.

## Conventions

- Python 3.10+; prefer `from __future__ import annotations`, `X | None` unions,
  and `pathlib.Path` over `os.path` for new code.
- `scripts/` files are importable as modules AND runnable as CLIs
  (`if __name__ == "__main__":` with `argparse`).
- Aggregations use duckdb SQL against pandas DataFrames (see `plot_metrics.py`).
- Plotting uses Plotly (via `pd.options.plotting.backend = "plotly"`), not matplotlib.
- Use the repo `venv/bin/python` — the system Python may lack pinned deps.
  Install deps from `scripts/requirements.txt` (which supersets root).
- Run results land in `scripts/runs/<run_id>/` (git-ignored); check `.gitignore`
  before committing.
- No tests or linter config exist in this repo; verify changes by running the
  affected script standalone or the pipeline step.
- `rpl/contiki-ng` is a submodule; never commit changes inside it.
- Cooja simulations are slow — prefer `is_simulate=False` when iterating on
  plotting/parsing, or reuse an existing run dir.
