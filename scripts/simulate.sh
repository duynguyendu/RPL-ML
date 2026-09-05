#!/usr/bin/env bash
# Run pipeline.py for a range of node counts / send rates / objective functions.
# Up to MAX_PARALLEL runs execute concurrently. Every concurrent run is given its
# own --build_dir_name (a free "slot") so their firmware build trees never collide.
#
# Usage: ./simulate.sh [label]
#   An optional label is folded into the run directory name alongside the
#   config, e.g. `./simulate.sh run1` -> runs/sim_..._run1_<timestamp>.
set -u

LABEL="${1:-}"
MAX_PARALLEL=1
PLATFORM=cooja

NODE_LIST=(40 60 80)
PPM_LIST=(60 40 30 20 15)
OF_LIST=(of0 mhrof mlof)
SEED_LIST=(12756 826352 927106)
BUFFER_SIZE=32
TOTAL=$(( ${#NODE_LIST[@]} * ${#PPM_LIST[@]} * ${#OF_LIST[@]} * ${#SEED_LIST[@]} ))
RUN=0

MAX_NODES=$(printf '%s\n' "${NODE_LIST[@]}" | sort -n | tail -1)
MIN_RATE=$(printf '%s\n' "${PPM_LIST[@]}" | sort -n | head -1)

# Every run from this invocation is written under its own timestamped directory
# instead of the default runs/ so different simulate.sh sweeps never mix.
RUN_DIR="runs/sim_${PLATFORM}_n${MAX_NODES}_ppm${MIN_RATE}_buffer${BUFFER_SIZE}${LABEL:+_$LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "=== Writing runs to $RUN_DIR ==="

SECONDS=0
fmt_dur() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

# Pool of build-dir slots and the pid -> slot bookkeeping for in-flight runs.
FREE_SLOTS=()
for ((s = 1; s <= MAX_PARALLEL; s++)); do FREE_SLOTS+=("build_${PLATFORM}${LABEL:+_$LABEL}_w${s}"); done
declare -A SLOT_OF_PID
INFLIGHT=0

reclaim() {                                    # return a finished pid's slot to the pool
    local pid=$1 slot=${SLOT_OF_PID[$1]:-}
    [[ -n $slot ]] || return 0
    FREE_SLOTS+=("$slot")
    unset 'SLOT_OF_PID[$pid]'
    (( INFLIGHT-- ))
}

# Ctrl+C (or a kill/TERM of this script) only signals simulate.sh itself -
# background jobs aren't in its foreground process group, so they'd otherwise
# keep running as orphans. Forward the signal to each job's whole process
# group (negative pid) instead, so pipeline.py/gradlew/java all die with us.
cleanup() {
    trap - INT TERM                            # don't re-enter on a second signal
    echo
    echo "=== Interrupted: terminating ${#SLOT_OF_PID[@]} in-flight run(s) ==="
    for pid in "${!SLOT_OF_PID[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null
    done
    sleep 2
    for pid in "${!SLOT_OF_PID[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null
    done
    exit 130
}
trap cleanup INT TERM

run_job() {
    local RUN=$1 TOTAL=$2 slot=$3 RPL_OF=$4 NUM_NODES=$5 PPM=$6 SEED=$7 RUN_DIR=$8 PLATFORM=$9 BUFFER_SIZE=${10}
    SECONDS=0
    echo "=== [$(date +%T)] START ($RUN/$TOTAL) $slot of=$RPL_OF nodes=$NUM_NODES ppm=$PPM seed=$SEED ==="
    python3 pipeline.py \
        --duration=1800 \
        --is_simulate=True \
        --packet_size=64 \
        --buffer_size="$BUFFER_SIZE" \
        --rpl_of="$RPL_OF" \
        --num_of_nodes="$NUM_NODES" \
        --ppm="$PPM" \
        --platform="$PLATFORM" \
        --seed="$SEED" \
        --base_output_dir="$RUN_DIR" \
        --build_dir_name="$slot" >/dev/null 2>&1
    echo "=== [$(date +%T)] DONE  ($RUN/$TOTAL) $slot of=$RPL_OF nodes=$NUM_NODES ppm=$PPM seed=$SEED (took $(fmt_dur $SECONDS)) ==="
}
export -f run_job fmt_dur

for NUM_NODES in "${NODE_LIST[@]}"; do
    for PPM in "${PPM_LIST[@]}"; do
        for RPL_OF in "${OF_LIST[@]}"; do
            for SEED in "${SEED_LIST[@]}"; do
                (( RUN++ ))
                # Block until a slot frees up (fewer than MAX_PARALLEL runs alive).
                while (( INFLIGHT >= MAX_PARALLEL )); do
                    wait -n -p done_pid
                    reclaim "$done_pid"
                done

                slot="${FREE_SLOTS[-1]}"
                FREE_SLOTS=("${FREE_SLOTS[@]:0:${#FREE_SLOTS[@]} - 1}")

                setsid bash -c 'run_job "$@"' _ \
                    "$RUN" "$TOTAL" "$slot" "$RPL_OF" "$NUM_NODES" "$PPM" "$SEED" "$RUN_DIR" "$PLATFORM" "$BUFFER_SIZE" &

                SLOT_OF_PID[$!]=$slot
                (( INFLIGHT++ ))
            done
        done
    done
done

while (( INFLIGHT > 0 )); do                    # let every run finish
    wait -n -p done_pid
    reclaim "$done_pid"
done

echo "=== All runs finished in $(fmt_dur $SECONDS) ==="

echo "=== Building cross-run comparison dashboard ==="
python3 plot_comparison.py --runs-dir "$RUN_DIR"

echo "=== Total elapsed (runs + dashboard): $(fmt_dur $SECONDS) ==="
echo "=== Results in $RUN_DIR ==="
