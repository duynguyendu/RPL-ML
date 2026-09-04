#!/usr/bin/env bash
# Run pipeline.py for a range of node counts / send rates / objective functions.
# Up to MAX_PARALLEL runs execute concurrently. Every concurrent run is given its
# own --build_dir_name (a free "slot") so their firmware build trees never collide.
set -u

MAX_PARALLEL=8

SECONDS=0
fmt_dur() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

# Pool of build-dir slots and the pid -> slot bookkeeping for in-flight runs.
FREE_SLOTS=()
for ((s = 1; s <= MAX_PARALLEL; s++)); do FREE_SLOTS+=("build_w$s"); done
declare -A SLOT_OF_PID
INFLIGHT=0

reclaim() {                                    # return a finished pid's slot to the pool
    local pid=$1 slot=${SLOT_OF_PID[$1]:-}
    [[ -n $slot ]] || return 0
    FREE_SLOTS+=("$slot")
    unset 'SLOT_OF_PID[$pid]'
    (( INFLIGHT-- ))
}

NODE_LIST=(20 40 60 80)
RATE_LIST=(1 2 3 5 7 10 15 20)
OF_LIST=(of0 mhrof)
TOTAL=$(( ${#NODE_LIST[@]} * ${#RATE_LIST[@]} * ${#OF_LIST[@]} ))
RUN=0

for NUM_NODES in "${NODE_LIST[@]}"; do
    for SEND_RATE in "${RATE_LIST[@]}"; do
        for RPL_OF in "${OF_LIST[@]}"; do
            (( RUN++ ))
            # Block until a slot frees up (fewer than MAX_PARALLEL runs alive).
            while (( INFLIGHT >= MAX_PARALLEL )); do
                wait -n -p done_pid
                reclaim "$done_pid"
            done

            slot="${FREE_SLOTS[-1]}"
            FREE_SLOTS=("${FREE_SLOTS[@]:0:${#FREE_SLOTS[@]} - 1}")

            (
                SECONDS=0
                echo "=== [$(date +%T)] START ($RUN/$TOTAL) $slot of=$RPL_OF nodes=$NUM_NODES rate=$SEND_RATE ==="
                python3 pipeline.py \
                    --duration=1800 \
                    --is_simulate=True \
                    --packet_size=64 \
                    --buffer_size=8 \
                    --rpl_of="$RPL_OF" \
                    --num_of_nodes="$NUM_NODES" \
                    --send_rate="$SEND_RATE" \
                    --platform=cooja \
                    --build_dir_name="$slot" >/dev/null 2>&1
                echo "=== [$(date +%T)] DONE  ($RUN/$TOTAL) $slot of=$RPL_OF nodes=$NUM_NODES rate=$SEND_RATE (took $(fmt_dur $SECONDS)) ==="
            ) &

            SLOT_OF_PID[$!]=$slot
            (( INFLIGHT++ ))
        done
    done
done

while (( INFLIGHT > 0 )); do                    # let every run finish
    wait -n -p done_pid
    reclaim "$done_pid"
done

echo "=== All runs finished in $(fmt_dur $SECONDS) ==="

echo "=== Building cross-run comparison dashboard ==="
python3 plot_comparison.py

echo "=== Total elapsed (runs + dashboard): $(fmt_dur $SECONDS) ==="
