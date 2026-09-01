# Run pipeline.py for a range of node counts

for NUM_NODES in 40 60 80; do
    for SEND_RATE in 5 10 15 20; do
        for RPL_OF in of0 mhrof; do
            echo "=== Running pipeline.py with num_of_nodes=$NUM_NODES, send_rate=$SEND_RATE ==="
            python3 pipeline.py \
                --duration=1800 \
                --is_simulate=True \
                --packet_size=64 \
                --buffer_size=8 \
                --rpl_of=$RPL_OF \
                --num_of_nodes=$NUM_NODES \
                --send_rate=$SEND_RATE
        done
    done
done
