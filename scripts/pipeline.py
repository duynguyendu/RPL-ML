import config
from cooja_simulation import simulate_csc
from csc_generator import generate_csc
from parse_cooja_log import parse_log
from plot_metrics import plot_metrics
from topology_generator import generate_topology

# from ..models.train import train_model


def pipeline(
    is_generate_topology: bool = True,
    is_simulate: bool = True,
    is_plot_metrics: bool = True,
    is_train_model: bool = False,
) -> None:
    etx = round(1 / (config.success_tx * config.success_rx), 2)
    run_id = f"node{config.num_of_nodes}_sendrate{config.send_rate}_buffer{config.buffer_size}_duration{config.duration}_etx{etx}_int_range{config.interference_range}_{config.topo_type}"
    output_dir = config.base_output_dir / run_id
    topology_json_file_name = output_dir / "topology.json"
    csc_file_name = output_dir / "topology.csc"

    if is_generate_topology:
        generate_topology(
            topo_type=config.topo_type,
            num_of_nodes=config.num_of_nodes,
            spacing=40,
            out_json=topology_json_file_name,
            platform="sky",
            seed=123456,
            tx_range=config.tx_range,
            interference_range=config.interference_range,
            success_tx=config.success_tx,
            success_rx=config.success_rx,
            duration=config.duration + config.ramp_up_duration,
            branching=2,
            mesh_jitter=0.2,
            sparse_max_attempts=20,
            add_overloading_client=config.add_overloading_client,
        )
        generate_csc(
            topology_json=topology_json_file_name,
            out_csc=csc_file_name,
            base_dir=config.base_mote_dir,
        )

    if is_simulate:
        simulate_csc(
            csc_file=csc_file_name,
            cooja_base=config.base_cooja,
            output_dir=output_dir,
        )

    if is_plot_metrics:
        parse_log(log_dir=output_dir, output_dir=output_dir)
        plot_metrics(df_dir=output_dir, output_dir=output_dir, dpi=150)

    print(f"The run results are saved in {run_id}")
    # if is_train_model:
    #     train_model()

    # convert model to C code
    # Plot and process data
    # Resimulate again
    pass


if __name__ == "__main__":
    pipeline(
        is_generate_topology=True,
        is_simulate=False,
        is_plot_metrics=True,
        is_train_model=False,
    )
