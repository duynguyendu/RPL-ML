from topology_generator import generate_topology
from csc_generator import generate_csc
from cooja_simulation import simulate_csc
from parse_cooja_log import parse_log
from plot_metrics import plot_metrics
import argparse

# from ..models.train import train_model
from pathlib import Path

BASE_OUTPUT_DIR = Path("runs").resolve()


def pipeline(
    is_generate_topology: bool = True,
    is_simulate: bool = True,
    is_plot_metrics: bool = True,
    is_train_model: bool = False,
    add_overloading_client: bool = True,
    topo_type: str = "tree",
    send_rate: int = 30,
    buffer_size: int = 10,
) -> None:
    num_of_nodes = 30
    duration = 900
    run_id = f"node{num_of_nodes}_sendrate{send_rate}_buffer{buffer_size}_duration{duration}_{topo_type}"
    output_dir = BASE_OUTPUT_DIR / run_id
    topology_json_file_name = output_dir / "topology.json"
    csc_file_name = output_dir / "topology.csc"

    if is_generate_topology:
        generate_topology(
            topo_type=topo_type,
            num_of_nodes=num_of_nodes,
            spacing=40,
            out_json=topology_json_file_name,
            platform="sky",
            seed=123456,
            tx_range=50.0,
            interference_range=100.0,
            success_tx=1.0,
            success_rx=1.0,
            duration=duration,
            branching=2,
            mesh_jitter=0.2,
            sparse_max_attempts=20,
            add_overloading_client=add_overloading_client,
        )
        generate_csc(
            topology_json=topology_json_file_name,
            out_csc=csc_file_name,
            base_dir="../rpl/motes/",
            platform="sky",
            buffer_size=buffer_size,
        )

    if is_simulate:
        simulate_csc(
            csc_file=csc_file_name,
            cooja_base="../rpl/contiki-ng/tools/cooja/",
            output_dir=output_dir,
        )

    parse_log(log_dir=output_dir, output_dir=output_dir)
    if is_plot_metrics:
        plot_metrics(df_dir=output_dir, output_dir=output_dir, dpi=150)

    # if is_train_model:
    #     train_model()

    # convert model to C code
    # Plot and process data
    # Resimulate again
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run the COOJA simulation pipeline: generate topology, simulate, parse data, plot metrics"
    )
    ap.add_argument(
        "--topo-type", type=str, default="tree", help="The type of topology"
    )
    args = ap.parse_args()
    # pipeline(
    #     is_generate_topology=False,
    #     is_simulate=False,
    #     is_plot_metrics=True,
    #     is_train_model=False,
    #     add_overloading_client=False,
    #     run_id=args.run_id,
    # )
    pipeline(
        is_generate_topology=True,
        is_simulate=True,
        is_plot_metrics=True,
        is_train_model=False,
        add_overloading_client=False,
        topo_type=args.topo_type,
    )
