from topology_generator import generate_topology
from csc_generator import generate_csc
from cooja_simulation import simulate_csc
from parse_cooja_log import parse_log
from plot_metrics import plot_metrics

# from ..models.train import train_model
from pathlib import Path

OUTPUT_DIR = Path("runs").resolve()
TOPOLOGY_JSON_FILE_NAME = OUTPUT_DIR / "topology.json"
CSC_FILE_NAME = OUTPUT_DIR / "topology.csc"


def pipeline(
    is_generate_topology: bool = True,
    is_simulate: bool = True,
    is_plot_metrics: bool = True,
    is_train_model: bool = False,
    add_overloading_client: bool = True,
) -> None:
    # TODO: maybe add run ID

    if is_generate_topology:
        generate_topology(
            topo_type="tree",
            num_of_nodes=10,
            spacing=40,
            out_json=TOPOLOGY_JSON_FILE_NAME,
            platform="sky",
            seed=123456,
            tx_range=50.0,
            interference_range=100.0,
            success_tx=1.0,
            success_rx=1.0,
            duration=300,
            branching=2,
            mesh_jitter=0.2,
            sparse_max_attempts=20,
            add_overloading_client=add_overloading_client,
        )
        generate_csc(
            topology_json=TOPOLOGY_JSON_FILE_NAME,
            out_csc=CSC_FILE_NAME,
            base_dir="../rpl/motes/",
            platform="sky",
        )

    if is_simulate:
        simulate_csc(
            csc_file=CSC_FILE_NAME,
            cooja_base="../rpl/contiki-ng/tools/cooja/",
            output_dir=OUTPUT_DIR,
        )
        parse_log(log_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR)

    if is_plot_metrics:
        plot_metrics(df_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR, dpi=150)

    # if is_train_model:
    #     train_model()

    # convert model to C code
    # Plot and process data
    # Resimulate again
    pass


if __name__ == "__main__":
    pipeline(
        is_generate_topology=True,
        is_simulate=True,
        is_plot_metrics=True,
        is_train_model=False,
        add_overloading_client=False
    )
