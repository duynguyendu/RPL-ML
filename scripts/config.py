import sys
from ast import literal_eval
from pathlib import Path

# topology config
spacing = 40
topo_type = "tree"
num_of_nodes = 30
success_tx = 0.9
success_rx = 0.9
interference_range = 60
tx_range = 50

# csc config
gather_metrics = 1
platform = "z1"
send_rate = 10
buffer_size = 8
ramp_up_duration = 120
duration = 900
with_dao_ack = 0

# output dir
base_output_dir = Path("runs").resolve()
base_mote_dir = "../rpl/motes"
base_cooja = "../rpl/contiki-ng/tools/cooja/"
add_overloading_client = False


for arg in sys.argv[1:]:
    if "=" not in arg:
        # assume it's the name of a config file
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith("--")
        key, val = arg.split("=")
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) is type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
