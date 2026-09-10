import sys
from ast import literal_eval
from pathlib import Path

# pipeline config
is_processing_data = True
is_training_model = True
is_porting = True

# data config
data_dir = Path("runs").resolve()

# training config
seeds = [0, 1, 2]

# training features (see rpl-mlof.c's "MLOF metrics" log line)
FEATURE_COLUMNS = [
    "is_new",
    "cpu",
    "p_cpu",
    "etx",
    "rssi",
    "ppm",
    "drop_rate",
    "hop_count",
    "nbr_count",
]

# training label
LABEL_COLUMN = "pdr"

features_to_test = list(FEATURE_COLUMNS)

n_estimators = 200
num_leaves = 31
min_child_samples = 20
min_split_gain = 0.0

param_grid = {
    "n_estimators": [50, 100, 200],
    "num_leaves": [7, 15, 31],
}


# capture the config variable names before applying any overrides
config_keys: list[str] = sorted(
    k
    for k in list(globals())
    if not k.startswith("_") and k not in ("sys", "literal_eval", "Path")
)


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
            current = globals()[key]
            if isinstance(current, Path):
                attempt = Path(val).resolve()
            else:
                try:
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    attempt = val
                assert type(attempt) is type(current)
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
