import sys
from ast import literal_eval
from pathlib import Path

# pipeline config
is_processing_data = True
is_analyse_data = True
is_training_model = True
is_porting = True

# data config
data_dir = Path("runs").resolve()
rpl_lite_dir = Path("../rpl/contiki-ng/os/net/routing/rpl-lite").resolve()

# training config
seeds = [0, 1, 2]
filter_unknown = True
exclude_features: list[str] = []

# training features (see rpl-mlof.c's "MLOF metrics" log line)
FEATURE_COLUMNS = [
    "is_new",
    "cpu",
    "p_cpu",
    "etx",
    "rssi",
    "ppm",
    "drop_rate",
    "parent_ppm",
    "parent_drop_rate",
    "hop_count",
]

FIXED_FEATURES = []

DYNAMIC_FEATURES = [f for f in FEATURE_COLUMNS if f not in FIXED_FEATURES]

# unknown-value sentinels (see rpl-mlof.c)
UNKNOWN_SENTINELS = {
    "ppm": 32767,
    "parent_ppm": 32767,
    "p_cpu": 255,
    "drop_rate": 255,
    "parent_drop_rate": 255,
    "hop_count": 255,
}

# training label
LABEL_COLUMN = "pdr"

n_jobs = -1
lgbm_n_jobs = 1
min_dynamic_features = 2
top_n_per_model = 5

lgbm_param_grid = {
    "n_estimators": [10, 20, 30, 50],
    "num_leaves": [7, 15, 31],
    # LightGBM's own "no limit" sentinel is <=0 (not None, unlike sklearn).
    "max_depth": [3, 5, 7],
    "min_data_in_leaf": [5, 10, 20],
}

ridge_param_grid = {
    "alpha": [0.1, 1.0, 10.0],
}

dtree_param_grid = {
    "max_depth": [7, 10, 12],
    "min_samples_leaf": [1, 5, 10],
    "splitter": ["best", "random"],
}

svr_param_grid = {
    "C": [0.1, 1.0, 10.0],
    "epsilon": [0.01, 0.1, 1.0],
}


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

if exclude_features:
    FEATURE_COLUMNS = [f for f in FEATURE_COLUMNS if f not in exclude_features]
    FIXED_FEATURES = [f for f in FIXED_FEATURES if f not in exclude_features]
    DYNAMIC_FEATURES = [f for f in FEATURE_COLUMNS if f not in FIXED_FEATURES]

if not FEATURE_COLUMNS:
    raise ValueError(
        "exclude_features excluded every training feature -- nothing left to train on"
    )

min_dynamic_features = min(min_dynamic_features, len(DYNAMIC_FEATURES))

if not FIXED_FEATURES:
    min_dynamic_features = max(min_dynamic_features, 1)
