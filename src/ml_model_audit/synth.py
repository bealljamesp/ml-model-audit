"""
Synthetic data generation for ml-model-audit.
Fixed random seed for reproducibility.
Effects will be recorded in data/ground_truth.json
"""

import numpy as np

# fixed random seed
SEED = 42
rng = np.random.default_rng(SEED)

# sample size
N_ROWS = 200

# range
X_LOW = 0
X_HIGH = 10

# planted effect: y = PLANTED_SLOPE * x + PLANTED_INTERCEPT + noise
PLANTED_SLOPE = 2.0
PLANTED_INTERCEPT = 5.0

# noise: random scatter for y. NOISE_STD controls the standard deviation of the noise
NOISE_STD = 1.0


def generate(
    seed=SEED,
    n_rows=N_ROWS,
    x_low=X_LOW,
    x_high=X_HIGH,
    slope=PLANTED_SLOPE,
    intercept=PLANTED_INTERCEPT,
    noise_std=NOISE_STD,
):
    """return x and y arrays with planted linear effect and added noise"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_low, x_high, size=n_rows)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_rows)
    y = slope * x + intercept + noise
    return x, y


import json
from pathlib import Path


def ground_truth(
    seed=SEED,
    n_rows=N_ROWS,
    x_low=X_LOW,
    x_high=X_HIGH,
    planted_slope=PLANTED_SLOPE,
    planted_intercept=PLANTED_INTERCEPT,
    noise_std=NOISE_STD,
):
    """return the ground truth dictionary for the planted effect"""
    return {
        "seed": seed,
        "n_rows": n_rows,
        "x_low": x_low,
        "x_high": x_high,
        "planted_slope": planted_slope,
        "planted_intercept": planted_intercept,
        "noise_std": noise_std,
        "numpy_version": np.__version__,
    }


def write_ground_truth(file_path="data/ground_truth.json"):
    """save the ground truth dictionary to a JSON file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(ground_truth(), indent=4))
    return file_path
