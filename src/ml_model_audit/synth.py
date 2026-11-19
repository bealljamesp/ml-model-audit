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
