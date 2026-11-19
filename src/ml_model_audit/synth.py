"""
Synthetic data generation for ml-model-audit.

Fixed random seed for reproducibility.

Effects are recorded in data/ground_truth.json
"""

# fixed random seed

SEED = 42

import numpy as np

np.random.seed(SEED)
