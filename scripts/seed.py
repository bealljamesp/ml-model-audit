"""Measure how well np.polyfit recovers the planted line across 1,000 seeds.

Draws x uniformly from 0 to 100, which is wider than the package default of
0 to 10. Fits a line with np.polyfit and prints the mean and spread of the
slope and intercept. Does not call generate(), so its results can differ
from the package defaults.

Run from the repo root (no PYTHONPATH needed, since it imports only numpy):
    python scripts/seed.py

Exploratory. Kept as the record of the early seed experiment.
"""

import numpy as np

slopes, intercepts = [], []
for seed in range(1000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, size=200)
    noise = rng.normal(0, 1, size=200)
    y = 2.0 * x + 5.0 + noise
    s, i = np.polyfit(x, y, 1)
    slopes.append(s)
    intercepts.append(i)

print(f"slope mean {np.mean(slopes):.4f} sd {np.std(slopes):.4f}")
print(f"intercept mean {np.mean(intercepts):.4f} sd {np.std(intercepts):.4f}")

# Expected
# slope mean 2.0 sd ≈ 0.0024
# intercept mean 5.0 sd ≈ 0.141
#   √(1/200 + x̄²/Sxx) with x̄ = 50 and Sxx = 200 × (100/√12)² ≈ 166,667, giving √(0.005 + 0.015) ≈ 0.141

# Observed
# slope mean 2.0001 sd 0.0024
# intercept mean 4.9959 sd 0.1375
