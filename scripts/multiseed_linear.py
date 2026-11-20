"""Repeat the linear regression pipeline over 1,000 synthetic datasets.

Each seed produces a fresh dataset, which is split 75/25 with a fixed split
seed, then fit with LinearRegression. Prints the mean and spread of the slope,
intercept, and test R^2 across seeds.

Run from the repo root:
    PYTHONPATH=src python scripts/multiseed_linear.py

Exploratory. The results are recorded in the README, not used by the package.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ml_model_audit.synth import generate

slopes, intercepts, r2s = [], [], []

for seed in range(1000):
    x, y = generate(seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x.reshape(-1, 1), y, test_size=0.25, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    slopes.append(model.coef_[0])
    intercepts.append(model.intercept_)
    r2s.append(model.score(x_test, y_test))

print(f"Mean slope: {np.mean(slopes):.4f}, sd: {np.std(slopes):.4f}")
print(f"Mean intercept: {np.mean(intercepts):.4f}, sd: {np.std(intercepts):.4f}")
print(f"Mean R^2: {np.mean(r2s):.4f}, sd: {np.std(r2s):.4f}")

# expected
# slope spread ≈ 1/(sqrt(150)*2.89) ≈ 0.028
# intercept spread ≈ 1/(sqrt(150)+25/1250) ≈ 0.163
# R2 mean near 0.97

# observed
# Mean slope: 2.0007, sd: 0.0279
# Mean intercept: 4.9945, sd: 0.1610
# Mean R^2: 0.9694, sd: 0.0075
