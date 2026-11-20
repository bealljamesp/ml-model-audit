"""Fit one linear regression on one synthetic dataset and report its test score.

Uses the default seed (42) and a 75/25 train/test split. A single split can
land far from the planted values, so this is a first look, not evidence.
See multiseed_linear.py for the spread across seeds.

Run from the repo root:
    PYTHONPATH=src python scripts/linear_baseline.py
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ml_model_audit.synth import generate

x, y = generate()
x_train, x_test, y_train, y_test = train_test_split(
    x.reshape(-1, 1), y, test_size=0.25, random_state=42
)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f"Slope: {model.coef_[0]:.4f}, intercept: {model.intercept_:.4f}")
print(f"test R^2: {model.score(x_test, y_test):.4f}")

# Observed
# Slope: 2.0687, intercept: 4.5807
# test R^2: 0.9690
# Slope is low, more seeds needed to get a reliable estimate -> multiseed_linear.py


# R^2 = var(signal)/(var(signal) + var(noise))
# signal = slope*x
# var(x) = (10^2)/12 = 100/12 ≈ 8.3333
# var(signal) = slope^2 * var(x) = 2^2 * 8.3333 ≈ 33.3333
# var(signal)/(var(signal) + var(noise)) = 33.3/(33.3 + 1) ≈ 0.971

# If R² were much lower, the model would be missing the pattern. If it were much higher, we’d suspect a bug or leakage.
