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

print(f"slope mean {np.mean(slopes)} sd {np.std(slopes)}")
print(f"intercept mean {np.mean(intercepts)} sd {np.std(intercepts)}")
