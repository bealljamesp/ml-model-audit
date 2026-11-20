import numpy as np

from ml_model_audit.synth import PLANTED_INTERCEPT, PLANTED_SLOPE, generate


def test_same_seed_gives_same_data():
    x1, y1 = generate(seed=42)
    x2, y2 = generate(seed=42)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_recovers_planted_effect_within_tolerance():
    x, y = generate(seed=42)
    slope, intercept = np.polyfit(x, y, deg=1)
    assert abs(slope - PLANTED_SLOPE) < 0.1
    assert abs(intercept - PLANTED_INTERCEPT) < 0.5


import json

from ml_model_audit.synth import ground_truth, write_ground_truth


def test_ground_truth_records_planted_values(tmp_path):
    file_path = write_ground_truth(tmp_path / "ground_truth.json")
    record = json.loads(file_path.read_text())
    assert record["planted_slope"] == PLANTED_SLOPE
    assert record["seed"] == 42


def test_ground_truth_matches_non_default_arguments():
    x, _ = generate(n_rows=50)
    record = ground_truth(n_rows=50)
    assert len(x) == record["n_rows"] == 50
