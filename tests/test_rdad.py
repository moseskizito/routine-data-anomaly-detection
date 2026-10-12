from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdad import simulate_series, DETECTORS, evaluate
from rdad.features import add_features


df = add_features(simulate_series(n_facilities=20, months=36, seed=3))


def test_simulation_has_labelled_anomalies():
    assert df["is_anomaly"].sum() > 0
    assert set(df["anomaly"].unique()) - {""} <= {"spike", "extra_digit", "drop", "flatline"}


def test_every_detector_returns_aligned_booleans():
    for name, fn in DETECTORS.items():
        f = fn(df)
        assert f.dtype == bool, name
        assert f.index.equals(df.index), name


def test_metrics_in_range():
    m = evaluate(df, DETECTORS["Modified z (median/MAD)"](df))
    for k in ("precision", "recall", "f1"):
        assert 0 <= m[k] <= 1


def test_extra_digit_errors_are_caught_by_robust_z():
    m = evaluate(df, DETECTORS["Modified z (median/MAD)"](df))
    assert m["recall_extra_digit"] >= 0.8
