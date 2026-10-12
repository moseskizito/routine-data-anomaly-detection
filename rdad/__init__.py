"""rdad: benchmark of statistical vs machine-learning anomaly detection on routine health data."""
from .simulate import simulate_series
from .detectors import DETECTORS
from .evaluate import evaluate, explain_flag

__all__ = ["simulate_series", "DETECTORS", "evaluate", "explain_flag"]
__version__ = "0.1.0"
