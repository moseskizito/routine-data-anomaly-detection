"""Simulate monthly facility indicator series with labelled anomalies.

Each facility has a level, a linear trend and annual seasonality, with Poisson
noise. Known data-entry errors are injected and labelled, so detectors can be
scored against ground truth. No real data are used.

Anomaly types (common in routine HMIS data):
- spike:        value multiplied 3-6x (double entry, wrong period)
- extra_digit:  value x10 (typing error)
- drop:         value cut to 0-20% (partial reporting)
- flatline:     the same value copied across 3+ months (copy-forward)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANOMALY_TYPES = ["spike", "extra_digit", "drop", "flatline"]


def simulate_series(
    n_facilities: int = 120,
    months: int = 48,
    anomaly_rate: float = 0.04,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t = np.arange(months)
    for f in range(n_facilities):
        level = rng.lognormal(4.0, 0.7)
        trend = rng.normal(0.004, 0.006)
        amp = rng.uniform(0.1, 0.4)
        phase = rng.uniform(0, 2 * np.pi)
        mu = level * (1 + trend * t) * (1 + amp * np.sin(2 * np.pi * t / 12 + phase))
        y = rng.poisson(np.clip(mu, 1, None)).astype(float)
        label = np.array([""] * months, dtype=object)

        n_anom = rng.binomial(months, anomaly_rate)
        for _ in range(n_anom):
            i = int(rng.integers(12, months))  # leave a 12-month history
            if label[i]:
                continue
            kind = rng.choice(ANOMALY_TYPES)
            if kind == "spike":
                y[i] *= rng.uniform(3, 6)
            elif kind == "extra_digit":
                y[i] = y[i] * 10 + rng.integers(0, 10)
            elif kind == "drop":
                y[i] *= rng.uniform(0, 0.2)
            elif kind == "flatline":
                j = min(months, i + int(rng.integers(3, 6)))
                y[i:j] = y[i - 1]
                label[i:j] = "flatline"
                continue
            label[i] = kind
        for k in range(months):
            rows.append(dict(facility_id=f"F{f:04d}", month=k, value=round(y[k]), expected=mu[k], anomaly=label[k]))
    df = pd.DataFrame(rows)
    df["is_anomaly"] = df["anomaly"] != ""
    return df
