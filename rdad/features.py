"""Interpretable features computed per facility from its own history."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["robust_z", "seasonal_ratio", "log_change", "repeat_run"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features using only each facility's own series.

    robust_z       deviation from the facility median in MAD units
    seasonal_ratio log ratio to the same month last year
    log_change     log ratio to the previous month
    repeat_run     length of the current run of identical values
    """
    out = []
    for _, g in df.sort_values(["facility_id", "month"]).groupby("facility_id", sort=False):
        g = g.copy()
        v = g["value"].astype(float)
        med = v.median()
        mad = np.median(np.abs(v - med)) or 1.0
        g["robust_z"] = 0.6745 * (v - med) / mad
        g["seasonal_ratio"] = np.log1p(v) - np.log1p(v.shift(12))
        g["log_change"] = np.log1p(v) - np.log1p(v.shift(1))
        same = v.eq(v.shift(1))
        g["repeat_run"] = same.groupby((~same).cumsum()).cumsum()
        out.append(g)
    res = pd.concat(out)
    res[FEATURES] = res[FEATURES].fillna(0)
    return res.sort_index()
