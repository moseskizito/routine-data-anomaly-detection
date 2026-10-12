"""Scoring against injected ground truth, and per-flag explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import ANOMALY_TYPES


def evaluate(df: pd.DataFrame, flags: pd.Series) -> dict:
    y, p = df["is_anomaly"], flags
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(precision=prec, recall=rec, f1=f1, flagged=int(p.sum()))
    for t in ANOMALY_TYPES:
        m = df["anomaly"] == t
        out[f"recall_{t}"] = float(p[m].mean()) if m.any() else np.nan
    return out


def explain_flag(row: pd.Series) -> str:
    """Plain-language reason for a flag, from the interpretable features."""
    reasons = []
    if abs(row["robust_z"]) > 3.5:
        reasons.append(f"{row['robust_z']:+.1f} MADs from this facility's usual level")
    if abs(row["seasonal_ratio"]) > np.log(3):
        reasons.append(f"{np.expm1(abs(row['seasonal_ratio'])) + 1:.1f}x {'above' if row['seasonal_ratio'] > 0 else 'below'} the same month last year")
    if row["repeat_run"] >= 2:
        reasons.append(f"same value repeated {int(row['repeat_run']) + 1} months in a row")
    return "; ".join(reasons) or "unusual combination of features"
