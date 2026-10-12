"""Anomaly detectors. Each takes the featured DataFrame and returns a boolean Series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

from .features import FEATURES


def zscore(df: pd.DataFrame, k: float = 3.0) -> pd.Series:
    """Classical z-score against the facility mean and SD (sensitive to the outliers it seeks)."""
    g = df.groupby("facility_id")["value"]
    z = (df["value"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z.abs() > k


def modified_z(df: pd.DataFrame, k: float = 3.5) -> pd.Series:
    """Median/MAD-based z-score (Iglewicz & Hoaglin)."""
    return df["robust_z"].abs() > k


def iqr(df: pd.DataFrame, k: float = 1.5) -> pd.Series:
    """Tukey fences per facility."""
    g = df.groupby("facility_id")["value"]
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    w = q3 - q1
    return (df["value"] < q1 - k * w) | (df["value"] > q3 + k * w)


def stl_residual(df: pd.DataFrame, k: float = 8.0) -> pd.Series:
    """Robust STL decomposition (log scale) per facility; flag residuals > k robust SDs.

    Smoothing windows and k were tuned on the simulation (seasonal=25, trend=37, k=8);
    see README for why STL still struggles on short (48-month) series.
    """
    flags = pd.Series(False, index=df.index)
    for _, g in df.groupby("facility_id"):
        g = g.sort_values("month")
        y = np.log1p(g["value"].to_numpy(dtype=float))
        res = STL(y, period=12, seasonal=25, trend=37, robust=True).fit().resid
        s = 1.4826 * np.median(np.abs(res - np.median(res))) or 1e-9
        flags.loc[g.index] = np.abs(res - np.median(res)) > k * s
    return flags


def isolation_forest(df: pd.DataFrame, contamination: float = 0.05, seed: int = 0) -> pd.Series:
    """Isolation Forest on the interpretable features (unsupervised)."""
    X = df[FEATURES].to_numpy()
    X = np.c_[np.abs(X[:, :3]), np.log1p(X[:, 3])]
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    return pd.Series(model.fit_predict(X) == -1, index=df.index)


def rule_ensemble(df: pd.DataFrame) -> pd.Series:
    """Transparent rules that combine the features: each rule is readable on its own."""
    return (
        (df["robust_z"].abs() > 3.5)
        | (df["seasonal_ratio"].abs() > np.log(3))
        | (df["repeat_run"] >= 2)
    )


DETECTORS = {
    "Z-score (mean/SD)": zscore,
    "Modified z (median/MAD)": modified_z,
    "IQR fences": iqr,
    "STL residual": stl_residual,
    "Isolation Forest": isolation_forest,
    "Rule ensemble": rule_ensemble,
}
