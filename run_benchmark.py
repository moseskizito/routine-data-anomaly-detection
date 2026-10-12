"""Run all detectors on simulated data and write results/.

Usage:  python run_benchmark.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rdad import simulate_series, DETECTORS, evaluate, explain_flag
from rdad.features import add_features
from rdad.simulate import ANOMALY_TYPES

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)

df = add_features(simulate_series())
print(f"{df['facility_id'].nunique()} facilities x {df['month'].nunique()} months = {len(df)} values; "
      f"{int(df['is_anomaly'].sum())} labelled anomalies ({df['is_anomaly'].mean():.1%})")

rows, flags_by = [], {}
for name, fn in DETECTORS.items():
    f = fn(df)
    flags_by[name] = f
    rows.append(dict(method=name, **evaluate(df, f)))
res = pd.DataFrame(rows).set_index("method")
res.round(3).to_csv(out / "benchmark.csv")

show = res[["precision", "recall", "f1"] + [f"recall_{t}" for t in ANOMALY_TYPES]].round(2)
print("\n" + show.to_string())
(out / "benchmark.md").write_text(show.to_markdown())

# Explanations for a sample of rule-ensemble flags
best = flags_by["Rule ensemble"]
sample = df[best & df["is_anomaly"]].groupby("anomaly").head(2)
sample = sample.assign(reason=sample.apply(explain_flag, axis=1))[["facility_id", "month", "value", "anomaly", "reason"]]
sample.to_csv(out / "example_explanations.csv", index=False)
print("\nExample explanations:\n" + sample.to_string(index=False))

# Figure 1: F1 by method
fig, ax = plt.subplots(figsize=(7, 3.4))
order = res.sort_values("f1")
ax.barh(order.index, order["f1"], color="#3B6EA8")
for i, v in enumerate(order["f1"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
ax.set_xlim(0, 1)
ax.set_xlabel("F1 score (higher is better)")
ax.set_title("Detecting injected data-entry errors — simulated facility data")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out / "f1_by_method.png", dpi=150)

# Figure 2: one facility series with flags
fid = df.loc[df["anomaly"].isin(["spike", "drop", "flatline"]), "facility_id"].value_counts().index[0]
g = df[df["facility_id"] == fid]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(g["month"], g["value"], color="#555", lw=1.4, label="reported value")
ax.plot(g["month"], g["expected"], color="#999", lw=1, ls="--", label="true expected level")
fl = g[best.loc[g.index]]
ax.scatter(fl["month"], fl["value"], s=60, facecolors="none", edgecolors="#C0392B", lw=1.6, label="flagged")
an = g[g["is_anomaly"]]
ax.scatter(an["month"], an["value"], s=14, color="#C0392B", label="injected error")
ax.set_xlabel("month")
ax.set_ylabel("monthly count")
ax.set_title(f"Facility {fid}: rule-ensemble flags vs injected errors")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out / "example_facility.png", dpi=150)
print(f"\nSaved results to {out}")
