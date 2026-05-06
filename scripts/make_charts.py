"""make_charts.py — Generate visual evidence charts for the PDF report.

Outputs three PNGs into evidence/screenshots/:
  - 08_pr_auc_leaderboard.png  : horizontal bar chart of 7 runs by PR-AUC
  - 09_r2_contents_pie.png      : pie chart of R2 bucket bytes by prefix
  - 10_pr_curves.png            : overlay PR curves of all 7 runs

These complement the MLflow UI screenshots and the R2 CLI evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MLOPS = Path(__file__).resolve().parents[1]
SHOTS = MLOPS / "evidence" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

# ── 1. PR-AUC leaderboard bar chart ──────────────────────────────────────────

metrics = json.loads((MLOPS / "metrics" / "metrics.json").read_text())
runs = sorted(metrics.items(), key=lambda kv: kv[1]["pr_auc"] or 0, reverse=True)
names = [r[0] for r in runs]
pr_aucs = [r[1]["pr_auc"] for r in runs]
roc_aucs = [r[1]["roc_auc"] for r in runs]

fig, ax = plt.subplots(figsize=(9, 4.2))
y = list(range(len(names)))
colors = ["#1f6feb" if i < 4 else "#cb2431" for i in range(len(names))]
ax.barh(y, pr_aucs, color=colors, alpha=0.85)
for i, (v, roc) in enumerate(zip(pr_aucs, roc_aucs)):
    ax.text(v + 0.012, i, f"{v:.4f}  (ROC {roc:.3f})", va="center", fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.02)
ax.set_xlabel("PR-AUC (Average Precision)")
ax.set_title("Leaderboard de los 7 modelos — ordenados por PR-AUC")
ax.axvline(0.0017, color="gray", ls=":", lw=0.8, label="random (0.17 % positivos)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=0.3)
fig.savefig(SHOTS / "08_pr_auc_leaderboard.png")
plt.close(fig)
print(f"wrote {SHOTS / '08_pr_auc_leaderboard.png'}")

# ── 2. R2 contents pie chart ─────────────────────────────────────────────────

r2_log = (MLOPS / "evidence" / "09_r2_list.txt").read_text()
sizes_by_prefix: dict[str, int] = {}
counts_by_prefix: dict[str, int] = {}
for line in r2_log.splitlines():
    m = re.match(r"\s*(\d+)\s+(\S+)", line)
    if not m:
        continue
    size = int(m.group(1))
    key = m.group(2)
    prefix = key.split("/", 1)[0]
    sizes_by_prefix[prefix] = sizes_by_prefix.get(prefix, 0) + size
    counts_by_prefix[prefix] = counts_by_prefix.get(prefix, 0) + 1

fig, ax = plt.subplots(figsize=(7, 5))
labels = []
sizes = []
for p, sz in sorted(sizes_by_prefix.items(), key=lambda kv: -kv[1]):
    mb = sz / 1024 / 1024
    labels.append(f"{p}\n{counts_by_prefix[p]} archivos  ·  {mb:.1f} MB")
    sizes.append(sz)
colors_pie = ["#1f6feb", "#28a745", "#cb2431"][:len(sizes)]
ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
       startangle=90, textprops={"fontsize": 9})
total_mb = sum(sizes) / 1024 / 1024
ax.set_title(
    f"Contenido del bucket R2 «luci-mlops-fraud»\n"
    f"{sum(counts_by_prefix.values())} objetos · {total_mb:.1f} MB total"
)
fig.savefig(SHOTS / "09_r2_contents_pie.png")
plt.close(fig)
print(f"wrote {SHOTS / '09_r2_contents_pie.png'}")

# ── 3. PR curves overlay ─────────────────────────────────────────────────────

pr_df = pd.read_csv(MLOPS / "plots" / "pr_curve.csv")
fig, ax = plt.subplots(figsize=(7, 5))
for run_name in pr_df["run_name"].unique():
    sub = pr_df[pr_df["run_name"] == run_name].sort_values("recall")
    ax.plot(sub["recall"], sub["precision"], label=run_name, lw=1.4, alpha=0.85)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Curvas Precision-Recall de los 7 modelos")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="lower left")
fig.savefig(SHOTS / "10_pr_curves.png")
plt.close(fig)
print(f"wrote {SHOTS / '10_pr_curves.png'}")
