"""
evaluate.py — Aggregate metrics from completed MLflow runs; no re-scoring.

Reads models/runs.json (manifest written by train_all.py), queries MLflow for
each run's logged metrics, downloads plot_data artifacts, and writes:
  - metrics/metrics.json        — 7-entry dict keyed by run_name
  - plots/roc_curve.csv         — multi-run ROC curve data
  - plots/pr_curve.csv          — multi-run PR curve data

NO model loading or re-scoring is performed here. All curve data comes from
the y_true/y_prob CSVs logged as artifacts during training.

Usage:
  python src/evaluate.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_recall_curve, roc_curve

# ── Constants ────────────────────────────────────────────────────────────────
PARAMS_FILE   = Path("params.yaml")
RUNS_MANIFEST = Path("models/runs.json")
METRICS_DIR   = Path("metrics")
PLOTS_DIR     = Path("plots")
EXPERIMENT_NAME = "credit-fraud"


def main() -> None:
    # Load tracking URI from params
    params = yaml.safe_load(PARAMS_FILE.read_text())
    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "file://./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Load run manifest written by train_all.py
    manifest = json.loads(RUNS_MANIFEST.read_text())
    print(f"Evaluating {len(manifest)} runs from {RUNS_MANIFEST}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_summary: dict = {}
    roc_rows: list[dict] = []
    pr_rows:  list[dict] = []

    for entry in manifest:
        run_name = entry["run_name"]
        run_id   = entry["run_id"]

        # Fetch metrics from MLflow (no re-scoring)
        run = client.get_run(run_id)
        m   = run.data.metrics

        metrics_summary[run_name] = {
            "run_id":                   run_id,
            "model":                    entry["model"],
            "pr_auc":                   m.get("pr_auc"),
            "roc_auc":                  m.get("roc_auc"),
            "best_threshold":           m.get("best_threshold"),
            "f1_at_threshold":          m.get("f1_at_threshold"),
            "precision_at_threshold":   m.get("precision_at_threshold"),
            "recall_at_threshold":      m.get("recall_at_threshold"),
        }

        # Download plot_data artifact and compute curve points
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_files = client.list_artifacts(run_id, path="plot_data")
            if not artifact_files:
                print(f"  [WARN] No plot_data artifact for {run_name}, skipping curves.")
                continue

            # There is exactly one CSV per run under plot_data/
            artifact_path = artifact_files[0].path
            local_dir = client.download_artifacts(run_id, artifact_path, tmpdir)
            plot_df = pd.read_csv(local_dir)

        y_true = plot_df["y_true"].values
        y_prob = plot_df["y_prob"].values

        # ROC curve: fpr, tpr at ~200 evenly-spaced thresholds
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr)), dtype=int)
        for i in idx:
            roc_rows.append({"run_name": run_name, "fpr": fpr[i], "tpr": tpr[i]})

        # PR curve: precision, recall
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        idx = np.linspace(0, len(prec) - 1, min(200, len(prec)), dtype=int)
        for i in idx:
            pr_rows.append({"run_name": run_name, "precision": prec[i], "recall": rec[i]})

        print(f"  {run_name:35s}  PR-AUC={m.get('pr_auc', 0):.4f}  ROC-AUC={m.get('roc_auc', 0):.4f}")

    # Write outputs
    metrics_path = METRICS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))
    print(f"\nMetrics written → {metrics_path}")

    roc_path = PLOTS_DIR / "roc_curve.csv"
    pd.DataFrame(roc_rows).to_csv(roc_path, index=False)
    print(f"ROC curve data  → {roc_path}")

    pr_path = PLOTS_DIR / "pr_curve.csv"
    pd.DataFrame(pr_rows).to_csv(pr_path, index=False)
    print(f"PR curve data   → {pr_path}")

    # Print ranking by PR-AUC
    ranked = sorted(metrics_summary.items(), key=lambda kv: kv[1]["pr_auc"] or 0, reverse=True)
    print("\n=== Runs ranked by PR-AUC ===")
    for rank, (name, vals) in enumerate(ranked, 1):
        print(f"  {rank}. {name:35s}  PR-AUC={vals['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
