"""Parity test: deployed Cloudflare Container vs local sklearn XGBClassifier.

Posts 100 test rows to the live Worker, fetches probabilities, compares against
the locally-loaded MLflow-registered model. Writes:
  evidence/m3/inference_parity.json   — full 100-row stats
  evidence/m3/curl_predictions.json   — 5 illustrative rows (probability + body)
  evidence/m3/worker_url.txt          — the live URL
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import requests

MLOPS_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = MLOPS_ROOT / "evidence" / "m3"
WORKER_URL = os.environ.get("WORKER_URL", "https://fraud-detector.eduardo-lalo1999.workers.dev")
TOLERANCE = 1e-4
N_ROWS = 100
N_CURL_EVIDENCE = 5


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLOPS_ROOT / 'mlflow.db'}")
    model = mlflow.xgboost.load_model("models:/fraud-detector@staging")

    X_test = pd.read_parquet(MLOPS_ROOT / "data" / "processed" / "X_test.parquet")
    y_test = pd.read_parquet(MLOPS_ROOT / "data" / "processed" / "y_test.parquet")
    rng = np.random.default_rng(0)
    y = y_test.iloc[:, 0].to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    take_pos = rng.choice(pos_idx, size=min(20, len(pos_idx)), replace=False)
    take_neg = rng.choice(neg_idx, size=N_ROWS - len(take_pos), replace=False)
    idx = np.concatenate([take_pos, take_neg])
    rng.shuffle(idx)

    X = X_test.iloc[idx].reset_index(drop=True)
    y_true = y[idx]
    sk_proba = model.predict_proba(X.to_numpy(dtype=np.float64))[:, 1]

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "worker_url.txt").write_text(WORKER_URL + "\n")

    print(f"Posting {N_ROWS} rows to {WORKER_URL}/predict ...")
    # Warm up — container cold start can take 30–60s
    print("Warming up...")
    for _ in range(3):
        try:
            requests.get(f"{WORKER_URL}/health", timeout=120)
            break
        except requests.exceptions.RequestException:
            time.sleep(2)
    worker_proba = np.empty(N_ROWS, dtype=np.float64)
    worker_pred = np.empty(N_ROWS, dtype=np.int64)
    latencies_ms = []
    curl_evidence = []

    for i in range(N_ROWS):
        feats = X.iloc[i].tolist()
        t0 = time.perf_counter()
        last_err = None
        body = None
        for attempt in range(4):
            try:
                r = requests.post(f"{WORKER_URL}/predict", json={"features": feats}, timeout=30)
                r.raise_for_status()
                body = r.json()
                break
            except requests.exceptions.RequestException as e:
                last_err = e
                time.sleep(1 + attempt)
        dt = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt)
        if body is None:
            raise RuntimeError(f"row {i} failed after retries: {last_err}")
        if i % 10 == 0:
            print(f"  row {i:3d}: {dt:.0f}ms  p={body['probability']:.4g}")
        worker_proba[i] = body["probability"]
        worker_pred[i] = body["prediction"]
        if i < N_CURL_EVIDENCE:
            curl_evidence.append({
                "row_index_in_X_test": int(idx[i]),
                "y_true": int(y_true[i]),
                "sklearn_proba": float(sk_proba[i]),
                "request": {"features": feats},
                "response": body,
                "latency_ms": round(dt, 1),
            })

    diff = np.abs(sk_proba - worker_proba)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    n_within = int((diff <= TOLERANCE).sum())

    threshold = 0.9274014830589294
    sk_pred = (sk_proba >= threshold).astype(int)
    pred_agreement = int((sk_pred == worker_pred).sum())

    print(f"max|Δp|={max_diff:.3e}  mean|Δp|={mean_diff:.3e}")
    print(f"within tol: {n_within}/{N_ROWS}   class agreement: {pred_agreement}/{N_ROWS}")
    print(f"latency p50={np.percentile(latencies_ms, 50):.1f}ms  p95={np.percentile(latencies_ms, 95):.1f}ms")

    parity = {
        "worker_url": WORKER_URL,
        "model_alias": "fraud-detector@staging",
        "n_rows": N_ROWS,
        "n_positives_sampled": int(len(take_pos)),
        "tolerance": TOLERANCE,
        "n_within_tolerance": n_within,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "class_agreement": pred_agreement,
        "threshold": threshold,
        "latency_ms": {
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "mean": float(np.mean(latencies_ms)),
        },
    }
    (EVIDENCE_DIR / "inference_parity.json").write_text(json.dumps(parity, indent=2))
    (EVIDENCE_DIR / "curl_predictions.json").write_text(json.dumps(curl_evidence, indent=2))
    assert n_within == N_ROWS, f"FAIL: only {n_within}/{N_ROWS} within {TOLERANCE}"
    assert pred_agreement == N_ROWS, f"FAIL: class agreement {pred_agreement}/{N_ROWS}"
    print("PARITY OK")


if __name__ == "__main__":
    main()
