"""
train_all.py — Run all 7 model configurations in-process, log to MLflow.

Reads configs from params.yaml and iterates over them, calling train_one()
from train.py for each. Writes models/runs.json as a manifest for evaluate.py.

Usage:
  python src/train_all.py

MLflow tracking URI preflight:
  - If URI starts with "http", pings with a 2s timeout; fails fast on error.
  - If URI starts with "file://", proceeds immediately.

DVC stage cmd: python src/train_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import requests
import yaml

# Import train_one from the same package
from train import train_one  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
PARAMS_FILE   = Path("params.yaml")
RUNS_MANIFEST = Path("models/runs.json")
EXPERIMENT_NAME = "credit-fraud"


# ── MLflow URI preflight ──────────────────────────────────────────────────────

def preflight_mlflow_uri(tracking_uri: str) -> None:
    """
    If tracking_uri starts with 'http', attempt a 2-second GET to verify the
    server is reachable. Fail loudly if not. For 'file://' URIs, proceed without
    any network check.
    """
    if tracking_uri.startswith("http"):
        print(f"[preflight] Checking MLflow server at {tracking_uri} …")
        try:
            resp = requests.get(tracking_uri, timeout=2)
            print(f"[preflight] MLflow server responded with HTTP {resp.status_code}. OK.")
        except requests.exceptions.RequestException as exc:
            print(
                f"\n[FATAL] Cannot reach MLflow tracking server at {tracking_uri}\n"
                f"        Error: {exc}\n"
                f"        Start the server with: mlflow server --host 127.0.0.1 --port 5000\n",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(f"[preflight] Using local file-based MLflow tracking: {tracking_uri}")


# ── Config flattening helpers ─────────────────────────────────────────────────

def build_configs(params: dict) -> list[dict]:
    """
    Parse params.yaml structure and return a flat list of 7 config dicts,
    each ready to pass into train_one().
    """
    configs = []
    models = params["models"]

    # 1. LogReg baseline
    lr_cfg = models["logreg"]
    configs.append({
        "run_name": lr_cfg["run_name"],
        "model": "logreg",
        "C": lr_cfg["C"],
    })

    # 2-4. XGBoost runs
    for xgb_cfg in models["xgboost_runs"]:
        configs.append({
            "run_name": xgb_cfg["run_name"],
            "model": "xgb",
            "max_depth": xgb_cfg["max_depth"],
            "lr": xgb_cfg["learning_rate"],
            "n_estimators": xgb_cfg["n_estimators"],
        })

    # 5-7. LightGBM runs
    for lgbm_cfg in models["lightgbm_runs"]:
        cfg: dict = {
            "run_name": lgbm_cfg["run_name"],
            "model": "lgbm",
            "num_leaves": lgbm_cfg["num_leaves"],
            "lr": lgbm_cfg["learning_rate"],
            "n_estimators": lgbm_cfg["n_estimators"],
        }
        # is_unbalance and scale_pos_weight are mutually exclusive
        if lgbm_cfg.get("is_unbalance", False):
            cfg["is_unbalance"] = True
        else:
            cfg["is_unbalance"] = False  # train_one will use scale_pos_weight
        configs.append(cfg)

    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load params
    params = yaml.safe_load(PARAMS_FILE.read_text())

    # Set tracking URI
    tracking_uri = params.get("mlflow", {}).get(
        "tracking_uri", "file://./mlruns"
    )
    mlflow.set_tracking_uri(tracking_uri)
    preflight_mlflow_uri(tracking_uri)

    # Build 7 configs
    configs = build_configs(params)
    assert len(configs) == 7, f"Expected 7 configs, got {len(configs)}"

    print(f"\nRunning {len(configs)} configurations under experiment '{EXPERIMENT_NAME}'\n")

    # Run all configs in-process
    run_manifest = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Starting: {cfg['run_name']}")
        run_id = train_one(cfg)
        run_manifest.append({
            "run_name": cfg["run_name"],
            "model": cfg["model"],
            "run_id": run_id,
        })

    # Write manifest for evaluate.py to consume
    RUNS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RUNS_MANIFEST.write_text(json.dumps(run_manifest, indent=2))
    print(f"\nManifest written → {RUNS_MANIFEST}")

    # Print summary
    print("\n=== All 7 runs complete ===")
    for entry in run_manifest:
        print(f"  {entry['run_name']:35s}  run_id={entry['run_id'][:8]}")


if __name__ == "__main__":
    main()
