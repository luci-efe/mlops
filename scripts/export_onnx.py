"""Export the registered fraud-detector@staging model to ONNX and verify parity.

Uses skl2onnx's `convert_sklearn` with a registered XGBoost converter, so we can
pass `zipmap=False` and produce a flat (N, 2) probability tensor — what JS/WASM
inference runtimes can consume directly.

Writes:
  models/fraud-detector-v1.onnx
  evidence/m3/onnx_export_parity.json
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost as xgb_convert_op,
)
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)
from xgboost import XGBClassifier

MLOPS_ROOT = Path(__file__).resolve().parents[1]
MODEL_OUT = MLOPS_ROOT / "models" / "fraud-detector-v1.onnx"
EVIDENCE_DIR = MLOPS_ROOT / "evidence" / "m3"
TOLERANCE = 1e-4
N_FEATURES = 29


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLOPS_ROOT / 'mlflow.db'}")
    model: XGBClassifier = mlflow.xgboost.load_model("models:/fraud-detector@staging")

    booster = model.get_booster()
    booster.feature_names = [f"f{i}" for i in range(N_FEATURES)]
    booster.feature_types = None

    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        xgb_convert_op,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    onnx_model = convert_sklearn(
        model,
        "fraud_detector",
        initial_types=[("input", FloatTensorType([None, N_FEATURES]))],
        target_opset={"": 15, "ai.onnx.ml": 3},
        options={id(model): {"zipmap": False}},
    )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, MODEL_OUT)
    size_kb = MODEL_OUT.stat().st_size / 1024
    print(f"Wrote {MODEL_OUT} ({size_kb:.1f} KB)")

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X_test = pd.read_parquet(MLOPS_ROOT / "data" / "processed" / "X_test.parquet")
    assert list(X_test.columns) == feature_cols
    sample = X_test.head(200).to_numpy(dtype=np.float32)

    sk_proba = model.predict_proba(X_test.head(200).to_numpy(dtype=np.float64))[:, 1]

    session = ort.InferenceSession(str(MODEL_OUT), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print("ONNX inputs:", input_name, "outputs:", output_names)

    raw_outputs = session.run(None, {input_name: sample})
    for name, arr in zip(output_names, raw_outputs):
        print(f"  out {name!r}: shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")

    proba_arr = next(
        (a for a in raw_outputs if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 2),
        None,
    )
    if proba_arr is None:
        raise RuntimeError(f"Could not find (N,2) probability tensor in outputs: {output_names}")
    onnx_proba = proba_arr[:, 1]

    diff = np.abs(sk_proba - onnx_proba)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    n_within = int((diff <= TOLERANCE).sum())
    n_total = int(diff.shape[0])
    print(f"max|Δ|={max_diff:.3e}  mean|Δ|={mean_diff:.3e}  within {TOLERANCE}: {n_within}/{n_total}")

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "onnx_export_parity.json").write_text(json.dumps({
        "model_alias": "fraud-detector@staging",
        "model_version": 1,
        "onnx_path": str(MODEL_OUT.relative_to(MLOPS_ROOT)),
        "onnx_size_kb": round(size_kb, 1),
        "onnx_input_name": input_name,
        "onnx_output_names": output_names,
        "n_test_rows": n_total,
        "tolerance": TOLERANCE,
        "n_within_tolerance": n_within,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "best_threshold": 0.9274014830589294,
    }, indent=2))
    assert n_within == n_total, f"Parity FAILED: only {n_within}/{n_total} within {TOLERANCE}"
    print("PARITY OK")


if __name__ == "__main__":
    main()
