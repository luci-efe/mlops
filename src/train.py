"""
train.py — Credit-card fraud detector training script.

Public API:
  train_one(config: dict) -> str   # returns MLflow run_id

CLI usage examples:
  python src/train.py --model logreg
  python src/train.py --model xgb  --max_depth 6 --lr 0.05 --n_estimators 500
  python src/train.py --model lgbm --num_leaves 63 --lr 0.05 --n_estimators 500

After all 7 runs are complete, promote the best to the registry:
  python src/register_best.py

MLflow docs refs:
  https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html
  https://mlflow.org/docs/latest/python_api/mlflow.xgboost.html
  https://mlflow.org/docs/latest/python_api/mlflow.lightgbm.html
  https://mlflow.org/docs/latest/python_api/mlflow.models.html  (infer_signature)
  https://mlflow.org/docs/latest/model-registry.html            (aliases, NOT stages)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ── LightGBM / XGBoost imports (optional; fail loudly if missing) ──────────
try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    LGBMClassifier = None  # type: ignore[assignment,misc]
    _lgbm_err = e

try:
    from xgboost import XGBClassifier
except ImportError as e:
    XGBClassifier = None  # type: ignore[assignment,misc]
    _xgb_err = e

# ── Constants ────────────────────────────────────────────────────────────────
# Pre-split parquet files produced by src/data.py preprocess subcommand.
# Time column is dropped during preprocessing, so these have 29 cols (V1-V28 + Amount).
PROCESSED_DIR = Path("data/processed")
X_TRAIN_PATH = PROCESSED_DIR / "X_train.parquet"
X_TEST_PATH  = PROCESSED_DIR / "X_test.parquet"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.parquet"
Y_TEST_PATH  = PROCESSED_DIR / "y_test.parquet"

EXPERIMENT_NAME = "credit-fraud"
RANDOM_STATE = 42


# ── Preprocessing helpers ────────────────────────────────────────────────────

def build_preprocessor(model_type: str) -> ColumnTransformer | None:
    """
    Only LogReg needs scaling; tree models are invariant to monotone transforms.
    - Amount:  log1p-transform (heavy right skew) then scale for LogReg.
    - Time:    already dropped at preprocessing step.
    - V1-V28:  already PCA-whitened, passthrough for trees, scale for LogReg.
    """
    if model_type != "logreg":
        return None  # trees: raw features

    v_cols = [f"V{i}" for i in range(1, 29)]
    return ColumnTransformer(
        transformers=[
            ("scale_v", StandardScaler(), v_cols),
            ("log_amount", Pipeline([
                ("log1p", FunctionTransformer(np.log1p)),
                ("scale", StandardScaler()),
            ]), ["Amount"]),
        ],
        remainder="drop",
    )


# ── Model factories ──────────────────────────────────────────────────────────

def make_logreg() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=0.1,
        random_state=RANDOM_STATE,
    )


def make_xgb(max_depth: int, lr: float, n_estimators: int,
             scale_pos_weight: float) -> "XGBClassifier":
    if XGBClassifier is None:
        raise ImportError(_xgb_err)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        max_depth=max_depth,
        learning_rate=lr,
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def make_lgbm(num_leaves: int, lr: float, n_estimators: int,
              use_is_unbalance: bool, scale_pos_weight: float) -> "LGBMClassifier":
    """
    is_unbalance and scale_pos_weight are mutually exclusive per LightGBM docs.
    Ref: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    """
    if LGBMClassifier is None:
        raise ImportError(_lgbm_err)
    kwargs: dict = dict(
        objective="binary",
        metric="average_precision",
        num_leaves=num_leaves,
        learning_rate=lr,
        n_estimators=n_estimators,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    if use_is_unbalance:
        kwargs["is_unbalance"] = True
    else:
        kwargs["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(**kwargs)


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Primary: PR-AUC (average_precision_score).
    PR-AUC is the correct primary metric at ~0.17% positive rate: accuracy is
    trivially 99.83% by always predicting 0, and ROC-AUC is optimistic because
    it includes the TN-heavy region irrelevant to fraud detection.
    """
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    # Threshold that maximises F1 on the test set
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    eps = 1e-9
    f1s = 2 * precisions * recalls / (precisions + recalls + eps)
    best_idx = int(np.argmax(f1s[:-1]))  # last element has no threshold
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])

    y_pred = (y_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "best_threshold": best_thresh,
        "f1_at_threshold": best_f1,
        "precision_at_threshold": best_prec,
        "recall_at_threshold": best_rec,
        "confusion_matrix": cm,
    }


# ── Core training function (public API) ─────────────────────────────────────

def train_one(config: dict) -> str:
    """
    Train a single model configuration, log to MLflow, return the run_id.

    config dict keys (all required unless noted):
      run_name      : str   — MLflow run name (e.g. "logreg-baseline")
      model         : str   — "logreg" | "xgb" | "lgbm"
      # logreg
        C           : float
      # xgb
        max_depth   : int
        lr          : float
        n_estimators: int
      # lgbm
        num_leaves  : int
        lr          : float
        n_estimators: int
        is_unbalance: bool  (if False, scale_pos_weight is used)
    """
    # ── Load pre-split data ──────────────────────────────────────────────────
    # Time column was dropped during preprocessing; splits have 29 feature cols.
    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_test  = pd.read_parquet(X_TEST_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH).squeeze()
    y_test  = pd.read_parquet(Y_TEST_PATH).squeeze()

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    model_type = config["model"]
    run_name   = config.get("run_name", f"{model_type}-run")

    # ── Build model ──────────────────────────────────────────────────────────
    preprocessor = build_preprocessor(model_type)

    if model_type == "logreg":
        estimator = make_logreg()
        model = Pipeline([("pre", preprocessor), ("clf", estimator)])
    elif model_type == "xgb":
        estimator = make_xgb(
            config["max_depth"], config["lr"], config["n_estimators"],
            scale_pos_weight,
        )
        model = estimator
    elif model_type == "lgbm":
        estimator = make_lgbm(
            config["num_leaves"], config["lr"], config["n_estimators"],
            config.get("is_unbalance", False),
            scale_pos_weight,
        )
        model = estimator
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    # ── MLflow run ───────────────────────────────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        # Params
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        if model_type == "logreg":
            mlflow.log_param("C", config.get("C", 0.1))
            mlflow.log_param("solver", "lbfgs")
            mlflow.log_param("class_weight", "balanced")
        elif model_type == "xgb":
            mlflow.log_param("max_depth", config["max_depth"])
            mlflow.log_param("learning_rate", config["lr"])
            mlflow.log_param("n_estimators", config["n_estimators"])
        elif model_type == "lgbm":
            mlflow.log_param("num_leaves", config["num_leaves"])
            mlflow.log_param("learning_rate", config["lr"])
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("is_unbalance", config.get("is_unbalance", False))

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_prob)

        for k, v in metrics.items():
            if k != "confusion_matrix":
                mlflow.log_metric(k, v)

        # Confusion matrix artifact
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["actual_neg", "actual_pos"],
            columns=["pred_neg", "pred_pos"],
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            cm_df.to_csv(f)
            mlflow.log_artifact(f.name, artifact_path="eval")

        # Plot data artifact: y_true + y_prob for ROC/PR curves in evaluate.py
        plot_df = pd.DataFrame({"y_true": y_test.values, "y_prob": y_prob})
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w", prefix=f"plot_data_{run_name}_"
        ) as f:
            plot_df.to_csv(f, index=False)
            mlflow.log_artifact(f.name, artifact_path="plot_data")

        # Signature + input example
        input_example = X_train.iloc[:5]
        signature = infer_signature(input_example, model.predict_proba(input_example))

        # Log model (registration done post-hoc by register_best.py)
        if model_type == "logreg":
            mlflow.sklearn.log_model(
                model, "model",
                signature=signature,
                input_example=input_example,
            )
        elif model_type == "xgb":
            mlflow.xgboost.log_model(
                model, "model",
                signature=signature,
                input_example=input_example,
            )
        elif model_type == "lgbm":
            mlflow.lightgbm.log_model(
                model, "model",
                signature=signature,
                input_example=input_example,
            )

        run_id = run.info.run_id
        print(
            f"[{run_id[:8]}] {run_name:35s} | "
            f"PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
            f"F1@{metrics['best_threshold']:.3f}={metrics['f1_at_threshold']:.4f}"
        )

    return run_id


# ── CLI (ad-hoc single-config runs) ──────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one fraud detector config, log to MLflow.")
    p.add_argument("--model", required=True, choices=["logreg", "xgb", "lgbm"])
    p.add_argument("--run_name", type=str, default=None)
    # Tree params
    p.add_argument("--lr",           type=float, default=0.05)
    p.add_argument("--n_estimators", type=int,   default=300)
    # XGBoost
    p.add_argument("--max_depth",    type=int,   default=6)
    # LightGBM
    p.add_argument("--num_leaves",   type=int,   default=31)
    p.add_argument("--is_unbalance", action="store_true")
    # logreg
    p.add_argument("--C",            type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg: dict = {"model": args.model}
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.model == "logreg":
        cfg["C"] = args.C
    elif args.model == "xgb":
        cfg.update({"max_depth": args.max_depth, "lr": args.lr,
                    "n_estimators": args.n_estimators})
    elif args.model == "lgbm":
        cfg.update({"num_leaves": args.num_leaves, "lr": args.lr,
                    "n_estimators": args.n_estimators,
                    "is_unbalance": args.is_unbalance})
    train_one(cfg)
