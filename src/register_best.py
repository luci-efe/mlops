"""
register_best.py — Post-hoc model registration.

After all 7 training runs complete, this script:
  1. Queries the EXPERIMENT_NAME experiment for the run with max pr_auc.
  2. Creates a registered model version under "fraud-detector".
  3. Sets the alias "staging" (MLflow 3.x recommended approach — see NOTE below).

NOTE on stages vs aliases (MLflow 3.12, May 2026):
  The old stage-based API (transition_model_version_stage → "Staging") is
  DEPRECATED and will be removed in a future major release.
  Ref: https://mlflow.org/docs/latest/model-registry.html
  Use MlflowClient.set_registered_model_alias() instead:
    client.set_registered_model_alias(model_name, "staging", version)
  To load by alias: mlflow.pyfunc.load_model("models:/fraud-detector@staging")

Usage:
  python src/register_best.py [--experiment credit-fraud] [--model-name fraud-detector]
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "credit-fraud"
MODEL_NAME = "fraud-detector"


def register_best(experiment_name: str, model_name: str) -> None:
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.pr_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No completed runs found.")

    best = runs[0]
    best_pr_auc = best.data.metrics.get("pr_auc", 0.0)
    model_uri = f"runs:/{best.info.run_id}/model"

    print(f"Best run: {best.info.run_id[:8]}  PR-AUC={best_pr_auc:.4f}")
    print(f"Registering as '{model_name}' from {model_uri}")

    # create_model_version registers the artifact and returns the version object
    mv = mlflow.register_model(model_uri, model_name)

    # Set alias "staging" — replaces deprecated transition_model_version_stage
    # Ref: https://mlflow.org/docs/latest/model-registry.html#model-aliases
    client.set_registered_model_alias(model_name, "staging", mv.version)
    print(f"Registered version {mv.version} with alias 'staging'.")
    print(f"Load with: mlflow.pyfunc.load_model('models:/{model_name}@staging')")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=EXPERIMENT_NAME)
    p.add_argument("--model-name", default=MODEL_NAME)
    args = p.parse_args()
    register_best(args.experiment, args.model_name)
