"""
data.py — Data acquisition and preprocessing for the fraud detection pipeline.

Subcommands:
  python src/data.py pull         -- Download raw data from OpenML (dataset id 1597)
  python src/data.py preprocess   -- Clean + split into 4 parquet files

Output of preprocess:
  data/processed/X_train.parquet  (~228k rows x 29 cols, Time dropped)
  data/processed/X_test.parquet   (~57k rows  x 29 cols)
  data/processed/y_train.parquet  (Class column only)
  data/processed/y_test.parquet   (Class column only)

OpenML dataset id 1597 is the canonical Kaggle creditcard fraud dataset.
Ref: https://www.openml.org/d/1597
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Constants ────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_FILE      = RAW_DIR / "creditcard.parquet"

OPENML_DATASET_ID = 1597   # creditcard fraud, ~284k rows
TEST_SIZE          = 0.20
RANDOM_STATE       = 42

# Stratified but NOT temporal split — acceptable for this academic exercise.
# A production deployment should respect transaction time ordering.


# ── pull ─────────────────────────────────────────────────────────────────────

def pull() -> None:
    """Download raw dataset from OpenML and save as parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading OpenML dataset id={OPENML_DATASET_ID} …")
    dataset = openml.datasets.get_dataset(
        OPENML_DATASET_ID,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Combine into a single DataFrame matching the Kaggle schema
    df = X.copy()
    df["Class"] = y.astype(int)

    df.to_parquet(RAW_FILE, index=False)
    print(f"Saved raw data → {RAW_FILE}  shape={df.shape}")


# ── preprocess ───────────────────────────────────────────────────────────────

def preprocess() -> None:
    """
    Load raw parquet, drop Time, stratified-split, save 4 split parquets.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_FILE} …")
    df = pd.read_parquet(RAW_FILE)

    # Drop Time: weakly informative elapsed seconds; adds noise for gradient
    # methods and collinearity risk for LogReg.
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    print(f"Full dataset: {X.shape[0]:,} rows × {X.shape[1]} features  "
          f"| fraud rate = {y.mean()*100:.3f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Save splits
    X_train.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(PROCESSED_DIR  / "X_test.parquet",  index=False)
    pd.DataFrame({"Class": y_train}).to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
    pd.DataFrame({"Class": y_test}).to_parquet(PROCESSED_DIR  / "y_test.parquet",  index=False)

    print(
        f"Train: {X_train.shape[0]:,} rows  |  "
        f"Test: {X_test.shape[0]:,} rows  |  "
        f"Features: {X_train.shape[1]}"
    )
    print(f"Saved splits → {PROCESSED_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Data pipeline for fraud detection.")
    p.add_argument("subcommand", choices=["pull", "preprocess"])
    args = p.parse_args()

    if args.subcommand == "pull":
        pull()
    elif args.subcommand == "preprocess":
        preprocess()


if __name__ == "__main__":
    main()
