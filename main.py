#!/usr/bin/env python3
"""End-to-end CLI for network anomaly detection.

Usage
-----
    python main.py --model both --data data/cicids_sample.csv --output outputs/metrics.json
    python main.py --model isolation_forest --data data/cicids_sample.csv
    python main.py --model autoencoder --data data/cicids_sample.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import (
    load_data,
    drop_low_variance,
    drop_correlated,
    split_data,
    scale_features,
)
from isolation_forest import (
    train_isolation_forest,
    predict_anomalies,
    evaluate as evaluate_if,
)
from autoencoder import (
    train_autoencoder,
    get_reconstruction_errors,
    threshold_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ID_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Protocol"]


def preprocess(data_path: str) -> Dict[str, Any]:
    """Shared preprocessing: load → clean → split → scale."""
    df_raw = load_data(data_path)
    df = df_raw.drop(columns=[c for c in ID_COLS if c in df_raw.columns])

    X_full = df.drop(columns=["Label"])
    X_lv, dropped_lv = drop_low_variance(X_full)
    X_clean, dropped_corr = drop_correlated(X_lv)

    df_clean = X_clean.copy()
    df_clean["Label"] = df["Label"]

    X_train, X_test, y_train, y_test = split_data(
        df_clean, label_col="Label", test_size=0.2, random_state=42
    )
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)

    return {
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "n_features": X_train_s.shape[1],
        "dropped_low_variance": dropped_lv,
        "dropped_correlated": dropped_corr,
    }


def run_isolation_forest(prep: Dict[str, Any]) -> Dict[str, float]:
    """Train Isolation Forest and return metrics."""
    X_train_s = prep["X_train"]
    X_test_s = prep["X_test"]
    y_test = prep["y_test"]
    contamination = float(prep["y_train"].mean())

    model = train_isolation_forest(X_train_s, contamination=contamination, random_state=42)
    y_pred, scores = predict_anomalies(model, X_test_s)
    metrics = evaluate_if(y_test, y_pred, scores)

    model_path = os.path.join("outputs", "isolation_forest.pkl")
    joblib.dump(model, model_path)

    return {f"isolation_forest_{k}": round(v, 4) for k, v in metrics.items()}


def run_autoencoder(prep: Dict[str, Any]) -> Dict[str, float]:
    """Train AutoEncoder and return metrics."""
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    X_train_s = prep["X_train"]
    X_test_s = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    X_train_benign = X_train_s[y_train == 0]

    model, _ = train_autoencoder(
        X_train_benign, epochs=50, lr=0.001, batch_size=256
    )

    errors_train = get_reconstruction_errors(model, X_train_s)
    errors_test = get_reconstruction_errors(model, X_test_s)
    _, threshold = threshold_predictions(errors_train, percentile=95)
    y_pred = (errors_test > threshold).astype(int)

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, errors_test),
    }

    weights_path = os.path.join("outputs", "autoencoder.pth")
    torch.save(model.state_dict(), weights_path)

    return {
        f"autoencoder_{k}": round(v, 4) for k, v in metrics.items()
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Network anomaly detection pipeline"
    )
    parser.add_argument(
        "--model",
        choices=["isolation_forest", "autoencoder", "both"],
        default="both",
        help="Which model(s) to run (default: both)",
    )
    parser.add_argument(
        "--data",
        default=os.path.join("data", "cicids_sample.csv"),
        help="Path to input CSV (default: data/cicids_sample.csv)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "metrics.json"),
        help="Path for metrics JSON output (default: outputs/metrics.json)",
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # ── Preprocess ──
    print(f"Loading data from {args.data} ...")
    prep = preprocess(args.data)
    print(f"Features after preprocessing: {prep['n_features']}")
    if prep["dropped_low_variance"]:
        print(f"Dropped zero-variance: {prep['dropped_low_variance']}")
    if prep["dropped_correlated"]:
        print(f"Dropped correlated: {prep['dropped_correlated']}")

    results: Dict[str, float] = {}

    # ── Isolation Forest ──
    if args.model in ("isolation_forest", "both"):
        print("\n--- Isolation Forest ---")
        if_results = run_isolation_forest(prep)
        results.update(if_results)
        for k, v in if_results.items():
            print(f"  {k.replace('isolation_forest_', ''):<12s}: {v:.4f}")

    # ── AutoEncoder ──
    if args.model in ("autoencoder", "both"):
        print("\n--- AutoEncoder ---")
        ae_results = run_autoencoder(prep)
        results.update(ae_results)
        for k, v in ae_results.items():
            print(f"  {k.replace('autoencoder_', ''):<12s}: {v:.4f}")

    # ── Save metrics ──
    results["n_features"] = prep["n_features"]
    results["n_train"] = len(prep["y_train"])
    results["n_test"] = len(prep["y_test"])
    results["attack_ratio"] = round(float(prep["y_train"].mean()), 4)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
