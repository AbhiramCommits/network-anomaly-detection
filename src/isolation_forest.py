"""Isolation Forest model training, prediction, and evaluation.

Functions
---------
train_isolation_forest – Fit an IsolationForest on training data.
predict_anomalies       – Return binary anomaly labels and raw scores.
evaluate                – Compute precision, recall, F1, and ROC-AUC.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    n_jobs: int = -1,
) -> IsolationForest:
    """Train an Isolation Forest anomaly detector.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.  Should be pre-scaled and free of labels.
    contamination : float, optional
        Expected proportion of anomalies in the data (default 0.05).
    random_state : int, optional
        Random seed for reproducibility (default 42).
    n_jobs : int, optional
        Number of parallel jobs; ``-1`` uses all available cores.

    Returns
    -------
    IsolationForest
        Fitted scikit-learn IsolationForest model.
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train)
    return model


def predict_anomalies(
    model: IsolationForest,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate binary anomaly predictions and raw anomaly scores.

    Parameters
    ----------
    model : IsolationForest
        A fitted IsolationForest instance.
    X : pd.DataFrame
        Feature matrix to score.

    Returns
    -------
    y_pred : np.ndarray  (shape: [n_samples,])
        Binary predictions: ``1`` = anomaly, ``0`` = normal.
    scores : np.ndarray  (shape: [n_samples,])
        Raw anomaly scores.  **Higher values indicate stronger anomaly
        likelihood** (computed by negating ``decision_function``).
    """
    raw = model.predict(X)                # -1 = outlier,  1 = inlier
    y_pred = (raw == -1).astype(int)      #  1 = anomaly,  0 = normal

    scores = -model.decision_function(X)  # higher → more anomalous
    return y_pred, scores


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics for anomaly detection results.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (``1`` = attack, ``0`` = benign).
    y_pred : np.ndarray
        Predicted binary labels (``1`` = anomaly, ``0`` = normal).
    scores : np.ndarray
        Raw anomaly scores (higher = more anomalous).

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``precision``, ``recall``, ``f1``, and
        ``roc_auc``.
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }
