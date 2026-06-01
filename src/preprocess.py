"""Reusable preprocessing pipeline for network anomaly detection.

Functions
---------
load_data       – Read CSV into a DataFrame.
drop_low_variance – Remove features whose variance falls below a threshold.
drop_correlated   – Remove one member of each pair with |r| > threshold.
encode_labels     – Binary-encode labels: BENIGN → 0, everything else → 1.
scale_features    – Fit a StandardScaler on training data and transform both
                    train and test sets.
split_data        – Stratified train/test split.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Read a CSV file and return a pandas DataFrame.

    Parameters
    ----------
    path : str
        Filesystem path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    return pd.read_csv(path)


def drop_low_variance(
    df: pd.DataFrame,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop numeric columns whose variance is ≤ *threshold*.

    Non-numeric columns are left untouched.  A threshold of ``0.0`` removes
    columns that are constant across all rows (zero-variance features).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  May contain mixed dtypes; only numeric columns
        are evaluated.
    threshold : float, optional
        Minimum variance a column must have to be retained (default 0.0).

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with low-variance numeric columns removed.
    dropped : List[str]
        Names of the dropped columns.

    Notes
    -----
    This function does **not** modify *df* in place.
    """
    numeric = df.select_dtypes(include=[np.number])
    variances = numeric.var()
    low_var = variances[variances <= threshold].index.tolist()
    df_out = df.drop(columns=low_var, errors="ignore")
    return df_out, low_var


def drop_correlated(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop one feature from each pair of numeric columns whose absolute
    Pearson correlation exceeds *threshold*.

    For every pair with ``|r| > threshold``, the column with the **lower**
    variance is dropped.  Non-numeric columns are passed through unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Only numeric columns participate in correlation
        analysis.
    threshold : float, optional
        Correlation magnitude above which one column is dropped (default
        0.95).

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with redundant columns removed.
    dropped : List[str]
        Names of the dropped columns.

    Notes
    -----
    This function does **not** modify *df* in place.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return df.copy(), []

    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop: set = set()

    for col in upper.columns:
        high = upper[col][upper[col] > threshold].index.tolist()
        for other in high:
            if numeric[col].var() >= numeric[other].var():
                to_drop.add(other)
            else:
                to_drop.add(col)

    dropped = sorted(to_drop)
    df_out = df.drop(columns=dropped, errors="ignore")
    return df_out, dropped


def encode_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    """Binary-encode a categorical label column.

    * ``BENIGN`` (case-insensitive) → ``0``
    * Any other value → ``1``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the label column.
    label_col : str
        Name of the column to encode.

    Returns
    -------
    pd.Series
        Integer series of 0s (benign) and 1s (attack).

    Raises
    ------
    KeyError
        If *label_col* is not present in *df*.
    """
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' not found in DataFrame.")

    return df[label_col].apply(
        lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
    )


def split_data(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split.

    Features are all columns except *label_col*.  Labels are binary-encoded
    via :func:`encode_labels` before splitting.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including features and labels.
    label_col : str
        Name of the label column.
    test_size : float, optional
        Fraction of data reserved for the test set (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training labels (0 / 1).
    y_test : pd.Series
        Test labels (0 / 1).
    """
    y = encode_labels(df, label_col)
    X = df.drop(columns=[label_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on the training set and transform both sets.

    Only numeric columns are scaled; non-numeric columns (if any) are
    dropped before scaling.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (must not contain labels).
    X_test : pd.DataFrame
        Test features (must not contain labels).

    Returns
    -------
    X_train_scaled : pd.DataFrame
        Scaled training features (same shape and column names).
    X_test_scaled : pd.DataFrame
        Scaled test features.
    scaler : StandardScaler
        The fitted scaler, available for inverse transforms later.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    Xt = X_train[numeric_cols].copy()
    Xs = X_test[numeric_cols].copy()

    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(Xt)
    X_test_arr = scaler.transform(Xs)

    X_train_scaled = pd.DataFrame(X_train_arr, columns=numeric_cols, index=Xt.index)
    X_test_scaled = pd.DataFrame(X_test_arr, columns=numeric_cols, index=Xs.index)

    return X_train_scaled, X_test_scaled, scaler
