"""UMAP-based 2-D visualisation of anomaly detection results.

Functions
---------
plot_umap – Reduce features to 2-D via UMAP and colour by prediction label.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


def plot_umap(
    X: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray,
    title: str = "UMAP projection",
    save_path: Optional[str] = None,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project high-dimensional features to 2-D with UMAP and scatter by
    predicted label.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix to project.
    y_pred : np.ndarray  (shape: [n_samples,])
        Binary predictions: ``0`` = normal, ``1`` = anomalous.
    title : str, optional
        Plot title.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    random_state : int, optional
        UMAP random seed (default 42).
    n_neighbors : int, optional
        UMAP ``n_neighbors`` parameter (default 30).
    min_dist : float, optional
        UMAP ``min_dist`` parameter (default 0.1).

    Returns
    -------
    np.ndarray  (shape: [n_samples, 2])
        The 2-D embedding coordinates.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values.astype(np.float32)

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    for label_val, label_name, color in [
        (0, "Predicted BENIGN", "#2ecc71"),
        (1, "Predicted ATTACK", "#e74c3c"),
    ]:
        mask = y_pred == label_val
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=label_name,
            alpha=0.4,
            s=6,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    plt.show()
    return embedding
