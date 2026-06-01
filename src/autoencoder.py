"""PyTorch AutoEncoder for unsupervised anomaly detection on network flows.

Architecture
------------
encoder : input_dim → 64 → 32 → 16  (ReLU after each dense layer)
decoder : 16 → 32 → 64 → input_dim  (ReLU after first two, linear output)

Functions
---------
train_autoencoder       – Train with MSE loss, optionally on GPU.
get_reconstruction_errors – Per-sample MSE between input and reconstruction.
threshold_predictions   – Binary anomaly labels via percentile threshold.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AutoEncoder(nn.Module):
    """Vanilla undercomplete autoencoder with ReLU activations.

    Parameters
    ----------
    input_dim : int
        Number of features in the input layer.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(
    X_train: np.ndarray | pd.DataFrame,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Tuple[AutoEncoder, List[float]]:
    """Train an undercomplete AutoEncoder with MSE loss.

    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training data (must be numeric and pre-scaled).
    epochs : int, optional
        Number of passes over the full dataset (default 50).
    lr : float, optional
        Adam learning rate (default 0.001).
    batch_size : int, optional
        Mini-batch size (default 256).
    device : str or None, optional
        Torch device string (e.g. ``"cuda"``, ``"mps"``).  Auto-detected
        when ``None``.

    Returns
    -------
    model : AutoEncoder
        Trained model (on the chosen device).
    losses : list[float]
        Average MSE loss recorded after every epoch.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values.astype(np.float32)
    else:
        X_train = X_train.astype(np.float32)

    tensor = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor, tensor), batch_size=batch_size, shuffle=True
    )

    input_dim = X_train.shape[1]
    model = AutoEncoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: List[float] = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(X_train)
        losses.append(avg_loss)
        print(f"Epoch {epoch:3d}/{epochs}  |  MSE loss = {avg_loss:.6f}")

    return model, losses


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_reconstruction_errors(
    model: AutoEncoder,
    X: np.ndarray | pd.DataFrame,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute per-sample reconstruction MSE.

    Parameters
    ----------
    model : AutoEncoder
        Trained autoencoder.
    X : np.ndarray or pd.DataFrame
        Input data to reconstruct.
    device : str or None, optional
        Torch device.  Auto-detected when ``None``.

    Returns
    -------
    np.ndarray  (shape: [n_samples,])
        Per-sample MSE  (higher → harder to reconstruct → more anomalous).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if isinstance(X, pd.DataFrame):
        X = X.values.astype(np.float32)
    else:
        X = X.astype(np.float32)

    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(tensor)
        # MSE per sample (mean over feature axis)
        se = ((tensor - reconstructed) ** 2).mean(dim=1)
        errors = se.cpu().numpy()

    return errors


def threshold_predictions(
    errors: np.ndarray,
    percentile: float = 95,
) -> Tuple[np.ndarray, float]:
    """Flag samples whose reconstruction error exceeds a percentile threshold.

    Parameters
    ----------
    errors : np.ndarray
        Per-sample reconstruction errors (higher = more anomalous).
    percentile : float, optional
        Percentile used as the anomaly threshold (default 95).

    Returns
    -------
    y_pred : np.ndarray  (shape: [n_samples,])
        Binary predictions: ``1`` = anomaly, ``0`` = normal.
    threshold : float
        The computed threshold value.
    """
    threshold = float(np.percentile(errors, percentile))
    y_pred = (errors > threshold).astype(int)
    return y_pred, threshold
