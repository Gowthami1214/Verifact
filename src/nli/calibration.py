"""
src/nli/calibration.py
------------------------
Confidence calibration and temperature scaling for TruthSync NLI.

Provides:
  - TemperatureScaler: PyTorch module learning optimal scalar temperature T
    on validation logits to calibrate output softmax probabilities.
  - compute_ece: Calculates Expected Calibration Error across confidence bins.
  - compute_brier_score: Multi-class Brier score.
  - get_reliability_data: Binned confidence vs. empirical accuracy for reliability plots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to logits:
        scaled_logits = logits / T
    where T > 0 is learned on a validation set to minimize Cross-Entropy (NLL).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.clamp(min=0.01)
        return logits / temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> float:
        """
        Fit temperature parameter using L-BFGS on validation logits and labels.

        Args:
            logits: Tensor of shape (N, num_classes)
            labels: Tensor of shape (N,)
        Returns:
            Optimal temperature value (float)
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """
    Compute multi-class Brier score:
    Brier = (1/N) sum_{i=1}^N sum_{k=1}^K (p_{ik} - y_{ik})^2
    Lower is better (0 = perfect).
    """
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    for i, l in enumerate(labels):
        one_hot[i, int(l)] = 1.0

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(round(brier, 4))


def get_reliability_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Collect bin-level data for reliability diagram plotting.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins_data = []

    for i in range(n_bins):
        bin_lower = float(bin_boundaries[i])
        bin_upper = float(bin_boundaries[i + 1])
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(round(np.mean(accuracies[in_bin]), 4))
            bin_conf = float(round(np.mean(confidences[in_bin]), 4))
        else:
            bin_acc = 0.0
            bin_conf = float(round((bin_lower + bin_upper) / 2, 4))

        bins_data.append({
            "bin": i + 1,
            "range": [round(bin_lower, 2), round(bin_upper, 2)],
            "count": bin_size,
            "accuracy": bin_acc,
            "confidence": bin_conf
        })

    return {
        "ece": compute_ece(probs, labels, n_bins=n_bins),
        "brier_score": compute_brier_score(probs, labels),
        "bins": bins_data
    }
