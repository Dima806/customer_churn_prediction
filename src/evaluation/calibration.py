"""Calibration analysis: calibration curve and Expected Calibration Error."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def compute_calibration(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration curve data and the Expected Calibration Error.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Number of equal-width bins for the calibration curve.

    Returns:
        Dict with:
        - ``fraction_of_positives`` – observed positive rate per bin
        - ``mean_predicted_value``  – mean predicted probability per bin
        - ``ece``                   – Expected Calibration Error (scalar)
    """
    y_true_arr = np.asarray(y_true, dtype=int)

    fraction_pos, mean_pred = calibration_curve(y_true_arr, y_prob, n_bins=n_bins)

    # ECE: bin-count-weighted absolute calibration error
    counts, _ = np.histogram(y_prob, bins=n_bins, range=(0.0, 1.0))
    n_populated = min(len(counts), len(fraction_pos))
    weights = counts[:n_populated] / max(counts[:n_populated].sum(), 1)
    ece = float(np.sum(weights * np.abs(fraction_pos[:n_populated] - mean_pred[:n_populated])))

    return {
        "fraction_of_positives": fraction_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
        "ece": round(ece, 6),
        "n_bins": n_bins,
    }
