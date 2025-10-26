from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics_dict(
    y_true_bin: np.ndarray,
    prob_spam: np.ndarray,
    best_hyperparams: dict[str, Any],
    n_samples: int,
    best_f1_threshold: float,
) -> dict[str, Any]:
    """
    Compute evaluation metrics for a spam classifier and package them
    (along with run metadata) into a dictionary.

    Parameters
    ----------
    y_true_bin : np.ndarray (n_samples,)
        Ground-truth binary labels (0 = ham, 1 = spam).
    prob_spam : np.ndarray (n_samples,)
        Model-predicted spam probabilities for each sample (float in [0, 1]).
    best_hyperparams : dict[str, Any]
        Hyperparameters of the trained model (e.g. from GridSearchCV.best_params_).
    n_samples : int
        Number of samples in this evaluation split.
    best_f1_threshold : float
        The best probability cutoff to consider "spam". If you pass the value
        returned by `tune_f1_threshold`, you can interpret that as the
        F1-optimal decision threshold.

    Metrics (returned keys)
    -----------------------
    roc_auc : float
        Area under the ROC curve — how well the model ranks spam above ham.
    pr_auc : float
        Area under the precision–recall curve (Average Precision). More
        informative for imbalanced data like spam detection.
    f1_at_0.5 : float
        F1 score using a 0.5 decision threshold.
    acc_at_0.5 : float
        Accuracy using a 0.5 decision threshold.
    best_f1_threshold : float
        The best threshold that maximizes F1.
    f1_at_best_threshold : float
        F1 score using the `best_f1_threshold` decision threshold.
    acc_at_best_threshold : float
        Accuracy using the `best_f1_threshold` decision threshold.
    best_hyperparams : dict[str, Any]
        The hyperparameters associated with this model.
    n_samples : int
        The number of evaluated samples.
    """
    y_pred_best: np.ndarray = (prob_spam >= best_f1_threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true_bin, prob_spam),
        "pr_auc": average_precision_score(y_true_bin, prob_spam),
        "f1_at_0.5": f1_score(y_true_bin, (prob_spam >= 0.5).astype(int)),
        "acc_at_0.5": accuracy_score(y_true_bin, (prob_spam >= 0.5).astype(int)),
        "best_f1_threshold": best_f1_threshold,
        "f1_at_best_threshold": f1_score(y_true_bin, y_pred_best),
        "acc_at_best_threshold": accuracy_score(y_true_bin, y_pred_best),
        "best_hyperparams": best_hyperparams,
        "n_samples": n_samples,
    }


def tune_f1_threshold(
    y_true_bin: np.ndarray, prob_spam: np.ndarray
) -> tuple[float, float]:
    """
    Find the probability threshold that maximizes F1 score.

    For thresholds t in [0.1, 0.9] (inclusive), we:
      - classify as spam if prob_spam >= t
      - compute F1(y_true_bin, y_pred_t)
    and return the t that yields the highest F1.

    Returns
    -------
    tuple[float, float]
        (best_threshold, best_f1), where:
        - best_threshold is the probability cutoff that maximizes F1.
        - best_f1 is the F1 score achieved at that cutoff.
            spam/ham decision on this dataset and the f1 at that threshold.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_true_bin, (prob_spam >= t).astype(int)) for t in thresholds]
    best_idx: int = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])
