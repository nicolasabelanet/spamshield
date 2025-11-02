from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


def _plot_precision_recall_curve(
    y_true_bin: np.ndarray,
    prob_spam: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save a precision–recall curve plot.

    Parameters
    ----------
    y_true_bin : array of shape (n_samples,)
        True binary labels (1=spam, 0=ham).
    prob_spam : array of shape (n_samples,)
        Predicted P(spam|text).
    out_path : Path
        Where to save the PNG.
    """
    precision, recall, _ = precision_recall_curve(y_true_bin, prob_spam)
    spam_rate = y_true_bin.mean()
    ap = average_precision_score(y_true_bin, prob_spam)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)

    # Plot main curve
    ax.plot(
        recall,
        precision,
        color="#0072B2",
        lw=2,
        label=f"Model (AP = {ap:.3f})",
    )

    # Shaded area under the curve
    ax.fill_between(recall, precision, alpha=0.1, color="#0072B2")

    # Baseline (no-skill line)
    ax.hlines(
        y=spam_rate,
        xmin=0.0,
        xmax=1.0,
        color="gray",
        linestyles="dashed",
        linewidth=1.5,
        label=f"Baseline (spam rate = {spam_rate:.2f})",
    )

    # Clean up axes
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")

    ax.grid(alpha=0.3, linestyle=":", linewidth=0.6)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(
    y_true_bin: np.ndarray,
    prob_spam: np.ndarray,
    out_path: Path,
    threshold: float = 0.5,
) -> None:
    """
    Save a confusion matrix plot at a given threshold.

    Parameters
    ----------
    y_true_bin : array of shape (n_samples,)
        True binary labels (1=spam, 0=ham).
    prob_spam : array of shape (n_samples,)
        Predicted P(spam|text).
    out_path : Path
        Where to save the PNG.
    threshold : float
        Classification cutoff for spam.
    """
    y_pred = (prob_spam >= threshold).astype(int)

    # cm[row=true_class, col=pred_class]
    # rows: 0=ham, 1=spam
    # cols: 0=ham, 1=spam
    cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=180)

    # Show matrix
    im = ax.imshow(cm, cmap="Blues")

    # Axes tick labels (predicted is x, true is y)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ham", "spam"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ham", "spam"])

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Title: include threshold, but keep it on one line and not clipped
    ax.set_title(f"Confusion Matrix @ Threshold = {threshold:.2f}")

    # Annotate each cell with count, and adjust text color for contrast
    max_val = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=11,
                color=("white" if max_val and val > max_val * 0.5 else "black"),
            )

    # Add cell borders for readability
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colorbar on the side with label
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_plots(
    y_true_bin: np.ndarray,
    prob_spam: np.ndarray,
    report_dir: Path,
    threshold: float = 0.5,
) -> None:
    """
    High-level convenience: write all diagnostic plots.

    Parameters
    ----------
    y_true_bin : np.ndarray
        Ground-truth binary labels (1=spam, 0=ham).
    prob_spam : np.ndarray
        Predicted P(spam|text) for each sample.
    report_dir : Path
        Directory where plots will be saved. Must already exist.
    threshold : float
        Classification cutoff for confusion-matrix rendering.
    """
    _plot_precision_recall_curve(
        y_true_bin=y_true_bin,
        prob_spam=prob_spam,
        out_path=report_dir / "precision_recall_curve.png",
    )

    _plot_confusion_matrix(
        y_true_bin=y_true_bin,
        prob_spam=prob_spam,
        out_path=report_dir / "confusion_matrix.png",
        threshold=threshold,
    )
