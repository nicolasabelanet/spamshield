import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve


def save_plots(y_true_bin: np.ndarray, prob_spam: np.ndarray, report_dir: Path):
    precision, recall, _ = precision_recall_curve(y_true_bin, prob_spam)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "pr_curve.png", dpi=160)
    plt.close()

    y_pred = (prob_spam >= 0.5).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["ham", "spam"])
    plt.yticks([0, 1], ["ham", "spam"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.title("Confusion Matrix @0.5")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close()
