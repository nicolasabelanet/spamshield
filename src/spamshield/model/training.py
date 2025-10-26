import json
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from spamshield.common import signature
from spamshield.model import data, metrics, pipeline, plots
from spamshield.model.types import ModelMetadata

import numpy as np
import pandas as pd


def train_model(
    train_test_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series],
    plots_dir: Path | None,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train the spam classification pipeline and compute evaluation metrics.

    Steps:
      1. Train the text classification pipeline (TF-IDF + LogisticRegression, etc.).
      2. Get spam probabilities on the test split.
      3. Find the probability threshold that maximizes F1.
      4. Compute final metrics at both 0.5 and the tuned threshold.
      5. Optionally write diagnostic plots to `plots_dir`.

    Parameters
    ----------
    train_test_data : tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        Same (X_train, X_test, y_train, y_test) tuple passed to `create_model_package`.
    plots_dir : Path | None
        Directory to write diagnostic plots (precision-recall curve, score
        histograms, etc.). If None, no plots are generated.

    Returns
    -------
    tuple[Pipeline, dict[str, Any]]
        - The trained inference pipeline (ready to be serialized).
        - A dict of computed metrics, including the best F1 threshold.
    """
    # Unpack input splits.
    X_train, X_test, y_train, y_test = train_test_data

    # Train the pipeline (vectorizer + classifier) and get best hyperparams.
    spam_pipeline, best_hyperparams = pipeline.train_pipeline(X_train, y_train)

    # The pipeline must expose .classes_ with ["ham", "spam"] in some order.
    classes: list[str] = list(spam_pipeline.classes_)
    # Ensure classes are as expected
    assert set(classes) == {"ham", "spam"}, "unexpected classes are present"
    spam_idx: int = classes.index("spam")

    # Use the held-out test set to get predicted spam probabilities.
    prob_spam: np.ndarray = spam_pipeline.predict_proba(X_test)[:, spam_idx]

    # Convert ground truth labels (Series of "spam"/"ham") to binary 1/0.
    y_true_bin: np.ndarray = (y_test == "spam").astype(int).to_numpy()

    # Pick the probability threshold that maximizes F1 score.
    best_f1_threshold, _ = metrics.tune_f1_threshold(y_true_bin, prob_spam)

    # Compute metrics at default 0.5 and at the tuned threshold.
    model_metrics: dict[str, Any] = metrics.compute_metrics_dict(
        y_true_bin, prob_spam, best_hyperparams, len(y_test), best_f1_threshold
    )

    # Optionally generate plots (e.g. PR curve, score distribution).
    if plots_dir is not None:
        plots.save_plots(y_true_bin, prob_spam, plots_dir, best_f1_threshold)

    print("=== Training complete ===")
    print(json.dumps(model_metrics, indent=2))

    return spam_pipeline, model_metrics


def create_model_package(
    train_test_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series],
    models_dir: Path,
    version: str,
    plots: bool = False,
) -> None:
    """
    Train a model and write a versioned bundle to disk.

    The bundle is written to: {models_dir}/{version}/
    and contains:
      - model_<sha8>.joblib       (the trained Pipeline)
      - model_metadata.joblib     (metadata needed at runtime)
      - model_metrics.json        (evaluation metrics for audit/debug)
      - optional plots            (if `plots=True`)

    The version directory is treated as immutable. If it already exists,
    the function prints a message and returns without overwriting anything.

    Parameters
    ----------
    train_test_data : tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test) where:
          - X_* are pd.Series of email/SMS text
          - y_* are pd.Series of labels "spam"/"ham"
        Typically returned by `data.load_test_train_data(...)`.
    models_dir : Path
        Root directory for all model versions. Typically Path("models").
    version : str
        Version identifier for this model package, e.g. "v1.0.4".
        This will become the folder name under `models_dir`.
    plots : bool
        If True, save diagnostic plots in the same version directory.
    """

    version_path = models_dir / version

    # Don't overwrite an existing model version.
    if version_path.exists():
        print(version_path, "already exists. Aborting")
        return

    version_path.mkdir()

    # Paths for artifacts associated with this version.
    model_path = version_path / "model.joblib"
    model_metadata_path = version_path / "model_metadata.joblib"
    metrics_path = version_path / "model_metrics.json"

    # If plots were requested, write them into the version directory.
    plots_dir = version_path if plots else version_path

    for path in (model_path, model_metadata_path, metrics_path):
        if path.exists():
            print(path, "already exists. Aborting")

    # Train model, compute metrics, optionally generate plots.
    spam_pipeline, model_metrics = train_model(train_test_data, plots_dir)

    # Serialize the trained pipeline to disk (temporary filename).
    joblib.dump(spam_pipeline, model_path)

    # Compute SHA-256 for integrity + traceability.
    # This lets runtime verify it is loading the exact expected bits.
    model_sha256 = signature.sha256_hash_file(model_path)

    # Rename model file to include its hash prefix. This gives us stable,
    # content-addressable artifacts like model_ab12cd34.joblib.
    model_filename = f"model_{model_sha256[:8]}.joblib"
    model_path_with_sha = version_path / model_filename
    shutil.move(model_path, model_path_with_sha)

    # Prepare metadata describing the packaged model.
    # The API service will load this metadata at startup.
    metadata: ModelMetadata = {
        "version": version,
        "model_sha256": model_sha256,
        "model_filename": model_filename,
        "threshold": model_metrics["best_f1_threshold"],
    }

    # Save metadata in joblib form (fast to load at runtime).
    joblib.dump(
        metadata,
        model_metadata_path,
    )

    (metrics_path).write_text(json.dumps(model_metrics, indent=2))


def main():
    """
    CLI entry point for training and packaging a model.

    Example:
        train-spam-model -m  \
            --version v1.0.4 \
            --dataset dataset/spam.csv \
            --plots

    Arguments:
      --version   required  Version label for this model bundle (e.g. v1.0.4)
      --dataset   required  Path to cleaned dataset CSV for training
      --plots     optional  If provided, save diagnostic training plots
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--version", type=str, help="model version like v1.0.4", required=True
    )
    parser.add_argument("--dataset", type=Path, help="dataset location", required=True)
    parser.add_argument(
        "--plots", action="store_true", help="generate training for inspection"
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        print("Dataset does not exist")
        return

    # Ensure the root models/ directory exists.
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Get stratified train/test splits (downloads/creates dataset if needed).
    test_train_data = data.load_test_train_data(args.dataset)

    # Train model and emit a versioned artifact bundle in models/<version>/.
    create_model_package(test_train_data, models_dir, args.version, args.plots)
