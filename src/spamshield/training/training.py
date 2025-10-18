from pathlib import Path
import shutil
from typing import Any
import pandas as pd
import numpy as np
import json
from argparse import ArgumentParser
import joblib


from spamshield.core.signature import sha256_hash_file
from spamshield.training import data, pipeline, plots, metrics


def train(
    train_test_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series],
    models_dir: Path,
    version: str,
    reports: bool = False,
) -> None:
    version_path = models_dir / version
    if version_path.exists():
        print(version_path, "already exists. Aborting")
        return

    version_path.mkdir()

    model_path = version_path / "model.joblib"

    if model_path.exists():
        print(model_path, "already exists. Aborting")
        return

    model_metadata_path = version_path / "model_metadata.joblib"

    if model_metadata_path.exists():
        print(model_metadata_path, "already exists. Aborting")
        return

    X_train, X_test, y_train, y_test = train_test_data

    spam_pipeline, best_hyperparams = pipeline.train_pipeline(X_train, y_train)

    classes: list[str] = list(spam_pipeline.classes_)

    # Ensure classes are as expected
    assert set(classes) == {"ham", "spam"}, "unexpected classes are present"

    spam_idx: int = classes.index("spam")

    # Make inferences using test data
    prob_spam: np.ndarray = spam_pipeline.predict_proba(X_test)[:, spam_idx]

    y_true_bin: np.ndarray = (y_test == "spam").astype(int).to_numpy()

    best_threshold, best_f1 = metrics.tune_f1_threshold(y_true_bin, prob_spam)

    model_metrics: dict[str, Any] = metrics.compute_metrics_dict(
        y_true_bin, prob_spam, best_hyperparams, len(y_test), best_threshold
    )

    print("=== Training complete ===")
    print(json.dumps(model_metrics, indent=2))

    joblib.dump(spam_pipeline, model_path)

    model_sha256 = sha256_hash_file(model_path)
    model_filename = f"model_{model_sha256[:8]}.joblib"
    model_path_with_sha = version_path / model_filename
    shutil.move(model_path, model_path_with_sha)

    joblib.dump(
        {
            "version": version,
            "model_sha256": model_sha256,
            "model_filename": model_filename, 
            "threshold": best_threshold,
        },
        model_metadata_path,
    )

    if reports:
        reports_dir: Path = models_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        metrics_path = reports_dir / "metrics.json"

        (metrics_path).write_text(json.dumps(model_metrics, indent=2))
        plots.save_plots(y_true_bin, prob_spam, reports_dir)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--version", type=str, help="model version like v1.0.4", required=True
    )
    parser.add_argument(
        "--reports", action="store_true", help="generate training reports"
    )

    args = parser.parse_args()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    data_dir = models_dir / "data"
    data_dir.mkdir(exist_ok=True)

    test_train_data = data.load_test_train_data(data_dir)

    train(test_train_data, models_dir, args.version, args.reports)


if __name__ == "__main__":
    main()
