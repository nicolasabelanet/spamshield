from collections.abc import Callable
from importlib import resources
from importlib.resources.abc import Traversable
from fastapi import Request
import joblib
from typing import Final, Literal

from pydantic import TypeAdapter
from sklearn.pipeline import Pipeline
import spamshield.api.models
from spamshield.common.signature import sha256_hash_file
from spamshield.model.types import ModelMetadata

# Anchor to find versioned spam models at.
MODELS_PATH: Final[Traversable] = resources.files(spamshield.api.models)

# Data adapter to validate loaded metadata adheres to expected shape.
MODEL_METADATA_ADAPTER: Final[TypeAdapter[ModelMetadata]] = TypeAdapter(ModelMetadata)

type ModelLoader = Callable[[str], SpamModel]


class SpamModel:
    """
    Wrapper around a trained spam classification pipeline and its metadata.

    This class exposes a clean interface for running spam/ham predictions
    using a serialized scikit-learn `Pipeline` and its associated metadata.
    """

    def __init__(self, model: Pipeline, metadata: ModelMetadata):
        # Store the trained scikit-learn pipeline
        self._pipeline: Pipeline = model

        # Metadata includes filename, checksum, and decision threshold
        self._metadata: ModelMetadata = metadata

        # Cache frequently used values for convenience
        self._hash: str = metadata["model_sha256"]
        self._threshold: float = metadata["threshold"]

    @property
    def version(self) -> str:
        """Version string combinining model version and unique SHA identifier."""
        return f"1.0.0+{self._metadata['model_sha256']}"

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        """
        Predict spam/ham labels and probabilities for input texts.

        Parameters
        ----------
        texts : list[str]
            List of raw text messages to classify.

        Returns
        -------
        list[tuple[str, float]]
            A list of (label, probability) pairs where:
              - label ∈ {"spam", "ham"}
              - probability is the model's spam probability in [0, 1]
        """
        # Predict probability distributions for each message
        probabilities = self._pipeline.predict_proba(texts)

        # Determine the column index corresponding to the "spam" class
        spam_index: int = int(self._pipeline.classes_.tolist().index("spam"))
        predictions: list[tuple[str, float]] = []

        for p in probabilities:
            # Extract the spam probability for this sample
            probability_spam: float = float(p[spam_index])

            # Apply model threshold to decide spam vs ham
            label: Literal["spam", "ham"] = (
                "spam" if probability_spam >= self._threshold else "ham"
            )

            # Append tuple of label and score
            predictions.append((label, probability_spam))

        return predictions


def load_model(model_version: str) -> SpamModel:
    """
    Load and validate the spam model and its associated metadata.

    This method:
    1. Reads the model metadata from `model_metadata.joblib`.
    2. Validates the model file’s SHA-256 checksum against the metadata.
    3. Loads the serialized scikit-learn `Pipeline` from disk.
    4. Ensures the loaded object is a valid `Pipeline`.

    Returns
    -------
    tuple[Pipeline, ModelMetadata]
        A tuple `(pipeline, metadata)` where:
        - `pipeline` is a scikit-learn `Pipeline` object used for inference.
        - `metadata` is a validated `ModelMetadata` dictionary.

    Raises
    ------
    RuntimeError
        If checksum validation fails or the loaded object is not a valid
        scikit-learn `Pipeline`.
    """
    model_metadata_path: Traversable = (
        MODELS_PATH / model_version / "model_metadata.joblib"
    )

    # 1. Load model metadata and validate schema
    model_metadata: ModelMetadata = MODEL_METADATA_ADAPTER.validate_python(
        joblib.load(model_metadata_path)
    )

    model_filename: str = model_metadata["model_filename"]
    model_filepath: Traversable = MODELS_PATH / model_version / model_filename

    # 2. Validate the model file's signature against the metadata
    valid_signature: bool = model_metadata["model_sha256"] == sha256_hash_file(
        model_filepath
    )

    if not valid_signature:
        raise RuntimeError("Could not validate integrity of spam model.")

    # 3. Load the spam model
    model = joblib.load(model_filepath)

    # 4. Validate type
    if not isinstance(model, Pipeline):
        raise RuntimeError("Model loaded from file was not a Pipeline.")

    return SpamModel(model, model_metadata)


def get_model_loader() -> ModelLoader:
    def impl(version: str) -> SpamModel:
        return load_model(version)

    return impl


def get_model(request: Request) -> SpamModel:
    return request.app.state.model
