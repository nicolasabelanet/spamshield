from collections.abc import Callable
import functools
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
        """
        Return a stable runtime version identifier for the loaded model.

        The identifier concatenates the human-readable model version with a
        content hash, e.g. "v1.0.0+ab12cd34...".
        """
        return f"{self._metadata['version']}+{self._metadata['model_sha256']}"

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


# Expose a callable signature representing "load a model by version".
type ModelLoader = Callable[[str], SpamModel]


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
    SpamModel
        A runtime-safe model wrapper that exposes `predict()` and `version`.

    Raises
    ------
    ValidationError
        The model metadata is not the correct shape.
    RuntimeError
        The model file checksum does not match the metadata or
        the loaded object is not a scikit-learn `Pipeline`.
    """
    model_metadata_path: Traversable = (
        MODELS_PATH / model_version / "model_metadata.joblib"
    )

    # Load model metadata and validate schema
    model_metadata: ModelMetadata = MODEL_METADATA_ADAPTER.validate_python(
        joblib.load(model_metadata_path)
    )

    model_filename: str = model_metadata["model_filename"]
    model_filepath: Traversable = MODELS_PATH / model_version / model_filename

    # Validate the model file's signature against the metadata
    valid_signature: bool = model_metadata["model_sha256"] == sha256_hash_file(
        model_filepath
    )

    if not valid_signature:
        raise RuntimeError("Could not validate integrity of spam model.")

    # Load the spam model
    model = joblib.load(model_filepath)

    # Validate type
    if not isinstance(model, Pipeline):
        raise RuntimeError("Model loaded from file was not a Pipeline.")

    return SpamModel(model, model_metadata)


@functools.cache
def get_model_loader() -> ModelLoader:
    """
    Dependency provider that returns a callable which loads a model by version.

    This indirection allows tests to override model loading cleanly:
    - In production, FastAPI will call `get_model_loader()` and receive a
      function that can be used to load the model.
    - In tests, dependency_overrides can replace `get_model_loader()` with
      a lambda that returns a fake model instead of reading from disk.
    """

    def impl(version: str) -> SpamModel:
        return load_model(version)

    return impl


def get_model(request: Request) -> SpamModel:
    """
    FastAPI dependency that returns the pre-loaded model from app state.

    The lifespan function stores the active model instance on
    `app.state.model` during startup. Routes can depend on this instead
    of reloading the model per request.
    """
    return request.app.state.model
