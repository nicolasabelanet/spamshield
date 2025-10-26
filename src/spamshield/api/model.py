from importlib import resources
from importlib.resources.abc import Traversable
import joblib
from typing import Final, Literal

from pydantic import TypeAdapter
from sklearn.pipeline import Pipeline
from spamshield.api.config import settings
import spamshield.api.models
from spamshield.common.signature import sha256_hash_file
from spamshield.model.types import ModelMetadata

MODELS_PATH = resources.files(spamshield.api.models)

MODEL_METADATA_ADAPTER = TypeAdapter(ModelMetadata)


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


class SpamModelDependency:
    """
    FastAPI dependency for loading and caching the SpamShield inference model.

    This dependency encapsulates the logic for safely loading a specific
    version of the spam classification model from disk, validating its integrity,
    and exposing it as a `SpamModel` instance. It loads the model lazily
    (on first use) and caches it for subsequent requests.

    Parameters
    ----------
    model_version : str
        The version identifier of the model to load. Used to locate both the
        serialized pipeline and its metadata under `MODELS_PATH`.

    Usage
    -----
    This class is typically instantiated once and registered as a FastAPI
    dependency:

        spam_model_dependency = SpamModelDependency(settings.MODEL_VERSION)

        @app.post("/predict")
        def predict(
            request: PredictRequest,
            model: Annotated[SpamModel, Depends(spam_model_dependency)],
        ):
            return model.predict(request.texts)

    Notes
    -----
    - The model directory must contain both:
        - `model_metadata.joblib`: validated metadata including SHA-256 checksum
        - `<model_filename>`: serialized scikit-learn `Pipeline`
    - The model’s SHA-256 hash is verified against the metadata before loading.
    - The dependency will raise a `RuntimeError` if integrity or type checks fail.
    """

    def __init__(self, model_version: str) -> None:
        """
        Initialize the dependency with a specific model version.

        The model is not loaded until the first dependency invocation.
        """
        # Model version is used to locate the correct model directory under MODELS_PATH
        self._model_version: str = model_version
        # Cache for the loaded model — populated on first dependency call
        self._model: SpamModel | None = None

    def __call__(self) -> SpamModel:
        """
        Return the cached `SpamModel` instance, loading it if necessary.

        On the first call, this method loads and validates the model from disk,
        wraps it in a `SpamModel` container (including metadata), and caches
        the result for reuse across subsequent requests.

        Returns
        -------
        SpamModel
            The loaded spam classification model, ready for inference.
        """
        # Lazily load and cache the model for reuse
        if self._model is None:
            model, metadata = self._load_model()
            self._model = SpamModel(model, metadata)

        return self._model

    def _load_model(self) -> tuple[Pipeline, ModelMetadata]:
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
            MODELS_PATH / self._model_version / "model_metadata.joblib"
        )

        # 1. Load model metadata and validate schema
        model_metadata: ModelMetadata = MODEL_METADATA_ADAPTER.validate_python(
            joblib.load(model_metadata_path)
        )

        model_filename: str = model_metadata["model_filename"]
        model_filepath: Traversable = MODELS_PATH / self._model_version / model_filename

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

        return model, model_metadata

# Global singleton dependency instance bound to the configured model version.
# Imported by FastAPI routes to inject the SpamShield model at runtime.
spam_model_dependency: Final[SpamModelDependency] = SpamModelDependency(
    settings.MODEL_VERSION
)
