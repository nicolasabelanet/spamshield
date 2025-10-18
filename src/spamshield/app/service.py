import os
from typing import Any
import joblib
from importlib import resources
import spamshield.app.models
from spamshield.core.signature import sha256_hash_file


MODELS_PATH = resources.files(spamshield.app.models)


def load_model() -> tuple[Any, Any]:
    model_version = os.environ.get(
        "SPAMSHIELD_MODEL_VERSION",
    )

    if model_version is None:
        raise RuntimeError("Cannot load model without SPAMSHIELD_MODEL_VERSION environment variable.")

    model_metadata_path = MODELS_PATH / model_version / "model_metadata.joblib"

    model_metadata = joblib.load(model_metadata_path)
    model_filename = model_metadata["model_filename"]

    model_filepath = MODELS_PATH / model_version / model_filename

    assert model_metadata["model_sha256"] == sha256_hash_file(model_filepath)

    model = joblib.load(model_filepath)

    return model, model_metadata


class SpamModel:
    def __init__(self, model, metadata):
        self._pipeline = model
        self._metadata = metadata
        self._hash = metadata["model_sha256"]
        self._threshold = metadata["threshold"]

    @property
    def version(self) -> str:
        return f"1.0.0+{self._metadata['model_sha256']}"

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        proba = self._pipeline.predict_proba(texts)
        spam_idx = int(self._pipeline.classes_.tolist().index("spam"))
        out = []
        for p in proba:
            prob_spam = float(p[spam_idx])
            label = "spam" if prob_spam >= self._threshold else "ham"
            out.append((label, prob_spam))
        return out


_model_singleton = None

def get_model() -> SpamModel:
    global _model_singleton

    if _model_singleton is None:
        model, metadata = load_model()
        _model_singleton = SpamModel(model, metadata)

    return _model_singleton
