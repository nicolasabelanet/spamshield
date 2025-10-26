from typing import TypedDict


class ModelMetadata(TypedDict):
    version: str
    model_sha256: str
    model_filename: str
    threshold: float
