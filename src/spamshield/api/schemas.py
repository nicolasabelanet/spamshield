from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Schema for `/predict` POST requests.

    Attributes
    ----------
    texts : list[str]
        A list of text messages to classify. Each entry is expected to be
        a single message or email body. The list may contain multiple
        samples for batch inference.
    """

    texts: list[str] = Field(examples=["Free entry in 2 a wkly comp..."])


class Prediction(BaseModel):
    """
    Schema representing an individual spam prediction result.

    Attributes
    ----------
    label : str
        The predicted class label, either `'spam'` or `'ham'`.
    score : float
        The model's estimated probability that the text is spam,
        in the range [0.0, 1.0].
    """

    label: str
    score: float


class PredictResponse(BaseModel):
    """
    Schema for `/predict` responses.

    Attributes
    ----------
    predictions : list[Prediction]
        The list of spam classification results corresponding
        to each input text provided in the request.
    """

    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """
    Schema for `/health` responses.

    Attributes
    ----------
    status : str
        The current health status of the service. Usually `'ok'` when operational.
    model_version : str
        Version string (and hash suffix) of the currently loaded spam model.
    """

    status: str
    model_version: str
