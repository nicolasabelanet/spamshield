from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(examples=["Free entry in 2 a wkly comp..."])


class Prediction(BaseModel):
    label: str
    prob_spam: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_version: str
