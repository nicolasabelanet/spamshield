from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request, Response
import prometheus_client

from spamshield.api import limiting
from spamshield.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
)
from spamshield.api import model 
from spamshield.api.config import settings
from spamshield.api import metrics
from spamshield.api import auth


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    spam_model: Annotated[model.SpamModel, Depends(model.spam_model_dependency)],
):
    return {"status": "ok", "model_version": spam_model.version}


@router.get("/ready")
def ready(_: Annotated[model.SpamModel, Depends(model.spam_model_dependency)]):
    return {"ready": True}


@router.get("/metrics")
def app_metrics():
    return Response(
        prometheus_client.generate_latest(),
        media_type=prometheus_client.CONTENT_TYPE_LATEST,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(auth.require_api_key)],
)
@limiting.limiter.limit(settings.RATE_LIMIT)
def predict(
    request: Request,
    prediction_request: PredictRequest,
    spam_model: Annotated[model.SpamModel, Depends(model.spam_model_dependency)],
):
    if len(prediction_request.texts) > settings.MAX_TEXTS_PER_REQUEST:
        raise HTTPException(status_code=413, detail="Too many items")

    if any(len(t) > settings.MAX_TEXT_LEN for t in prediction_request.texts):
        raise HTTPException(status_code=413, detail="Item too large")

    with metrics.INFER_TIME.time():
        preds = spam_model.predict(prediction_request.texts)

    return {
        "predictions": [
            Prediction(label=label, prob_spam=prob) for label, prob in preds
        ]
    }
