from typing import Annotated, Final
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


router: Final[APIRouter] = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    spam_model: Annotated[model.SpamModel, Depends(model.spam_model_dependency)],
):
    """
    Return a basic health status for the API and the loaded spam model.

    This endpoint can be used by load balancers or uptime monitors to verify
    that the service is responsive and the model dependency is initialized.

    Parameters
    ----------
    spam_model : SpamModel
        The loaded spam detection model (injected via dependency).

    Returns
    -------
    HealthResponse
        A simple payload with `"status": "ok"` and the current model version.
    """
    return HealthResponse(status="ok", model_version=spam_model.version)


@router.get("/ready")
def ready(_: Annotated[model.SpamModel, Depends(model.spam_model_dependency)]):
    """
    Return a readiness signal indicating that the spam model is available.

    This endpoint is intended for container orchestration systems (e.g. ECS, Kubernetes)
    to confirm that the application has finished loading its dependencies and is ready
    to serve inference traffic.

    Returns
    -------
    dict
        A JSON object with `"ready": True` once the model is loaded successfully.
    """
    return {"ready": True}


@router.get("/metrics")
def app_metrics():
    """
    Expose runtime metrics for Prometheus to scrape.

    Returns
    -------
    Response
        A plaintext response in Prometheus exposition format, containing
        all metric values (e.g., request counts, latency histograms, etc.)
        registered in `spamshield.api.metrics`.
    """
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
    """
    Predict whether input text samples are spam or ham.

    This endpoint accepts a batch of text strings, runs them through the
    trained spam classification pipeline, and returns predicted labels
    with corresponding spam probabilities.

    The endpoint enforces authentication, rate limiting, and input size constraints.

    Parameters
    ----------
    request : Request
        The FastAPI request context.
    prediction_request : PredictRequest
        The input payload containing one or more text strings to classify.
    spam_model : SpamModel
        The loaded spam detection model used for inference.

    Returns
    -------
    PredictResponse
        A response containing a list of predictions (each containing `label` and 
        `prob_spam` fields).

    Raises
    ------
    HTTPException
        - 413 if the number of texts exceeds `MAX_TEXTS_PER_REQUEST`.
        - 413 if any text exceeds `MAX_TEXT_LEN`.
        - 401 if authentication fails (via dependency).
    """
    # Reject requests with too many text samples.
    if len(prediction_request.texts) > settings.MAX_TEXTS_PER_REQUEST:
        raise HTTPException(status_code=413, detail="Too many items")
        
    # Reject individual text samples that exceed the maximum allowed length.
    if any(len(t) > settings.MAX_TEXT_LEN for t in prediction_request.texts):
        raise HTTPException(status_code=413, detail="Item too large")

    # Measure inference duration with Prometheus histogram context.
    with metrics.INFER_TIME.time():
        preds = spam_model.predict(prediction_request.texts)

    # Convert model predictions into response schema objects.
    return PredictResponse(
        predictions=[Prediction(label=label, prob_spam=prob) for label, prob in preds]
    )
