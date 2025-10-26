from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import Response

from spamshield.api.auth import require_api_key
from spamshield.api.config import settings
from spamshield.api.logs import configure_logging
from spamshield.api.metrics import INFER_TIME
from spamshield.api.middleware import RequestContextMiddleware
from spamshield.api.schemas import (
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from spamshield.api.service import get_model

configure_logging(json_logs=settings.LOG_JSON, level=settings.LOG_LEVEL)

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)
app.add_middleware(RequestContextMiddleware)

limiter = Limiter(get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore


app.add_middleware(SlowAPIMiddleware)


if settings.ENABLE_CORS and settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.CORS_ORIGINS],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )


@app.get("/health", response_model=HealthResponse)
def health():
    m = get_model()
    return {"status": "ok", "model_version": m.version}


@app.get("/ready")
def ready():
    _ = get_model()
    return {"ready": True}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)]
)
@limiter.limit(settings.RATE_LIMIT)
def predict(request: Request, prediction_request: PredictRequest):
    if len(prediction_request.texts) > settings.MAX_TEXTS_PER_REQUEST:
        raise HTTPException(status_code=413, detail="Too many items")

    if any(len(t) > settings.MAX_TEXT_LEN for t in prediction_request.texts):
        raise HTTPException(status_code=413, detail="Item too large")

    spam_model = get_model()

    with INFER_TIME.time():
        preds = spam_model.predict(prediction_request.texts)

    return {
        "predictions": [
            Prediction(label=label, prob_spam=prob) for label, prob in preds
        ]
    }
