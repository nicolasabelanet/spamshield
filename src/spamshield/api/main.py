from contextlib import asynccontextmanager
import logging
from typing import Final
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from spamshield.api import model
from spamshield.api.config import settings
from spamshield.api import limiting
from spamshield.api.logs import configure_logging
from spamshield.api.middleware import RequestContextMiddleware
from spamshield.api import routes

logger = logging.getLogger(f"{settings.API_TITLE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle the lifespan spamshield API.

    Ensure that logging is configured and the spam ML model is primed
    and ready to make inferences.
    """
    configure_logging(json_logs=settings.LOG_JSON, level=settings.LOG_LEVEL)

    # Prime the spam model dependency
    spam_model: model.SpamModel = app.dependency_overrides.get(
        model.spam_model_dependency, model.spam_model_dependency
    )()

    logger.info(f"Loaded spam model {spam_model.version}")

    yield


app: Final[FastAPI] = FastAPI(
    lifespan=lifespan, title=settings.API_TITLE, version=settings.API_VERSION
)

app.state.limiter = limiting.limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

app.add_middleware(RequestContextMiddleware)
app.add_middleware(SlowAPIMiddleware)

if settings.ENABLE_CORS and settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.CORS_ORIGINS],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

app.include_router(routes.router)
