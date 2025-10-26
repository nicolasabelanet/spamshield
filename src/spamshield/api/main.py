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
    Manage the full application lifespan for the SpamShield API.

    Ensures that logging is configured before serving requests
    and that the spam classification model is loaded and ready to
    make inferences.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance passed by the framework.

    Notes
    -----
    - The spam model dependency is explicitly invoked here to "warm up"
      the model before handling requests.
    - Logging is configured once globally using app-level settings.
    """
    # Configure structured logging early in the startup process.
    configure_logging(json_logs=settings.LOG_JSON, level=settings.LOG_LEVEL)

    # Retrieve and invoke the spam model dependency to preload the model into memory.
    # If dependency overrides are provided (e.g., during testing), use those instead.
    spam_model: model.SpamModel = app.dependency_overrides.get(
        model.spam_model_dependency, model.spam_model_dependency
    )()

    logger.info(f"Loaded spam model {spam_model.version}")

    # Yield control back to FastAPI — the app is now fully initialized.
    yield

# Instantiate the FastAPI app with the lifespan manager.
app: Final[FastAPI] = FastAPI(
    lifespan=lifespan, title=settings.API_TITLE, version=settings.API_VERSION
)

# Attach a rate limiter instance to app state for global use.
app.state.limiter = limiting.limiter

# Register global exception handler for rate limiting violations.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

# Add contextual and performance middlewares.
app.add_middleware(RequestContextMiddleware)
app.add_middleware(SlowAPIMiddleware)

# Enable CORS for configured origins (if enabled in settings).
if settings.ENABLE_CORS and settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.CORS_ORIGINS],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

# Register API routes.
app.include_router(routes.router)
