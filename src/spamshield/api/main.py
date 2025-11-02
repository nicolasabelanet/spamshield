from contextlib import asynccontextmanager
import logging
from typing import Final
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from spamshield.api import config, model, static_config
from spamshield.api import metrics
from spamshield.api.logs import configure_logging
from spamshield.api.middleware import RequestContextMiddleware
from spamshield.api import routes

logger = logging.getLogger(f"{static_config.API_TITLE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for the SpamShield API.

    Initializes and tears down global runtime components such as logging,
    model loading, and observability dependencies.

    This function is executed once during application startup and shutdown
    by FastAPI, ensuring that all shared resources are prepared before
    serving requests.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Notes
    -----
    - Settings, metrics, and model dependencies respect any test-time
      dependency overrides defined via `app.dependency_overrides`.
    - The spam model is explicitly loaded ("warmed up") here so that
      the first inference request does not incur load-time latency.
    - Structured logging is configured before any request handling.
    """
    # Retrieve and invoke the settings dependency to preload the settings.
    # If dependency overrides are provided (e.g. during testing), use those instead.
    settings: config.Settings = app.dependency_overrides.get(
        config.get_settings, config.get_settings
    )()
    logger.info("Loaded settings")

    app.state.settings = settings

    # Configure structured logging early in the startup process.
    configure_logging(json_logs=settings.LOG_JSON, level=settings.LOG_LEVEL)

    # Enable CORS for configured origins (if enabled in settings).
    if settings.ENABLE_CORS and settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(o) for o in settings.CORS_ORIGINS],
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["*"],
        )

    # Retrieve and invoke the metrics manager dependency to preload the metrics
    # manager. If dependency overrides are provided (e.g. during testing), use those instead.
    metrics_manager: metrics.MetricsManager = app.dependency_overrides.get(
        metrics.get_metrics_manager, metrics.get_metrics_manager
    )()
    logger.info("Loaded metrics manager")

    # Add the metrics manager to app state for use in middleware.
    app.state.metrics_manager = metrics_manager

    model_loader: model.ModelLoader = app.dependency_overrides.get(
        model.get_model_loader, model.get_model_loader
    )()
    spam_model = model_loader(settings.MODEL_VERSION)
    app.state.model = spam_model

    logger.info(f"Loaded spam model {spam_model.version}")

    # Yield control back to FastAPI — the app is now fully initialized.
    yield


# Instantiate the FastAPI app with the lifespan manager.
app: Final[FastAPI] = FastAPI(
    lifespan=lifespan, title=static_config.API_TITLE, version=static_config.API_VERSION
)

# Add contextual and performance middlewares.
app.add_middleware(RequestContextMiddleware)

# Register API routes.
app.include_router(routes.router)
