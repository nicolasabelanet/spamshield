import logging
import time
from typing import Final
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from spamshield.api import metrics

logger = logging.getLogger("app")

REQUEST_ID_HEADER: Final[str] = "x-request-id"
MAX_PAYLOAD_BYTES: Final[int] = 1_000_000  # ~1MB


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds structured request context, logging, and performance metrics
    for every incoming HTTP request to the SpamShield API.

    This middleware provides consistent observability across all requests by:
      - Assigning and propagating a unique `x-request-id` for distributed tracing.
      - Measuring and recording request latency, payload size, and response status codes.
      - Enforcing a configurable payload size limit to prevent large or malicious uploads.
      - Emitting structured JSON logs for both requests and responses, including timing and status info.

    Metrics are recorded using the `prometheus_client` library, following Prometheus conventions
    (e.g., durations in seconds, counters for request totals). These metrics are exposed through
    the `/metrics` endpoint and can be scraped by Prometheus for monitoring and alerting.

    Logging is handled via Python’s standard `logging` library, and each log entry includes
    contextual metadata (request ID, route, client IP, method, and response duration in ms)
    for easier correlation across systems.

    Notes
    -----
    - Because this middleware reads the request body, any later middleware or route handlers that
      also read the body must use `request.stream()` or reattach the body explicitly.
    - The middleware adds an `x-request-id` header to all responses, even if the client didn’t provide one.
    - Latency metrics use seconds for Prometheus recording but milliseconds in logs for readability.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Intercept and process incoming HTTP requests for logging, metrics, and request tracing.

        This middleware performs several key functions before and after the request is handled
        by the downstream FastAPI route:

        1. Request identification - Assigns a unique `x-request-id` (or uses the provided one)
           to enable end-to-end traceability across logs and responses.
        2. Payload inspection - Reads and records the request body size, enforcing a maximum
           limit to protect the model from excessively large requests.
        3. Request logging - Emits structured logs with request path, method, client IP, and payload size.
        4. Metrics collection - Records request latency, total requests, and payload size in
           Prometheus-style metrics (`REQ_TIME`, `REQUESTS`, `PAYLOAD_SIZE`).
        5. Response logging - Logs response status and total duration in milliseconds, and adds
           the request ID to the response headers for correlation.

        Parameters
        ----------
        request : Request
            The incoming FastAPI request object.
        call_next : Callable[[Request], Awaitable[Response]]
            The next ASGI handler in the middleware chain (typically the route handler).

        Returns
        -------
        Response
            The HTTP response returned by the downstream handler, with the `x-request-id` header added.

        Raises
        ------
        Response(status_code=413)
            If the request payload exceeds the maximum allowed size.

        Notes
        -----
        - Reading `request.body()` consumes the stream; avoid adding other middlewares that depend
          on the unconsumed body unless you reattach it.
        - Request duration is recorded in **seconds** for metrics and in **milliseconds** for logs.
        - The middleware ensures structured, correlated logging for both requests and responses.
        """

        # Use the metrics manager that was attached to app state in the lifecycle function.
        metrics: metrics.MetricsManager | None = getattr(  # type: ignore
            request.app.state, "metrics_manager", None
        )

        if metrics is None:
            raise RuntimeError("Could not find metrics manager in app state.")

        # Use provided request ID if present (for trace continuity),
        # otherwise generate a new one.
        request_id: str = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))

        # Capture start time for latency measurement.
        request_start: float = time.perf_counter()

        # Read request body once so we can both size-check and (optionally) log/metric it.
        # NOTE: This will consume the body; if you later need the body downstream,
        # you'll need to re-attach it to the request's stream.
        payload: bytes = await request.body()
        payload_size: int = len(payload)

        # Record payload size metric.
        metrics.payload_size.observe(payload_size)

        # Enforce max body size to protect the model and avoid wasting inference cycles.
        if len(payload) > MAX_PAYLOAD_BYTES:
            return Response(status_code=413, content="Payload too large.")

        # Log inbound request metadata.
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else None,
            },
        )

        # Call the downstream handler (actual endpoint).
        response: Response = await call_next(request)

        request_duration_sec: float = time.perf_counter() - request_start
        request_duration_ms: float = round(request_duration_sec * 1000, 2)

        request_path: str = request.url.path

        # Emit Prometheus metrics.
        metrics.request_time.labels(route=request_path, method=request.method).observe(
            request_duration_sec
        )
        metrics.requests.labels(
            route=request_path, method=request.method, status=str(response.status_code)
        ).inc()

        # Propagate the request ID back to the caller for tracing/debug.
        response.headers[REQUEST_ID_HEADER] = request_id

        # Log response metadata.
        logger.info(
            "response",
            extra={
                "request_id": request_id,
                "path": request_path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": request_duration_ms,
            },
        )
        return response
