import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from .metrics import REQUESTS, REQ_TIME, PAYLOAD_SIZE
import logging

log = logging.getLogger("app")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        payload = await request.body()
        PAYLOAD_SIZE.observe(len(payload))
        if len(payload) > 2_000_000:
            return Response(status_code=413, content="Payload too large")
        log.info(
            "request",
            extra={
                "request_id": req_id,
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else None,
            },
        )
        response = await call_next(request)
        dur = time.perf_counter() - start
        route = request.url.path
        REQ_TIME.labels(route=route, method=request.method).observe(dur)
        REQUESTS.labels(
            route=route, method=request.method, status=str(response.status_code)
        ).inc()
        response.headers["x-request-id"] = req_id
        log.info(
            "response",
            extra={
                "request_id": req_id,
                "path": route,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": round(dur * 1000, 2),
            },
        )
        return response
