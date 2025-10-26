from prometheus_client import Counter, Histogram

# Count of all HTTP requests served by the API.
# Labeled by route, HTTP method, and response status code.
REQUESTS = Counter(
    "api_requests_total", "Total requests", ["route", "method", "status"]
)

# Distribution of model inference durations (seconds).
# Used to track how long the spam model takes to compute predictions.
INFER_TIME = Histogram(
    "model_inference_seconds",
    "Latency of model inference",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

# Distribution of full API request latency (seconds).
# This measures end-to-end time from request receipt to response sent.
# Labels include route and HTTP method for granular performance insights.
REQ_TIME = Histogram(
    "request_latency_seconds",
    "End-to-end request latency",
    ["route", "method"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

# Distribution of request payload sizes (bytes).
# Helps detect unusually large incoming requests that may impact performance.
PAYLOAD_SIZE = Histogram(
    "request_payload_bytes",
    "Payload size in bytes",
    buckets=(128, 512, 1024, 4096, 16384, 65536),
)
