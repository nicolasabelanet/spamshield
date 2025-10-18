from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "api_requests_total", "Total requests", ["route", "method", "status"]
)
INFER_TIME = Histogram(
    "model_inference_seconds",
    "Latency of model inference",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
REQ_TIME = Histogram(
    "request_latency_seconds",
    "End-to-end request latency",
    ["route", "method"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
PAYLOAD_SIZE = Histogram(
    "request_payload_bytes",
    "Payload size in bytes",
    buckets=(128, 512, 1024, 4096, 16384, 65536),
)
