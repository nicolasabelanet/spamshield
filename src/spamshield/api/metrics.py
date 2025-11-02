from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
import functools


class MetricsManager:
    """
    Centralized Prometheus metrics manager for the SpamShield API.

    This class encapsulates all Prometheus metric definitions used by the service,
    including counters for request tracking and histograms for latency and payload
    size distributions. It provides a single interface to initialize, record, and
    render metrics, ensuring consistent labeling and isolated registry management.

    Attributes
    ----------
    registry : CollectorRegistry
        Prometheus registry that holds all defined metrics. If not provided,
        a new isolated registry is created.
    requests : Counter
        Counter tracking total HTTP requests served by the API, labeled by
        route, HTTP method, and response status code.
    infer_time : Histogram
        Histogram measuring latency of model inference operations, in seconds.
    request_time : Histogram
        Histogram measuring end-to-end API request latency, labeled by route
        and method.
    payload_size : Histogram
        Histogram tracking distribution of incoming request payload sizes, in bytes.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        """
        Initialize and register all metrics with the given Prometheus registry.

        Parameters
        ----------
        registry : CollectorRegistry | None
            Custom Prometheus registry to register metrics on. If omitted,
            a new `CollectorRegistry` is created for isolation, which is useful
            in testing environments.
        """
        self.registry: CollectorRegistry = registry or CollectorRegistry()

        # Count of all HTTP requests served by the API.
        # Labeled by route, HTTP method, and response status code.
        self.requests = Counter(
            "api_requests_total",
            "Total requests",
            ["route", "method", "status"],
            registry=self.registry,
        )

        # Distribution of model inference durations (seconds).
        # Used to track how long the spam model takes to compute predictions.
        self.infer_time: Histogram = Histogram(
            "model_inference_seconds",
            "Latency of model inference",
            buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )

        # Distribution of full API request latency (seconds).
        # This measures end-to-end time from request receipt to response sent.
        # Labels include route and HTTP method for granular performance insights.
        self.request_time: Histogram = Histogram(
            "request_latency_seconds",
            "End-to-end request latency",
            ["route", "method"],
            buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )

        # Distribution of request payload sizes (bytes).
        # Helps detect unusually large incoming requests that may impact performance.
        self.payload_size: Histogram = Histogram(
            "request_payload_bytes",
            "Payload size in bytes",
            buckets=(128, 512, 1024, 4096, 16384, 65536),
            registry=self.registry,
        )

    def render(self) -> bytes:
        """
        Render all registered metrics in Prometheus' text exposition format.

        Returns
        -------
        bytes
            The encoded metrics output suitable for HTTP response content
            in a `/metrics` endpoint.
        """
        return generate_latest(self.registry)


@functools.cache
def get_metrics_manager() -> MetricsManager:
    """
    Retrieve a cached global instance of the MetricsManager.

    This function provides a lightweight singleton for accessing metrics
    throughout the application. The instance is memoized using
    `functools.cache`, ensuring that all routes share the same
    Prometheus registry unless explicitly overridden.

    Returns
    -------
    MetricsManager
        The cached global metrics manager instance.
    """
    return MetricsManager()
