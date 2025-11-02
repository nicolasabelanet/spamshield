from collections.abc import Generator
import json
from time import time
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry
import pytest
from fastapi.testclient import TestClient

from spamshield.api import config, metrics, model
from spamshield.api.main import app
from spamshield.client.client import SpamShieldAPIClient
from spamshield.common import signature
from spamshield.common.signature import hash_api_key

# Constants used accross tests
TEST_API_KEY = "test-key"
TEST_SECRET = "test-secret"

HEADERS = {"x-api-key": TEST_API_KEY}

MODEL_VERSION = "v1.0.0"
MODEL_HASH = "b5bb9d8014a0f9b1d61e21e796d78dccdf1352f23cd32812f4850b878ae4944c"


class FakeModel:
    """Lightweight stub that mimics the real SpamShield model."""

    def __init__(self, model_version: str, sha256: str) -> None:
        self._version = model_version
        self._hash = sha256

    @property
    def version(self) -> str:
        """Expose the same composite version string as the real model."""
        return f"{self._version}+{self._hash}"

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        """Return a deterministic spam/ham label based on keyword matching."""
        results: list[tuple[str, float]] = []

        for text in texts:
            if "free cash" in text.lower():
                results.append(("spam", 0.95))
            else:
                results.append(("ham", 0.05))

        return results


@pytest.fixture()
def fake_model() -> FakeModel:
    """Provide a deterministic, in-memory fake model instance."""
    return FakeModel(MODEL_VERSION, MODEL_HASH)


@pytest.fixture()
def metrics_manager() -> metrics.MetricsManager:
    """Use an isolated CollectorRegistry per test for clean metric state."""
    return metrics.MetricsManager(registry=CollectorRegistry())


@pytest.fixture()
def settings() -> config.Settings:
    """Create test settings with HMAC disabled for default tests."""
    return config.Settings(
        MODEL_VERSION="v1.0.0",
        API_KEY_HASH=hash_api_key(TEST_API_KEY),
        API_SECRET=TEST_SECRET,
        SECONDARY_API_KEY_HASH="",
        REQUIRE_HMAC=False,
        ENABLE_CORS=False,
        LOG_JSON=True,
    )


@pytest.fixture()
def secure_settings() -> config.Settings:
    """Create test settings with HMAC disabled for signature based tests."""
    return config.Settings(
        MODEL_VERSION="v1.0.0",
        API_KEY_HASH=hash_api_key(TEST_API_KEY),
        API_SECRET=TEST_SECRET,
        SECONDARY_API_KEY_HASH="",
        REQUIRE_HMAC=True,
        ENABLE_CORS=False,
        LOG_JSON=True,
    )


@pytest.fixture
def client(
    fake_model: FakeModel,
    metrics_manager: metrics.MetricsManager,
    settings: config.Settings,
) -> Generator[TestClient]:
    """
    Provide a FastAPI TestClient with all core dependencies overridden.

    - Injects a fake model loader and in-memory metrics manager
    - Runs the app lifespan context to trigger startup hooks
    """
    app.dependency_overrides[metrics.get_metrics_manager] = lambda: metrics_manager
    app.dependency_overrides[model.get_model_loader] = lambda: lambda _: fake_model
    app.dependency_overrides[config.get_settings] = lambda: settings
    client = TestClient(app)

    # Use the client as a context to run the lifecycle function.
    with client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def client_with_hmac(
    fake_model: FakeModel,
    metrics_manager: metrics.MetricsManager,
    secure_settings: config.Settings,
) -> Generator[TestClient]:
    """Same as `client` but with HMAC authentication required."""
    app.dependency_overrides[metrics.get_metrics_manager] = lambda: metrics_manager
    app.dependency_overrides[model.get_model_loader] = lambda: lambda _: fake_model
    app.dependency_overrides[config.get_settings] = lambda: secure_settings

    client = TestClient(app)
    with client:
        yield client

    app.dependency_overrides.clear()


def test_lifespan_initializes_runtime_state(
    client: TestClient,
    settings: config.Settings,
    fake_model: FakeModel,
    metrics_manager: metrics.MetricsManager,
):
    """Ensure FastAPI startup correctly loads settings, model, and metrics."""
    app: FastAPI = client.app  # type: ignore

    assert app.state.metrics_manager is metrics_manager
    assert app.state.settings is settings
    assert app.state.model is fake_model
    assert app.state.model.version == fake_model.version

    # metrics manager loaded
    assert hasattr(app.state, "metrics_manager")


def test_health_reports_model_version(client: TestClient, fake_model: FakeModel):
    """Verify /health returns model metadata."""
    resp = client.get("/health", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["model_version"] == fake_model.version


def test_predict_returns_expected_label_and_score(client: TestClient):
    """Smoke test: single spam text yields spam label and fixed score."""
    res = client.post(
        "/predict", json={"texts": ["Click here for FREE CASH now!"]}, headers=HEADERS
    )
    assert res.status_code == 200

    predictions = res.json()["predictions"]

    assert len(predictions) == 1

    prediction = predictions[0]
    assert prediction["label"] == "spam"
    assert prediction["score"] == 0.95


def test_predict_classifies_spam_vs_ham(client: TestClient):
    """Multi-text request: ensures both spam and ham are classified correctly."""
    body = {"texts": ["hello friend", "Click here for FREE CASH now!"]}
    res = client.post("/predict", json=body, headers=HEADERS)
    assert res.status_code == 200

    results = res.json()["predictions"]
    ham_result = results[0]
    spam_result = results[1]

    assert ham_result["label"] == "ham"
    assert spam_result["label"] == "spam"
    assert spam_result["score"] > ham_result["score"]


def test_client_with_signed_hmac_request(client_with_hmac: TestClient) -> None:
    """Full HMAC client request should succeed."""
    spamshield_client = SpamShieldAPIClient(
        str(client_with_hmac.base_url), TEST_API_KEY, TEST_SECRET, client_with_hmac
    )

    response = spamshield_client.predict(["some text"])
    assert "predictions" in response


def test_handles_missing_signature_hmac_request(client_with_hmac: TestClient):
    """Reject requests missing the HMAC signature header."""
    timestamp = int(time())

    res = client_with_hmac.post(
        "/predict",
        json={"texts": ["some payload"]},
        headers={
            **HEADERS,
            "x-timestamp": str(timestamp),
        },
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Unauthorized"


def test_handles_missing_timestamp_hmac_request(client_with_hmac: TestClient):
    """Reject requests missing timestamp header."""
    timestamp = int(time())

    payload = json.dumps({"texts": ["Click here for free cash now!"]}).encode()
    message_signature = signature.compute_message_signature(
        method="POST",
        path="/predict",
        timestamp=timestamp,
        content=payload,
        api_key=TEST_API_KEY,
        secret=TEST_SECRET,
    )

    res = client_with_hmac.post(
        "/predict",
        content=payload,
        headers={
            **HEADERS,
            "x-signature": message_signature,
        },
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Unauthorized"


def test_handles_incorrectly_signed_hmac_request(client_with_hmac: TestClient):
    """Reject requests where signature is computed for a different payload."""
    timestamp = int(time())

    payload = json.dumps({"texts": ["Click here for free cash now!"]}).encode()
    message_signature = signature.compute_message_signature(
        method="POST",
        path="/predict",
        timestamp=timestamp,
        content=payload,
        api_key=TEST_API_KEY,
        secret=TEST_SECRET,
    )

    res = client_with_hmac.post(
        "/predict",
        json={"texts": ["some other payload"]},
        headers={
            **HEADERS,
            "x-timestamp": str(timestamp),
            "x-signature": message_signature,
        },
    )
    assert res.status_code == 401
    assert res.json()["detail"] == "Unauthorized"


def test_predict_handles_empty_batch(client: TestClient):
    """Empty input list should succeed with empty predictions."""
    res = client.post("/predict", json={"texts": []}, headers=HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert data["predictions"] == []


def test_predict_handles_maximum_texts(client: TestClient, settings: config.Settings):
    """Reject requests exceeding MAX_TEXTS_PER_REQUEST."""
    texts: list[str] = ["some text"] * (settings.MAX_TEXTS_PER_REQUEST + 1)
    res = client.post("/predict", json={"texts": texts}, headers=HEADERS)
    assert res.status_code == 413
    data = res.json()
    assert data["detail"] == "Too many items"


def test_predict_handles_maximum_text_length(
    client: TestClient, settings: config.Settings
):
    """Reject individual text inputs exceeding MAX_TEXT_LEN."""
    text: str = "x" * (settings.MAX_TEXT_LEN + 1)
    res = client.post("/predict", json={"texts": [text]}, headers=HEADERS)
    assert res.status_code == 413
    data = res.json()
    assert data["detail"] == "Item too large"


def test_predict_rejects_incorrect_api_key(client: TestClient):
    """Unauthorized if an incorrect API key header is provided."""
    res = client.post("/predict", json={}, headers={"x-api-key": "garbage"})
    assert res.status_code == 401
    body = res.json()
    assert body["detail"] == "Unauthorized"


def test_predict_rejects_missing_api_key(client: TestClient):
    """Unauthorized if no API key header is provided."""
    res = client.post("/predict", json={})
    assert res.status_code == 401
    body = res.json()
    assert body["detail"] == "Unauthorized"


def test_predict_rejects_missing_texts(client: TestClient):
    """422 validation error if required field is missing."""
    res = client.post("/predict", json={}, headers=HEADERS)
    assert res.status_code == 422  # FastAPI validation
    body = res.json()
    assert body["detail"][0]["loc"][-1] == "texts"


def test_metrics_reflect_prediction_call(
    client: TestClient, metrics_manager: metrics.MetricsManager
):
    """
    Ensure that model inference updates all expected Prometheus metrics.

    Verifies counters and histograms for:
    - Total request count
    - Model inference latency
    - End-to-end request latency
    - Payload size distribution
    """
    # trigger one request to /predict
    res = client.post(
        "/predict", json={"texts": ["Click here for FREE CASH now!"]}, headers=HEADERS
    )
    assert res.status_code == 200

    # Grab the registry directly from the metrics_manager fixture
    registry: CollectorRegistry = metrics_manager.registry

    # Collect metric samples
    metrics_samples = {}
    for metric_family in registry.collect():
        for sample in metric_family.samples:
            metrics_samples[sample.name] = sample

    assert metrics_samples["api_requests_total"].value == 1
    assert metrics_samples["model_inference_seconds_count"].value == 1
    assert metrics_samples["request_latency_seconds_count"].value == 1
    assert metrics_samples["request_payload_bytes_count"].value == 1

    # Payload histogram sum should match request body size
    assert metrics_samples["request_payload_bytes_sum"].value == len(
        res.request.content
    )


def test_metrics_endpoint(client: TestClient):
    """Verify /metrics returns Prometheus-compatible output."""
    resp = client.get("/metrics", headers=HEADERS)
    assert resp.status_code == 200

    # Prometheus expects this content type exactly
    assert resp.headers["content-type"] == CONTENT_TYPE_LATEST

    body = resp.text.strip()
    assert len(body) > 0
