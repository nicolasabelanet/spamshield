import time
import json
from typing import Any
import requests

from spamshield.app.schemas import PredictResponse
from spamshield.core import signature


class SpamShieldAPIClient:
    def __init__(self, url: str, api_key: str, api_secret: str | None = None) -> None:
        self._url: str = url
        self._api_key: str = api_key
        self._api_secret: str = api_key if api_secret is None else api_secret

    def predict(self, texts: list[str]) -> PredictResponse:
        return PredictResponse.model_validate_json(
            self._request("/predict", "POST", {"texts": texts}).content
        )

    def _request(
        self, path: str, method: str, payload: dict[str, Any] | None = None
    ) -> requests.Response:
        timestamp = int(time.time())

        content = json.dumps(payload).encode()

        sig = signature.compute_message_signature(
            method,
            path,
            timestamp,
            content,
            self._api_key,
            self._api_secret,
        )

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "x-timestamp": str(timestamp),
            "x-signature": sig,
        }

        return requests.request(
            method, f"{self._url}{path}", headers=headers, data=content
        )
