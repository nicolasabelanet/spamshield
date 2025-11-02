import json
import time
from typing import Any

import httpx

from spamshield.common import signature


class SpamShieldAPIClient:
    """
    Lightweight client for interacting with the SpamShield API.

    This class wraps the API’s authentication and signing logic, providing
    a convenient interface for sending authenticated requests (e.g. to
    `/predict`). It automatically generates HMAC signatures consistent with
    the server’s expectations, ensuring that every request can be verified
    for integrity and authenticity.

    Examples
    --------
    >>> client = SpamShieldAPIClient("https://api.spamshield.dev", "dev-key", "dev-secret")
    >>> resp = client.predict(["Win a free iPhone!"])
    >>> resp
    {'predictions': [{'label': 'spam', 'prob_spam': 0.94}]}
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        _client: httpx.Client | None = None,
    ) -> None:
        """
        Initialize a new SpamShield API client.

        Parameters
        ----------
        url : str
            Base URL of the SpamShield API (no trailing slash).
        api_key : str
            Public API key used for identification and signature generation.
        api_secret : str
            Secret key used for signing requests.
        _client : httpx.Client | None
            Optional underlying httpx client used to make requests.
        """
        self._client: httpx.Client = _client or httpx.Client()
        self._url: str = url
        self._api_key: str = api_key
        self._api_secret: str = api_key if api_secret is None else api_secret

    def predict(self, texts: list[str]) -> dict[Any, Any]:
        """
        Send a prediction request to the SpamShield API.

        Parameters
        ----------
        texts : list[str]
            One or more text messages to classify as spam or ham.

        Returns
        -------
        dict[Any, Any]
            Parsed JSON response from the `/predict` endpoint, typically in the
            form:
            {
                "predictions": [
                    {"label": "spam", "prob_spam": 0.97},
                    ...
                ]
            }
        """
        return self._request("/predict", "POST", {"texts": texts}).json()

    def _request(
        self, path: str, method: str, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        """
        Internal helper for sending signed HTTP requests to the API.

        This method constructs the full request, including:
          - HMAC signature via `compute_message_signature`
          - timestamp header (`x-timestamp`)
          - authentication headers (`x-api-key`, `x-signature`)
          - JSON-encoded payload

        Parameters
        ----------
        path : str
            API route path (e.g. "/predict").
        method : str
            HTTP method to use ("POST", "GET", etc.).
        payload : dict[str, Any] | None
            JSON-serializable request body (optional).

        Returns
        -------
        httpx.Response
            Raw HTTP response object from the SpamShield API.

        Raises
        ------
        requests.RequestException
            If the underlying HTTP request fails.
        """
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

        return self._client.request(
            method, f"{self._url}{path}", headers=headers, content=content
        )
