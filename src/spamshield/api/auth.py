import hmac
import time
from fastapi import Header, HTTPException, Request

from spamshield.core import signature
from spamshield.api.config import settings


def _compare_digest(a: str, b: str) -> bool:
    try:
        return hmac.compare_digest(a, b)
    except Exception:
        return False


def _find_first_matching_api_key(x_api_key: str | None) -> str | None:
    if x_api_key is None:
        return None

    valid_keys = (
        key for key in (settings.API_KEY, settings.SECONDARY_API_KEY) if len(key) > 0
    )

    for key in valid_keys:
        if _compare_digest(x_api_key, key):
            return key

    return None


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    x_timestamp: str | None = Header(default=None),
    x_signature: str | None = Header(default=None),
):
    print("KEY", x_api_key)
    print("TIMESTAMP", x_timestamp)
    matching_key = _find_first_matching_api_key(x_api_key)

    if not x_api_key or not matching_key:
        print("no api key")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Signature signing not required.
    if not settings.REQUIRE_HMAC:
        return

    # Ensure that the signature and timestamp are present.
    if not (x_timestamp and x_signature):
        print("missing timestamp")
        raise HTTPException(status_code=401, detail="Unauthorized")
    now = int(time.time())

    try:
        timestamp = int(x_timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if abs(now - timestamp) > 300:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Build message signature
    secret: str = matching_key
    method: str = request.method.upper()
    path: str = request.url.path

    content: bytes = await request.body()

    computed_signature: str = signature.compute_message_signature(
        method, path, timestamp, content, x_api_key, secret
    )

    if not _compare_digest(computed_signature, x_signature):
        raise HTTPException(status_code=401, detail="Unauthorized")
