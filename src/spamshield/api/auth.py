import hmac
import time
from typing import Annotated, Final

from fastapi import Depends, Header, HTTPException, Request

from spamshield.api import config
from spamshield.common import signature

ALLOWED_TIME_SKEW_SEC: Final[int] = 300


def _compare_digest(a: str, b: str) -> bool:
    """
    Perform a constant-time string comparison to prevent timing attacks.

    Returns True if the two digests match exactly, False otherwise.
    Falls back safely if `hmac.compare_digest` raises an exception.
    """
    try:
        return hmac.compare_digest(a, b)
    except Exception:
        return False


def _find_first_matching_api_key(
    x_api_key: str | None, valid_key_hashes: tuple[str, ...]
) -> str | None:
    """
    Validate the provided API key against the configured keys.

    Returns the matching key if valid, or None if the key is missing
    or does not match.
    """
    if x_api_key is None:
        return None

    candidate_hash: str = signature.hash_api_key(x_api_key)

    # Filter out unset or empty keys from configuration
    valid_hashes = (key for key in valid_key_hashes if len(key) > 0)

    # Compare each valid key in constant time
    for hash in valid_hashes:
        if _compare_digest(candidate_hash, hash):
            return hash

    # No valid key found
    return None


async def require_api_key(
    request: Request,
    settings: Annotated[config.Settings, Depends(config.get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
    x_timestamp: Annotated[str | None, Header()] = None,
    x_signature: Annotated[str | None, Header()] = None,
):
    """
    FastAPI dependency enforcing API key + HMAC authentication.

    1. Verifies that the provided API key matches either the primary or
       secondary configured key.
    2. If HMAC verification is required (controlled by `settings.REQUIRE_HMAC`),
       validates the timestamp and cryptographic signature.
    3. Rejects requests with expired timestamps (>5 min skew), invalid formats,
       or mismatched HMAC signatures.

    Raises
    ------
    HTTPException(401)
        If the authentication fails at any stage.
    """
    # Verify that the API key exists and matches a configured one
    matching_key: str | None = _find_first_matching_api_key(
        x_api_key, (settings.API_KEY_HASH, settings.SECONDARY_API_KEY_HASH)
    )

    if not x_api_key or not matching_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # If HMAC signing is disabled, only the API key check is required
    if not settings.REQUIRE_HMAC:
        return

    # Ensure required headers are present for HMAC validation
    if not (x_timestamp and x_signature):
        raise HTTPException(status_code=401, detail="Unauthorized")

    now: int = int(time.time())

    # Parse timestamp and validate its freshness (max ±5 minutes)
    try:
        timestamp = int(x_timestamp)

    except ValueError:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if abs(now - timestamp) > ALLOWED_TIME_SKEW_SEC:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Compute HMAC signature for request body and metadata
    method: str = request.method.upper()
    path: str = request.url.path

    content: bytes = await request.body()

    computed_signature: str = signature.compute_message_signature(
        method, path, timestamp, content, x_api_key, settings.API_SECRET
    )

    # Reject if computed signature does not match provided one
    if not _compare_digest(computed_signature, x_signature):
        raise HTTPException(status_code=401, detail="Unauthorized")
