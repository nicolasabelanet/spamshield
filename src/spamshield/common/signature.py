import hashlib
import hmac
from importlib.resources.abc import Traversable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def hash_api_key(raw_key: str) -> str:
    """Compute a deterministic SHA-256 hash of an API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def sha256_hash_file(path: Path | Traversable) -> str:
    """
    Compute the SHA-256 hash of a file’s contents.

    This function reads the file in fixed-size chunks (8 KB) to efficiently
    compute its SHA-256 digest without loading the entire file into memory.

    Parameters
    ----------
    path : Path | Traversable
        The file path or traversable object representing the file to hash.

    Returns
    -------
    str
        The lowercase hexadecimal SHA-256 digest of the file’s contents.

    Notes
    -----
    - Uses incremental updates (`hash.update(chunk)`) for memory efficiency.
    - Suitable for verifying file integrity or computing deterministic
      identifiers for model artifacts and datasets.
    """

    hash = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash.update(chunk)

    return hash.hexdigest()


def compute_message_signature(
    method: str, path: str, timestamp: int, content: bytes, api_key: str, secret: str
) -> str:
    """
    Compute an HMAC-SHA256 signature for a request message.

    The signature authenticates a request by hashing the concatenation of the
    HTTP method, path, timestamp, content hash, and API key using the shared
    secret. This ensures both integrity and authenticity of the request body
    and metadata.

    Parameters
    ----------
    method : str
        The HTTP method (e.g. "POST", "GET") used in the request.
    path : str
        The request path (e.g. "/predict").
    timestamp : int
        The UNIX timestamp (in seconds) when the request was created.
    content : bytes
        The raw request body to hash (use b"" for GET requests or empty bodies).
    api_key : str
        The client’s public API key (used for identification).
    secret : str
        The client’s private secret key (used to compute the HMAC).

    Returns
    -------
    str
        A lowercase hexadecimal HMAC-SHA256 digest representing the computed
        message signature.

    Notes
    -----
    - The message is constructed as:

      ```
      {method}
      {path}
      {timestamp}
      {SHA256(content)}
      {api_key}
      ```

    - The resulting signature can be compared against a client-provided
      signature header to verify authenticity.
    """

    content_sha256 = hashlib.sha256(content).hexdigest()

    message = f"{method}\n{path}\n{timestamp}\n{content_sha256}\n{api_key}"

    return hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).hexdigest()
