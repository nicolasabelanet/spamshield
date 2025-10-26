import hashlib
import hmac
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path



def sha256_hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_message_signature(
    method: str, path: str, timestamp: int, content: bytes, api_key: str, secret: str
) -> str:
    content_sha256 = hashlib.sha256(content).hexdigest()

    message = f"{method}\n{path}\n{timestamp}\n{content_sha256}\n{api_key}"

    return hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).hexdigest()

