"""Small HTTP trust-boundary checks shared by local clients."""

from urllib.parse import urlsplit


def validated_backend_url(value: str) -> str:
    """Accept only explicit HTTP(S) API origins without embedded credentials."""
    normalized = value.rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("backend URL must be an HTTP(S) origin without credentials")
    return normalized
