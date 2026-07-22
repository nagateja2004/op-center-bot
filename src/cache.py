"""Small Redis caches for stable RAG work; failures never affect answers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from redis import Redis

_client: Redis | None = None


def configure_cache(redis_url: str) -> None:
    global _client
    _client = Redis.from_url(redis_url, decode_responses=True, socket_timeout=1)


def close_cache() -> None:
    global _client
    if _client:
        _client.close()
    _client = None


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def get(namespace: str, payload: Any) -> Any | None:
    if not _client:
        return None
    try:
        raw = _client.get(_key(namespace, payload))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def set(namespace: str, payload: Any, value: Any, ttl: int = 900) -> None:
    if _client:
        try:
            _client.setex(_key(namespace, payload), ttl, json.dumps(value))
        except Exception:
            pass


def _key(namespace: str, payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return f"opcenter:cache:{namespace}:{hashlib.sha256(data.encode()).hexdigest()}"
