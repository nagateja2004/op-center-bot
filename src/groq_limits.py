"""Redis-backed, per-model asynchronous admission control for Groq requests."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import random
import re
import time
from typing import Any, Awaitable, Callable, Iterator, TypeVar
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError
from src.observability import gauge

from src.config import Settings, settings


ResultT = TypeVar("ResultT")
_request_id: ContextVar[str | None] = ContextVar("groq_request_id", default=None)


class GroqLimitError(RuntimeError):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(kind)


@dataclass(frozen=True, slots=True)
class ModelLimits:
    concurrency: int
    requests_per_minute: int
    tokens_per_minute: int


_ADMIT = """
local active, request_key, token_key = KEYS[1], KEYS[2], KEYS[3]
local now, lease, concurrency, requests, tokens = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4]), tonumber(ARGV[5])
local token_cost, slot = tonumber(ARGV[6]), ARGV[7]
redis.call('ZREMRANGEBYSCORE', active, '-inf', now)
if redis.call('ZCARD', active) >= concurrency then return {0, 0} end
local current_requests = tonumber(redis.call('GET', request_key) or '0')
local current_tokens = tonumber(redis.call('GET', token_key) or '0')
if current_requests + 1 > requests or current_tokens + token_cost > tokens then
  return {0, math.max(redis.call('TTL', request_key), redis.call('TTL', token_key), 1)}
end
redis.call('INCR', request_key)
redis.call('EXPIRE', request_key, 60)
redis.call('INCRBY', token_key, token_cost)
redis.call('EXPIRE', token_key, 60)
redis.call('ZADD', active, now + lease, slot)
redis.call('EXPIRE', active, math.ceil(lease / 1000) + 5)
return {1, 0}
"""

_RELEASE = "redis.call('ZREM', KEYS[1], ARGV[1]); return 1"


def _model_key(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.casefold()).strip("_")


class RedisGroqLimiter:
    def __init__(self, client: Redis, config: Settings = settings) -> None:
        self.client = client
        self.config = config

    @classmethod
    async def connect(cls, config: Settings = settings) -> "RedisGroqLimiter":
        client = Redis.from_url(config.redis_url, decode_responses=True, socket_timeout=2)
        await client.ping()
        return cls(client, config)

    async def close(self) -> None:
        await self.client.aclose()

    def limits_for(self, model: str) -> ModelLimits:
        prefix = f"GROQ_MODEL_{_model_key(model).upper()}"
        return ModelLimits(
            concurrency=int(os.getenv(f"{prefix}_MAX_CONCURRENCY", self.config.groq_model_max_concurrency)),
            requests_per_minute=int(os.getenv(f"{prefix}_REQUESTS_PER_MINUTE", self.config.groq_model_requests_per_minute)),
            tokens_per_minute=int(os.getenv(f"{prefix}_TOKENS_PER_MINUTE", self.config.groq_model_tokens_per_minute)),
        )

    async def set_status(self, request_id: str | None, status: str, **details: Any) -> None:
        if not request_id:
            return
        try:
            key = f"opcenter:groq:request:{request_id}"
            lock = self.client.lock(f"{key}:lock", timeout=3, blocking_timeout=0.1)
            if not await lock.acquire(blocking=True):
                return
            try:
                values = {"status": status, "updated_at": f"{time.time():.3f}"}
                values.update({name: str(value) for name, value in details.items()})
                await self.client.hset(key, mapping=values)
                await self.client.expire(key, self.config.groq_request_status_ttl_seconds)
            finally:
                try:
                    await lock.release()
                except RedisError:
                    pass
        except RedisError:
            return

    async def run(
        self, model: str, token_cost: int, operation: Callable[[], Awaitable[ResultT]]
    ) -> ResultT:
        limits = self.limits_for(model)
        request_id = _request_id.get() or uuid4().hex
        model_key = _model_key(model)
        queue_key = f"opcenter:groq:queue:{model_key}"
        try:
            queue_depth = int(await self.client.incr(queue_key))
            gauge("opcenter_groq_queue_depth", queue_depth, model=model_key)
            await self.client.expire(queue_key, self.config.groq_max_queue_wait_seconds + 5)
        except RedisError as exc:
            raise GroqLimitError("redis_unavailable") from exc
        if queue_depth > self.config.groq_max_queue_depth:
            try:
                await self.client.decr(queue_key)
                gauge("opcenter_groq_queue_depth", max(0, queue_depth - 1), model=model_key)
            except RedisError:
                pass
            await self.set_status(request_id, "rejected", model=model, reason="queue_full")
            raise GroqLimitError("queue_full")

        slot = f"{request_id}:{uuid4().hex}"
        active_key = f"opcenter:groq:active:{model_key}"
        request_key = f"opcenter:groq:requests:{model_key}"
        token_key = f"opcenter:groq:tokens:{model_key}"
        lease_ms = max(5_000, self.config.groq_max_queue_wait_seconds * 1_000 + 120_000)
        deadline = time.monotonic() + self.config.groq_max_queue_wait_seconds
        admitted = False
        try:
            await self.set_status(request_id, "queued", model=model, queue_depth=queue_depth)
            while time.monotonic() < deadline:
                try:
                    result = await self.client.eval(
                        _ADMIT,
                        3,
                        active_key,
                        request_key,
                        token_key,
                        int(time.time() * 1_000),
                        lease_ms,
                        limits.concurrency,
                        limits.requests_per_minute,
                        limits.tokens_per_minute,
                        max(1, token_cost),
                        slot,
                    )
                except RedisError as exc:
                    raise GroqLimitError("redis_unavailable") from exc
                if int(result[0]):
                    admitted = True
                    await self.set_status(request_id, "running", model=model)
                    return await operation()
                delay = min(1.0, max(0.05, float(result[1] or 0)))
                await asyncio.sleep(delay + random.uniform(0.01, 0.1))
            await self.set_status(request_id, "rejected", model=model, reason="queue_timeout")
            raise GroqLimitError("queue_timeout")
        finally:
            try:
                await self.client.decr(queue_key)
                gauge("opcenter_groq_queue_depth", max(0, queue_depth - 1), model=model_key)
            except RedisError:
                pass
            if admitted:
                try:
                    await self.client.eval(_RELEASE, 1, active_key, slot)
                except RedisError:
                    pass


class _NoopGroqLimiter:
    async def set_status(self, request_id: str | None, status: str, **details: Any) -> None:
        return None

    async def run(
        self, model: str, token_cost: int, operation: Callable[[], Awaitable[ResultT]]
    ) -> ResultT:
        return await operation()


_limiter: RedisGroqLimiter | _NoopGroqLimiter = _NoopGroqLimiter()


def configure_groq_limiter(limiter: RedisGroqLimiter) -> None:
    global _limiter
    _limiter = limiter


def get_groq_limiter() -> RedisGroqLimiter | _NoopGroqLimiter:
    return _limiter


def current_groq_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def groq_request_scope(request_id: str) -> Iterator[None]:
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)
