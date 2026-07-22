"""Bounded asynchronous admission for local embedding and reranker inference."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator

from src.config import settings
from src.observability import gauge, observe


class InferenceBusyError(RuntimeError):
    pass


class InferenceGate:
    def __init__(self, name: str) -> None:
        self.name = name
        self.semaphore = asyncio.Semaphore(settings.inference_max_concurrency)
        self.max_queue_depth = settings.inference_max_queue_depth
        self.waiters = 0
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        started = perf_counter()
        async with self.lock:
            if self.semaphore.locked() and self.waiters >= self.max_queue_depth:
                raise InferenceBusyError(f"{self.name}_queue_full")
            self.waiters += 1
            gauge("opcenter_inference_queue_depth", self.waiters, kind=self.name)
        try:
            await self.semaphore.acquire()
        finally:
            async with self.lock:
                self.waiters -= 1
                gauge("opcenter_inference_queue_depth", self.waiters, kind=self.name)
        queue_time = perf_counter() - started
        observe("opcenter_inference_queue_seconds", queue_time, kind=self.name)
        inference_started = perf_counter()
        try:
            yield
        finally:
            self.semaphore.release()
            observe(
                "opcenter_inference_duration_seconds",
                perf_counter() - inference_started,
                kind=self.name,
            )


embedding_gate = InferenceGate("embedding")
reranker_gate = InferenceGate("reranker")
