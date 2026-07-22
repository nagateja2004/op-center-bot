"""Small process-local Prometheus metrics and timing helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
import logging
from threading import Lock
from time import perf_counter
from typing import Any, TypeVar


StateT = TypeVar("StateT")
logger = logging.getLogger(__name__)
_lock = Lock()
_counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
_gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
_summaries: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(
    lambda: [0.0, 0.0]
)


def _key(name: str, labels: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((key, str(value)) for key, value in labels.items()))


def increment(name: str, amount: float = 1, **labels: Any) -> None:
    with _lock:
        _counters[_key(name, labels)] += amount


def gauge(name: str, value: float, **labels: Any) -> None:
    with _lock:
        _gauges[_key(name, labels)] = value


def observe(name: str, value: float, **labels: Any) -> None:
    with _lock:
        stats = _summaries[_key(name, labels)]
        stats[0] += value
        stats[1] += 1


def timed_sync_node(name: str, operation: Callable[[StateT], StateT]) -> Callable[[StateT], StateT]:
    def wrapped(state: StateT) -> StateT:
        started = perf_counter()
        status = "success"
        try:
            return operation(state)
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - started
            observe("opcenter_langgraph_node_duration_seconds", duration, node=name, status=status)
            logger.info("langgraph_node node=%s status=%s duration_ms=%.1f", name, status, duration * 1000)

    return wrapped


def timed_async_node(
    name: str, operation: Callable[[StateT], Awaitable[StateT]]
) -> Callable[[StateT], Awaitable[StateT]]:
    async def wrapped(state: StateT) -> StateT:
        started = perf_counter()
        status = "success"
        try:
            return await operation(state)
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - started
            observe("opcenter_langgraph_node_duration_seconds", duration, node=name, status=status)
            logger.info("langgraph_node node=%s status=%s duration_ms=%.1f", name, status, duration * 1000)

    return wrapped


def render_prometheus() -> str:
    lines: list[str] = []
    with _lock:
        counters = dict(_counters)
        gauges = dict(_gauges)
        summaries = {key: list(value) for key, value in _summaries.items()}
    for metric_type, values in (("counter", counters), ("gauge", gauges)):
        for name in sorted({key[0] for key in values}):
            lines.append(f"# TYPE {name} {metric_type}")
            for (metric, labels), value in sorted(values.items()):
                if metric == name:
                    lines.append(f"{metric}{_labels(labels)} {value}")
    for name in sorted({key[0] for key in summaries}):
        lines.append(f"# TYPE {name} summary")
        for (metric, labels), (total, count) in sorted(summaries.items()):
            if metric == name:
                rendered = _labels(labels)
                lines.append(f"{metric}_sum{rendered} {total}")
                lines.append(f"{metric}_count{rendered} {int(count)}")
    return "\n".join(lines) + "\n"


def _labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    escaped = [f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for key, value in labels]
    return "{" + ",".join(escaped) + "}"
