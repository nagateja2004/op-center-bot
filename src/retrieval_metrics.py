"""Context-local stage profiling; does not log queries or evidence text."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
from time import perf_counter

_profile = ContextVar('retrieval_profile', default=None)
logger = logging.getLogger(__name__)


@contextmanager
def profile_retrieval():
    values = {}
    token = _profile.set(values)
    try:
        yield values
    finally:
        _profile.reset(token)


@contextmanager
def stage(name):
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = (perf_counter() - start) * 1000
        values = _profile.get()
        if values is not None:
            values.setdefault(name, []).append(elapsed)
        logger.debug('retrieval_stage', extra={'stage': name, 'duration_ms': elapsed})
