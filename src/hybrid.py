"""Independent async retrieval branches with deadlines and safe partial results."""
import asyncio
import logging
from time import perf_counter

logger = logging.getLogger(__name__)


class RetrievalUnavailable(RuntimeError):
    pass


async def run_branches(dense, bm25, dense_timeout, bm25_timeout):
    metrics = {'errors': {}}
    async def measured(name, operation, timeout):
        start = perf_counter()
        try:
            return await asyncio.wait_for(operation(), timeout)
        except Exception as exc:
            metrics['errors'][name] = type(exc).__name__
            logger.warning('retrieval_branch_failed', extra={'branch': name, 'error_type': type(exc).__name__})
            return []
        finally:
            metrics[f'{name}_ms'] = (perf_counter() - start) * 1000
    start = perf_counter()
    results = await asyncio.gather(measured('dense', dense, dense_timeout), measured('bm25', bm25, bm25_timeout))
    metrics['hybrid_wall_ms'] = (perf_counter() - start) * 1000
    if len(metrics['errors']) == 2:
        raise RetrievalUnavailable('Both retrieval branches failed')
    return *results, metrics
