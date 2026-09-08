import asyncio
from dataclasses import replace
from types import SimpleNamespace
from time import perf_counter

import pytest


def test_one_embedding_per_query_across_body_and_representation_branches(monkeypatch):
    from src import retrieval as r
    calls = []
    model = SimpleNamespace(embed_query=lambda query: calls.append(query) or [1, 0, 0])
    collection = SimpleNamespace(query=lambda **kw: {'ids': [[]], 'distances': [[]]})
    resources = SimpleNamespace(embedding_model=model, chroma_collection=collection,
                                representation_collection=collection, segments_by_id={'s': {}},
                                representations_by_id={'r': {}}, manual_names=('Guide',))
    monkeypatch.setattr(r, 'load_resources', lambda _: resources)
    monkeypatch.setattr(r, 'cache_get', lambda *a: None)
    monkeypatch.setattr(r, 'cache_set', lambda *a, **kw: None)
    config = replace(r.settings, reuse_query_embeddings=True)
    with r.query_embedding_scope():
        r.vector_search('Question', config=config)
        r.representation_search('Question', config=config)
        r._preferred_vector_search('Question', ['Guide'], config)
        r.representation_search('Other question', config=config)
    assert calls == ['Question', 'Other question']
    with r.query_embedding_scope():
        r.vector_search('Question', config=config)
    assert calls == ['Question', 'Other question', 'Question']


def test_parallel_branches_overlap_and_report_timings():
    from src.hybrid import run_branches
    async def exercise():
        entered = set()
        async def branch(name, delay):
            entered.add(name)
            await asyncio.sleep(delay)
            assert entered == {'dense', 'bm25'}
            return [name]
        start = perf_counter()
        dense, bm25, timings = await run_branches(lambda: branch('dense', .2), lambda: branch('bm25', .15), 1, 1)
        elapsed = perf_counter() - start
        assert dense == ['dense'] and bm25 == ['bm25']
        assert .19 <= elapsed < .32
        assert timings['hybrid_wall_ms'] < timings['dense_ms'] + timings['bm25_ms'] - 80
    asyncio.run(exercise())


@pytest.mark.parametrize('failed', ['dense', 'bm25', 'both', 'timeout'])
def test_branch_failures_are_isolated(failed, caplog):
    from src.hybrid import run_branches, RetrievalUnavailable
    async def good():
        return ['ok']
    async def bad():
        if failed == 'timeout':
            await asyncio.sleep(1)
        raise ValueError('sensitive-query-do-not-log')
    async def exercise():
        if failed == 'both':
            with pytest.raises(RetrievalUnavailable):
                await run_branches(bad, bad, .02, .02)
        else:
            dense, bm25, metrics = await run_branches(bad if failed != 'bm25' else good,
                                                      bad if failed == 'bm25' else good, .02, .02)
            expected = (['ok'], []) if failed == 'bm25' else ([], ['ok'])
            assert (dense, bm25) == expected
            assert metrics['errors']
    asyncio.run(exercise())
    assert 'sensitive-query-do-not-log' not in caplog.text


def test_cli_closes_its_default_executor(monkeypatch):
    import threading
    from unittest.mock import AsyncMock
    from src import vector_store, embeddings
    monkeypatch.setattr(embeddings, 'create_embedding_model', lambda _: SimpleNamespace(embed_query=lambda _: [1, 0, 0]))
    monkeypatch.setattr(vector_store, 'create_pool', lambda _: SimpleNamespace(open=AsyncMock(), close=AsyncMock()))
    config = SimpleNamespace(embedding_model='test', embedding_model_revision='revision', db_pool_timeout=30)
    with vector_store.standalone_client(config) as client:
        worker = client.call(asyncio.to_thread(threading.current_thread))
        assert worker.is_alive()
    assert not worker.is_alive()


def test_stage_profile_records_rerank_work_without_query_text():
    from src.retrieval_metrics import profile_retrieval, stage
    with profile_retrieval() as metrics:
        with stage('rerank_ms'):
            pass
    assert set(metrics) == {'rerank_ms'}
    assert len(metrics['rerank_ms']) == 1
    assert metrics['rerank_ms'][0] >= 0
