"""Regression tests for defects found in the production-readiness review."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_cli_pool_configuration_failure_does_not_start_background_loop(monkeypatch):
    from src import vector_store, embeddings
    events = []
    monkeypatch.setattr(embeddings, "create_embedding_model", lambda _: object())
    monkeypatch.setattr(vector_store, "embedding_dimension", lambda _: 3)
    monkeypatch.setattr(vector_store, "Thread", lambda **kw: SimpleNamespace(start=lambda: events.append("start"), join=lambda: None))
    def invalid_pool(config):
        raise ValueError("invalid pool configuration")
    monkeypatch.setattr(vector_store, "create_pool", invalid_pool)
    with pytest.raises(ValueError, match="invalid pool"):
        with vector_store.standalone_client(object()):
            pass
    assert events == []


def test_startup_cancellation_closes_opened_pool(monkeypatch):
    from backend import dependencies
    from fastapi import FastAPI
    pool = SimpleNamespace(close=AsyncMock())
    async def opening(stack):
        stack.push_async_callback(pool.close)
        raise asyncio.CancelledError()
    monkeypatch.setattr(dependencies, "settings", SimpleNamespace(validate=lambda: None, checkpoint_backend="postgres"))
    monkeypatch.setattr(dependencies, "_open_postgres_pool", opening)
    async def exercise():
        with pytest.raises(asyncio.CancelledError):
            async with dependencies.lifespan(FastAPI()):
                pass
        pool.close.assert_awaited_once()
    asyncio.run(exercise())


def test_backfill_rejects_overlapping_body_and_representation_ids(monkeypatch):
    from src import pgvector_migrate
    from test_pgvector_ingest import MemoryClient, records
    body, reps = records()
    reps[0]["representation_id"] = body[0]["segment_id"]
    monkeypatch.setattr(pgvector_migrate, "create_embedding_model", lambda _: SimpleNamespace(embed_documents=lambda texts: [[1, 0, 0] for _ in texts]))
    client = MemoryClient()
    with pytest.raises(ValueError, match="overlap"):
        pgvector_migrate.build_pgvector_indexes(body, reps, SimpleNamespace(chroma_collection="body"), client)
    assert client.publications == 0


def test_blank_document_identity_is_rejected():
    from src.vector_store import make_chunk
    with pytest.raises(ValueError, match="document identity"):
        make_chunk({"segment_id": "s1", "searchable_text": "text", "metadata": {"source_file": "  ", "manual": "  "}}, "body", [1, 0, 0])


@pytest.mark.parametrize("dimension", [True, 3.5])
def test_schema_dimension_requires_a_real_integer(dimension):
    from src.vector_store import PgVectorRepository
    with pytest.raises(ValueError, match="dimension"):
        PgVectorRepository(None, dimension, "model", "revision")


def test_compression_failure_preserves_bounded_evidence_and_citations(monkeypatch, caplog):
    from src import nodes
    doc = {"chunk_id": "s1", "text": "relevant evidence " * 200, "content_type": "concept",
           "retrieval_scores": {"reranker_score": 1},
           "metadata": {"evidence_id": "e1", "manual": "Manual", "pdf_page": 7,
                        "printed_page": "2-3", "parent_section_id": "p1", "source_file": "manual.pdf"}}
    original = deepcopy(doc)
    def fail(*a, **kw):
        raise RuntimeError("sensitive provider/query contents")
    monkeypatch.setattr(nodes, "compress_evidence", fail)
    output = nodes._with_compressed_view(doc, "aspect", "question", [], include_complete_procedure=False)
    view = output["metadata"]["compressed_views"]["aspect"]
    assert view["compressed_text"] == doc["text"][:700]
    assert view["source_metadata"]["pdf_page"] == 7
    assert view["source_metadata"]["parent_section_id"] == "p1"
    assert nodes._answer_content(doc, "question", False) == doc["text"][:700]
    assert doc == original
    assert "sensitive provider" not in caplog.text


def test_duplicate_in_one_source_does_not_inflate_rrf_score():
    from src import retrieval
    doc = {"chunk_id": "s1", "text": "text", "content_type": "concept", "metadata": {}, "retrieval_scores": {"vector_rank": 1}}
    fused = retrieval._rank_fuse([[doc, doc]], [1.0], score_name="rrf_score")
    assert fused[0]["retrieval_scores"]["rrf_score"] == pytest.approx(1 / 61)
