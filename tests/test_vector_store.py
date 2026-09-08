"""Unit tests plus opt-in real PostgreSQL tests in disposable, isolated schemas."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, sql

from src.config import Settings
from src.vector_store import (
    PgVectorClient, PgVectorRepository, VectorChunk, create_pool,
    embedding_dimension, make_chunk, vector_literal,
)


def test_dimension_is_measured_not_hardcoded():
    assert embedding_dimension(SimpleNamespace(embed_query=lambda _: [1.0] * 7)) == 7


@pytest.mark.parametrize("vector,dimension", [([1], 2), ([float("nan")], 1), ([float("inf")], 1), ([0, 0], 2)])
def test_invalid_vectors_rejected(vector, dimension):
    with pytest.raises(ValueError):
        vector_literal(vector, dimension)


def test_chunk_mapping_preserves_nested_metadata_and_parent_identity():
    record = {"segment_id": "s1", "evidence_id": "e1", "segment_index": 2,
              "searchable_text": "Factory setup", "content_type": "table",
              "metadata": {"manual": "Guide", "source_file": "Guide.pdf", "chapter": "C",
                           "section": "S", "heading_path": ["C", "S"], "pdf_page": 9,
                           "printed_page": "2-1", "table_rows": [["a", "b"]]}}
    chunk = make_chunk(record, "body", [1, 0, 0])
    assert chunk.document_id == "Guide.pdf"
    assert chunk.metadata["table_rows"] == [["a", "b"]]
    assert chunk.metadata["evidence_id"] == "e1"
    assert chunk.chunk_index == 2
    other = {**record, "metadata": {**record["metadata"], "source_file": "Other.pdf"}}
    assert make_chunk(other, "body", [1, 0, 0]).parent_section_id != chunk.parent_section_id


def test_sync_adapter_rejects_event_loop_use_and_works_from_worker():
    async def exercise():
        repo = SimpleNamespace(ids=AsyncMock(return_value=["a"]))
        client = PgVectorClient(repo, asyncio.get_running_loop())
        with pytest.raises(RuntimeError, match="to_thread"):
            client.get_collection("body").get()
        assert await asyncio.to_thread(client.get_collection("body").get) == {"ids": ["a"]}
    asyncio.run(exercise())


def test_existing_checkpointer_reuses_supplied_pool(monkeypatch):
    from backend import dependencies
    from contextlib import AsyncExitStack
    pool = object()
    monkeypatch.setattr(dependencies, "settings", SimpleNamespace(checkpoint_backend="postgres"))
    monkeypatch.setattr(dependencies, "_open_postgres_pool", AsyncMock(side_effect=AssertionError("second pool")))
    monkeypatch.setattr("langgraph.checkpoint.postgres.aio.AsyncPostgresSaver", lambda supplied, **_: supplied)
    async def exercise():
        async with AsyncExitStack() as stack:
            saver, supplied = await dependencies._open_checkpointer(stack, pool)
            assert saver is supplied is pool
    asyncio.run(exercise())


def test_pool_configuration_is_reused(monkeypatch):
    from src import vector_store
    captured = {}
    monkeypatch.setattr(vector_store, "AsyncConnectionPool", lambda **kwargs: captured.update(kwargs))
    create_pool(SimpleNamespace(database_url="postgresql://localhost/test", db_pool_min_size=2,
                               db_pool_max_size=7, db_pool_timeout=11))
    assert (captured["min_size"], captured["max_size"], captured["timeout"], captured["open"]) == (2, 7, 11, False)


def test_pgvector_lifespan_shares_pool_and_closes_without_chroma(monkeypatch):
    from dataclasses import asdict
    from fastapi import FastAPI
    from backend import dependencies
    config = SimpleNamespace(**{**asdict(Settings()), "vector_store": "pgvector", "checkpoint_backend": "postgres"})
    config.validate = lambda: None
    monkeypatch.setattr(dependencies, "settings", config)
    pool = SimpleNamespace(close=AsyncMock())
    async def open_pool(stack):
        stack.push_async_callback(pool.close)
        return pool
    opener = AsyncMock(side_effect=open_pool)
    monkeypatch.setattr(dependencies, "_open_postgres_pool", opener)
    checkpointer = SimpleNamespace(setup=AsyncMock())
    saver = AsyncMock(return_value=(checkpointer, pool))
    monkeypatch.setattr(dependencies, "_open_checkpointer", saver)
    monkeypatch.setattr(dependencies, "create_embedding_model", lambda _: object())
    monkeypatch.setattr(dependencies, "embedding_dimension", lambda _: 7)
    monkeypatch.setattr(dependencies, "create_reranker", lambda _: object())
    captured = []
    def repo_factory(supplied_pool, dimension, model, revision, **options):
        captured.append((supplied_pool, dimension))
        return SimpleNamespace(health_check=AsyncMock(return_value=True))
    monkeypatch.setattr(dependencies, "PgVectorRepository", repo_factory)
    monkeypatch.setattr(dependencies, "create_chroma_client", lambda _: pytest.fail("Chroma must not be created"))
    monkeypatch.setattr(dependencies, "validate_search_indexes", lambda client: 1)
    lock = SimpleNamespace(acquire=AsyncMock(return_value=True), release=AsyncMock())
    limiter = SimpleNamespace(client=SimpleNamespace(lock=lambda *a, **kw: lock), close=AsyncMock())
    monkeypatch.setattr(dependencies.RedisGroqLimiter, "connect", AsyncMock(return_value=limiter))
    for name in ("configure_cache", "configure_groq_limiter", "initialize_async_groq_client"):
        monkeypatch.setattr(dependencies, name, lambda *a: None)
    monkeypatch.setattr(dependencies, "close_cache", lambda: None)
    monkeypatch.setattr(dependencies, "close_async_groq_client", AsyncMock())
    graph = object()
    monkeypatch.setattr("src.graph.build_graph", lambda _: graph)
    async def exercise():
        app = FastAPI()
        async with dependencies.lifespan(app):
            assert app.state.ready_error is None
            assert app.state.graph is graph
            assert app.state.postgres_pool is pool
            assert isinstance(app.state.chroma_client, PgVectorClient)
            assert saver.await_args.args[1] is pool
            assert captured == [(pool, 7)]
        pool.close.assert_awaited_once()
        opener.assert_awaited_once()
        assert app.state.graph is None
    asyncio.run(exercise())


def test_backfill_publishes_both_collections_without_touching_chroma(monkeypatch):
    from src import pgvector_migrate
    from src.ingest import REPRESENTATION_COLLECTION
    monkeypatch.setattr(pgvector_migrate, "create_embedding_model", lambda _: SimpleNamespace(embed_documents=lambda texts: [[1, 0, 0] for _ in texts]))
    class Client:
        timeout = 30
        def __init__(self):
            self.repository = SimpleNamespace(dimension=3, health_check=lambda: True, replace_collections=self.publish)
        def call(self, result):
            return result
        def publish(self, collections, chunks):
            assert collections == ["body", REPRESENTATION_COLLECTION]
            assert [c.id for c in chunks] == ["s1", "r1"]
            assert chunks[1].metadata["representation_type"] == "heading"
    body = {"segment_id": "s1", "evidence_id": "e1", "searchable_text": "text", "metadata": {"manual": "A", "pdf_page": 1}}
    representation = {"representation_id": "r1", "evidence_id": "e1", "text": "heading", "representation_type": "heading", "metadata": {"manual": "A", "pdf_page": 1}}
    client = Client()
    assert pgvector_migrate.build_pgvector_indexes([body], [representation], SimpleNamespace(chroma_collection="body"), client) == 1
    assert client.timeout == 30


def test_unsupported_filters_rejected_before_database_access():
    async def exercise():
        repo = PgVectorRepository(None, 3, "model", "revision")
        with pytest.raises(ValueError, match="Only metadata"):
            await repo.search("body", [1, 0, 0], 3, {"manual": {"$ne": "A"}})
    asyncio.run(exercise())


DATABASE_URL = os.getenv("PGVECTOR_TEST_DATABASE_URL")
integration = pytest.mark.skipif(not DATABASE_URL, reason="Set PGVECTOR_TEST_DATABASE_URL to a disposable pgvector database")


@integration
def test_hnsw_migration_search_filters_and_dimension_validation():
    async def exercise():
        async with repository() as repo:
            await repo.create_hnsw_index()
            await repo.create_hnsw_index()
            await repo.upsert_chunks([
                VectorChunk('a', 'body', 'doc', 'near', [1, 0, 0], {'manual': 'A'}, 'p', 0),
                VectorChunk('b', 'body', 'doc', 'far', [0, 1, 0], {'manual': 'B'}, 'p', 1),
            ])
            exact = await repo.search('body', [1, 0, 0], 2, mode='exact')
            ann = await repo.search('body', [1, 0, 0], 2, mode='hnsw')
            assert [r['id'] for r in ann] == [r['id'] for r in exact] == ['a', 'b']
            assert ann[0]['distance'] == 0
            assert [r['id'] for r in await repo.search('body', [1, 0, 0], 2, {'manual': 'B'}, mode='hnsw')] == ['b']
            with pytest.raises(ValueError, match='dimension'):
                await repo.search('body', [1, 0], 2, mode='hnsw')
            async with repo.pool.connection() as conn:
                cur = await conn.execute("SELECT indexdef FROM pg_indexes WHERE schemaname=current_schema() AND indexname='document_chunks_embedding_hnsw'")
                assert 'vector_l2_ops' in (await cur.fetchone())['indexdef']
    asyncio.run(exercise())


@integration
def test_hnsw_health_requires_a_valid_index():
    async def exercise():
        async with repository() as repo:
            repo.search_mode = 'hnsw'
            assert not await repo.health_check()
            await repo.create_hnsw_index()
            assert await repo.health_check()
    asyncio.run(exercise())


@asynccontextmanager
async def repository():
    # Never use the application's DATABASE_URL implicitly. Only a test-created schema is dropped.
    schema = "test_vectors_" + uuid4().hex
    admin = await AsyncConnection.connect(DATABASE_URL, autocommit=True)
    pool = None
    try:
        await admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        config = replace(Settings(), database_url=DATABASE_URL, db_pool_min_size=1, db_pool_max_size=1)
        async def configure(connection):
            await connection.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        pool = create_pool(config, configure=configure)
        await pool.open(wait=True)
        repo = PgVectorRepository(pool, 3, "fixture-model", "fixture-revision")
        await repo.migrate()
        yield repo
    finally:
        if pool is not None:
            await pool.close()
        await admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        await admin.close()


def chunk(id="a", manual="A", vector=None, index=0):
    return VectorChunk(id, "body", manual + ".pdf", "content " + id, vector or [1, 0, 0],
                       {"manual": manual, "chapter": "chapter", "section": "section", "pdf_page": 2,
                        "printed_page": "1-2", "nested": {"rows": [[1, 2]]}}, "parent-" + manual, index)


@integration
def test_database_connection_and_extension():
    async def exercise():
        async with repository() as repo:
            assert await repo.health_check()
            await repo.migrate()  # Idempotent schema migration.
    asyncio.run(exercise())


@integration
def test_chunk_batch_insert_upsert_and_metadata_persistence():
    async def exercise():
        async with repository() as repo:
            await repo.upsert_chunks([chunk(), chunk("b")])
            row = await repo.get_chunk("body", "a")
            assert row["metadata"]["nested"] == {"rows": [[1, 2]]}
            assert row["page_number"] == 2 and row["manual_name"] == "A"
            assert row["created_at"] and row["chunk_id"] == "a"
            await repo.upsert_chunks([replace(chunk(), content="updated")])
            assert (await repo.get_chunk("body", "a"))["content"] == "updated"
            assert len(await repo.ids("body")) == 2
            assert await repo.get_chunk("body", "missing") is None
    asyncio.run(exercise())


@integration
def test_vector_search_and_collection_isolation():
    async def exercise():
        async with repository() as repo:
            await repo.upsert_chunks([chunk(), chunk("b", vector=[0, 1, 0]), replace(chunk("c"), collection="representations")])
            result = await repo.search("body", [1, 0, 0], 10)
            assert [r["id"] for r in result] == ["a", "b"]
            assert result[0]["distance"] == pytest.approx(0)
            assert result[1]["distance"] == pytest.approx(2)
    asyncio.run(exercise())


@integration
def test_filtered_search_equal_in_and_hostile_metadata():
    async def exercise():
        async with repository() as repo:
            await repo.upsert_chunks([chunk(), chunk("b", manual="B", vector=[0, 1, 0])])
            for filters in ({"manual": "B"}, {"manual": {"$eq": "B"}}, {"manual": {"$in": ["B"]}}):
                assert [r["id"] for r in await repo.search("body", [1, 0, 0], 10, filters)] == ["b"]
            assert await repo.search("body", [1, 0, 0], 10, {"manual": "'; DROP TABLE document_chunks;--"}) == []
            assert await repo.search("body", [1, 0, 0], 10, {"manual": {"$in": []}}) == []
            assert await repo.health_check()
    asyncio.run(exercise())


@integration
def test_parent_retrieval_and_document_deletion():
    async def exercise():
        async with repository() as repo:
            await repo.upsert_chunks([chunk("b", index=1), chunk("a", index=0), chunk("c", manual="B")])
            assert [r["id"] for r in await repo.parent_chunks("body", "parent-A")] == ["a", "b"]
            await repo.delete_document("A.pdf")
            assert await repo.ids("body") == ["c"]
    asyncio.run(exercise())


@integration
def test_connection_pool_reuse_and_close():
    async def exercise():
        async with repository() as repo:
            pool = repo.pool
            async with pool.connection() as connection:
                pid = connection.info.backend_pid
            for _ in range(3):
                assert await repo.health_check()
                async with pool.connection() as connection:
                    assert connection.info.backend_pid == pid
            assert pool.get_stats()["pool_size"] == 1
        assert pool.closed
    asyncio.run(exercise())


@integration
def test_model_identity_mismatch_rejected():
    async def exercise():
        async with repository() as repo:
            for dimension, model, revision in ((4, repo.model, repo.revision), (3, "different", repo.revision), (3, repo.model, "different")):
                other = PgVectorRepository(repo.pool, dimension, model, revision)
                with pytest.raises(ValueError, match="differs"):
                    await other.migrate()
            assert await repo.health_check()
    asyncio.run(exercise())


@integration
def test_atomic_publication_rolls_back_on_invalid_embedding():
    async def exercise():
        async with repository() as repo:
            await repo.upsert_chunks([chunk()])
            with pytest.raises(ValueError):
                await repo.replace_collections(["body"], [chunk("new"), replace(chunk("bad"), embedding=[1])])
            assert await repo.ids("body") == ["a"]
            await repo.replace_collections(["body"], [chunk("new")])
            assert await repo.ids("body") == ["new"]
    asyncio.run(exercise())


@integration
def test_repeated_real_import_has_valid_database_summary(monkeypatch):
    from src import pgvector_migrate
    from test_pgvector_ingest import records
    monkeypatch.setattr(pgvector_migrate, "create_embedding_model", lambda _: SimpleNamespace(embed_documents=lambda texts: [[1, 0, 0] for _ in texts]))
    async def exercise():
        async with repository() as repo:
            client = PgVectorClient(repo, asyncio.get_running_loop())
            body, reps = records(257)
            for _ in range(2):
                await asyncio.to_thread(pgvector_migrate.build_pgvector_indexes, body, reps, SimpleNamespace(chroma_collection="body"), client)
            report = await repo.ingestion_report(["body", "opcenter_manual_representations"])
            assert report == {"total_documents": 1, "total_chunks": 258, "missing_embeddings": 0,
                              "missing_page_numbers": 0, "missing_document_ids": 0,
                              "missing_parent_ids": 0, "duplicate_chunk_ids": 0, "embedding_dimensions": [3]}
            pgvector_migrate.validate_report(report, 3)
    asyncio.run(exercise())
