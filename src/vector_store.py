"""Async vector storage and a worker-thread adapter for the existing sync retriever.

SQL lives here, never in retrieval. psycopg's async pool is shared with checkpoints.
Vector literals use bound parameters, avoiding per-connection type registration.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from threading import Thread
from typing import Any, Protocol

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool


def create_pool(config, *, configure=None):
    return AsyncConnectionPool(
        conninfo=config.database_url, min_size=config.db_pool_min_size,
        max_size=config.db_pool_max_size, timeout=config.db_pool_timeout,
        open=False, configure=configure,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                "connect_timeout": config.db_pool_timeout},
    )


def embedding_dimension(model) -> int:
    """Probe the configured, pinned model; no environment dimension override."""
    vector = model.embed_query("embedding dimension validation")
    dimension = len(vector)
    vector_literal(vector, dimension)
    return dimension


def vector_literal(vector, dimension):
    values = [float(value) for value in vector]
    if len(values) != dimension or not 0 < dimension <= 16000:
        raise ValueError("Embedding dimension mismatch")
    if not all(math.isfinite(v) for v in values) or not any(values):
        raise ValueError("Embedding must be finite and nonzero")
    return json.dumps(values, allow_nan=False)


@dataclass
class VectorChunk:
    id: str
    collection: str
    document_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    parent_section_id: str
    chunk_index: int = 0


class VectorStore(Protocol):
    async def upsert_chunks(self, chunks: list[VectorChunk]) -> None: ...
    async def delete_document(self, document_id: str) -> None: ...
    async def search(self, collection: str, embedding: list[float], limit: int, filters: dict | None = None) -> list[dict]: ...
    async def get_chunk(self, collection: str, chunk_id: str) -> dict | None: ...
    async def parent_chunks(self, collection: str, parent_section_id: str) -> list[dict]: ...
    async def health_check(self) -> bool: ...


class PgVectorRepository:
    def __init__(self, pool, dimension: int, model: str, revision: str, *, search_mode='exact', ef_search=40):
        if type(dimension) is not int or not 0 < dimension <= 16000:
            raise ValueError("Invalid embedding dimension")
        self.pool, self.dimension = pool, dimension
        self.model, self.revision = model, revision
        if search_mode not in {'exact', 'hnsw'} or type(ef_search) is not int or not 1 <= ef_search <= 1000:
            raise ValueError('Invalid search mode or HNSW ef_search')
        self.search_mode, self.ef_search = search_mode, ef_search

    async def create_hnsw_index(self, m=16, ef_construction=64):
        """Explicit setup operation; preserve L2 ranking and existing Chroma scores."""
        if type(m) is not int or type(ef_construction) is not int or not 2 <= m <= 100 or not 4 <= ef_construction <= 1000 or ef_construction < 2 * m:
            raise ValueError('Invalid HNSW build parameters')
        if self.dimension > 2000:
            raise ValueError('HNSW vector indexing supports at most 2000 dimensions')
        template = (Path(__file__).parent / 'sql' / '002_hnsw.sql').read_text()
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute('SELECT pg_advisory_xact_lock(73819102)')
                await connection.execute(template.replace('__M__', str(m)).replace('__EF_CONSTRUCTION__', str(ef_construction)))
            await connection.execute('ANALYZE document_chunks')

    async def migrate(self):
        """Explicit administrative operation; never called at API startup."""
        template = (Path(__file__).parent / "sql" / "001_pgvector.sql").read_text()
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock(73819102)")
                for statement in template.replace("__DIMENSION__", str(self.dimension)).split(";"):
                    if statement.strip():
                        await connection.execute(statement)
                await connection.execute(
                    "INSERT INTO vector_model (id, model, revision, dimension) VALUES (1, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (self.model, self.revision, self.dimension),
                )
                await self._validate(connection)

    async def _validate(self, connection):
        cursor = await connection.execute("SELECT model, revision, dimension FROM vector_model WHERE id=1")
        row = await cursor.fetchone()
        if row != {"model": self.model, "revision": self.revision, "dimension": self.dimension}:
            raise ValueError("Stored embedding model/revision/dimension differs; use a new database and re-embed")
        cursor = await connection.execute(
            "SELECT format_type(atttypid, atttypmod) AS type FROM pg_attribute "
            "WHERE attrelid='document_chunks'::regclass AND attname='embedding' AND NOT attisdropped"
        )
        if (await cursor.fetchone())["type"] != f"vector({self.dimension})":
            raise ValueError("Stored vector column dimension differs from configured model")

    async def health_check(self):
        async with self.pool.connection() as connection:
            cursor = await connection.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
            if not await cursor.fetchone():
                return False
            await self._validate(connection)
            if self.search_mode == 'hnsw':
                cursor = await connection.execute(
                    "SELECT i.indisvalid FROM pg_index i "
                    "JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am a ON a.oid=c.relam "
                    "JOIN pg_opclass o ON o.oid=i.indclass[0] "
                    "WHERE i.indexrelid=to_regclass('document_chunks_embedding_hnsw') "
                    "AND i.indrelid='document_chunks'::regclass AND a.amname='hnsw' AND o.opcname='vector_l2_ops'"
                )
                index = await cursor.fetchone()
                if not index or not index['indisvalid']:
                    return False
                cursor = await connection.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
                if tuple(int(part) for part in (await cursor.fetchone())['extversion'].split('.')[:3]) < (0, 8, 0):
                    return False
            return True

    def _parameters(self, chunk):
        m = chunk.metadata
        return (chunk.collection, chunk.id, chunk.document_id, chunk.id, chunk.content,
                vector_literal(chunk.embedding, self.dimension), m.get("manual", ""),
                m.get("chapter", ""), m.get("section", ""), m.get("pdf_page"),
                chunk.parent_section_id, chunk.chunk_index, m.get("content_type", "text"), Jsonb(m))

    async def _upsert(self, connection, chunks):
        rows = [self._parameters(chunk) for chunk in chunks]
        if not rows:
            return
        async with connection.cursor() as cursor:
            await cursor.executemany(
                "INSERT INTO document_chunks (collection,id,document_id,chunk_id,content,embedding,manual_name,chapter,section,page_number,parent_section_id,chunk_index,content_type,metadata) "
                "VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (collection,id) DO UPDATE SET document_id=excluded.document_id,chunk_id=excluded.chunk_id,content=excluded.content,embedding=excluded.embedding,"
                "manual_name=excluded.manual_name,chapter=excluded.chapter,section=excluded.section,page_number=excluded.page_number,"
                "parent_section_id=excluded.parent_section_id,chunk_index=excluded.chunk_index,content_type=excluded.content_type,metadata=excluded.metadata",
                rows,
            )

    async def upsert_chunks(self, chunks):
        """Batch content and embeddings atomically, keeping metadata intact."""
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await self._upsert(connection, chunks)

    async def replace_collections(self, collections, chunks):
        """Atomic snapshot publication, including deletion of stale chunk IDs."""
        if not collections or any(c.collection not in collections for c in chunks):
            raise ValueError("Invalid replacement scope")
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock(73819103)")
                await connection.execute("DELETE FROM document_chunks WHERE collection = ANY(%s)", (list(collections),))
                for start in range(0, len(chunks), 256):
                    await self._upsert(connection, chunks[start:start + 256])

    async def delete_document(self, document_id):
        async with self.pool.connection() as connection:
            await connection.execute("DELETE FROM document_chunks WHERE document_id=%s", (document_id,))

    async def get_chunk(self, collection, chunk_id):
        async with self.pool.connection() as connection:
            cursor = await connection.execute("SELECT * FROM document_chunks WHERE collection=%s AND id=%s", (collection, chunk_id))
            return await cursor.fetchone()

    async def parent_chunks(self, collection, parent_section_id):
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM document_chunks WHERE collection=%s AND parent_section_id=%s ORDER BY page_number NULLS LAST, chunk_index, id",
                (collection, parent_section_id),
            )
            return await cursor.fetchall()

    async def ids(self, collection):
        async with self.pool.connection() as connection:
            cursor = await connection.execute("SELECT id FROM document_chunks WHERE collection=%s ORDER BY id", (collection,))
            return [row["id"] for row in await cursor.fetchall()]

    async def ingestion_report(self, collections):
        """Validate persisted data, scoped to the imported collection namespaces."""
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT count(DISTINCT document_id) AS total_documents, count(*) AS total_chunks, "
                "count(*) FILTER (WHERE embedding IS NULL) AS missing_embeddings, "
                "count(*) FILTER (WHERE page_number IS NULL OR page_number < 1) AS missing_page_numbers, "
                "count(*) FILTER (WHERE document_id IS NULL OR btrim(document_id)='') AS missing_document_ids, "
                "count(*) FILTER (WHERE parent_section_id IS NULL OR btrim(parent_section_id)='') AS missing_parent_ids, "
                "count(*) - count(DISTINCT (collection, chunk_id)) AS duplicate_chunk_ids, "
                "array_agg(DISTINCT vector_dims(embedding)) FILTER (WHERE embedding IS NOT NULL) AS embedding_dimensions "
                "FROM document_chunks WHERE collection=ANY(%s)", (list(collections),),
            )
            return await cursor.fetchone()

    async def search(self, collection, embedding, limit, filters=None, *, mode=None, explain=False):
        mode = mode or self.search_mode
        if mode not in {'exact', 'hnsw'}:
            raise ValueError('Invalid search mode')
        if not 0 < limit <= 10000:
            raise ValueError("Search limit must be between 1 and 10000")
        clauses, params = [], [vector_literal(embedding, self.dimension), collection]
        for key, condition in (filters or {}).items():
            if isinstance(condition, dict):
                if len(condition) != 1 or next(iter(condition)) not in {"$eq", "$in"}:
                    raise ValueError("Only metadata equality and $in filters are supported")
                values = condition.get("$in", [condition.get("$eq")])
                if not isinstance(values, list):
                    raise ValueError("$in requires a list")
            else:
                values = [condition]
            clauses.append("(" + " OR ".join("metadata @> %s" for _ in values) + ")" if values else "FALSE")
            params.extend(Jsonb({key: value}) for value in values)
        where = " AND " + " AND ".join(clauses) if clauses else ""
        order = 'distance, id'
        if mode == 'hnsw':
            # ANN requires a bare distance operator in ORDER BY, not power().
            order = 'embedding <-> %s::vector'
            params.append(vector_literal(embedding, self.dimension))
        params.append(limit)
        statement = (
            'SELECT id, content, metadata, power(embedding <-> %s::vector, 2) AS distance '
            'FROM document_chunks WHERE collection=%s' + where + ' ORDER BY ' + order + ' LIMIT %s'
        )
        async with self.pool.connection() as connection:
            async with connection.transaction():
                if mode == 'hnsw':
                    await connection.execute("SELECT set_config('hnsw.ef_search', %s, true), set_config('hnsw.iterative_scan', 'strict_order', true)", (str(self.ef_search),))
                cursor = await connection.execute(('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ' if explain else '') + statement, params)
                rows = await cursor.fetchall()
        if not explain and mode == 'hnsw' and len(rows) < limit:
            # Iterative scans have resource bounds; don't silently starve a filter.
            return await self.search(collection, embedding, limit, filters, mode='exact')
        return rows


class PgVectorClient:
    """Narrow compatibility boundary. Call only from a worker, never the DB loop."""
    def __init__(self, repository, loop, timeout=30):
        self.repository, self.loop, self.timeout = repository, loop, timeout

    def call(self, coroutine):
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            coroutine.close()
            raise RuntimeError("Sync vector adapter requires asyncio.to_thread")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("Vector operation timed out") from None

    def get_collection(self, name):
        return PgVectorCollection(self, name)


class PgVectorCollection:
    def __init__(self, client, name):
        self.client, self.name = client, name

    def get(self, include=None):
        return {"ids": self.client.call(self.client.repository.ids(self.name))}

    def count(self):
        return len(self.get()["ids"])

    def query(self, *, query_embeddings, n_results, include=None, where=None):
        results = [self.client.call(self.client.repository.search(self.name, vector, n_results, where)) for vector in query_embeddings]
        return {"ids": [[r["id"] for r in rows] for rows in results],
                "distances": [[r["distance"] for r in rows] for rows in results]}


@contextmanager
def standalone_client(config):
    """One pool and loop for a complete synchronous CLI operation."""
    from src.embeddings import create_embedding_model
    dimension = embedding_dimension(create_embedding_model(config))
    # Validate everything that can fail synchronously before starting a background loop.
    pool = create_pool(config)
    repository = PgVectorRepository(pool, dimension, config.embedding_model, config.embedding_model_revision,
                                    search_mode=getattr(config, 'pgvector_search_mode', 'exact'),
                                    ef_search=getattr(config, 'hnsw_ef_search', 40))
    timeout = config.db_pool_timeout
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, name="pgvector-cli", daemon=True)
    try:
        thread.start()
    except BaseException:
        loop.close()
        raise
    client = PgVectorClient(repository, loop, timeout)
    try:
        client.call(pool.open(wait=True, timeout=config.db_pool_timeout))
        yield client
    finally:
        try:
            client.call(pool.close())
            client.call(loop.shutdown_asyncgens())
            # Python 3.11 (Docker) has no timeout argument; the adapter bounds this wait.
            client.call(loop.shutdown_default_executor())
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()


def make_chunk(record, collection, embedding, representation=False):
    metadata = dict(record["metadata"])
    metadata.update({key: value for key, value in record.items() if key not in {"metadata", "searchable_text", "text"}})
    document_id = str(metadata.get("source_file") or metadata.get("manual") or "").strip()
    if not document_id:
        raise ValueError("Chunk requires source_file or manual document identity")
    parent = hashlib.sha256(json.dumps([document_id, metadata.get("heading_path", []), metadata.get("chapter"), metadata.get("section")], sort_keys=True).encode()).hexdigest()
    return VectorChunk(str(record["representation_id" if representation else "segment_id"]), collection,
                       document_id, record["text" if representation else "searchable_text"], embedding,
                       metadata, parent, int(record.get("segment_index", 0)))
