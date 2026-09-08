"""FastAPI application resources."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.config import settings
from src.groq_limits import RedisGroqLimiter, configure_groq_limiter
from src.llm import close_async_groq_client, initialize_async_groq_client
from src.ingest import validate_indexes
from src.retrieval import configure_chroma_client, create_chroma_client
from src.retrieval import load_resources
from src.cache import close_cache, configure_cache
from src.embeddings import create_embedding_model, create_reranker
from src.request_store import RedisRequestStore
from src.vector_store import PgVectorRepository, PgVectorClient, embedding_dimension


logger = logging.getLogger(__name__)


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=[("src.schemas", "SourceInfo")])


async def _open_postgres_pool(stack: AsyncExitStack):
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=getattr(settings, "db_pool_min_size", 1),
        max_size=getattr(settings, "db_pool_max_size", 10),
        timeout=getattr(settings, "db_pool_timeout", 30),
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                "connect_timeout": getattr(settings, "db_pool_timeout", 30)},
    )
    stack.push_async_callback(pool.close)
    await pool.open(wait=True)
    return pool


async def _open_checkpointer(stack: AsyncExitStack, pool=None):
    if settings.checkpoint_backend == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await stack.enter_async_context(aiosqlite.connect(settings.sqlite_path))
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA synchronous=NORMAL")
        await connection.execute("PRAGMA busy_timeout=30000")
        await connection.commit()
        return AsyncSqliteSaver(connection, serde=_serializer()), pool

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    if pool is None:
        pool = await _open_postgres_pool(stack)
    return AsyncPostgresSaver(pool, serde=_serializer()), pool


def validate_search_indexes(chroma_client) -> int:
    """Validate deployment prerequisites without rebuilding any indexes."""
    settings.validate()
    if not settings.manuals_dir.is_dir() or not any(settings.manuals_dir.glob("*.pdf")):
        raise FileNotFoundError("manuals")
    required = (
        settings.evidence_units_path,
        settings.retrieval_segments_path,
        settings.indexes_dir / "manifest.json",
    )
    if settings.vector_store == "chroma" and settings.chroma_mode == "local":
        required += (settings.chroma_dir / "chroma.sqlite3",)
    if any(not path.exists() for path in required):
        raise FileNotFoundError("indexes")
    chroma_client.get_collection(settings.chroma_collection)
    return validate_indexes(chroma_client=chroma_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open checkpoint storage and compile the graph once per API process."""
    app.state.graph = None
    app.state.ready_error = None
    stack = AsyncExitStack()
    try:
        settings.validate()
        postgres_pool = None
        if settings.checkpoint_backend == "postgres" or settings.vector_store == "pgvector":
            postgres_pool = await _open_postgres_pool(stack)
        app.state.postgres_pool = postgres_pool
        embedding_model = await asyncio.to_thread(create_embedding_model, settings)
        if settings.vector_store == "pgvector":
            dimension = await asyncio.to_thread(embedding_dimension, embedding_model)
            repository = PgVectorRepository(postgres_pool, dimension, settings.embedding_model, settings.embedding_model_revision,
                                            search_mode=settings.pgvector_search_mode, ef_search=settings.hnsw_ef_search)
            if not await repository.health_check():
                raise RuntimeError("pgvector extension unavailable; run python -m src.pgvector_migrate")
            app.state.vector_repository = repository
            chroma_client = PgVectorClient(repository, asyncio.get_running_loop(), settings.db_pool_timeout)
        else:
            chroma_client = await asyncio.to_thread(create_chroma_client, settings)
        configure_chroma_client(chroma_client)
        stack.callback(configure_chroma_client, None)
        await asyncio.to_thread(validate_search_indexes, chroma_client)
        # Load CPU models before accepting requests; retrieval only reuses these caches.
        await asyncio.to_thread(create_reranker, settings)
        limiter = await RedisGroqLimiter.connect(settings)
        app.state.redis = limiter.client
        app.state.request_store = RedisRequestStore(
            limiter.client,
            settings.chat_request_ttl_seconds,
            settings.thread_ownership_ttl_seconds,
        )
        configure_cache(settings.redis_url)
        stack.callback(close_cache)
        configure_groq_limiter(limiter)
        stack.push_async_callback(limiter.close)
        initialize_async_groq_client(settings)
        stack.push_async_callback(close_async_groq_client)
        checkpointer, postgres_pool = await _open_checkpointer(stack, postgres_pool)
        app.state.postgres_pool = postgres_pool
        app.state.chroma_client = chroma_client
        setup_lock = limiter.client.lock(
            "opcenter:startup:checkpoint-setup",
            timeout=120,
            blocking_timeout=120,
        )
        if not await setup_lock.acquire(blocking=True):
            raise TimeoutError("Checkpoint setup lock timed out")
        try:
            await checkpointer.setup()
        finally:
            await setup_lock.release()
        from src.graph import build_graph

        app.state.graph = build_graph(checkpointer)
    except Exception as exc:
        logger.error("Backend startup failed error=%s", type(exc).__name__)
        app.state.ready_error = str(exc)
        await stack.aclose()
        yield
        return
    except BaseException:
        # Cancellation during startup bypasses Exception and must still release resources.
        await stack.aclose()
        raise

    try:
        yield
    finally:
        app.state.graph = None
        app.state.request_store = None
        await stack.aclose()


def get_graph(request: Request):
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="RAG backend is not ready")
    return graph


async def readiness_checks(app: FastAPI) -> dict[str, bool]:
    """Probe live dependencies without rebuilding or changing indexes."""
    checks = {
        "graph": getattr(app.state, "graph", None) is not None,
        "postgres": settings.checkpoint_backend == "sqlite",
        "redis": False,
        settings.vector_store: False,
        "bm25": False,
        "evidence_units": False,
        "embedding_model": create_embedding_model.cache_info().currsize == 1,
        "reranker_model": create_reranker.cache_info().currsize == 1,
    }
    try:
        checks["redis"] = bool(await app.state.redis.ping())
    except Exception:
        logger.warning("Readiness Redis probe failed")
    pool = getattr(app.state, "postgres_pool", None)
    if pool is not None:
        try:
            async with pool.connection() as connection:
                await connection.execute("SELECT 1")
            checks["postgres"] = True
        except Exception:
            logger.warning("Readiness PostgreSQL probe failed")
    try:
        collection = await asyncio.to_thread(
            app.state.chroma_client.get_collection, settings.chroma_collection
        )
        checks[settings.vector_store] = await asyncio.to_thread(collection.count) > 0
        if settings.vector_store == "pgvector":
            checks["pgvector"] = checks["pgvector"] and await app.state.vector_repository.health_check()
    except Exception:
        logger.warning("Readiness Chroma probe failed")
    try:
        resources = await asyncio.to_thread(load_resources, settings)
        checks["bm25"] = bool(resources.bm25_ids)
        checks["evidence_units"] = bool(resources.evidence_units_by_id)
    except Exception:
        logger.warning("Readiness local index probe failed")
    return checks
