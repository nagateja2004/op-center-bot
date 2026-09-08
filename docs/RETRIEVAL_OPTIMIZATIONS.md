# Retrieval optimization implementation

This increment adds HNSW, parallel dense/BM25 retrieval, request-local embedding
reuse and stage profiling. It does not replace the embedding model, change RRF
weights, shrink the reranker candidate set, or change citation construction.

## Activated local configuration

The backend was restarted with these values after the repeated quality checks:

```dotenv
VECTOR_STORE=pgvector
PGVECTOR_SEARCH_MODE=hnsw
HNSW_EF_SEARCH=200
PARALLEL_HYBRID=true
REUSE_QUERY_EMBEDDINGS=true
DENSE_TIMEOUT=10
BM25_TIMEOUT=10
```

The previous exact-search `.env` is preserved as `.env.rollback-exact-20260907`
(Git-ignored, mode 600). Readiness verifies pgvector and all dependencies. The
six-round benchmark measured hybrid p50 131.97 → 85.65 ms (35.10% reduction versus
exact pgvector), not a speedup over Chroma. Recall@5/MRR were unchanged on the
44 labeled questions; effective exact-top-20 overlap was 96.82%.

The full suite passed **331 tests, zero skipped**, with six existing deprecation
warnings (48.17 s). This includes real SQL/index tests, eight exact/optimized live
corpus checks, deadline/failure/concurrency tests and executor-cleanup tests.

Final live checks: the optimized API returned a sufficient-evidence Factory answer
with Modeling guide citation `[S1]`, physical PDF page 111 (one 9.60-second full
request, not a performance benchmark). The Resource field-definition prompt
returned `in_scope_insufficient` without citations in **both** optimized and exact
API instances. This existing generation/evidence-grading limitation remains;
retrieval Recall@5 alone is not proof of complete answer quality. The temporary
exact comparison server on port 8001 was stopped; the optimized server remains
on port 8000. Python compilation, Compose validation and whitespace checks passed.
After the Python 3.11-compatible executor-cleanup adjustment, its affected
optimization test module also passed all eight tests.

## Database and search

`src/sql/002_hnsw.sql` creates `document_chunks_embedding_hnsw` using
`vector_l2_ops`. The existing model produces normalized embeddings, but existing
scores use squared L2, so L2 is retained for compatibility rather than changing
operators to cosine. The ANN query orders by the bare `<->` operator so PostgreSQL
can use the index, while returning the existing squared-distance score.

Create the index explicitly; it is not built during API startup:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m src.pgvector_migrate --hnsw
```

Optional setup flags: `--hnsw-m 16 --hnsw-ef-construction 64`. These are the tested
build defaults. Repeating setup does not rebuild an existing index or change its
parameters. Build during an ingestion maintenance window; regular index creation
can block writers. No table, Chroma collection, or existing index is deleted.

`PGVECTOR_SEARCH_MODE=exact` retains the exact-search path and avoids the ANN index.
`hnsw` uses a transaction-local `HNSW_EF_SEARCH` and pgvector's strict-order iterative
scan. This requires pgvector >=0.8; readiness validates extension/model/dimension
and a valid L2 HNSW index. If bounded ANN filtering underfills top-k, the repository
retries exact search with the same filter. This can increase latency for narrow
filters. Parent, document and JSONB indexes remain unchanged.

The implementation follows [pgvector's HNSW and iterative-scan documentation](https://github.com/pgvector/pgvector#hnsw).
It probes dimensions; HNSW on `vector` rejects dimensions above 2,000 before setup.

## Parallel execution and embedding reuse

`PARALLEL_HYBRID=true` connects the existing synchronous retrieval worker to its
application-owned async PostgreSQL loop. Dense search awaits the async repository;
BM25 runs via `asyncio.to_thread`. `asyncio.gather` overlaps only independent
branches. RRF, neighbor expansion, reranking and evidence resolution remain ordered.
Chroma rollback continues using its existing sequential branch implementation.

Both branches have independent deadlines (`DENSE_TIMEOUT` and `BM25_TIMEOUT`,
10 seconds by default, below the 30-second adapter/pool timeout). One branch failure
retains the other branch's results; both failures raise a controlled retrieval
error. Cancellation is propagated. A timed-out worker thread cannot be forcibly
stopped; its CPU work may finish after the timeout. Thread count is bounded by
the event loop's executor and normal API inference admission still applies.
The CLI closes its pool, asynchronous generators and default executor on exit.

`REUSE_QUERY_EMBEDDINGS=true` shares query vectors inside a bounded context-local
scope spanning the existing multi-query retrieval call. Body, representation,
preferred-manual and semantic-heading searches reuse the same vector for identical
query text and model instance. Distinct rewritten queries get separate vectors.
No global user-query dictionary was introduced. Redis embedding-cache keys retain
exact query text/model revision; retrieval cache keys include optimization settings.

## Profiling and quality safeguards

`src/retrieval_metrics.py` provides reusable context-local stage profiling.
The benchmark records embedding, dense/BM25 and reranker timing; the parallel path
also records branch wall time and safe exception-class diagnostics. No query text
or secrets are added to these timing logs. Sequential dense timing includes query
embedding; parallel dense timing excludes the preceding shared embedding. Use
total adapter latency, not those unequal stage boundaries, for speed comparisons.

The reranker remains the same pinned local cross-encoder, bounded to 20 prepared
candidates with unchanged final limits. Profiling is not permission to shrink
the candidate set without a separate recall evaluation.

## Reproduce validation

```bash
PGVECTOR_TEST_DATABASE_URL='postgresql:///opcenter_vector_test?host=/tmp' \
PGVECTOR_CORPUS_DATABASE_URL='postgresql:///opcenter?host=/tmp' \
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q --tb=short

HNSW_EF_SEARCH=200 .venv/bin/python -m benchmarks.run_retrieval_benchmark --rounds 6 --optimized

HNSW_EF_SEARCH=200 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LANGSMITH_TRACING=false \
  .venv/bin/python -m benchmarks.profile_query_reuse
```

The benchmark now compares Chroma, exact pgvector, HNSW sequential, HNSW parallel,
and parallel with reuse. It also stores exact-vs-ANN top-20 IDs/overlap and actual
`EXPLAIN ANALYZE` JSON, rejecting a representative plan that does not use HNSW.
The simplified adapter queries one dense branch, so its reuse-on/off difference
is not evidence of savings across multiple branches; the separate full-path reuse
check measures inference counts and verifies unchanged IDs for its three queries.

Initial ef_search=40 lost a labeled table-definition result, and a probe at 100
did not recover it. Those settings were not activated. A probe at 200 recovered
that query's exact top-20; the full repeated benchmark is the release gate.
See the benchmark report for measured results and their limitations.

## Rollback

Set `PGVECTOR_SEARCH_MODE=exact`, `PARALLEL_HYBRID=false` and
`REUSE_QUERY_EMBEDDINGS=false`, then restart the API. This does not require dropping
the HNSW index. The broader `VECTOR_STORE=chroma` rollback also remains available.
Do not remove either store until the full production quality/load gates and a
rollback soak period are complete.
