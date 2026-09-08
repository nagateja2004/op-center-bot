# PostgreSQL + pgvector Migration Plan

## Optimization increment — 2026-09-07

HNSW setup (L2), index-compatible query ordering, exact fallback, parallel async
dense/threaded BM25, branch deadlines, request-local query embedding reuse and
reranker profiling are implemented. Candidate budgets and RRF weights were retained.
The six-round local benchmark measured 35.10% lower hybrid p50 than exact pgvector,
with unchanged Recall@5/MRR; it did not beat Chroma. Full-path checks reduced
embedding computations from four to one on three queries without changing IDs.
See [optimization details](RETRIEVAL_OPTIMIZATIONS.md) and
[benchmark results](../benchmarks/BENCHMARK_RESULTS.md). Earlier "pending HNSW"
status entries below are historical. Full production migration gates remain open.

## Verified local cutover — 2026-09-07

This update supersedes the earlier Docker-blocked status below. Native PostgreSQL
17.11 + pgvector 0.8.6 now serve the running local backend. `.env` selects pgvector;
the code default and Chroma data remain available for rollback. Existing JSON was
backfilled with the same pinned model (probed dimension 384): eight documents,
15,721 body chunks and 21,121 search representations. Persisted checks found zero
missing embeddings/pages/document IDs/parent IDs or duplicate chunk IDs, and both
ID sets match JSON.

Final verification: **314 passed, zero skipped**, including nine disposable SQL
integration tests and three read-only live corpus retrieval/citation checks. A
real chat request returned sufficient evidence with the Modeling guide citation,
physical page 111. Backend `/ready` verifies pgvector and all other dependencies.

Post-cutover fixes isolate tests from live `.env` credentials/backend selection
and redact credential-bearing fields from the settings representation. Native
services bypass Docker without resetting it or deleting volumes. SQLite chat
history remains unchanged. Detailed restart, security, test and rollback notes:
[LOCAL_PGVECTOR_CUTOVER.md](LOCAL_PGVECTOR_CUTOVER.md).

This is a storage cutover, not completion of all target phases: HNSW, query-wide
filter relaxation, parallel hybrid retrieval, PostgreSQL context expansion and
comprehensive before/after instrumentation/benchmarks remain pending. Historical
checklists below must not be read as a claim that these phases were implemented.

## Phase 3 ingestion increment

- `python -m src.pgvector_migrate --manuals` explicitly selects pgvector for the
  job, creates/validates its schema, and runs the existing PDF ingestion function.
  The same job pool/client is reused for index checks, writes and final validation.
  Parsing, OCR, chunking, model configuration, deterministic source IDs, and
  Chroma ingestion remain unchanged. `--backfill` still imports canonical JSON
  without PDF parsing. The two modes are mutually exclusive.
- Existing model caching and normalized embeddings are retained. Inference uses
  256-record batches; publication uses 256-row database batches in one transaction.
  Repeated full-corpus snapshots replace the same IDs and remove stale rows in the
  two selected namespaces. Input dimensions/pages are checked before publication.
- `PgVectorRepository.ingestion_report` aggregates persisted document/chunk totals,
  missing embeddings/pages/document IDs/parent IDs, duplicate chunk IDs per
  namespace, and actual stored embedding dimensions. The importer prints this
  report and rejects invalid/empty results. Existing JSON-to-store ID validation
  runs as well. The ordinary `VECTOR_STORE=pgvector python -m src.ingest` path also
  prints and validates the report, including on unchanged reruns.
- `tests/test_pgvector_ingest.py` covers deterministic repeated imports, metadata,
  batched inference/publication, prepublication rejection, summaries and the PDF
  command route. A new opt-in real-database test repeats a multi-batch import and
  checks the persisted summary and absence of duplicates.
- Docker was checked again and reports “Docker Desktop is unable to start”.
  No real corpus import was attempted, and real SQL execution remains a release
  gate. No new HNSW optimization or retrieval behavior changes are included.

Exact local and Docker commands are in README's phase 3 ingestion section.

Verification: relevant ingestion/storage/embedding tests **83 passed, 9 skipped**;
full regression suite **293 passed, 9 skipped**. The skipped cases are the opt-in
database integration tests. Compose configuration and whitespace checks passed.

## Phase 2 implementation status (2026-09-06)

The section below records the implemented storage increment requested in PROMPT 2.
It supersedes the earlier audit's speculative schema, environment names and phase
ordering where they differ. The remaining audit is retained as future-work context,
not as a claim that HNSW or the full target architecture has been implemented.

### Delivered storage changes

- `src/vector_store.py`: `VectorStore` protocol, `VectorChunk`, async
  `PgVectorRepository`, configurable psycopg pool factory, validated vector
  encoding, model-dimension probe, and a narrow synchronous collection adapter.
  The adapter schedules database work on the pool's owning loop and rejects calls
  from that loop to prevent deadlocks. Existing async retrieval already runs in a
  worker thread. Standalone jobs own one dedicated loop and pool for the job.
- `src/sql/001_pgvector.sql`: explicit, transactional/idempotent migration;
  `CREATE EXTENSION IF NOT EXISTS vector`; model identity table; `document_chunks`
  with collection/id composite key, document_id, chunk_id, content,
  `vector(__DIMENSION__)`, manual_name, chapter, section, page_number,
  parent_section_id, chunk_index, content_type, JSONB metadata and created_at.
  Only ordinary document/parent and metadata GIN indexes. **No ANN index.**
- The current pinned self-hosted model was probed offline again: 384 dimensions.
  Schema creation substitutes that measured integer, not a hardcoded 384. Stored
  model name/revision/dimension and actual column type are checked on startup.
- Batch upsert, transactional two-collection snapshot replacement, document
  deletion, ID lookup, exact/filtered dense search, ordered parent lookup and
  extension/schema/model health checks are implemented. Metadata predicates use
  parameterized JSONB equality/`$eq`/`$in`; unsupported operators fail closed.
- `src/pgvector_migrate.py`: explicit schema and optional JSON backfill CLI.
  Re-embeds body segments and search representations with the existing pinned
  model. Preserves source IDs and all nested metadata. Completes inference before
  transactional publication; failed publication rolls back stale-ID deletion.
- `src/config.py`, `.env.example`: `VECTOR_STORE=chroma|pgvector` (default chroma),
  `DB_POOL_MIN_SIZE=1`, `DB_POOL_MAX_SIZE=10`, `DB_POOL_TIMEOUT=30`. Reuses
  `DATABASE_URL`, existing model settings and collection names.
- `backend/dependencies.py`: one pool per application process shared by pgvector
  and PostgreSQL checkpoints, including support for SQLite checkpoints plus
  pgvector. Models and synchronous index checks are offloaded during startup.
  Shared pool closes through the lifespan exit stack on shutdown/failure.
  Readiness checks the selected backend; no schema creation occurs at startup.
- `src/ingest.py`, `src/retrieval.py`: selected-store ingestion/validation,
  collection compatibility and backend/model-aware retrieval cache keys. The
  established BM25, fusion, reranking, compression, parent EvidenceUnit resolution,
  LLM, citations, API and UI stay unchanged. Some internal names still say
  `chroma_collection`; those now refer to the narrow compatibility interface.
- `docker-compose.yml`: upstream `pgvector/pgvector:pg16`, retained persistent
  `postgres_data` volume definition, ingestion database URL/dependency. Chroma is
  retained. `requirements-backend.txt` documents reuse of existing psycopg pins;
  no redundant PostgreSQL driver or vector Python codec is needed.
- `README.md`: startup, explicit migration, cutover, host tests and rollback.
- `tests/test_vector_store.py`: dimension/vector validation, metadata mapping,
  worker adapter, shared checkpoint pool, configuration and eight opt-in isolated
  database integration tests covering connection/extension, writes, metadata,
  search/filtering, parents/deletion, connection reuse, model mismatch and rollback.

### Verification and release gate

- Fresh offline full suite: **284 passed, 8 skipped** (27.43 seconds), including
  application lifespan/pool sharing and two-collection backfill unit coverage.
- Existing backend/retrieval/ingestion subset: **69 passed**.
- Compose configuration validation succeeded; model probe returned 384.
- Docker's local endpoint reports **“Docker Desktop is unable to start”**.
  No local PostgreSQL server was available. The eight database integration tests
  were collected but skipped without `PGVECTOR_TEST_DATABASE_URL`; SQL execution,
  image startup, real backfill and end-to-end pgvector behavior are **not yet
  verified**. Do not switch production until these tests and a staging backfill pass.
- No application database migration or Chroma deletion was performed by this work.

### Migration, rollback and risks for this increment

1. Back up and test restore of checkpoint DB, Chroma, manuals and `indexes/`.
   For an existing Alpine PostgreSQL volume, use a separate pgvector instance and
   logical restore; account for libc/collation changes. Do not blindly replace its
   image or remove its volume. Pin the validated upstream image digest in production.
2. Start pgvector PostgreSQL; run `python -m src.pgvector_migrate --backfill` with
   the application database URL and current JSON/model cache mounted. This does
   not reingest PDF/OCR content, alter JSON, or touch Chroma.
3. Run the real-database tests against a disposable database and verify staging
   corpus IDs, filtered results, full answers/citations and baseline latency.
4. Set `VECTOR_STORE=pgvector`, restart API processes, check readiness and soak.
   Roll back by restoring `VECTOR_STORE=chroma` and restarting. If JSON/manuals
   changed after cutover, restore their matching Chroma-era snapshot as well.
5. Subsequent ingestion updates only the selected backend. Pause API traffic
   during ingestion and restart afterward; JSON plus database publication is not
   globally atomic. Run only one ingestion/backfill writer at a time.

Exact squared-L2 search intentionally matches default Chroma scoring, but ANN
candidate differences and tie ordering can still change answers. Measure those
changes before cutover. Backfill currently holds generated vectors in memory
before publication (appropriate to the audited corpus, not unbounded datasets).
Large corpora should later use staging/COPY publication. Long backfills have a
one-hour publication wait limit; query adapter waits use DB_POOL_TIMEOUT.
Parent-section IDs hash document plus heading path/chapter/section; repeated
identical headings in one manual share a logical parent. Occurrence-level section
modeling is deferred and the current answer expansion does not use this new field.
Pool limits are per process; budget checkpoint, vector and ingestion concurrency
together. Full-content JSON remains mandatory; this is not a standalone corpus DB.

### Next implementation order (not part of this change)

1. Clear the real-database/staging release gate above.
2. Establish corpus/answer parity and p50/p95/p99 exact-search baseline.
3. Add measured HNSW optimization with recall and filter selectivity tests.
4. Convert hybrid orchestration to direct async dense + parallel BM25 retrieval,
   retaining existing fusion rules and eliminating the compatibility bridge.
5. Introduce versioned corpus publication and occurrence-aware parent sections.
6. Evaluate expansion/compression ordering changes independently, then benchmark
   concurrent load, cancellation, pool contention and end-to-end latency again.

---

## Original audit and full target design (future phases)

**Goal:** Replace Chroma dense retrieval with PostgreSQL 16 plus pgvector HNSW while preserving the current Opcenter ingestion semantics, BM25 path, reranker, answer generation, citations, API, and Streamlit UI.

**Architecture:** Keep PostgreSQL as the deployment database and add versioned retrieval tables with 384-dimensional pgvector columns. Run asynchronous pgvector retrieval and the existing in-memory BM25 search concurrently, fuse ranks with the current weighted RRF rules, rerank segments, expand selected results to their stored parent evidence/section context, compress that context, and leave the downstream LangGraph answer path unchanged.

**Tech stack:** Python 3.11/3.12, FastAPI, LangGraph, psycopg 3, psycopg-pool, PostgreSQL 16, pgvector, SentenceTransformers, `rank-bm25`, Redis, Streamlit, Groq, pytest.

**Spec:** User request “PROMPT 1: AUDIT EXISTING OPCENTER RAG PROJECT,” received 2026-09-06.

## Global constraints

- Do not implement the migration while preparing this document.
- Preserve the existing UI and answer payloads.
- Keep Chroma available as a rollback backend until pgvector passes the soak period.
- Reuse the existing embedding model, BM25 implementation, weighted RRF, cross-encoder, evidence structures, compression, grading, citation validation, and generation flow.
- Use the dimension measured from the configured embedding model and stored vectors. The measured value is **384**.
- Keep CPU-heavy SentenceTransformer and cross-encoder calls away from the asyncio event loop.
- Do not rebuild indexes during API startup or a chat request.

---

## Audit scope and repository state

The audit covers the active repository at `/Users/nagatejay/Documents/op centre bot`, commit `f2787af` on `main`. It includes application source, deployment files, index manifests and representative records, benchmarks, and all collected tests. The PDF manuals and generated figure binaries were treated as corpus data rather than source code.

The working tree contained one unrelated untracked file before this audit:

```text
output/pdf/yerasingu_nagateja_ats_resume.pdf
```

This plan does not touch that file.

Measured corpus and test inventory:

| Item | Current value |
| --- | ---: |
| Manuals | 8 |
| EvidenceUnits | 12,733 |
| RetrievalSegments | 15,721 |
| SearchRepresentations | 21,121 |
| Heading records | 6,932 |
| Concept records | 12,473 |
| Index schema | 8 |
| Ingestion pipeline | `text-only-pymupdf-hierarchical-v8.0` |
| Collected pytest tests | 272 |

## Executive findings

1. The project already implements most of the requested logical RAG pipeline. Query planning, manual hints, local embeddings, BM25, weighted RRF, local cross-encoder reranking, evidence expansion, deterministic contextual compression, grading, citations, and latency timing all exist.
2. Chroma owns only the dense body-segment and search-representation indexes. JSON files remain the canonical application-side records, and the backend rebuilds BM25 from `retrieval_segments.json` at startup.
3. PostgreSQL already stores LangGraph checkpoints through one `AsyncConnectionPool`. The migration can reuse the database, driver, and pool lifecycle instead of adding a second database stack.
4. The current request path offloads synchronous retrieval and reranking to worker threads. This prevents direct event-loop blocking, but vector, representation, BM25, heading, concept, and preferred-manual searches execute sequentially inside that thread.
5. The current `expand_context()` adds previous and next child segments **before** reranking. `resolve_evidence_units()` expands selected segments or representations to an `EvidenceUnit` **after** reranking. The repository does not persist a separate parent-section entity.
6. Metadata-aware retrieval is partial. The planner produces manual preferences; Chroma applies manual filters only in preferred-manual side searches. The main dense and BM25 searches remain global, and other fields such as release, content type, chapter, and section are not first-class query filters.
7. The retrieval benchmark already reports p50 and p95. It must add p99, stage timing, pgvector/Chroma comparison, concurrency profiles, and cold/warm runs.

## Current architecture

```text
Browser
  -> Streamlit app.py
  -> FastAPI POST /v1/chat and SSE GET /v1/chat/{request_id}/stream
  -> LangGraph
       -> understand_question
       -> retrieve_documents
            -> Chroma body-vector search
            -> Chroma search-representation vector search
            -> in-memory rank-bm25 search
            -> heading and concept lookup paths
            -> preferred-manual side searches
            -> weighted RRF across paths and query variants
       -> expand_context (previous/next RetrievalSegments)
       -> rerank_documents (local cross-encoder)
            -> resolve child candidates to complete EvidenceUnits
            -> deterministic contextual compression
       -> grade evidence, with one bounded retry
       -> generate with Groq
       -> verify citations and groundedness
       -> optional diagram
  -> answer, sources, evidence status, figures, optional diagram

PostgreSQL: LangGraph checkpoints only
Redis: request handoff, ownership, rate limits, and caches
Chroma: dense segment and representation vectors
JSON: evidence, segments, representations, headings, concepts, manifest, audit
Process memory: BM25 and record maps
```

## Current component inventory

| Concern | Current implementation | Audit result | Migration action |
| --- | --- | --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Self-hosted through `HuggingFaceEmbeddings`; normalized output | Reuse unchanged |
| Embedding dimension | Model reports 384; probe produced 384 values; both Chroma collections store 384-value vectors | Determined from the configured model and stored data, not inferred from the model name | Declare `vector(384)` and validate at startup/backfill |
| Vector database | Chroma 1.5.9; HTTP server in Compose, persistent local client in tests/dev | Two collections: body segments and search representations | Add pgvector adapter, shadow it, then remove Chroma after soak |
| Ingestion | Explicit offline `python -m src.ingest` | PyMuPDF, selective Tesseract, PaddleOCR fallback, hierarchical evidence construction, stable hashed IDs, incremental file-hash reuse | Preserve extraction and record construction; replace the dense index sink |
| Chunk schema | `EvidenceUnit` parent plus one or more `RetrievalSegment` children; separate `SearchRepresentation` children | Strong existing parent/child contract | Map directly to relational tables; add a stable section key and unit order |
| Stored metadata | Manual, source file, release, chapter, section, subsection, heading path, content type, printed page, PDF page, TOC flag; table/step/annotation data on EvidenceUnits | Sufficient for citations and useful metadata filters | Store typed filter columns plus complete JSONB metadata |
| BM25 | `rank_bm25.BM25Okapi` rebuilt from segment text at backend startup | `indexes/bm25.pkl` exists but current code does not load it | Reuse BM25; build it from pgvector/PostgreSQL rows after cutover |
| Hybrid retrieval | Body vectors, representation vectors, BM25, headings, concepts, preferred-manual variants, and multi-query fusion | More capable than the target’s two-path minimum | Preserve all paths; make dense and BM25 execution concurrent |
| RRF | `_rank_fuse()` with `RRF_K = 60`, per-path weights, original-query weight 1.15 | Fuses ranks rather than incompatible raw scores | Reuse unchanged |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, pinned revision `c5ee24cb16019beea0893ab7796b1df96625c6b8` | Local cached `CrossEncoder`; finite-score validation and fallback | Reuse unchanged |
| Parent retrieval | Neighbor segments before rerank; child-to-EvidenceUnit resolution after rerank | No stored parent-section record | Move expansion after rerank and add bounded section membership expansion |
| Contextual compression | `src/compression.py::compress_evidence` | Deterministic sentence, procedure, and table compression; preserves source metadata | Reuse unchanged after parent expansion |
| LLM provider | Groq, with role-specific primary/fallback models and async client | Planner, grader, broadener, answer, verifier, diagram, evaluation judge | No migration change |
| API | FastAPI with async lifespan and SSE streaming | Stable external contract | Preserve routes and payloads |
| UI | Streamlit | Calls only the backend and renders citations, tables, figures, and diagrams | No migration change |
| Configuration | Frozen `Settings` dataclass populated from `.env` via `python-dotenv` | Chroma and PostgreSQL settings share one object | Add retrieval-backend and HNSW/pool settings; retain Chroma settings during rollout |
| Runtime state | PostgreSQL checkpoints, Redis distributed request state/cache | PostgreSQL already uses async psycopg and pooling | Reuse and centralize the pool |
| Tests | 272 unit/integration-style tests plus 50-case evaluation and 44 scored retrieval cases | Strong behavior and citation coverage; Chroma-specific tests exist | Keep behavioral tests, add pgvector integration/parity/concurrency tests |
| Instrumentation | Per-node and total LangGraph timings; process-local Prometheus summaries | Only sum/count at runtime; retrieval stages are not separated | Add stage labels and structured timing; calculate percentiles in benchmarks |

## Embedding dimension evidence

The audit loaded the configured model in offline mode and checked both its declared output and an actual normalized probe:

```text
model=sentence-transformers/all-MiniLM-L6-v2
revision=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
reported_dimension=384
probe_dimension=384
normalized_probe_norm=1.0000000153962765
```

Chroma inspection returned:

```text
opcenter_manuals:                 count=15721 dimension=384
opcenter_manual_representations:  count=21121 dimension=384
```

The migration must fail before loading data if the configured model, a probe vector, stored corpus metadata, and the SQL column dimension disagree.

## Existing data contracts

### EvidenceUnit

`src/schemas.py::EvidenceUnit` contains:

- `evidence_id`
- full `text`
- `content_type`
- source `metadata`
- `token_count`
- optional structured table
- ordered procedure steps
- connected annotations

### RetrievalSegment

`src/schemas.py::RetrievalSegment` contains:

- `segment_id` and parent `evidence_id`
- embedding/search text in `searchable_text`
- `content_type` and source metadata
- `segment_index`
- previous/next segment IDs within the EvidenceUnit
- word count, embedding token count, and effective model limit

### SearchRepresentation

`src/schemas.py::SearchRepresentation` contains:

- `representation_id` and parent `evidence_id`
- one of `heading`, `definition`, `procedure_title`, `table_title_headers`, `field_name_description`, or `acronym_alias`
- representation text, source metadata, and embedding token count

### Metadata fields

All three record types carry the source metadata needed to preserve the API:

```text
manual
source_file
release
chapter
section
subsection
heading_path
content_type
printed_page
pdf_page
is_toc
```

EvidenceUnits additionally preserve structured tables, procedure steps, and annotations. Retrieval segments may add the table rows needed for the specific searchable segment.

## Complete ingestion trace

1. `src.ingest.main()` calls `ingest_manuals()` as an explicit offline job.
2. `ingest_manuals()` validates parser configuration, enumerates top-level PDFs, hashes each file, and loads the version-8 manifest and JSON artifacts.
3. Unchanged PDF hashes reuse their existing EvidenceUnits and RetrievalSegments. Changed or new PDFs call `_ingest_pdf()`.
4. `_ingest_pdf()` opens the PDF with PyMuPDF and extracts the title and release from PDF metadata.
5. `_extract_page_data()` obtains page blocks and outline context. `_extract_page_content()` uses native text first, Tesseract for low-text pages, and PaddleOCR structure parsing for weak or complex OCR layouts.
6. The pipeline detects TOC/index pages, repeated margins, printed page numbers, headings, tables, procedures, warnings, and other content elements.
7. `_extract_groups()` assembles `SectionGroup` objects with chapter, section, subsection, heading path, source page, and ordered content elements.
8. `_evidence_specs()` and `_build_evidence()` create stable hashed EvidenceUnit IDs. They preserve full logical tables, ordered procedures, conditions, warnings, and citations.
9. `_segments_for_unit()` splits each EvidenceUnit into embedding-safe children. It prefixes manual/chapter/section/type context, keeps table headers with rows, and links sibling segments with previous/next IDs.
10. `_build_heading_index()` and `_build_concept_index()` create deterministic manual-derived lookup records.
11. `_build_search_representations()` creates small alternate vector targets for headings, definitions, procedures, tables, fields, and aliases.
12. The CLI writes EvidenceUnits, RetrievalSegments, SearchRepresentations, heading/concept indexes, figures, audit data, and manifest through temporary-file replacement.
13. `build_indexes()` loads the pinned local embedding model and embeds body segments and search representations in batches of 256.
14. It recreates two Chroma collections and writes text, flattened metadata, stable IDs, and vectors.
15. `validate_indexes()` checks duplicate IDs, parent references, embedding token limits, and exact ID agreement between JSON and both Chroma collections.
16. The completed manifest records counts, model-safe token limit, processed/skipped/removed manuals, and index generation time.

## Complete query-time trace

1. Streamlit sends `POST /v1/chat`, then consumes the SSE stream endpoint. Redis stores the pending request and binds its thread to the anonymous session.
2. `backend/routes/chat.py` starts `graph.astream()` with the current conversation/thread checkpoint key.
3. `aunderstand_question()` handles greetings without retrieval. For manual questions it uses deterministic planning or the Groq planner, resolves follow-up context, extracts entities/aliases, builds aspects and search variants, and records preferred manuals.
4. `aretrieve_documents()` obtains the embedding inference gate and offloads the synchronous `retrieve_documents()` node to one worker thread.
5. For each aspect, `retrieve_multiple_queries()` executes up to four search queries. Each `_retrieve_query_paths()` currently calls these paths sequentially:
   - Chroma body-segment vector search
   - Chroma search-representation vector search
   - in-memory BM25 search
   - exact/fuzzy/semantic heading search
   - concept/alias search
   - preferred-manual vector, representation, BM25, heading, and concept searches
6. `_rank_fuse()` applies the existing path weights and `RRF_K=60`. A second RRF pass fuses query variants, with weight 1.15 for the original question.
7. Retrieval adds small entity and manual-preference bonuses after rank fusion, deduplicates exact records, and caches the result in Redis.
8. The LangGraph `expand_context` node adds previous/next RetrievalSegments before reranking.
9. `arerank_documents()` offloads the cross-encoder work to a worker thread under the reranker inference gate.
10. Each aspect reranks at most 20 diverse candidates, keeps six or eight, and falls back to RRF order if the local cross-encoder fails.
11. `resolve_evidence_units()` converts selected segment/representation children into complete EvidenceUnits and merges scores/aspect metadata.
12. `compress_evidence()` selects relevant sentences, ordered procedure steps, table rows, and connected annotations without mutating the parent EvidenceUnit.
13. The grader checks each required aspect. Missing evidence may trigger one broadened retrieval pass.
14. The answer model receives capped compressed evidence. The verifier normalizes citation IDs, removes invalid citations, and checks groundedness.
15. The API maps cited EvidenceUnits to the existing source payload and returns the answer, evidence status, manual figures, and optional validated diagram.

## Components to reuse

Keep these components and their current behavior:

- PDF parsing, OCR selection, hierarchy detection, table/procedure preservation, figure extraction, and ingestion audit
- stable EvidenceUnit, RetrievalSegment, and SearchRepresentation identifiers
- local normalized SentenceTransformer embeddings and pinned revision
- `rank-bm25` tokenization and scoring
- heading, concept, alias, preferred-manual, and multi-query paths
- `_rank_fuse()` and its established path weights
- exact-only deduplication and per-parent representation diversity
- cross-encoder and its finite-score/fallback protections
- deterministic contextual compression
- evidence grading, one-retry policy, answer prompts, citation validation, source payloads, and diagram behavior
- FastAPI routes, SSE events, Streamlit UI, Redis request state, and LangGraph checkpoint semantics
- current golden datasets and evaluation gates

## Chroma coupling map

| Location | Coupling | Required change |
| --- | --- | --- |
| `src/config.py:77-82, 228-233` | Chroma directory/mode/host/port/SSL/collection and validation | Add `VECTOR_STORE_BACKEND`; retain Chroma fields until retirement; add HNSW and retrieval pool settings |
| `src/ingest.py:79-80` | Hard-coded body and representation collection names | Move backend-neutral names to the dense-store boundary |
| `src/ingest.py:1927-1955` | Chroma metadata flattening | Keep only in the Chroma adapter during rollout |
| `src/ingest.py:1958-2042` | Direct Chroma client creation, collection deletion, embedding, and batched inserts | Route records/vectors to a storage writer; add pgvector bulk loader |
| `src/ingest.py:2045-2115` | Validation compares JSON IDs to Chroma collections | Add backend-neutral index validation and PostgreSQL count/parent/dimension checks |
| `src/ingest.py:2229-2279` | Rebuild detection and manifest fields depend on Chroma | Record backend, corpus ID, model revision, dimension, and load status |
| `src/retrieval.py:13, 35, 42-56` | Chroma import, global client, and resource fields | Replace with an injected dense-store abstraction |
| `src/retrieval.py:59-150` | Chroma client lifecycle and collection loading/alignment | Split static lexical resources from async dense-store resources |
| `src/retrieval.py:242-313` | Direct Chroma body and representation queries | Call async dense-store methods with typed filters |
| `src/retrieval.py:967-993` | Preferred-manual Chroma `where` queries | Express through `RetrievalFilter` and the dense-store interface |
| `backend/dependencies.py:17-19, 63-95, 111, 148-184` | Startup, readiness, and app state require Chroma | Create the selected store from the shared PostgreSQL pool; report selected-store readiness |
| `docker-compose.yml:28-38, 53-55, 78-79, 120-130, 164` | Chroma service, dependency, environment, and volume | Keep under a rollback profile, then remove after soak; use a pgvector PostgreSQL image |
| `.env.example:46-47` | Chroma-only retrieval settings | Add backend switch and pgvector/HNSW settings |
| `requirements-backend.txt` | `chromadb` and `langchain-chroma` | Add the Python `pgvector` package; remove Chroma packages only after rollback retirement |
| `retrieval_benchmark.py` | Labels pipelines as Chroma and imports synchronous dense search | Parameterize backend, use async pipeline, and add p99/stage/concurrency results |
| `README.md` | Architecture, deployment, ingestion, benchmark, and operations describe Chroma | Update at cutover; retain rollback instructions during transition |
| `tests/conftest.py` | Forces local Chroma mode | Introduce explicit unit/integration fixtures for both stores during rollout |
| `tests/test_retrieval.py`, `tests/test_ingest.py`, `tests/test_final_regressions.py` | Client construction, Chroma metadata, ID parity, and live collections | Preserve backend-neutral assertions; move Chroma-only checks to adapter tests; add pgvector equivalents |

## Blocking and synchronous work

### Request path

| Operation | Current behavior | Migration requirement |
| --- | --- | --- |
| Query embedding | `embed_query()` is synchronous CPU/model work | Keep under `embedding_gate`; execute in `asyncio.to_thread()` or batch all query variants in one thread call |
| Chroma body/representation queries | Synchronous client calls | Replace with awaited psycopg operations |
| BM25 scoring and Python sorting | Synchronous CPU work over 15,721 rows | Run in `asyncio.to_thread()` and overlap it with dense SQL |
| Semantic heading fallback | Synchronous document embedding, query embedding, and Python dot products | Precompute heading embeddings at startup; keep query work under the embedding gate/thread pool |
| Cross-encoder reranking | Synchronous model inference | Keep the current `reranker_gate` plus `asyncio.to_thread()` |
| Redis cache in `src/cache.py` | Uses synchronous `redis.Redis` | Convert retrieval-cache calls to `redis.asyncio` before the retrieval orchestrator becomes natively async, or explicitly offload them |
| `_available_manual_names()` and `_concept_catalog()` | Synchronous manifest/alias file reads reached from async planning; caches reduce repetition | Load immutable catalogs during lifespan or offload first load |
| Manual figure lookup | `Path.stat()` and `read_bytes()` run inside the async SSE generator | Offload file reads or preload the small catalog; keep image byte limits |
| Context expansion/compression | Synchronous Python over small bounded lists | Keep synchronous; offload only if profiling shows material event-loop delay |

### Startup and offline work

- `validate_search_indexes()`, Chroma collection checks, JSON loading, BM25 construction, and model loading run synchronously inside the async lifespan. Startup accepts no requests yet, but these calls delay readiness and block lifespan progress.
- Build static catalogs and BM25 with `asyncio.to_thread()` during startup. Fetch PostgreSQL rows through the async pool.
- PDF parsing, OCR, section construction, embedding batches, and bulk loading are offline ingestion work. They may remain CPU-bound, but database writes should use async psycopg and bounded batches.
- Do not move index construction into API startup.

## Abstractions to introduce or modify

Introduce only boundaries required for pgvector rollout and rollback.

### `RetrievalFilter`

Add an immutable typed value to `src/schemas.py`:

```python
@dataclass(frozen=True, slots=True)
class RetrievalFilter:
    manuals: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    releases: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    chapters: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
```

Only planner-supported filters should populate this type. SQL must bind all values as parameters; never interpolate metadata values or identifiers.

### `DenseStore`

Create `src/dense_store.py` with one narrow protocol:

```python
class DenseStore(Protocol):
    async def search_segments(
        self, embedding: Sequence[float], limit: int, filters: RetrievalFilter
    ) -> list[RetrievedDocument]: ...

    async def search_representations(
        self, embedding: Sequence[float], limit: int, filters: RetrievalFilter
    ) -> list[RetrievedDocument]: ...

    async def fetch_evidence_units(
        self, evidence_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]: ...

    async def expand_parent_sections(
        self, evidence_ids: Sequence[str], *, per_parent_limit: int
    ) -> list[RetrievedDocument]: ...

    async def health(self) -> dict[str, bool | int | str]: ...
```

`src/chroma_store.py` should wrap the existing Chroma calls behind this interface during the rollback window. Its synchronous client calls must run in a worker thread.

`src/pgvector_store.py` should implement the interface with `AsyncConnectionPool`, parameterized SQL, cosine distance, metadata predicates, bounded overfetch, and stable rank output.

### PostgreSQL pool ownership

Create `src/postgres.py` to open and close one application-owned `AsyncConnectionPool`. Pass that pool to both `AsyncPostgresSaver` and `PgVectorStore`. Keep SQLite checkpoint mode independent from retrieval backend selection.

The initial pool budget should reserve capacity for both retrieval and checkpoints. Add settings rather than hard-coded values:

```text
POSTGRES_POOL_MIN_SIZE=2
POSTGRES_POOL_MAX_SIZE=20
PGVECTOR_HNSW_EF_SEARCH=100
PGVECTOR_OVERFETCH_FACTOR=4
VECTOR_STORE_BACKEND=chroma|pgvector
```

Tune these values from load-test evidence. Do not add a separate pool until metrics show checkpoint starvation or retrieval queueing.

### Retrieval orchestration

Convert the public query path in `src/retrieval.py` to native async functions. Keep pure fusion, deduplication, scoring, and compression helpers synchronous.

For one set of query variants:

1. Batch-embed normalized unique queries once under `embedding_gate` in a worker thread.
2. Start BM25 scoring in a worker thread.
3. Await body and representation pgvector queries concurrently with BM25.
4. Run existing heading/concept lookups concurrently where safe.
5. Apply the existing weighted RRF only after all paths complete.
6. Rerank fused child candidates.
7. Resolve selected children to EvidenceUnits and expand bounded siblings from the same parent section.
8. Run existing contextual compression.

Do not call `asyncio.run()` from the async request path. Keep the synchronous wrappers at the bottom of `src/nodes.py` only for tests/local synchronous graph usage.

## Target database schema

Use the existing application database. The Compose PostgreSQL image must include pgvector, and migration SQL must execute `CREATE EXTENSION IF NOT EXISTS vector` with an appropriately privileged migration role.

The SQL migration should create the following tables. The exact column dimension is 384.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE rag_corpora (
    corpus_id text PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('staging', 'active', 'previous', 'failed')),
    schema_version integer NOT NULL,
    pipeline_version text NOT NULL,
    embedding_model text NOT NULL,
    embedding_revision text NOT NULL,
    embedding_dimension integer NOT NULL CHECK (embedding_dimension = 384),
    manifest jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    activated_at timestamptz
);

CREATE UNIQUE INDEX rag_one_active_corpus
    ON rag_corpora ((status)) WHERE status = 'active';

CREATE TABLE rag_sections (
    corpus_id text NOT NULL REFERENCES rag_corpora(corpus_id) ON DELETE CASCADE,
    section_id text NOT NULL,
    manual text NOT NULL,
    source_file text NOT NULL,
    release text,
    chapter text NOT NULL,
    section text NOT NULL,
    subsection text,
    heading_path text[] NOT NULL,
    first_pdf_page integer,
    last_pdf_page integer,
    PRIMARY KEY (corpus_id, section_id)
);

CREATE TABLE rag_evidence_units (
    corpus_id text NOT NULL REFERENCES rag_corpora(corpus_id) ON DELETE CASCADE,
    evidence_id text NOT NULL,
    section_id text NOT NULL,
    unit_index integer NOT NULL,
    text text NOT NULL,
    content_type text NOT NULL,
    manual text NOT NULL,
    source_file text NOT NULL,
    release text,
    chapter text NOT NULL,
    section text NOT NULL,
    subsection text,
    heading_path text[] NOT NULL,
    printed_page text,
    pdf_page integer,
    is_toc boolean NOT NULL DEFAULT false,
    metadata jsonb NOT NULL,
    token_count integer NOT NULL,
    structured_table jsonb,
    procedure_steps jsonb NOT NULL,
    annotations jsonb NOT NULL,
    PRIMARY KEY (corpus_id, evidence_id),
    FOREIGN KEY (corpus_id, section_id)
        REFERENCES rag_sections(corpus_id, section_id) ON DELETE CASCADE
);

CREATE TABLE rag_retrieval_segments (
    corpus_id text NOT NULL REFERENCES rag_corpora(corpus_id) ON DELETE CASCADE,
    segment_id text NOT NULL,
    evidence_id text NOT NULL,
    section_id text NOT NULL,
    searchable_text text NOT NULL,
    content_type text NOT NULL,
    manual text NOT NULL,
    source_file text NOT NULL,
    release text,
    chapter text NOT NULL,
    section text NOT NULL,
    subsection text,
    pdf_page integer,
    metadata jsonb NOT NULL,
    segment_index integer NOT NULL,
    previous_segment_id text,
    next_segment_id text,
    word_count integer NOT NULL,
    embedding_token_count integer NOT NULL,
    effective_embedding_limit integer NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (corpus_id, segment_id),
    FOREIGN KEY (corpus_id, evidence_id)
        REFERENCES rag_evidence_units(corpus_id, evidence_id) ON DELETE CASCADE,
    FOREIGN KEY (corpus_id, section_id)
        REFERENCES rag_sections(corpus_id, section_id) ON DELETE CASCADE
);

CREATE TABLE rag_search_representations (
    corpus_id text NOT NULL REFERENCES rag_corpora(corpus_id) ON DELETE CASCADE,
    representation_id text NOT NULL,
    evidence_id text NOT NULL,
    section_id text NOT NULL,
    representation_type text NOT NULL,
    text text NOT NULL,
    manual text NOT NULL,
    source_file text NOT NULL,
    release text,
    chapter text NOT NULL,
    section text NOT NULL,
    subsection text,
    pdf_page integer,
    metadata jsonb NOT NULL,
    embedding_token_count integer NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (corpus_id, representation_id),
    FOREIGN KEY (corpus_id, evidence_id)
        REFERENCES rag_evidence_units(corpus_id, evidence_id) ON DELETE CASCADE,
    FOREIGN KEY (corpus_id, section_id)
        REFERENCES rag_sections(corpus_id, section_id) ON DELETE CASCADE
);

CREATE INDEX rag_segments_embedding_hnsw
    ON rag_retrieval_segments USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX rag_representations_embedding_hnsw
    ON rag_search_representations USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX rag_segments_manual_filter
    ON rag_retrieval_segments (corpus_id, manual);
CREATE INDEX rag_segments_source_filter
    ON rag_retrieval_segments (corpus_id, source_file);
CREATE INDEX rag_segments_type_filter
    ON rag_retrieval_segments (corpus_id, content_type);
CREATE INDEX rag_segments_section_filter
    ON rag_retrieval_segments (corpus_id, section_id);
CREATE INDEX rag_representations_manual_filter
    ON rag_search_representations (corpus_id, manual);
CREATE INDEX rag_evidence_section_order
    ON rag_evidence_units (corpus_id, section_id, unit_index);
```

Generate `section_id` deterministically from `source_file` plus the normalized `heading_path`. Assign `unit_index` from source order. This adds the missing parent-section relationship without changing existing EvidenceUnit IDs or citation metadata.

HNSW filtering needs explicit validation. pgvector may scan approximate neighbors before applying metadata predicates, depending on installed pgvector behavior and query shape. The implementation must overfetch, apply typed filters in SQL, and benchmark recall for narrow manual/content filters. Use iterative scans only if the pinned pgvector version supports them and tests confirm the query plan.

## Target query architecture

```text
User query
  -> existing deterministic/LLM preprocessing
  -> RetrievalFilter from validated planner metadata
  -> batch local query embeddings
  -> concurrent paths
       -> async pgvector HNSW body + representation retrieval
       -> rank-bm25 lexical retrieval in worker thread
       -> existing heading/concept/alias paths
  -> existing weighted RRF
  -> existing cross-encoder reranking
  -> selected EvidenceUnit resolution
  -> bounded parent-section sibling expansion
  -> existing deterministic contextual compression
  -> existing grading / retry / Groq generation / verification
  -> unchanged answer and citation payload
```

The target order intentionally moves contextual expansion after reranking. This limits database reads and cross-encoder work to child candidates. Parent expansion should return the winning EvidenceUnit plus nearby units in the same `section_id`, ordered by `unit_index`, with a strict per-parent and total character/token cap. It should keep each EvidenceUnit separate so citations retain a single source page rather than citing an artificial merged section.

## Files to modify

| File | Planned responsibility |
| --- | --- |
| `src/config.py` | Backend selection, pool sizing, HNSW search/overfetch settings, dimension validation, and rollout flags |
| `src/schemas.py` | `RetrievalFilter`, section identity/order fields, and backend-neutral dense result types |
| `src/embeddings.py` | One dimension probe/validation helper and batch query embedding entrypoint; retain model constructors |
| `src/ingest.py` | Emit section identity/order; route dense index output through a writer; retain Chroma writer during rollout |
| `src/retrieval.py` | Native async orchestration, parallel dense/BM25 paths, backend injection, post-rerank parent expansion, and async cache use |
| `src/nodes.py` | Await native async retrieval; place rerank before parent expansion/compression; preserve sync wrappers for tests |
| `src/cache.py` | Use `redis.asyncio` for request-path retrieval cache calls |
| `src/observability.py` | Add retrieval-stage duration and result-count metrics/logs |
| `backend/dependencies.py` | Own shared pool/store lifecycle and pgvector readiness checks |
| `docker-compose.yml` | Use pgvector-enabled PostgreSQL; keep Chroma under rollback profile during soak |
| `.env.example` | Document backend, pool, HNSW, and rollout settings |
| `requirements-backend.txt` | Add `pgvector`; retain Chroma packages until rollback retirement |
| `retrieval_benchmark.py` | Async backends, p50/p95/p99, stage and concurrency reports, Chroma/pgvector parity |
| `load_test.py` | Add p50 and p99 for total latency and time to first token |
| `README.md` | Target architecture, migration, operations, readiness, benchmark, and rollback commands |
| `tests/conftest.py` | Backend-neutral fixtures and pgvector integration marker/config |
| `tests/test_retrieval.py` | Preserve fusion/rerank behavior while testing concurrent orchestration and typed filters |
| `tests/test_ingest.py` | Section IDs/order, pgvector payloads, dimension failures, and relational parity |
| `tests/test_backend.py` | Shared pool/store lifecycle and readiness state |
| `tests/test_final_regressions.py` | Replace live-Chroma-only parity with selected-backend parity; keep Chroma case during rollout |
| `tests/test_retrieval_benchmark.py` | p99, backend labels, stage timing, and comparison report contract |

## New files to create

| File | Purpose |
| --- | --- |
| `src/postgres.py` | Application-owned async pool construction and lifecycle |
| `src/dense_store.py` | Narrow `DenseStore` protocol and backend factory |
| `src/chroma_store.py` | Temporary rollback adapter around existing Chroma behavior |
| `src/pgvector_store.py` | Async HNSW queries, filters, evidence/section fetches, health, and count validation |
| `db/migrations/001_pgvector_retrieval.sql` | Extension, tables, constraints, HNSW, and filter indexes |
| `scripts/backfill_pgvector.py` | Idempotent staging-corpus load from existing version-8 artifacts and Chroma embeddings |
| `tests/test_dense_store.py` | Store contract and backend-factory unit tests |
| `tests/test_pgvector_store.py` | SQL construction/filter/result mapping unit tests with a fake pool |
| `tests/integration/test_pgvector_retrieval.py` | Real PostgreSQL/pgvector schema, HNSW, parity, rollback, and query-plan tests |

## Migration strategy

1. Capture the current 272-test result, 44-case retrieval report, 50-case answer evaluation, and 50-user load report before changing retrieval.
2. Add the schema and application pool without changing `VECTOR_STORE_BACKEND=chroma`.
3. Backfill a `staging` corpus from the current version-8 JSON artifacts. Read existing Chroma embeddings in ID-keyed batches so the first comparison holds vectors constant. Fail on missing IDs, duplicates, non-finite values, non-unit norms outside tolerance, or dimensions other than 384.
4. Build HNSW indexes after bulk load, run `ANALYZE`, validate row counts and parent foreign keys, then execute deterministic probe queries.
5. Implement the pgvector store behind the same dense result contract. Keep weighted RRF and all downstream logic unchanged.
6. Run shadow retrieval: serve Chroma results while executing pgvector on sampled internal requests. Log IDs, rank overlap, filter behavior, stage latency, and errors without logging question text or evidence.
7. Activate the corpus by switching its status to `active`. Set `VECTOR_STORE_BACKEND=pgvector` in staging first, then production.
8. During the soak period, keep the previous Chroma service, packages, settings, and artifacts intact. Continue loading both stores during explicit ingestion if manuals change.
9. After quality, latency, and operational gates hold through the agreed soak period, stop dual writes, remove the Chroma dependency/service in a separate change, and retain one recoverable database backup plus version-8 JSON artifacts.

The backfill should not reparse PDFs. It should use current stable IDs and metadata. A `--reembed` mode may recompute vectors with the pinned local model when Chroma export is unavailable, but it must record the model revision, measured dimension, and vector source in `rag_corpora.manifest`.

## Rollback strategy

Rollback must require configuration and deployment changes only:

1. Set `VECTOR_STORE_BACKEND=chroma` and redeploy/restart backend replicas.
2. Keep the existing version-8 JSON files and Chroma collections readable for the full soak window.
3. Include backend name and corpus generation in retrieval cache keys so rollback cannot return pgvector-cached rankings.
4. Keep pgvector rows; do not delete them during an emergency rollback.
5. If the active pgvector corpus itself is faulty, atomically mark the last known corpus `active` and the faulty corpus `failed`, then restart or invalidate retrieval caches.
6. Restore PostgreSQL from backup only for database corruption. A ranking regression should use the backend/corpus switch rather than database restoration.

## Testing strategy

### Unit tests

- Verify the 384-dimension guard rejects short, long, and non-finite vectors.
- Verify every `RetrievalFilter` field maps to parameterized SQL and no value enters SQL text.
- Verify pgvector distance order maps to the same `RetrievedDocument` score/rank fields used by RRF.
- Verify Chroma and pgvector adapters satisfy the same store contract.
- Verify BM25 and dense tasks overlap with controlled async events rather than executing sequentially.
- Preserve all existing RRF weights, original-query weighting, deduplication, reranker fallback, evidence resolution, and citation tests.
- Verify parent expansion occurs after reranking, uses `section_id`/`unit_index`, respects caps, and preserves individual EvidenceUnit citations.
- Verify retrieval cache keys include backend and corpus ID.

### PostgreSQL integration tests

- Start PostgreSQL with pgvector, apply `001_pgvector_retrieval.sql`, and load a small deterministic corpus.
- Assert extensions, columns, constraints, HNSW indexes, filter indexes, and foreign keys exist.
- Run dense searches with no filters and with manual, release, content type, chapter, and section filters.
- Use `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` on a representative corpus to confirm HNSW use for the production query shape.
- Test empty result sets, unavailable active corpus, pool exhaustion timeout, transaction rollback, and cancellation.
- Test corpus activation and rollback without deleting either corpus.

### Ingestion and parity tests

- Assert PostgreSQL IDs exactly match version-8 EvidenceUnit, RetrievalSegment, and SearchRepresentation artifacts.
- Assert every segment and representation references an existing EvidenceUnit and section.
- Assert the database preserves structured tables, procedure steps, annotations, heading paths, and source pages.
- Compare exported Chroma and pgvector vectors by ID during initial backfill.
- Re-run the current indirect-definition, table, procedure, relationship, hard-question, and final-regression cases against pgvector.

### End-to-end tests

- Run all 272 existing tests with Chroma during rollout.
- Run the backend-neutral suite and pgvector integration suite with pgvector selected.
- Run the 50-case answer evaluation and require the current citation, evidence-status, manual-routing, and semantic-quality gates.
- Confirm Streamlit receives the same SSE event types and final payload fields.

## Benchmarking strategy

### Existing measured baseline

The checked-in 44-case scored retrieval benchmark currently reports:

| Pipeline | p50 | p95 | Macro F1@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chroma dense only | 7.36 ms | 9.47 ms | 18.72% | 48.86% | 39.36% |
| Current Chroma hybrid | 141.22 ms | 175.00 ms | 29.22% | 73.86% | 67.12% |

The current benchmark does not report p99. The current end-to-end and load tools report median/p50 and p95 but not p99.

### Required benchmark changes

For each backend, run one unmeasured warmup and at least 30 measured repetitions of each of the 44 scored questions. This produces enough observations for a meaningful p99. Record:

- query embedding
- body-vector SQL/client time
- representation-vector SQL/client time
- BM25 time
- heading/concept time
- RRF time
- reranker time
- parent expansion time
- compression time
- total retrieval time
- result counts and cache hit/miss

Run these profiles:

1. Chroma baseline, warm cache, concurrency 1.
2. pgvector exact scan baseline, warm cache, concurrency 1.
3. pgvector HNSW, warm cache, concurrency 1.
4. pgvector HNSW at concurrency 5 and 20.
5. Metadata-filtered HNSW for broad and narrow manual/content filters.
6. Cold process start plus first query, reported separately from steady state.
7. Existing 50-user end-to-end load test with p50/p95/p99 total latency and time to first token.

Write raw per-run JSON plus summary p50/p95/p99. Do not mix cold-start samples with steady-state percentiles.

### Acceptance gates

- No statistically observed decrease in the existing macro F1@5, Recall@5, or MRR beyond the project’s existing 3-percentage-point regression allowance.
- Exact citation/source metadata parity for matching EvidenceUnit IDs.
- pgvector hybrid p95 no more than 10% or 20 ms above the same-machine Chroma baseline, whichever allowance is larger.
- pgvector hybrid p99 below 250 ms on the current local benchmark host, unless the checked-in baseline run establishes a documented hardware-specific threshold.
- No pool timeouts or request errors at concurrency 20 in the retrieval benchmark.
- Existing end-to-end evaluation and load-test gates pass.

## Implementation phases and exact order

### Phase 0: Freeze contracts and baselines

- [ ] Run `pytest -q` and save the result.
- [ ] Run `python retrieval_benchmark.py` and preserve the raw report.
- [ ] Run the 50-case evaluation and 50-user load test against the current stack.
- [ ] Add p99 and raw stage timing support to benchmarks before comparing stores.
- [ ] Commit only benchmark/test contract changes.

### Phase 1: Add schema and shared pool

- [ ] Write failing schema and lifecycle tests.
- [ ] Add `db/migrations/001_pgvector_retrieval.sql` with `vector(384)`, HNSW, filter indexes, and constraints.
- [ ] Change Compose PostgreSQL to a pgvector-enabled PostgreSQL 16 image.
- [ ] Add `src/postgres.py` and pass one pool to the checkpoint saver.
- [ ] Add settings and validation while leaving Chroma selected by default.
- [ ] Run unit plus PostgreSQL integration tests and commit.

### Phase 2: Add the store boundary and pgvector reader

- [ ] Write the `DenseStore` contract tests and async pgvector query tests.
- [ ] Extract current Chroma calls into `ChromaStore` without changing rankings.
- [ ] Implement `PgVectorStore` with parameterized metadata filters, cosine distance, HNSW search settings, evidence fetch, section expansion, and readiness.
- [ ] Add backend selection in the application lifespan.
- [ ] Run both adapter suites and commit.

### Phase 3: Backfill a versioned corpus

- [ ] Write failing backfill tests for counts, IDs, vectors, parent keys, and idempotency.
- [ ] Implement `scripts/backfill_pgvector.py` against current JSON and Chroma embeddings.
- [ ] Load a staging corpus, build HNSW, run `ANALYZE`, and validate all counts.
- [ ] Store the pinned model, revision, dimension, manifest, and vector source in `rag_corpora`.
- [ ] Run deterministic dense-query parity and commit.

### Phase 4: Make retrieval natively async and parallel

- [ ] Write a concurrency test that proves BM25 overlaps dense retrieval.
- [ ] Convert retrieval cache access to async Redis.
- [ ] Batch query embeddings under the existing inference gate.
- [ ] Run pgvector body/representation queries and threaded BM25 concurrently.
- [ ] Preserve heading/concept/preferred-manual paths and RRF weights.
- [ ] Keep CPU embedding and reranking in bounded worker threads.
- [ ] Add per-stage metrics/logs and commit.

### Phase 5: Correct the expansion order

- [ ] Write tests for `RRF -> rerank -> EvidenceUnit/section expansion -> compression`.
- [ ] Remove pre-rerank neighbor expansion from the active pgvector path.
- [ ] Resolve winning children to EvidenceUnits.
- [ ] Add bounded sibling EvidenceUnits from the same `section_id` in source order.
- [ ] Apply the existing deterministic compressor and source caps.
- [ ] Run citation, procedure, table, relationship, and hard-question regressions; commit.

### Phase 6: Shadow, benchmark, and cut over

- [ ] Run sampled shadow retrieval while serving Chroma results.
- [ ] Compare IDs, rank overlap, filters, errors, and p50/p95/p99 stage latency.
- [ ] Run the 44-case retrieval benchmark, 50-case answer evaluation, and concurrency/load profiles.
- [ ] Tune `ef_search`, overfetch, and pool size only from measured results.
- [ ] Activate the validated corpus and select pgvector in staging.
- [ ] Repeat all gates, then select pgvector in production; commit deployment/docs changes.

### Phase 7: Integrate ingestion and retire Chroma later

- [ ] Teach explicit ingestion to write a staging PostgreSQL corpus and activate it after validation.
- [ ] Keep dual writes during the agreed rollback soak period.
- [ ] Exercise the configuration-only rollback at least once.
- [ ] After the soak gate, remove the Chroma service, packages, adapter, settings, and Chroma-only tests in a separate reversible change.
- [ ] Retain version-8 JSON artifacts and a database backup according to the deployment retention policy.

## Risks and mitigations

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Filtered HNSW loses recall | Narrow metadata filters return fewer or worse neighbors | Typed filters, bounded overfetch, iterative scans where supported, filtered-recall benchmarks |
| Distance semantics change | Normalized Chroma/L2 rankings differ from pgvector cosine ordering | Reuse existing vectors initially; use `vector_cosine_ops`; run ID/rank parity tests |
| Model dimension drift | Inserts or queries fail, or vectors target the wrong schema | Pin revision; probe dimension; store dimension in corpus metadata; fail before load/startup |
| Pool contention | Retrieval starves checkpoints or vice versa | Shared measured pool, timeout metrics, concurrency tests; split pools only with evidence |
| Async rewrite exposes blocking cache/model calls | Event-loop stalls under load | Async Redis; bounded `to_thread()` for embedding, BM25, reranker, and file reads |
| Parent sections are too large | Prompt bloat and lower answer quality | Expand bounded sibling EvidenceUnits, keep separate citations, compress before generation |
| Corpus activation exposes partial data | Missing vectors or broken foreign keys reach users | Stage status, transactional activation, count/ID/dimension/parent validation |
| HNSW build or bulk load locks production | Ingestion delays or query disruption | Load a staging corpus, build indexes before activation, schedule and observe bulk jobs |
| Replicas hold stale BM25/catalog state | Dense and lexical paths use different corpus versions | Include corpus ID in resources/cache; reload or restart replicas after atomic activation |
| BM25 duplicates memory per replica | Higher memory use as the corpus grows | Accept at current 15,721 segments; revisit a shared lexical service only after measurement |
| Chroma rollback drifts after new ingestion | Emergency rollback serves an older corpus | Dual-write explicit ingestion during soak and validate both backends before activation |
| SQLite checkpoint development mode becomes coupled to PostgreSQL retrieval | Local tests become harder to run | Keep checkpoint and retrieval backend settings separate; use fake store fixtures for unit tests |
| Existing source payload changes | UI or citation evaluation regresses | Preserve EvidenceUnit IDs and metadata; run exact source-payload regression tests |
| Custom metrics lack runtime quantiles | Operators cannot see tail latency from `/metrics` alone | Export stage sum/count for monitoring and compute p50/p95/p99 from raw benchmark/load samples; adopt histogram buckets only if operations require live quantiles |

## Completion criteria

The migration is complete only when pgvector serves production retrieval, every existing behavior/evaluation gate passes, p50/p95/p99 evidence meets the acceptance gates, rollback has been exercised, explicit ingestion can publish a validated pgvector corpus, and the team has completed the Chroma soak period. Chroma removal belongs to the final cleanup phase, not the cutover change.
