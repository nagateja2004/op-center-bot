# Opcenter Chatbot

An asynchronous RAG chatbot grounded in indexed Opcenter PDF manuals.

## Production review status — 2026-09-07

**The final optimized architecture is not implemented or production-verified.**
**This Mac now runs retrieval on PostgreSQL + pgvector.** Local `.env` selects
`VECTOR_STORE=pgvector`; the code default remains `chroma` for compatibility.
Native PostgreSQL 17.11 and pgvector 0.8.6 bypass the broken Docker daemon.
All 36,842 body/representation records from eight manuals were imported and
validated, and a live API answer retained its manual/page citation. Chroma data
and its selector remain available for rollback. HNSW, parallel dense/BM25 retrieval
and request-local embedding reuse are now implemented and quality-benchmarked.
See [optimization configuration and rollback](docs/RETRIEVAL_OPTIMIZATIONS.md).

See [local cutover and restart instructions](docs/LOCAL_PGVECTOR_CUTOVER.md).

Latest matched six-round benchmark: optimized pgvector hybrid p50 **85.65 ms**
versus exact pgvector **131.97 ms** (**35.10% reduction**), with equal Recall@5 and
MRR on 44 labeled questions. Chroma was **80.30 ms**, so this is **not a speedup
over Chroma**. These are local retrieval-only results, not LLM response times. See
[measured results and reproduction commands](benchmarks/BENCHMARK_RESULTS.md).

Current behavior and the intended architecture are documented separately in
[FINAL_ARCHITECTURE.md](docs/FINAL_ARCHITECTURE.md), including review findings,
test results, operational risks and the release checklist.

Optimization suite: **331 passed, zero skipped**, including real SQL/HNSW tests,
eight exact/optimized live retrieval checks, concurrency and embedding-reuse tests. The
fresh local Chroma hybrid benchmark measured p50 **92.33 ms** and p95 **111.95 ms**
at `k=5` (44 scored cases, one warm pass, simplified retrieval adapter, no LLM).
These are not pgvector, production-load or end-to-end measurements.

| Component | Implemented behavior |
| --- | --- |
| HNSW | Explicit L2 HNSW migration, bare-distance indexed ORDER BY, transaction-local ef_search and strict iterative scans. Exact-search rollback remains. Tested local ef_search=200; exact top-20 overlap 96.82%. |
| Metadata filtering | Parameterized equality/`$in` in the repository; existing manual preferences in retrieval. No structured query-wide prefilter or relaxation policy yet. |
| Parallel hybrid retrieval | Dense async PostgreSQL retrieval overlaps worker-thread BM25 with independent deadlines and partial-failure handling. Auxiliary retrieval paths and dependent ranking stages retain their order. |
| RRF | Existing weighted reciprocal-rank fusion, `k=60`; repeated IDs within one list now contribute once. Dense/BM25/fused defaults remain 12/12/18. |
| Reranking | Cached local cross-encoder, at most 20 candidates, existing per-aspect output limits and RRF-order failure fallback. |
| Context | Neighbor expansion before reranking, JSON EvidenceUnit resolution after it, deterministic compression; compression exceptions now use bounded original text with citation metadata. PostgreSQL parent lookup is not connected to generation. |
| Pooling | One async PostgreSQL pool per API process, shared with checkpoints; cleanup on normal shutdown, startup failure and startup cancellation. |
| Benchmarking | Repeated five-variant benchmark, 50 cases/44 scored labels, exact/ANN overlap, actual EXPLAIN and reranker profiles. No production load or full-answer improvement claim. |

Local offline retrieval benchmark (no LLM generation or paid API calls):

```bash
CHROMA_MODE=local VECTOR_STORE=chroma HF_HUB_OFFLINE=1 LANGSMITH_TRACING=false \
  .venv/bin/python retrieval_benchmark.py --output-dir evaluation_results/local-review --k 5
```

The benchmark's hybrid adapter is a simplified dense+BM25+context+reranker path,
not the complete production multi-aspect router with every auxiliary retrieval
list. Its latency is retrieval-only. Do not describe these numbers as end-to-end
request performance or as a pgvector improvement.

The Streamlit frontend communicates only with a FastAPI backend. The backend
runs the LangGraph workflow, hybrid pgvector (or rollback Chroma)/BM25 retrieval, reciprocal-rank
fusion, cross-encoder reranking, evidence grading, citation validation, answer
verification, and optional diagram generation.

No user account or authentication is required. Anonymous `session_id`,
`conversation_id`, and `thread_id` UUIDs isolate browser conversations, while
LangGraph checkpoints preserve conversation memory in PostgreSQL.

## Architecture

### Ingest Opcenter manuals into pgvector (phase 3)

The explicit importer selects pgvector for that job without changing `.env` or
the API's rollback selector. With `DATABASE_URL` pointing to a pgvector-capable
server, use either:

```bash
# Parse the PDFs in MANUALS_DIRECTORY using the existing extraction/chunking pipeline:
.venv/bin/python -m src.pgvector_migrate --manuals

# Import existing canonical JSON without parsing PDFs again:
.venv/bin/python -m src.pgvector_migrate --backfill
```

Docker equivalents (after starting PostgreSQL as described below):

```bash
docker compose --profile tools build ingest
docker compose --profile tools run --rm ingest python -m src.pgvector_migrate --manuals
# Or reuse the backend image for a JSON-only import:
docker compose run --rm --no-deps backend python -m src.pgvector_migrate --backfill
```

Both commands create/validate the schema before ingestion. `--manuals` honors
the existing unchanged-document/hash logic; `--backfill` deliberately re-embeds
the JSON snapshot. The cached, pinned self-hosted embedding model is loaded once
per job. Texts are submitted in batches of at most 256. Existing normalization
is retained: unit-length embeddings and squared-L2 distance give the same ranking
as cosine distance. IDs, chunk ordering, citation page labels and nested metadata
are preserved; parsing/chunking and the Chroma path are unchanged.

Database writes use batched `executemany` inside one transaction to publish the
two collection snapshots, not one transaction per chunk. Repeating an import
retains the same collection/chunk IDs and removes stale IDs in those collections;
it does not append duplicates. These are **full-corpus imports**, not a
single-document update command. Preserve the complete JSON corpus when backfilling.

After ingestion, a JSON summary reports `total_documents`, `total_chunks`
(body chunks plus search representations), `missing_embeddings`,
`missing_page_numbers`, `missing_document_ids`, `missing_parent_ids`,
`duplicate_chunk_ids` (within collection namespaces), and `embedding_dimensions`.
Counts come from PostgreSQL, not the input lists; stored IDs are also checked
against local JSON. Invalid reports raise an error and the command exits nonzero.
Invalid PDF pages and embedding dimensions are rejected before publication.
Every imported PDF chunk receives a deterministic logical parent-section ID;
printed page labels remain separate from the positive physical PDF page number.

Run offline and restart API workers after ingestion, including unchanged imports,
because JSON/DB publication and in-process caches are separate resources. Keep
the rollback snapshots described below. For the relevant tests:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_pgvector_ingest.py tests/test_vector_store.py tests/test_ingest.py tests/test_llm_embeddings.py -q
```

Real-database tests require `PGVECTOR_TEST_DATABASE_URL`; without it they skip.

### Optional PostgreSQL vector storage (migration phase 2)

`VECTOR_STORE=chroma` remains the default. Set `VECTOR_STORE=pgvector` only after
backfilling and validating PostgreSQL. The same async psycopg pool serves vectors
and PostgreSQL checkpoints, once per API process; SQLite checkpoints also work
with pgvector. `DB_POOL_MIN_SIZE=1`, `DB_POOL_MAX_SIZE=10`, and
`DB_POOL_TIMEOUT=30` configure pool capacity, acquisition timeout and the sync
retriever's database wait limit (seconds). Two API replicas therefore allow up to
20 connections, plus explicit offline jobs. Existing requirements already contain
the necessary driver; no asyncpg or SQLAlchemy dependency is added.

For a **new database volume**, configure `.env` using `.env.example`, then:

```bash
docker compose up -d postgres
docker compose build backend
docker compose run --rm --no-deps backend python -m src.pgvector_migrate --backfill
```

The command probes the pinned local SentenceTransformer, creates the extension
and tables, embeds the existing JSON records, and verifies both collection ID
sets. It does not re-extract PDFs or touch Chroma. Omitting `--backfill` creates
and validates the schema only. Existing model caches must be available, or the
pinned model must be downloaded on the first run.

After the command succeeds, set `VECTOR_STORE=pgvector` in `.env` and restart
the API with `docker compose up -d backend`. Keep the existing normal Redis,
Chroma, frontend and proxy deployment. For subsequent PDF ingestion, run
`docker compose --profile tools run --rm ingest`; the selected vector backend is
updated. During ingestion, stop API traffic and restart the API afterward: local
JSON and the vector database are not one atomic resource, and workers cache the
local records. Do not run concurrent ingestion/backfill jobs.

**Existing PostgreSQL volumes:** Compose now uses the upstream maintained
`pgvector/pgvector:pg16` image rather than `postgres:16-alpine`. Do not blindly
restart an existing database against a different image. Take and verify logical
backups of checkpoint data, restore into a separate pgvector PostgreSQL 16
instance/volume, validate compatibility (including libc/collation differences),
and switch the connection URL under a maintenance window. Never delete the old
volume. Pin the validated image digest for production deployments.

Schema: `src/sql/001_pgvector.sql` creates `vector_model` and `document_chunks`.
The vector dimension is generated from the configured model, measured locally
as **384** for `all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. A different model, revision or dimension
fails validation instead of mixing incompatible embeddings. Both body segments
and search representations retain their IDs in separate collection namespaces;
all metadata, including printed pages and nested table data, is retained in JSONB.
`page_number` is the physical PDF page; printed labels stay in metadata.

This phase uses exact squared-L2 search to retain the current Chroma distance
convention. No HNSW/IVFFlat index, parallel hybrid retrieval, or new reranking
behavior is introduced. Full parent evidence and BM25 remain backed by the local
JSON files, which are still required. The repository also supports ordered
parent-section chunk lookup; it does not yet change the existing expansion flow.

Rollback: restore `VECTOR_STORE=chroma` and restart the API. Chroma is never
deleted by pgvector backfill. If ingestion ran with pgvector selected, restore
the matching pre-cutover `indexes/` and manuals snapshot before rollback, or
explicitly rebuild Chroma against the new JSON first. Keep source snapshots,
database backups and Chroma storage together; the selector alone cannot make
different corpus generations consistent.

Tests (real database tests never implicitly use the application's DATABASE_URL):

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
# Supply a disposable database with pgvector installed and schema/extension privileges:
PGVECTOR_TEST_DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/TEST_DB \
  .venv/bin/python -m pytest tests/test_vector_store.py -q
```

The nine integration tests create and drop uniquely named test schemas only.
They skip explicitly when `PGVECTOR_TEST_DATABASE_URL` is absent. Compose keeps
PostgreSQL on its private network; a host-side test URL requires a separately
provisioned test instance or explicitly configured localhost port forwarding.
For host-side CLI use, set `DATABASE_URL` to that reachable server and run
`.venv/bin/python -m src.pgvector_migrate --backfill`.

See `docs/PGVECTOR_MIGRATION_PLAN.md` for phase status and remaining work.
Implementation references: [pgvector](https://github.com/pgvector/pgvector) and
[psycopg async pools](https://www.psycopg.org/psycopg3/docs/advanced/pool.html).

### System architecture

```mermaid
flowchart TB
    subgraph Client["Client layer"]
        User["User"] --> UI["Streamlit chat UI"]
    end

    subgraph Application["Application layer · Docker Compose"]
        UI -->|"POST /v1/chat"| API["FastAPI · 2 replicas"]
        API -->|"SSE progress and answer"| UI
        API <--> Requests[("Redis<br/>pending requests · ownership · cache")]
        API --> Graph["Compiled LangGraph workflow"]
        Graph <--> Memory[("PostgreSQL<br/>conversation checkpoints")]
    end

    subgraph RAG["RAG and model layer"]
        Graph --> Retrieval["Sequential hybrid retrieval<br/>Chroma or exact pgvector + BM25 + weighted RRF"]
        Retrieval --> Expansion["Neighbor-segment context expansion"]
        Expansion --> Reranker["Local cross-encoder reranker"]
        Reranker --> Context["EvidenceUnit resolution<br/>deterministic compression"]
        Context -->|"ranked evidence"| Graph
        Graph --> Groq["Role-specific Groq LLMs<br/>planner · grader · answer · verifier · diagram"]
        Groq <--> Limits[("Redis per-model admission control")]
    end

    subgraph Knowledge["Knowledge preparation · offline"]
        PDFs["Opcenter PDF manuals"] --> Ingest["Explicit ingestion job"]
        Ingest --> Units[("EvidenceUnits and retrieval segments")]
        Units --> Vectors[("Chroma or opt-in pgvector vectors")]
        Units --> Lexical[("BM25 runtime index")]
        Ingest --> Figures[("Extracted manual figures")]
        Vectors --> Retrieval
        Lexical --> Retrieval
        Figures --> Graph
    end

    subgraph Operations["Quality and operations"]
        Graph -. "nested traces" .-> LangSmith["LangSmith"]
        API -. "health · readiness · latency · queues" .-> Endpoints["/health · /ready · /metrics"]
        Golden["50-case golden dataset"] --> Evaluator["Deterministic checks + LLM judge"]
        Evaluator -->|"evaluation questions"| API
        API -->|"answers + bounded evidence"| Evaluator
        Evaluator --> Reports["JSON + HTML evaluation reports"]
    end
```

| Layer | Main implementation | Responsibility |
| --- | --- | --- |
| Interface | Streamlit | Anonymous chat UI, session identifiers, streamed rendering |
| API | FastAPI | Request validation, replica-safe request acceptance, SSE, health and readiness |
| Orchestration | LangGraph | Conditional RAG flow, bounded retry, checkpoints and node-level tracing |
| Retrieval | Chroma, BM25 and weighted RRF | Semantic and lexical candidate retrieval |
| Evidence processing | PyMuPDF, Tesseract, PP-StructureV3, EvidenceUnits and cross-encoder | Extract native or scanned content, preserve structure, compress deterministically and rerank |
| Generation | Role-specific Groq models | Planning, grading, grounded answering, verification and optional diagrams |
| State | PostgreSQL and Redis | Short-term conversation checkpoints, request handoff, ownership, caches and limits |
| Observability | LangSmith and built-in endpoints | Traces, latency, errors, queue depth, token and cost inspection |
| Quality | Golden dataset and local evaluator | Deterministic metrics, optional LLM-as-a-judge and JSON/HTML reports |

Important design boundaries:

- **Offline knowledge plane:** ingestion is an explicit command; startup and chat requests never rebuild indexes.
- **Short-term memory only:** PostgreSQL persists a conversation thread so follow-up questions can use earlier turns. The bot does not build a permanent user profile or recall unrelated conversations.
- **Deterministic context compression:** structured sentences, procedure steps and table rows are selected from EvidenceUnits. The runtime does not use `LLMChainExtractor` or `LLMChainFilter` for compression.
- **Protected evaluation evidence:** only requests authenticated with `EVALUATION_API_TOKEN` can receive bounded evidence excerpts for the judge; normal users cannot request them.
- **Failure containment:** one retrieval broadening attempt is allowed, diagram failure does not discard a grounded answer, and queue/provider retries are bounded.

## Main features

- Grounded answers with `[S1]`, `[S2]`, and similar citations
- Direct definitions, procedures, comparisons, troubleshooting, and follow-ups
- Multi-aspect retrieval with independent evidence coverage
- Relationship-aware retrieval for hierarchy and parent-child questions
- Selective OCR for scanned pages with PP-StructureV3 fallback for complex layouts and tables
- Original diagrams extracted from cited manual pages
- Generated hierarchy, relationship, process, decision, and architecture diagrams
- Basic greetings without unnecessary retrieval or Groq calls
- Anonymous browser-session isolation without user accounts
- Two backend replicas in Docker Compose
- Bounded Groq and local inference queues
- Built-in latency, inference, and queue metrics without a monitoring service dependency
- Optional LangSmith traces for the complete LangGraph request path
- A 50-case golden dataset with JSON and visual HTML evaluation reports
- Optional local threshold and accepted-baseline regression checks

## Requirements

- Docker Desktop with Docker Compose
- A Groq API key beginning with `gsk_`
- Text-based PDF manuals in `manuals/`
- Approximately 8 GB of available memory for local models and containers

Python 3.11 is used in the production containers. Python 3.11 or 3.12 can be
used for local development.

## Quick start with Docker

From the repository root:

```bash
git clone https://github.com/nagateja2004/op-center-bot.git
cd op-center-bot
cp .env.example .env
chmod 600 .env
```

Add at least the following to `.env`:

```dotenv
GROQ_API_KEY=gsk_replace_with_your_key
TOKENIZERS_PARALLELISM=false
```

To send LangGraph and Groq traces to LangSmith, add:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_replace_with_your_key
LANGSMITH_PROJECT=opcenter-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Tracing is optional. Leave `LANGSMITH_TRACING` unset or set it to `false` when
manual content must not leave the deployment environment.

For manual semantic evaluation, also add a long random secret:

```dotenv
EVALUATION_API_TOKEN=replace_with_a_long_random_evaluation_secret
GROQ_JUDGE_MODEL=openai/gpt-oss-20b
```

Use the same `EVALUATION_API_TOKEN` in the backend and the local evaluator. It
protects the bounded evidence excerpts used by the judge. Normal chat requests
cannot request or receive those excerpts. Do not expose this token publicly.

Place the PDF manuals directly in `manuals/`. Subdirectories are not scanned.

For a new installation, start Chroma and run ingestion once:

```bash
docker compose up -d chroma
docker compose --profile tools run --rm ingest
```

Build and start the complete application:

```bash
docker compose up -d --build --wait
```

Open the UI:

```text
http://localhost:8501
```

Verify the services:

```bash
docker compose ps
curl --fail http://localhost:8501/_stcore/health
```

The backend is internal to the Compose network. Its `/ready` endpoint verifies:

- compiled LangGraph
- PostgreSQL
- Redis
- Chroma collection
- BM25 index
- EvidenceUnit store
- embedding model
- reranker model

## Common Docker commands

View application logs:

```bash
docker compose logs -f backend frontend
```

Rebuild after changing source code:

```bash
docker compose build backend frontend
docker compose up -d --force-recreate --wait backend frontend
```

Stop the application while keeping persistent data:

```bash
docker compose down
```

Stop and delete the PostgreSQL, Chroma, and model-cache volumes:

```bash
docker compose down -v
```

Do not use `down -v` unless deleting those persistent volumes is intentional.

If port 8501 is already allocated:

```bash
APP_PORT=8502 docker compose up -d
```

## Services

| Service | Purpose | Profile | Host access |
| --- | --- | --- | --- |
| `frontend` | Streamlit chat UI | Default | `127.0.0.1:8501` (configurable) |
| `backend` | FastAPI and compiled LangGraph RAG, two replicas | Default | Docker network only, port `8000` |
| `postgres` | Short-term conversation checkpoints | Default | Docker network only, port `5432` |
| `redis` | Request queue, cache, rate limits, status and ownership | Default | Docker network only, port `6379` |
| `chroma` | Dense-vector storage and retrieval | Default | Docker network only, port `8000` |
| `ingest` | Offline PyMuPDF, Tesseract and PP-StructureV3 ingestion | `tools` | None; one-shot job |
| `evaluate` | 50-case deterministic evaluation and optional LLM judge | `tools` | None; one-shot job |
| `load-test` | Bounded concurrent SSE and latency test | `tools` | None; one-shot job |

PostgreSQL and Chroma use persistent named volumes. Manuals and index metadata
are bind-mounted from the repository and are not copied into container images.
The frontend binds to loopback by default. Put an authenticated, TLS-enabled
reverse proxy with client rate limits in front of it for public access; do not
publish PostgreSQL, Redis, Chroma, or the backend directly.

## Environment variables

### Required

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Groq API key; must begin with `gsk_` |
| `POSTGRES_PASSWORD` | Strong PostgreSQL password required by Docker Compose |
| `DATABASE_URL` | PostgreSQL checkpoint URL when running the backend outside Compose |
| `REDIS_URL` | Redis URL when running the backend outside Compose |
| `CHROMA_HOST` | Chroma host when `CHROMA_MODE=server` outside Compose |

Docker Compose supplies `DATABASE_URL`, `REDIS_URL`, `CHROMA_HOST`, and
`CHROMA_PORT` to the backend. `GROQ_API_KEY` and `POSTGRES_PASSWORD` are
required in `.env` for the default local Compose deployment. PostgreSQL and
Chroma are private Docker services and are not published on host ports.

### Storage and services

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHECKPOINT_BACKEND` | `postgres` | `postgres`, or `sqlite` only for explicit local development |
| `CHROMA_MODE` | `server` | `server`, or `local` only for explicit development |
| `CHROMA_PORT` | `8000` | Chroma server port as seen by the backend |
| `CHROMA_SSL` | `false` | Use HTTPS for Chroma |
| `CHROMA_COLLECTION` | `opcenter_manuals` | Main vector collection |
| `MANUALS_DIRECTORY` | `manuals` | PDF manual directory |
| `INDEXES_DIRECTORY` | `indexes` | Local index metadata directory |
| `EVIDENCE_UNITS_PATH` | `indexes/evidence_units.json` | EvidenceUnit store |
| `RETRIEVAL_SEGMENTS_PATH` | `indexes/retrieval_segments.json` | Retrieval segments |
| `CHAT_MEMORY_PATH` | `data/chat_memory.sqlite` | SQLite path in local SQLite mode |

### Models

| Variable | Default |
| --- | --- |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_MODEL_REVISION` | Pinned trusted model commit |
| `EMBEDDING_DEVICE` | `cpu` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANKER_MODEL_REVISION` | Pinned trusted model commit |
| `INFERENCE_MAX_CONCURRENCY` | `4` |
| `INFERENCE_MAX_QUEUE_DEPTH` | `32` |
| `GROQ_REQUEST_TIMEOUT` | `90` seconds |
| `GROQ_JUDGE_MODEL` | `openai/gpt-oss-20b` (evaluation process only) |

Each Groq role has a primary model and at most one fallback. Override a role
with:

```dotenv
GROQ_ANSWER_PRIMARY_MODEL=openai/gpt-oss-120b
GROQ_ANSWER_FALLBACK_MODEL=qwen/qwen3.6-27b
GROQ_ANSWER_TIMEOUT=90
GROQ_ANSWER_MAX_OUTPUT_TOKENS=4096
```

The supported default role assignment is:

| Role | Primary | Fallback |
| --- | --- | --- |
| Planner | `openai/gpt-oss-20b` | `openai/gpt-oss-120b` |
| Query broadening | `openai/gpt-oss-20b` | None |
| Evidence grader | `openai/gpt-oss-20b` | `openai/gpt-oss-120b` |
| Answer generation | `openai/gpt-oss-120b` | `qwen/qwen3.6-27b` |
| Answer verifier | `qwen/qwen3.6-27b` | `openai/gpt-oss-20b` |
| Diagram generation | `openai/gpt-oss-20b` | `openai/gpt-oss-120b` |

Docker Compose passes these model IDs explicitly to the backend, while local
development reads the same values from `.env`. The retired
`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, and
`meta-llama/llama-4-scout-17b-16e-instruct` IDs are not used.

Supported role prefixes are `PLANNER`, `QUERY_BROADENING`, `GRADER`, `ANSWER`,
`VERIFIER`, and `DIAGRAM`.

### LangSmith tracing

| Variable | Default | Purpose |
| --- | --- | --- |
| `LANGSMITH_TRACING` | `false` | Enable automatic LangGraph and LLM tracing |
| `LANGSMITH_API_KEY` | None | LangSmith API key; keep it only in `.env` or a secrets manager |
| `LANGSMITH_PROJECT` | `default` | Trace project; use `opcenter-rag` for this application |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith API endpoint |

Each graph trace is named `opcenter_chat` and tagged `opcenter-rag` and
`streaming-api`. Request, session, conversation, client thread, and diagram
settings are attached as searchable metadata. LangSmith automatically nests
the LangGraph nodes and LangChain model calls under the graph run.

LangSmith may capture graph inputs, outputs, and retrieved manual evidence.
Enable it only after confirming that the deployment's data policy permits this.

### Limits and request safety

| Variable | Default |
| --- | --- |
| `GROQ_MODEL_MAX_CONCURRENCY` | `4` |
| `GROQ_MODEL_REQUESTS_PER_MINUTE` | `30` |
| `GROQ_MODEL_TOKENS_PER_MINUTE` | `60000` |
| `GROQ_MAX_QUEUE_DEPTH` | `20` |
| `GROQ_MAX_QUEUE_WAIT_SECONDS` | `30` |
| `MAX_REQUEST_BYTES` | `16384` |
| `CHAT_REQUEST_TTL_SECONDS` | `300` |
| `THREAD_OWNERSHIP_TTL_SECONDS` | `2592000` |
| `CORS_ORIGINS` | Empty |

`CORS_ORIGINS` is a comma-separated list, for example:

```dotenv
CORS_ORIGINS=https://chat.example.company
```

Per-model Groq limits can override the global values by using the normalized
model name:

```dotenv
GROQ_MODEL_OPENAI_GPT_OSS_120B_MAX_CONCURRENCY=2
GROQ_MODEL_OPENAI_GPT_OSS_120B_REQUESTS_PER_MINUTE=20
GROQ_MODEL_OPENAI_GPT_OSS_120B_TOKENS_PER_MINUTE=50000
```

Never commit `.env`, API keys, passwords, manuals, or generated indexes.

## Ingestion

Run ingestion only after adding, removing, or changing manuals, or when the
index schema changes:

```bash
docker compose --profile tools run --rm ingest
```

### Document ingestion and hybrid retrieval

```mermaid
flowchart TD
    PDF["PDF manuals"] --> Detect{"Page/content analysis"}
    Detect -->|"sufficient selectable text"| Native["PyMuPDF native extraction"]
    Detect -->|"scanned or insufficient text"| Tesseract["PyMuPDF + Tesseract OCR"]
    Tesseract --> Layout{"OCR result usable and simple?"}
    Layout -->|"yes"| Normalize["Normalize text, headings, tables and source metadata"]
    Layout -->|"no · weak, tabular or multi-column"| Paddle["PaddleOCR PP-StructureV3"]
    Native --> Normalize
    Paddle --> Normalize

    Normalize --> Evidence["EvidenceUnits<br/>text · procedures · structured tables · citations"]
    Evidence --> Segments["RetrievalSegments + deterministic search representations"]
    Segments --> Dense[("Chroma dense vectors")]
    Segments --> Sparse[("In-memory BM25 sparse index")]

    Query["Question planned by LangGraph"] --> Dense
    Query --> Sparse
    Dense --> RRF["Weighted reciprocal-rank fusion"]
    Sparse --> RRF
    RRF --> Expand["Neighbor-segment context expansion"]
    Expand --> Rerank["Local cross-encoder reranker"]
    Rerank --> Resolve["Resolve complete EvidenceUnits"]
    Resolve --> Compress["Deterministic context compression"]
    Compress --> RAG["LangGraph grading, generation and verification"]
    RAG --> Answer["Grounded answer with citations"]
```

The current stored index schema remains version 8. Ingestion creates:

- `indexes/evidence_units.json`
- `indexes/retrieval_segments.json`
- `indexes/search_representations.json`
- `indexes/heading_index.json`
- `indexes/concept_index.json`
- `indexes/ingestion_audit.json`
- `indexes/manifest.json`
- `indexes/manual_figures.json`
- `indexes/manual_figures/`
- Chroma retrieval and search-representation collections

Every page first uses PyMuPDF's embedded-text extraction. Pages with fewer than
`OCR_MIN_NATIVE_CHARS` useful characters are passed to PyMuPDF's Tesseract OCR.
When that result is still weak or appears tabular/multi-column, PP-StructureV3
reprocesses the page and preserves structured tables. The heavy PaddleOCR
runtime is installed only in `Dockerfile.ingest`; the live backend image remains
unchanged. The first complex-page run downloads PP-StructureV3 models into the
persisted `paddle_cache` volume.

OCR configuration:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OCR_ENABLED` | `true` | Enable scanned-page fallback during explicit ingestion |
| `OCR_LANGUAGE` | `eng` | Tesseract language code; combine packs with `eng+deu` |
| `OCR_DPI` | `300` | Page-rendering resolution for OCR |
| `OCR_MIN_NATIVE_CHARS` | `40` | Minimum useful native/OCR characters before fallback |
| `PADDLE_OCR_ENABLED` | `true` | Enable PP-StructureV3 for weak or complex OCR pages |
| `PADDLE_OCR_DEVICE` | `cpu` | Paddle inference device; use `gpu` only in a compatible image |

`indexes/ingestion_audit.json` records native, Tesseract, Paddle Structure, and
failed OCR page counts for every manual. Existing version-8 indexes remain
readable during rollout, but run explicit ingestion once to rebuild unchanged
manuals with the new OCR pipeline.

Evidence IDs and retrieval metadata are preserved from ingestion through
Chroma, in-memory BM25, reranking, citations, and returned sources. BM25 is
rebuilt safely from `retrieval_segments.json` at backend startup; no pickle
artifact is loaded.

## API

### Create a request

```http
POST /v1/chat
Content-Type: application/json
```

```json
{
  "message": "Explain the resource modeling hierarchy and include a diagram.",
  "session_id": "3d692768-c03f-4d53-b106-893641dd16f2",
  "diagram_enabled": true,
  "diagram_type": "hierarchy"
}
```

The backend returns HTTP 202 with:

```json
{
  "request_id": "...",
  "session_id": "...",
  "conversation_id": "...",
  "thread_id": "..."
}
```

For the next message in the same conversation, send the same `session_id`,
`conversation_id`, and `thread_id`. To start a new conversation, omit
`conversation_id` and `thread_id`; the backend generates new UUIDs.

### Stream the response

```http
GET /v1/chat/{request_id}/stream
Accept: text/event-stream
X-Session-ID: {session_id returned by POST /v1/chat}
```

SSE event types:

- `progress` — completed LangGraph node
- `answer` — answer text chunk
- `complete` — final answer, sources, evidence, generated diagram, and manual figures
- `error` — safe client-facing failure

Other endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process liveness |
| `GET /ready` | Dependency and model readiness |
| `GET /metrics` | Built-in process-local metrics in text format |

## Anonymous sessions

The application does not use authentication or user accounts.

- The frontend generates a UUID `session_id` for each Streamlit browser session.
- The backend generates new conversation and thread UUIDs when they are omitted.
- Redis binds a `thread_id` to the originating `session_id`.
- A different browser session cannot claim and reuse that thread.
- LangGraph uses `conversation_id:thread_id` as its internal checkpoint key.
- Internal PostgreSQL checkpoint identifiers are never returned to the frontend.

Selecting **New conversation** in the UI clears the displayed messages and
causes the backend to create a new conversation and thread.

## Retrieval and answer flow

```mermaid
flowchart TD
    Start(("START")) --> Understand["understand_question<br/>basic-chat check · deterministic or LLM plan"]
    Understand -->|"basic chat"| Done(("END"))
    Understand -->|"manual question"| Retrieve["retrieve_documents<br/>Chroma + BM25 + weighted RRF"]
    Retrieve --> Expand["expand_context<br/>neighbors · complete EvidenceUnits · compression"]
    Expand --> Rerank["rerank_documents<br/>local cross-encoder"]
    Rerank --> Grade["grade_evidence<br/>one grade per required aspect"]
    Grade -->|"sufficient or partial"| Answer["generate_answer<br/>grounded response with source IDs"]
    Grade -->|"retry once"| Broaden["broaden_query<br/>target missing evidence"]
    Broaden --> Retrieve
    Grade -->|"insufficient or out of scope"| Fallback["generate_fallback<br/>safe limitation"]
    Fallback --> Done
    Answer --> Verify["verify_answer<br/>normalize citations · remove unsupported claims"]
    Verify -->|"grounded and diagram requested or useful"| Diagram["generate_diagram<br/>Graphviz DOT + deterministic validation"]
    Verify -->|"answer ready"| Done
    Diagram --> Done

    Memory[("PostgreSQL short-term thread memory")] <--> Understand
    Nodes["Every node"] -. "timing and errors" .-> Trace["One nested LangSmith trace"]
```

1. Classify basic chat, direct questions, multi-aspect questions, and follow-ups.
2. Preserve explicitly named concepts for comparisons and relationship questions.
3. Retrieve with Chroma and BM25, then combine candidates with weighted RRF.
4. Add neighboring segments and resolve results into complete EvidenceUnits.
5. Rerank query-document pairs with the shared cross-encoder.
6. Grade evidence independently for each required aspect.
7. Broaden missing evidence at most once.
8. Generate a grounded answer with citations.
9. Run deterministic citation validation before optional LLM verification.
10. Return original manual figures or a validated Graphviz diagram when useful.

Simple definition questions use deterministic planning. The LLM planner is
reserved for ambiguous or multi-aspect questions. Conversation-dependent
follow-ups are not stored in the final-answer cache.

### LLM-call profile

The number of LLM calls is dynamic; retrieval, context expansion, compression,
reranking, caching and citation parsing do not call an LLM.

| Role | Calls in one user request | When it runs |
| --- | ---: | --- |
| Planner | 0 or 1 | Ambiguous, follow-up or multi-aspect questions; simple direct questions use deterministic planning |
| Evidence grader | 1 per required aspect per retrieval pass | Checks whether each requested aspect is supported |
| Query broadener | 0 or 1 | Only when evidence is missing after the first retrieval |
| Answer model | 0 or 1 | Only for sufficient or partial evidence |
| Verifier | 0 or 1 | Validates and corrects a generated answer |
| Diagram model | 0 or 1 | Only when a grounded diagram is requested or useful |
| Evaluation judge | 1 per evaluated case, plus at most one error retry | Runs outside the user request during evaluation |

A typical single-aspect question with no retry or diagram uses **three RAG LLM
calls**: grader, answer and verifier. An ambiguous question normally adds the
planner. Multi-aspect grading, one retrieval retry or an optional diagram can
increase the total. Basic greetings use zero RAG LLM calls.

### Memory and context behavior

```mermaid
sequenceDiagram
    participant Browser as Browser session
    participant API as FastAPI
    participant Redis as Redis ownership/request store
    participant Graph as LangGraph
    participant PG as PostgreSQL checkpoints

    Browser->>API: question + session/conversation/thread UUIDs
    API->>Redis: validate thread ownership and store pending request
    API-->>Browser: request_id
    Browser->>API: open SSE stream using request_id
    API->>Graph: run question with conversation_id:thread_id
    Graph->>PG: load earlier turns from this thread
    Graph->>Graph: resolve follow-up into a standalone question
    Graph->>PG: save updated checkpoint
    Graph-->>API: node updates and final state
    API-->>Browser: SSE progress, answer, citations and optional diagram
```

This is durable **short-term conversational memory**, not long-term personal
memory. A related question such as “How do I define it?” can use the preceding
“What is a Factory?” turn in the same thread. A new conversation receives a new
thread and does not inherit that context.

## Diagrams

When the cited page contains an extracted manual diagram, the backend can return
the original image. Otherwise, it may generate Graphviz DOT when the user asks
for a diagram or a diagram is clearly useful.

Generated DOT is validated before it is returned. Diagram failure is nonfatal:
the grounded answer and sources are still returned.

Supported diagram types:

- `auto`
- `hierarchy`
- `relationship`
- `process`
- `decision`
- `architecture`

## Local development without containerized application processes

Start the infrastructure:

```bash
docker compose up -d postgres redis chroma
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

For host-run backend processes, use host ports in `.env`:

```dotenv
GROQ_API_KEY=gsk_replace_with_your_key
CHECKPOINT_BACKEND=postgres
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/opcenter
REDIS_URL=redis://localhost:6379/0
CHROMA_MODE=server
CHROMA_HOST=localhost
CHROMA_PORT=8001
TOKENIZERS_PARALLELISM=false
```

Run the backend and frontend in separate terminals:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
source .venv/bin/activate
BACKEND_URL=http://127.0.0.1:8000 streamlit run app.py
```

SQLite remains available only for explicit local development:

```dotenv
CHECKPOINT_BACKEND=sqlite
CHAT_MEMORY_PATH=data/chat_memory.sqlite
```

## Tests and evaluation

Run unit and integration tests:

```bash
source .venv/bin/activate
pytest -q
```

Run the 50-case golden evaluation corpus:

```bash
docker compose --profile tools run --rm evaluate
```

### Golden dataset

The versioned dataset in
[`tests/evaluation_questions.json`](tests/evaluation_questions.json) contains
an initial 50 cases covering direct and indirect questions, conversation-dependent
follow-ups, procedures, fields and tables, comparisons, insufficient evidence,
out-of-scope routing, Execution Electronics, and Execution Discrete.

| Category | Cases | What it validates |
| --- | ---: | --- |
| Direct | 7 | Definitions and explicit concepts |
| Indirect | 8 | Semantic retrieval without exact wording |
| Follow-up | 5 | Short-term conversation context and thread memory |
| Procedure | 5 | Ordered, grounded instructions |
| Table/field | 5 | Structured field and column retrieval |
| Comparison | 4 | Coverage of every requested concept |
| Unsupported | 3 | In-scope questions with insufficient evidence |
| Irrelevant | 3 | Out-of-scope routing |
| Execution Electronics | 5 | Product-specific manual routing |
| Execution Discrete | 5 | Product-specific manual routing |
| **Initial total** | **50** | End-to-end RAG regression coverage; human-approved production cases can extend it |

### Retrieval ablation benchmark

The golden questions also have exact EvidenceUnit, source-file, and PDF-page
labels in
[`tests/retrieval_ground_truth.json`](tests/retrieval_ground_truth.json). The
local benchmark compares the same questions and `K=5` cutoff across two paths:

- **Semantic-only baseline:** Chroma dense-vector retrieval
- **Hybrid system:** Chroma + BM25 + weighted reciprocal-rank fusion + neighbor
  context expansion + a local cross-encoder reranker

Run the benchmark without Groq or LangSmith calls:

```bash
source .venv/bin/activate
python retrieval_benchmark.py
```

The current run scores the 44 answerable cases; the six unsupported and
irrelevant cases are retained in the 50-case report but excluded from retrieval
F1 because they intentionally have no relevant manual evidence.

| Metric at K=5 | Semantic-only | Hybrid |
| --- | ---: | ---: |
| Precision@5 | 11.82% | 18.64% |
| Recall@5 | 48.86% | 73.86% |
| Macro F1@5 | 18.72% | 29.22% |
| Mean reciprocal rank | 39.36% | 67.12% |
| p50 retrieval latency | 7 ms | 141 ms |
| p95 retrieval latency | 9 ms | 175 ms |

Hybrid retrieval produced a **56.07% relative macro F1@5 lift** over the
semantic-only baseline. This is a measured local ablation result, not an
estimated claim. See the
[`JSON report`](evaluation_results/retrieval-benchmark-latest.json) for exact
configuration and per-case rankings, or open the
[`HTML dashboard`](evaluation_results/retrieval-benchmark-latest.html) for a
visual comparison.

Resume-ready wording:

> Built a grounded Opcenter manuals Q&A assistant with per-source citations;
> hybrid retrieval (Chroma + BM25 + weighted RRF + cross-encoder reranking)
> improved macro retrieval F1@5 by 56.1% over a dense-only baseline on 44
> scored questions from a 50-case golden dataset.

### Manual evaluation workflow and metrics

```mermaid
flowchart LR
    Gold["Golden dataset<br/>question + expected output"] --> RAG["Running RAG API"]
    RAG --> Actual["Actual model answer"]
    RAG --> Context["Retrieved and cited context"]
    Gold --> Rules["Deterministic evaluator"]
    Actual --> Rules
    Context --> Rules
    Gold --> Judge["LLM-as-a-judge"]
    Actual --> Judge
    Context --> Judge

    Rules --> Quality["Quality metrics<br/>terms · manual · status<br/>citations · diagrams"]
    Rules --> Latency["Latency metrics<br/>mean · median/p50 · p95"]
    Rules --> Cases["Per-case deterministic checks"]
    Judge --> Semantic["Semantic metrics<br/>relevance · correctness · faithfulness<br/>completeness · clarity"]

    Quality --> JSON["Machine-readable JSON report"]
    Latency --> JSON
    Cases --> JSON
    Quality --> Dashboard["Shareable HTML dashboard"]
    Latency --> Dashboard
    Cases --> Dashboard
    Semantic --> JSON
    Semantic --> Dashboard
```

| Metric | Evaluation rule |
| --- | --- |
| Overall pass rate | Every applicable check for a case must pass |
| Answer-term accuracy | Any expected term, or all explicitly required terms, appears in the answer or source metadata |
| Manual-routing accuracy | Retrieved sources contain every expected manual label |
| Evidence-status accuracy | Actual status matches `sufficient`, `in_scope_insufficient`, or `out_of_scope` |
| Citation-ID accuracy | For sufficient answers, answer citation IDs exactly match returned source IDs |
| Diagram-render accuracy | Generated-diagram state matches the golden expectation where specified |
| Latency | End-to-end mean, median/p50, and p95 response time |
| Semantic quality | Separate LLM judge scores relevance, correctness, faithfulness, completeness, and clarity from 1 to 5 |
| Judge tokens | Input, output, and total tokens used by semantic evaluation |

### Manual evaluation thresholds

The evaluator writes its JSON and HTML reports before returning a non-zero exit
code when a selected threshold fails. This command runs only when invoked
manually and does not deploy the application. The default thresholds are:

| Check | Default threshold |
| --- | ---: |
| Overall pass rate | At least 85% |
| Answer-term accuracy | At least 85% |
| Manual-routing accuracy | At least 90% |
| Evidence-status accuracy | At least 90% |
| Citation-ID accuracy | At least 95% |
| p95 latency | At most 75 seconds |
| Evaluation errors | 0 |
| Semantic-quality average | At least 4.0/5 when the judge is enabled |
| Semantic pass rate | At least 85% when the judge is enabled |
| Judge errors | 0 |
| Normalized drop from accepted baseline | At most 3 percentage points |

Run the threshold check against a local or test backend:

```bash
EVALUATION_API_TOKEN=replace_with_the_evaluation_secret \
GROQ_API_KEY=gsk_replace_with_your_key \
python evaluation.py --backend-url http://127.0.0.1:8000 --llm-judge --gate
```

After reviewing a successful report, it can be accepted as the historical
baseline:

```bash
cp evaluation_results/golden-50-latest.json \
  evaluation_results/accepted-baseline.json
python evaluation.py --backend-url http://127.0.0.1:8000 --gate \
  --llm-judge --baseline evaluation_results/accepted-baseline.json
```

Every threshold can be overridden with command-line options such as
`--min-citation-id-accuracy`, `--max-p95-latency`, and `--max-regression`.

The default outputs are stored at `evaluation_results/golden-50-latest.json`
and `evaluation_results/golden-50-latest.html`. The JSON contains the aggregate
metrics and machine-readable test cases. Open the HTML report in a browser to
show summary cards and expandable cases with the input, gold expectations,
actual model answer, cited retrieval context, checks, and latency.
For a locally running backend, use `python evaluation.py`; `--dataset` and
`--output` can override the input and report paths.

Run the bounded 50-user load test:

```bash
docker compose --profile tools run --rm load-test
```

The README intentionally does not publish unexecuted scores. Run the evaluator
against the indexed manuals to populate the HTML metric cards with reproducible
results. The load test separately reports completion rate, latency, and time to
first streamed answer token.

## Project layout

```text
app.py                     Streamlit frontend
backend/main.py            FastAPI app, middleware, health, readiness, metrics
backend/dependencies.py    startup, shutdown, pools, clients, graph compilation
backend/routes/chat.py     request acceptance and SSE streaming
src/graph.py               LangGraph workflow
src/nodes.py               RAG nodes and evidence logic
src/retrieval.py           Chroma/BM25 hybrid retrieval
src/ingest.py              explicit offline ingestion
Dockerfile.ingest          OCR-enabled offline ingestion image
requirements-ocr.txt       Tesseract/Paddle ingestion Python dependencies
src/llm.py                 asynchronous role-specific Groq calls
src/groq_limits.py         Redis per-model admission control
src/embeddings.py          shared embedding and reranker models
src/cache.py               Redis query/retrieval/answer caches
src/observability.py       logs and built-in process metrics
retrieval_benchmark.py     semantic-only versus hybrid retrieval ablation
evaluation.py              deterministic/semantic evaluator and HTML dashboard
scripts/add_golden_case.py manual validated golden-case intake
tests/evaluation_questions.json  versioned 50-case golden dataset
tests/retrieval_ground_truth.json exact EvidenceUnit and PDF-page labels
docker-compose.yml         local production-style deployment
docker-compose.staging.yml isolated staging backend port
```

## Operational notes

- Models are loaded once per backend process during startup.
- The LangGraph is compiled once per backend process.
- PostgreSQL and Groq HTTP connections are reused.
- Chroma clients are created during startup and reused.
- Embedding and reranker inference use bounded queues and semaphores.
- Provider retries are bounded to one retry/fallback attempt.
- HTTP 400, 401, and 403 responses are not retried unchanged.
- HTTP 429 respects `Retry-After` before one retry with jitter.
- Timeouts and server errors receive at most one retry.
- Logs exclude API keys, complete manual evidence, and full conversations.
- Health responses and client errors do not expose internal stack traces.

## Limitations

- Answers are limited to the supplied and indexed manuals.
- OCR accuracy depends on scan resolution, language packs and page complexity.
- Original figures are returned only when linked to retrieved or cited pages.
- Retrieval broadening is limited to one pass.
- Generated diagrams require sufficient verified evidence.
- Local model throughput depends on the available CPU, GPU, and memory.
- Production deployments must replace default PostgreSQL credentials and set
  company-approved CORS origins and secret injection.
