# Opcenter Chatbot

An asynchronous RAG chatbot grounded in indexed Opcenter PDF manuals.

The Streamlit frontend communicates only with a FastAPI backend. The backend
runs the LangGraph workflow, hybrid Chroma/BM25 retrieval, reciprocal-rank
fusion, cross-encoder reranking, evidence grading, citation validation, answer
verification, and optional diagram generation.

No user account or authentication is required. Anonymous `session_id`,
`conversation_id`, and `thread_id` UUIDs isolate browser conversations, while
LangGraph checkpoints preserve conversation memory in PostgreSQL.

## Architecture

```text
Browser
  -> Streamlit frontend
  -> POST /v1/chat
  -> GET /v1/chat/{request_id}/stream (SSE)
  -> FastAPI backend replicas
       -> LangGraph compiled once per backend process
       -> Chroma server + BM25 hybrid retrieval
       -> all-MiniLM-L6-v2 query embeddings
       -> ms-marco-MiniLM-L-6-v2 cross-encoder reranking
       -> role-specific asynchronous Groq models
       -> PostgreSQL conversation checkpoints
       -> Redis limits, queues, cache, request status, and thread ownership

Offline ingestion
  manuals/*.pdf
    -> EvidenceUnits and RetrievalSegments
    -> BM25 and metadata indexes in indexes/
    -> embeddings in the Chroma server
    -> extracted manual diagrams in indexes/manual_figures/
```

The application never rebuilds indexes during startup or a chat request.
Ingestion is always an explicit offline command.

## Main features

- Grounded answers with `[S1]`, `[S2]`, and similar citations
- Direct definitions, procedures, comparisons, troubleshooting, and follow-ups
- Multi-aspect retrieval with independent evidence coverage
- Relationship-aware retrieval for hierarchy and parent-child questions
- Original diagrams extracted from cited manual pages
- Generated hierarchy, relationship, process, decision, and architecture diagrams
- Basic greetings without unnecessary retrieval or Groq calls
- Anonymous browser-session isolation without user accounts
- Two backend replicas in Docker Compose
- Bounded Groq and local inference queues
- Prometheus-compatible metrics

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
cd ~/Desktop/opcenter-chatbot/opcenter-chatbot
touch .env
chmod 600 .env
```

Add at least the following to `.env`:

```dotenv
GROQ_API_KEY=gsk_replace_with_your_key
TOKENIZERS_PARALLELISM=false
```

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

| Service | Purpose | Host port |
| --- | --- | --- |
| `frontend` | Streamlit chat UI | `8501` |
| `backend` | FastAPI and LangGraph, two replicas | Internal `8000` |
| `postgres` | LangGraph conversation checkpoints | `5432` |
| `redis` | Limits, queues, cache, status, ownership | Internal `6379` |
| `chroma` | Vector database server | `8001` |
| `ingest` | Explicit offline ingestion profile | None |
| `evaluate` | Evaluation profile | None |
| `load-test` | Bounded concurrency test profile | None |

PostgreSQL and Chroma use persistent named volumes. Manuals and index metadata
are bind-mounted from the repository and are not copied into container images.

## Environment variables

### Required

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Groq API key; must begin with `gsk_` |
| `DATABASE_URL` | PostgreSQL checkpoint URL when running the backend outside Compose |
| `REDIS_URL` | Redis URL when running the backend outside Compose |
| `CHROMA_HOST` | Chroma host when `CHROMA_MODE=server` outside Compose |

Docker Compose supplies `DATABASE_URL`, `REDIS_URL`, `CHROMA_HOST`, and
`CHROMA_PORT` to the backend. Only the Groq key is required in `.env` for the
default local Compose deployment.

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
| `BM25_INDEX_PATH` | `indexes/bm25.pkl` | BM25 index |
| `EVIDENCE_UNITS_PATH` | `indexes/evidence_units.json` | EvidenceUnit store |
| `RETRIEVAL_SEGMENTS_PATH` | `indexes/retrieval_segments.json` | Retrieval segments |
| `CHAT_MEMORY_PATH` | `data/chat_memory.sqlite` | SQLite path in local SQLite mode |

### Models

| Variable | Default |
| --- | --- |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DEVICE` | `cpu` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `INFERENCE_MAX_CONCURRENCY` | `4` |
| `INFERENCE_MAX_QUEUE_DEPTH` | `32` |
| `GROQ_REQUEST_TIMEOUT` | `90` seconds |

Each Groq role has a primary model and at most one fallback. Override a role
with:

```dotenv
GROQ_ANSWER_PRIMARY_MODEL=openai/gpt-oss-120b
GROQ_ANSWER_FALLBACK_MODEL=qwen/qwen3.6-27b
GROQ_ANSWER_TIMEOUT=90
GROQ_ANSWER_MAX_OUTPUT_TOKENS=4096
```

Supported role prefixes are `PLANNER`, `QUERY_BROADENING`, `GRADER`, `ANSWER`,
`VERIFIER`, and `DIAGRAM`.

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

The current index schema is version 8. Ingestion creates:

- `indexes/evidence_units.json`
- `indexes/retrieval_segments.json`
- `indexes/search_representations.json`
- `indexes/heading_index.json`
- `indexes/concept_index.json`
- `indexes/ingestion_audit.json`
- `indexes/manifest.json`
- `indexes/bm25.pkl`
- `indexes/manual_figures.json`
- `indexes/manual_figures/`
- Chroma retrieval and search-representation collections

PyMuPDF extracts embedded text, tables, headings, procedures, warnings, and
usable manual figures. Scanned pages without embedded text require OCR before
ingestion.

Evidence IDs and retrieval metadata are preserved from ingestion through
Chroma, BM25, reranking, citations, and returned sources.

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
| `GET /metrics` | Prometheus-compatible metrics |

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

Run the production evaluation corpus:

```bash
docker compose --profile tools run --rm evaluate
```

Run the bounded 50-user load test:

```bash
docker compose --profile tools run --rm load-test
```

The evaluator checks answer terms, manual routing, evidence status, citation
IDs, diagrams, and latency. The load test reports completion rate, latency, and
time to first streamed answer token.

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
src/llm.py                 asynchronous role-specific Groq calls
src/groq_limits.py         Redis per-model admission control
src/embeddings.py          shared embedding and reranker models
src/cache.py               Redis query/retrieval/answer caches
src/observability.py       logs and Prometheus metrics
docker-compose.yml         local production-style deployment
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
- Scanned PDFs need OCR preprocessing.
- Original figures are returned only when linked to retrieved or cited pages.
- Retrieval broadening is limited to one pass.
- Generated diagrams require sufficient verified evidence.
- Local model throughput depends on the available CPU, GPU, and memory.
- Production deployments must replace default PostgreSQL credentials and set
  company-approved CORS origins and secret injection.
