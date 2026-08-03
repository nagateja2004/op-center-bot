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

```mermaid
flowchart LR
    User["User"] --> UI["Streamlit chat UI"]
    UI -->|"POST request"| API["FastAPI backend"]
    API -->|"SSE answer stream"| UI
    API --> Graph["LangGraph RAG workflow"]

    Graph --> Models["Role-specific Groq LLMs"]
    Graph --> Retrieval["Hybrid retrieval and reranking"]
    Retrieval --> Chroma[("Chroma vectors")]
    Retrieval --> BM25[("BM25 index")]
    Retrieval --> Encoder["Cross-encoder reranker"]
    Graph <--> Postgres[("PostgreSQL checkpoints")]
    API <--> Redis[("Redis queues, cache and limits")]

    Graph -. "traces" .-> LangSmith["LangSmith"]
    API -. "latency and queue metrics" .-> Metrics["Built-in /metrics endpoint"]

    PDFs["Opcenter PDF manuals"] --> Ingest["Offline ingestion"]
    Ingest --> Evidence["EvidenceUnits and retrieval segments"]
    Evidence --> Chroma
    Evidence --> BM25
    Ingest --> Figures["Extracted manual figures"]
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
- Built-in latency, inference, and queue metrics without a monitoring service dependency
- Optional LangSmith traces for the complete LangGraph request path
- A 50-case golden dataset with JSON and visual HTML evaluation reports
- CI quality gates with fixed thresholds and accepted-baseline regression checks

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
touch .env
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

For a candidate/staging deployment that will be scored by the semantic
evaluator, also add a long random secret:

```dotenv
EVALUATION_API_TOKEN=replace_with_a_long_random_staging_secret
GROQ_JUDGE_MODEL=openai/gpt-oss-20b
```

Use the same `EVALUATION_API_TOKEN` as a GitHub Actions repository secret. It
protects the bounded evidence excerpts used by the judge. Normal chat requests
cannot request or receive those excerpts. Do not enable this evaluation route
on the current production deployment.

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
| `GROQ_JUDGE_MODEL` | `openai/gpt-oss-20b` (evaluation process only) |

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
    Start(("Start")) --> Understand["Understand question"]
    Understand -->|"basic chat"| Done(("End"))
    Understand -->|"knowledge question"| Retrieve["Retrieve documents<br/>Chroma + BM25 + RRF"]
    Retrieve --> Expand["Expand neighboring context"]
    Expand --> Rerank["Cross-encoder reranking"]
    Rerank --> Grade["Grade evidence by required aspect"]
    Grade -->|"sufficient or partial"| Answer["Generate grounded answer"]
    Grade -->|"retry once"| Broaden["Broaden query"]
    Broaden --> Retrieve
    Grade -->|"insufficient"| Fallback["Generate evidence-aware fallback"]
    Fallback --> Done
    Answer --> Verify["Validate citations and verify answer"]
    Verify -->|"grounded and diagram useful"| Diagram["Generate and validate diagram"]
    Verify -->|"answer complete"| Done
    Diagram --> Done

    Memory[("PostgreSQL conversation memory")] <--> Understand
    Trace["LangSmith trace"] -.-> Understand
    Trace -.-> Retrieve
    Trace -.-> Grade
    Trace -.-> Answer
    Trace -.-> Verify
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

### Approved final LLMOps workflow

> **Deployment boundary:** Every automation in this diagram is implemented in
> the repository. It remains inactive until the repository is published and its
> GitHub environments, secrets, variables, and staging server are configured.
> The currently deployed production server was not changed while building it.

```mermaid
flowchart TD
    Change["1. Change code, prompt, retrieval or model locally"] --> LocalTest["2. Run local tests"]
    LocalTest --> Push["3. Commit and push a feature branch"]
    Push --> PR["4. Create or update pull request"]
    PR --> StagePR["5. Deploy the PR commit to staging"]
    StagePR --> PRCases["6. Run 10 representative golden cases"]
    PRCases --> PRRAG["7. Execute the LangGraph RAG pipeline"]

    PRRAG --> PRTrace["Store staging traces in LangSmith"]
    PRRAG --> PREval["Deterministic checks and LLM-as-a-judge"]
    PREval --> PRGate{"8. PR CI gate passed?"}

    PRGate -->|"No"| Diagnose["Inspect the HTML report and LangSmith trace"]
    Diagnose --> Fix["Fix locally, commit and push again"]
    Fix --> PR

    PRGate -->|"Yes"| Merge["9. Human reviews and merges the PR"]
    Merge --> StageMain["10. Deploy the merged commit to staging"]
    StageMain --> FullCases["11. Run the full 50+ case dataset"]
    FullCases --> FullEval["Deterministic checks, LLM judge and baseline comparison"]
    FullCases --> FullTrace["Store full evaluation traces in LangSmith"]
    FullEval --> ReleaseGate{"12. Release CI gate passed?"}

    ReleaseGate -->|"No"| Diagnose
    ReleaseGate -->|"Yes"| Production["13. Deploy the evaluated commit to production"]
    Production --> Monitor["14. Monitor production traces daily"]
    Monitor --> ProductionCheck{"Verified production problem found?"}

    ProductionCheck -->|"No"| Monitor
    ProductionCheck -->|"Yes"| Review["15. Human reviews the trace and manuals"]
    Review --> NewCase["16. Add an approved case to the golden dataset"]
    NewCase --> DatasetPR["17. Create a golden-dataset pull request"]
    DatasetPR --> PR
```

The **PR gate** is the fast pre-merge check: it evaluates 10 cases and blocks
the pull request when quality, latency, or error thresholds fail. The
**release gate** runs after merge against all 50+ cases on staging and blocks
production deployment when the full evaluation or accepted-baseline comparison
fails. A pull request is not the `git pull` command: developers push their
feature branch, then request that GitHub merge it into `main` after review.

The separate judge model runs at temperature `0` and scores relevance,
correctness, faithfulness, completeness, and clarity from 1 to 5. A case passes
when its average is at least 4.0 and its correctness and faithfulness scores are
both at least 4.0. The aggregate gate requires a semantic average of at least
4.0, an 85% semantic pass rate, and zero judge errors. The callable PR workflow
runs a 10-case subset with one case from every dataset category; manual and
nightly runs use all 50. Treat these as starting thresholds and calibrate them
with human-reviewed reports before making the semantic gate a required merge
check. Baseline comparison is skipped for the subset because a 10-case report
must not be compared with a 50-case baseline.

Only human-confirmed production failures may become new golden cases. A failed
gate must lead back through a code change, PR update, fresh deployment, and new
evaluation; it must never reuse the previously failed deployment.

| Capability | Status |
| --- | --- |
| 50-case golden dataset and deterministic evaluation | Implemented |
| JSON/HTML reports and fixed/baseline CI thresholds | Implemented |
| LangSmith trace configuration | Implemented and locally verified |
| Protected retrieval evidence for evaluation | Implemented |
| LLM-as-a-judge semantic scoring and token accounting | Implemented |
| Callable CI gate for a candidate/staging URL | Implemented |
| SSH deployment of each internal PR/merge commit to staging | Implemented; requires GitHub/server configuration |
| Automatic production release after the merged commit passes all 50 cases on staging | Implemented; requires GitHub/server configuration |
| Daily LangSmith error, latency, cost, and feedback monitoring | Implemented; requires LangSmith credentials |
| Production observation → GitHub review issue | Implemented |
| Human-approved issue → validated golden-dataset PR | Implemented |

### Activate the complete delivery pipeline

The workflow uses the existing Docker Compose deployment and standard SSH. The
server must have Docker Compose, and the SSH user must be allowed to run it.
Create separate staging and production directories before enabling the
workflow. Each directory must already contain its own `.env`, manuals, and
indexes. Deployment deliberately never copies or replaces those files.

Use different Compose project names and LangSmith projects:

```dotenv
# Staging server .env
COMPOSE_PROJECT_NAME=opcenter-staging
COMPOSE_FILE=docker-compose.yml:docker-compose.staging.yml
DEPLOYMENT_ENVIRONMENT=staging
APP_PORT=8502
POSTGRES_PORT=5433
CHROMA_EXPOSE_PORT=8002
BACKEND_EXPOSE_PORT=8003
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=replace_with_your_langsmith_key
LANGSMITH_PROJECT=opcenter-rag-staging
EVALUATION_API_TOKEN=the_same_random_value_stored_in_github
```

```dotenv
# Production server .env
COMPOSE_PROJECT_NAME=opcenter-production
DEPLOYMENT_ENVIRONMENT=production
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=replace_with_your_langsmith_key
LANGSMITH_PROJECT=opcenter-rag-production
```

Create GitHub `staging` and `production` environments. A required reviewer on
the production environment provides a final human release approval if desired.

Repository secrets:

| Secret | Purpose |
| --- | --- |
| `STAGING_SSH_HOST`, `STAGING_SSH_USER`, `STAGING_SSH_KEY`, `STAGING_SSH_KNOWN_HOSTS` | Deploy the candidate commit to staging with a pinned host key |
| `PRODUCTION_SSH_HOST`, `PRODUCTION_SSH_USER`, `PRODUCTION_SSH_KEY`, `PRODUCTION_SSH_KNOWN_HOSTS` | Release the evaluated merge commit with a pinned host key |
| `GROQ_API_KEY` | Run the separate semantic judge |
| `EVALUATION_API_TOKEN` | Authenticate protected staging evidence |
| `LANGSMITH_API_KEY` | Read production trace aggregates and trace links |
| `LANGSMITH_WORKSPACE_ID` | Optional; required for organization-scoped LangSmith keys |

Repository variables:

| Variable | Example/default |
| --- | --- |
| `STAGING_DEPLOY_PATH` | `/srv/opcenter-staging` |
| `STAGING_SSH_PORT` | `22` |
| `STAGING_RAG_BACKEND_URL` | Public or runner-accessible staging API URL |
| `PRODUCTION_DEPLOY_PATH` | `/srv/opcenter-production` |
| `PRODUCTION_SSH_PORT` | `22` |
| `LANGSMITH_PRODUCTION_PROJECT` | `opcenter-rag-production` |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` |
| `MONITOR_LOOKBACK_MINUTES` | `1440` |
| `MONITOR_MAX_ERROR_RATE` | `0.05` |
| `MONITOR_MAX_P95_LATENCY` | `30` |
| `MONITOR_MIN_FEEDBACK_SCORE` | `0.5` |
| `MONITOR_MAX_AVERAGE_COST_USD` | `0` disables the cost alert; cost is still reported |

The resulting behavior is:

1. An internal, non-draft PR is deployed to staging.
2. One case from every golden-dataset category runs through deterministic and
   semantic evaluation. Failure blocks the workflow; success permits merge.
3. A merge to `main` is deployed to staging again and evaluated with the full
   dataset. Production is released only after this gate passes.
4. LangSmith production traces are checked daily for errors, p95 latency,
   average cost, and low `correctness` or `user_feedback` scores.
5. A detected issue creates or updates a GitHub issue without copying prompts,
   answers, or manual content into GitHub.
6. A human validates the trace and expected answer, adds the JSON golden case,
   and applies `golden-approved`. GitHub then validates it and opens a dataset PR.

Configure GitHub branch protection after the first run and require the
candidate evaluation check. Merging remains a human decision; deployment after
the merge is automatic. Pull requests from forks are not deployed because
repository secrets are intentionally unavailable to them.

The staging override binds its backend to `127.0.0.1:8003` by default. Route a
staging HTTPS hostname to that port through the server's reverse proxy and use
that URL for `STAGING_RAG_BACKEND_URL`. If staging and production are on
different servers, the port overrides can be changed or omitted.

### Current evaluation workflow and metrics

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

    Quality --> Gate{"CI quality gate"}
    Latency --> Gate
    Cases --> Gate
    Semantic --> Gate
    Gate -->|"pass"| Merge["Allow merge or deployment"]
    Gate -->|"fail"| Block["Block regression"]

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

### CI-gated evaluation

The evaluator writes its JSON and HTML reports before returning a non-zero exit
code when a gate fails. The default gate is:

| Check | Default threshold |
| --- | ---: |
| Overall pass rate | At least 85% |
| Answer-term accuracy | At least 85% |
| Manual-routing accuracy | At least 90% |
| Evidence-status accuracy | At least 90% |
| Citation-ID accuracy | At least 95% |
| p95 latency | At most 30 seconds |
| Evaluation errors | 0 |
| Semantic-quality average | At least 4.0/5 when the judge is enabled |
| Semantic pass rate | At least 85% when the judge is enabled |
| Judge errors | 0 |
| Normalized drop from accepted baseline | At most 3 percentage points |

Run the gate against a local or candidate/staging backend:

```bash
EVALUATION_API_TOKEN=replace_with_the_staging_secret \
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

The `RAG evaluation gate` GitHub Actions workflow performs a deterministic gate
contract check on relevant pull requests and pushes. Its candidate job runs the full
50 cases nightly or manually, and 10 cases when called by a PR deployment
workflow. Configure the repository variable `STAGING_RAG_BACKEND_URL`, the
repository secrets `GROQ_API_KEY` and `EVALUATION_API_TOKEN`, or pass a
branch-specific candidate `backend_url` through `workflow_call`. Never configure
this variable with the production URL. Reports are uploaded as workflow
artifacts even when the gate fails.

For a true pull-request merge gate, deploy the PR branch first, call this
reusable workflow with that deployment URL, and make the caller's live evaluation
job a required status check in GitHub branch protection. This avoids evaluating
a shared backend that does not contain the proposed code.

[`delivery-pipeline.yml`](.github/workflows/delivery-pipeline.yml) now performs
that deployment and calls the reusable gate. On `main`, it repeats evaluation
with the full dataset before releasing the same commit to production.

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
src/llm.py                 asynchronous role-specific Groq calls
src/groq_limits.py         Redis per-model admission control
src/embeddings.py          shared embedding and reranker models
src/cache.py               Redis query/retrieval/answer caches
src/observability.py       logs and built-in process metrics
evaluation.py              deterministic/semantic evaluator and HTML dashboard
production_monitor.py      LangSmith production regression monitor
scripts/deploy_remote.sh   staging/production SSH deployment contract
scripts/add_golden_case.py validated human-review dataset intake
tests/evaluation_questions.json  versioned 50-case golden dataset
.github/workflows/evaluation-gate.yml  CI and scheduled quality gate
.github/workflows/delivery-pipeline.yml PR staging gate and production release
.github/workflows/production-monitor.yml trace-to-review issue automation
.github/workflows/golden-case-intake.yml approved issue-to-dataset PR automation
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
- Scanned PDFs need OCR preprocessing.
- Original figures are returned only when linked to retrieved or cited pages.
- Retrieval broadening is limited to one pass.
- Generated diagrams require sufficient verified evidence.
- Local model throughput depends on the available CPU, GPU, and memory.
- Production deployments must replace default PostgreSQL credentials and set
  company-approved CORS origins and secret injection.
