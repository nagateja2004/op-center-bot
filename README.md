# Opcenter Chatbot

Streamlit question-answering application grounded in local Opcenter PDF manuals.
It uses hybrid Chroma/BM25 retrieval, rank fusion, context expansion, cross-encoder
reranking, evidence grading, answer verification, optional Graphviz diagrams, and
SQLite-backed LangGraph conversation memory.

The ingestion and runtime paths are text-only. Images are not extracted, OCRed,
described, embedded, or displayed.

The current document/index schema is **version 4**. Application startup validates
the schema and ID alignment but never deletes or rebuilds existing indexes.

## Architecture

```text
manuals/*.pdf
    -> PyMuPDF text, heading, procedure, note, warning, and table extraction
    -> semantic EvidenceUnits (definitions, concepts, procedures, prerequisites, warnings, tables)
    -> 150-220 word RetrievalSegments validated with the all-MiniLM-L6-v2 tokenizer
    -> indexes/evidence_units.json
       indexes/retrieval_segments.json
       indexes/bm25.pkl
       indexes/chroma/ (collection: opcenter_manuals)

Streamlit question
    -> understand_question (latest 6 messages)
       -> single topic or 2-6 required aspects
       -> deterministic aspect queries, planner search queries, and required answer shape
    -> per-aspect Chroma + BM25 retrieval and weighted RRF
    -> add neighbouring RetrievalSegments and deduplicate
    -> per-aspect cross-encoder reranking
    -> resolve selected segments into complete EvidenceUnits and deduplicate by evidence_id
    -> round-robin merge (at most 10 final chunks)
    -> per-aspect evidence grading and coverage
       -> sufficient/partial: answer -> correct and verify -> optional diagram -> END
       -> retry once: broaden missing aspects only -> retrieve again
       -> insufficient/out of scope: fallback -> END

Conversation checkpoints -> data/chat_memory.sqlite (SqliteSaver)
```

At most 10 final EvidenceUnits are eligible for Groq prompts. Evidence is capped
per source and globally. Cross-aspect and cross-manual evidence is preserved, and
answers expose only sources actually cited as `[S1]`, `[S2]`, and so on.

For multi-part questions, every aspect is retrieved, expanded, reranked, and
graded independently. Missing aspects may be retried once without rerunning
retrieval for supported aspects. Supported aspects can still produce a clearly
labeled partial answer when another part lacks manual support.

## Installation

Python 3.11 or 3.12 is required. The local cross-encoder is disabled on
Python 3.13 because the current sentence-transformers/PyTorch combination can be
unstable there; retrieval falls back to RRF order.

```bash
cd ~/Desktop/opcenter-chatbot
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

The reranker stack is pinned to `torch==2.3.1`, `transformers==4.48.3`, and
`sentence-transformers==3.4.1`. On Apple Silicon, reranking uses MPS because the
PyTorch CPU/Accelerate matrix path can return non-finite scores for this model.
Other systems use CPU and retain RRF order if the startup score probe fails.

## Environment variables

Edit `.env`:

| Variable | Required | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | None |
| `GROQ_<ROLE>_MODEL` | No | Role default in `.env.example` |
| `GROQ_<ROLE>_FALLBACK_MODEL` | No | Role default; blank disables fallback |
| `GROQ_<ROLE>_TEMPERATURE` | No | Role-specific |
| `GROQ_<ROLE>_MAX_OUTPUT_TOKENS` | No | Role-specific |
| `GROQ_<ROLE>_TIMEOUT` | No | Role-specific |
| `GROQ_<ROLE>_ALLOW_STRUCTURED_OUTPUT` | No | `true` only for planner and grader |
| `GROQ_<ROLE>_INPUT_TOKEN_BUDGET` | No | Role-specific deterministic prompt cap |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DEVICE` | No | `cpu` |
| `RERANKER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CHROMA_DIRECTORY` | No | `indexes/chroma` |
| `BM25_INDEX_PATH` | No | `indexes/bm25.pkl` |
| `EVIDENCE_UNITS_PATH` | No | `indexes/evidence_units.json` |
| `RETRIEVAL_SEGMENTS_PATH` | No | `indexes/retrieval_segments.json` |
| `CHAT_MEMORY_PATH` | No | `data/chat_memory.sqlite` |

Startup requires a Groq key and reports a safe configuration error when it is
missing. Never commit `.env`, a real key, or credentials in source code.

## Add PDF manuals

Place PDFs directly in:

```text
opcenter-chatbot/manuals/
```

Subdirectories are not scanned. PDFs must contain embedded text; OCR is not
performed.

## Ingest and index

Schema changes require an explicit operator action. If Streamlit reports an old
index schema, stop the app and run:

```bash
python -m src.ingest
```

This is the only supported re-indexing command. It re-extracts the manuals and
replaces Chroma/BM25 only because the operator invoked it explicitly. Streamlit
and retrieval startup never call ingestion or delete existing artifacts.

This creates or updates:

- `indexes/evidence_units.json` - complete semantic units with source metadata,
  structured table JSON, ordered procedure steps, and attached notes/warnings
- `indexes/retrieval_segments.json` - searchable children with EvidenceUnit IDs,
  150-220 word targets, and embedding-token counts
- `indexes/manifest.json` - PDF SHA-256 hashes and index counts
- `indexes/bm25.pkl` - BM25 object and ordered RetrievalSegment IDs
- `indexes/chroma/` - persistent `opcenter_manuals` collection

Only RetrievalSegments are indexed. Unchanged PDF hashes reuse both stored levels
and do not rebuild Chroma or BM25. The command validates segment-to-evidence links,
tokenizer limits, duplicate IDs, and matching RetrievalSegment ID sets across
Chroma, BM25, and `retrieval_segments.json`.

After the schema-4 rebuild, start the application normally:

```bash
streamlit run app.py
```

## Docker

Docker uses Python 3.11, runs Streamlit as a non-root user, and includes the
Graphviz system binary. Copy `.env.example` to `.env` and set `GROQ_API_KEY`
before starting the container. The `.env` file must never be committed.

Build:

```bash
docker build -t opcenter-chatbot .
```

Run:

```bash
docker run --rm \
  --env-file .env \
  -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/indexes:/app/indexes" \
  -v "$(pwd)/manuals:/app/manuals" \
  opcenter-chatbot
```

Docker Compose:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

Logs:

```bash
docker compose logs -f opcenter-chatbot
```

Rebuild:

```bash
docker compose build --no-cache
docker compose up
```

Place proprietary PDF manuals in `manuals/`; they are bind-mounted and excluded
from Git and the image. Generated Chroma, BM25, EvidenceUnit, and RetrievalSegment
artifacts persist under `indexes/`. LangGraph conversation memory persists at
`data/chat_memory.sqlite`.

Run ingestion before the UI when indexes are absent or stale:

```bash
docker compose run --rm opcenter-chatbot python -m src.ingest
```

Embedding and reranker models download into the persistent Hugging Face cache on
first use. The UI is available on port 8501. Verify container health with:

```bash
curl --fail http://localhost:8501/_stcore/health
```

## Run tests

```bash
pytest -q
```

The test suite mocks Groq for graph scenarios and covers document-level splitting,
segment-only index alignment, compact prompts and budgets, role/model isolation,
rate-limit fallbacks, hard comparisons, cited diagrams, SQLite memory, and
Streamlit final-only streaming.

Run the real evaluation corpus after configuring `GROQ_API_KEY`:

```bash
python evaluation.py
```

The evaluator reports retrieval hit rate, fallback accuracy, citation coverage,
and mean/median/p95 latency.

## Run Streamlit

```bash
streamlit run app.py
```

The project disables Streamlit's source-file watcher because its module scanner
is incompatible with PyTorch's dynamic `torch.classes` namespace. Chat reruns and
sidebar controls continue to work; automatic reruns after editing source files are
disabled, so restart Streamlit after code changes.

The sidebar can start or delete a conversation, hide cited sources, and disable
diagram generation. Streamlit stores only the current thread ID and UI controls;
LangGraph conversation history is persisted in SQLite using
`configurable.thread_id`.

The live UI consumes `graph.stream(..., stream_mode="updates", version="v2")`
only for node progress. Planning, grading, draft generation, and verification
content is never rendered. After completion, Streamlit reads the final verified
checkpoint, streams only its answer, and then renders the diagram and cited source
cards.

## Supported question types

- Direct definitions and explanations
- Indirect or paraphrased manual questions
- Context-dependent follow-up questions
- Ordered procedures and how-to steps
- Tables, fields, columns, buttons, and configuration definitions
- Comparisons and cross-manual questions
- Multi-part troubleshooting questions with likely reasons and checks
- Configuration-time versus runtime explanations
- Data CDO versus Service CDO comparisons ending in a table
- Cross-release synthesis with an explicit release warning
- Hierarchy, process, architecture, and relationship diagrams when grounded
- Clearly labeled fallbacks for unsupported Opcenter and irrelevant questions

## Limitations

- Answers are limited to the supplied manuals and their extracted text/tables.
- Scanned PDFs without embedded text require external preprocessing.
- PDF images and screenshots are intentionally ignored.
- Contents, indexes, repeated page furniture, and image-only captions are removed.
- Procedure steps and complete table rows are indivisible; ingestion fails clearly
  if one indivisible item alone exceeds the embedding tokenizer limit.
- Retrieval broadening is limited to one retry; each aspect uses at most three
  focused query variations.
- Diagrams require sufficient verified evidence and are limited to 3-10 nodes.
- Reranking falls back to RRF order if the cross-encoder cannot load or returns
  invalid scores. A failed model is disabled for the rest of that process, so one
  warning is logged instead of one warning per aspect.
- A recovered Groq planner/grader failure is logged as a warning; unrecovered
  provider failures use sanitized role/error categories. Prompts, credentials,
  organization identifiers, and manual text are never shown in the UI error.
- SQLite memory is intended for a local, single-host Streamlit deployment; it is
  not a multi-server conversation store.
- Additions, schema changes, or manual changes require explicitly rerunning
  `python -m src.ingest`.
