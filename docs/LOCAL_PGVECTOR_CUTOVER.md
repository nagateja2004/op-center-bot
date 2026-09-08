# Local pgvector cutover — 2026-09-07

## Result

The running backend now uses PostgreSQL + pgvector for body and representation
dense searches. Local `.env` selects `VECTOR_STORE=pgvector` and
`DATABASE_URL=postgresql:///opcenter?host=/tmp`. The final URL assignment overrides
the older Docker-oriented assignment retained above it. The resolved configuration
and running API's `pgvector` readiness check were verified.

The pinned, normalized `sentence-transformers/all-MiniLM-L6-v2` model was reused;
its dimension was probed as 384. Existing JSON was imported without reparsing PDFs
or writing/deleting Chroma collections.

| Persisted validation | Result |
| --- | ---: |
| Documents | 8 |
| Body chunks | 15,721 |
| Search representations | 21,121 |
| Total rows | 36,842 |
| Embedding dimension | 384 |
| Missing embeddings / pages / document IDs / parent IDs | 0 / 0 / 0 / 0 |
| Duplicate chunk IDs within namespaces | 0 |

Both collection ID sets matched the canonical JSON artifacts.

## Infrastructure

- Native PostgreSQL 17.11, pgvector 0.8.6, Redis 8.10.1, Python 3.12.13.
- Cluster: `/opt/homebrew/var/opcenter-postgres`; application database approximately
  128 MB immediately after import, excluding cluster/WAL overhead.
- PostgreSQL listens only on a Unix socket under `/tmp`, using macOS user peer
  authentication. No TCP database port is exposed. This is a local development
  setup, not a least-privilege multi-user production deployment.
- Login service: `~/Library/LaunchAgents/local.opcenter.postgres.plist`, with
  `LC_ALL=en_US.UTF-8` required for PostgreSQL startup in this macOS environment.
  Redis uses its Homebrew login service and binds to loopback.
- `opcenter_vector_test` is a separate disposable database. Storage tests create
  and remove only unique test schemas there.
- SQLite checkpoints were retained to preserve existing conversation history.

Docker still cannot start. No Docker reset, volume deletion or image cleanup was
performed. Homebrew's PostgreSQL post-install also failed because its compiled
support directories were missing. Missing links under
`/opt/homebrew/share/postgresql@17` and `/opt/homebrew/lib/postgresql@17` were added
to the installed 17.11 support files without replacing existing entries. Recheck
these links after a PostgreSQL upgrade.

## Run locally

From `/Users/nagatejay/Documents/op centre bot`, the database and Redis normally
start at login. Restart the registered services if needed:

```bash
launchctl kickstart -k gui/$(id -u)/local.opcenter.postgres
brew services restart redis
```

Start backend and frontend in separate terminals; do not start duplicates if this
app already occupies ports 8000 and 8501:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LANGSMITH_TRACING=false \
  .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
BACKEND_URL=http://127.0.0.1:8000 .venv/bin/python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Open <http://127.0.0.1:8501>. Check readiness:

```bash
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8501/_stcore/health
```

The API returned `status=ready` with graph, PostgreSQL, Redis, pgvector, BM25,
EvidenceUnits and both models all true. There is no Chroma readiness check in the
active configuration.

## Ingestion and verification

Stop API traffic before reimporting the full canonical JSON snapshot:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LANGSMITH_TRACING=false \
  .venv/bin/python -u -m src.pgvector_migrate --backfill
```

Restart the backend afterward. JSON remains required for BM25, catalogs, neighbors
and EvidenceUnit resolution.

Run all tests with explicitly designated storage-test and read-only corpus databases:

```bash
PGVECTOR_TEST_DATABASE_URL='postgresql:///opcenter_vector_test?host=/tmp' \
PGVECTOR_CORPUS_DATABASE_URL='postgresql:///opcenter?host=/tmp' \
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q --tb=short
```

Final result: **314 passed, zero skipped, six existing deprecation warnings,
33.77 seconds**. This includes nine real SQL tests and three read-only live corpus
tests for dense ordering, labeled hybrid evidence and citation metadata.

An earlier post-selector run exposed 12 failures from tests inheriting live `.env`.
Test bootstrap now isolates its backend and credentials; PostgreSQL integration
still explicitly opts into designated databases. No retrieval expectation was
weakened. A settings credential-representation regression failed before its fix
and passed afterward; `Settings.__repr__` now excludes credentials/service URLs.

One actual `/v1/chat` + SSE request, “What is a Factory in Opcenter?”, completed in
**10,390.8 ms** client wall time. Its sufficient-evidence answer cited `[S1]`, the
Modeling guide, “Defining a Factory”, printed page 2-5, physical PDF page 111.
This is a single smoke test, not a latency benchmark or improvement claim.

Python compilation, Compose configuration validation and `git diff --check` passed.
Container runtime tests remain blocked by Docker itself.

## Rollback and limitations

The previous `.env` is saved as `.env.rollback-chroma-20260907`; both are mode 600
and Git-ignored. For rollback, change the final `VECTOR_STORE` assignment to
`chroma` and restart the backend. `CHROMA_MODE=local` and SQLite checkpoints were
preserved, so this rollback does not require PostgreSQL. Preserve matching
Chroma/JSON snapshots during subsequent ingestion; do not delete either store.

A failing test traceback exposed live Groq/evaluation credential values through
the old settings representation. They are not repeated here. Rotate the affected
credentials: future redaction does not revoke earlier exposed values. No
credentials were committed.

Pending: HNSW, structured metadata relaxation, parallel hybrid retrieval,
PostgreSQL-backed context expansion, comprehensive request metrics, repeated
before/after benchmarks and production concurrency/quality gates. Dense search
remains exact squared-L2. No PostgreSQL latency reduction is established.
