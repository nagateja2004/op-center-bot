# Matched storage benchmark — 2026-09-07

## Quality-checked optimization results

Latest matched six-round run, **44 labeled questions × 6 = 264 samples per variant**:
[raw results, profiles, ANN IDs and EXPLAIN](results/storage-comparison-20260907T090237Z.json).

| Hybrid variant | Mean ms | p50 ms | p95 ms | p99 ms | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chroma | 80.68 | 80.30 | 98.21 | 112.19 | 0.738636 | 0.671212 |
| Exact pgvector | 136.47 | 131.97 | 165.64 | 217.72 | 0.738636 | 0.671212 |
| HNSW sequential | 98.75 | 96.36 | 136.73 | 151.84 | 0.738636 | 0.671212 |
| HNSW parallel | 82.26 | 82.74 | 98.86 | 106.25 | 0.738636 | 0.671212 |
| HNSW parallel + reuse | 87.77 | 85.65 | 108.56 | 129.75 | 0.738636 | 0.671212 |

**Final versus exact pgvector:** p50 reduced **35.10%**, p95 **34.46%**, p99
**40.41%**, mean **35.68%**. **Final versus Chroma:** p50 is **6.67% slower**.
Do not describe this as a speedup over Chroma or as end-to-end LLM latency.

HNSW uses `vector_l2_ops`, `m=16`, `ef_construction=64`, `ef_search=200` and strict
iterative scans. The observed exact-top-20 overlap was **96.82%**, not 100%; final
hybrid Recall@5/MRR matched the baseline on this dataset. The recorded actual
EXPLAIN uses `document_chunks_embedding_hnsw`. Underfilled ANN filters can fall
back to exact retrieval, so this overlap describes the effective retrieval path.

The first [ef_search=40 run](results/storage-comparison-20260907T084800Z.json) had
90.91% top-20 overlap and lower hybrid Recall@5/MRR (0.715909/0.648485). It was
rejected. A probe at 100 did not recover the lost Resource table-definition case;
200 recovered it and passed the complete repeated quality comparison. No build
parameters or candidate limits were changed to manufacture the gain.

Reranker profiling: final mean reranker time **57.52 ms**, about **65.5%** of mean
hybrid latency. The existing model and 20-candidate preparation cap are retained.
Parallel branch means were dense **17.09 ms**, BM25 **22.80 ms**, wall **23.86 ms**:
the wall time is near the slower branch rather than their **39.89 ms** sum.

The simplified benchmark has only one dense branch, so it cannot demonstrate
multi-branch reuse savings. Its reuse-on/off timing difference must not be treated
as a causal regression or improvement. The separate
[full-path reuse check](results/query-embedding-reuse.json) covered three queries
with manual preferences: each dropped from **4 embedding computations to 1**,
with identical ordered retrieval IDs. Measured embedding time changed from roughly
19–21 ms to 5–6 ms. Those three one-pass timings are not a general speedup estimate.

Reproduce the optimized comparison and reuse check with the commands in
[RETRIEVAL_OPTIMIZATIONS.md](../docs/RETRIEVAL_OPTIMIZATIONS.md). Variant order is
rotated/reversed between rounds and questions are identically shuffled within each
round. Existing cache/warmup/hardware limitations below still apply. Tuning and
validation used the same small dataset, so a larger held-out evaluation remains
necessary before claiming generalization or production reliability.

### Defensible resume wording

> Optimized a PostgreSQL/pgvector RAG retriever with HNSW and parallel dense/BM25
> search, reducing median retrieval latency by 35% (132 ms to 86 ms) versus exact
> pgvector in a 44-question, six-round local benchmark, with unchanged Recall@5.

This supports a **local retrieval benchmark claim**, not a production SLA, a
Chroma speedup, or a total-answer latency reduction.

Activation verification: **331 tests passed, zero skipped** (48.17 s, six existing
deprecation warnings); eight affected optimization tests also passed after a
Python 3.11 cleanup compatibility adjustment. HNSW/parallel/reuse are enabled
locally with ef_search=200. The Factory API smoke returned its page citation.
The table-definition API prompt returned insufficient evidence in both exact and
optimized modes, an existing full-answer limitation outside the retrieval benchmark.

## Initial exact-storage baseline (historical)

The initial exact-only migration did **not** support a latency-improvement resume claim.
The simplified hybrid adapter using exact pgvector was **67.19% slower at p50**
than the same adapter using existing Chroma. Recall@5 and MRR were equal.
This is evidence about the current storage implementations, not HNSW or parallel
retrieval, which remain unimplemented.

Raw samples, input fingerprints, per-round results and environment:
[storage-comparison-20260907T075156Z.json](results/storage-comparison-20260907T075156Z.json).

## Matched results

All latency values are milliseconds, retrieval only, excluding model load and LLM.

| Metric | Chroma hybrid | Exact pgvector hybrid |
| --- | ---: | ---: |
| Samples | 264 | 264 |
| Mean | 84.58 | 135.50 |
| p50 | 80.58 | 134.73 |
| p95 | 98.86 | 161.75 |
| p99 | 141.99 | 179.23 |
| Recall@5 | 0.738636 | 0.738636 |
| MRR | 0.671212 | 0.671212 |

Reduction `(old - new) / old * 100`: p50 **-67.19%**, p95 **-63.62%**,
p99 **-26.22%**, mean **-60.20%**. Negative reduction means a slowdown.

| Metric | Chroma dense-only | Exact pgvector dense-only |
| --- | ---: | ---: |
| Mean | 6.50 | 46.43 |
| p50 | 5.74 | 45.73 |
| p95 | 8.83 | 56.17 |
| p99 | 14.69 | 65.29 |
| Recall@5 | 0.488636 | 0.511364 |
| MRR | 0.393561 | 0.404924 |

Per-round hybrid p50:

| Round | Chroma | Exact pgvector |
| --- | ---: | ---: |
| 1 | 85.98 | 133.35 |
| 2 | 78.83 | 138.70 |
| 3 | 80.46 | 134.57 |
| 4 | 80.71 | 134.85 |
| 5 | 79.07 | 135.10 |
| 6 | 81.14 | 132.00 |

The slowdown appeared in every paired round; it is not just one outlier.

## Protocol and environment

- Reused `retrieval_benchmark.py`'s existing simplified adapter: dense + BM25 +
  weighted RRF + neighbor expansion + cross-encoder + EvidenceUnit resolution.
  This is **not** the full multi-aspect production query router.
- Six rounds, 44 labeled questions per backend/pipeline/round, from the existing
  50-question dataset. Six cases without relevant-evidence labels are excluded
  from scored latency/quality samples. These are 44 distinct questions repeated,
  not 264 distinct questions. Follow-ups use existing standalone-query construction.
- Backend order alternates per round. Each paired round uses the same seeded
  question shuffle. Each adapter gets its existing warmup before measurement.
- Redis application caches disabled; OS/database pages and models warm. Query
  embedding and reranking remain inside measured retrieval. Concurrency is one.
- Same model revisions, corpus body IDs, candidate budgets and final k=5. Body ID
  parity is asserted before measuring; input/ID hashes are saved with raw output.
- Apple M3, 8 GiB RAM, macOS 15.5 arm64, Python 3.12.13; embedding on CPU and
  cross-encoder using the existing automatic device selection (MPS on this Mac).
- PostgreSQL 17.11, pgvector 0.8.6; 15,721 body chunks, dimension 384.
  Database also contains 21,121 search representations, unused by this simplified
  adapter. Dense/BM25/fused candidate defaults: 12/12/18.
- Pool min/max 1/10, timeout 30 seconds. Snapshot: one connection created,
  555 acquisitions, zero waiting at completion. This is not a concurrent load test.
- Exact pgvector search, no HNSW. Existing Chroma index configuration retained.
  This compares deployed implementations, not equivalent exact-search algorithms.

No total LLM latency, generated citation accuracy, concurrent-request performance,
or causal per-stage breakdown was measured. p99 from 264 correlated warm samples
is exploratory, not a production tail-latency guarantee. Other applications were
not stopped; this was not an isolated hardware performance laboratory.

## Reproduce

From the repository root, with the local pgvector corpus already imported:

```bash
.venv/bin/python -m benchmarks.run_retrieval_benchmark --rounds 6 --k 5
```

The harness runs read-only retrieval, changes neither `.env` nor indexes, uses
no LLM calls, and saves timestamped JSON under `benchmarks/results/`. It currently
uses macOS `sysctl` for hardware metadata. Avoid simultaneous chat/load-test traffic
while benchmarking. Partial runs lack the final `summary` and are not final results.

Verification after adding the harness: **316 tests passed, zero skipped, six
existing deprecation warnings (29.44 seconds)** with both explicit PostgreSQL test
URLs enabled. This includes percentile/reduction regression tests. Python
compilation and `git diff --check` passed; backend readiness remains healthy.

## Resume and next implementation

Valid now: describe the storage migration, indexed corpus, async pooling and
tested retrieval. If mentioning quality, qualify Recall@5 **0.739** and MRR **0.671**
as results on this **44-question local labeled benchmark**. Do not claim improved
latency, production performance, HNSW, or parallel retrieval from this run.

Next optimization experiment: implement correctly indexed HNSW with a matching
distance operator, verify EXPLAIN and exact-top-k recall overlap, then compare
sequential versus parallel hybrid execution with the same candidate budgets.
Keep these exact-search and Chroma samples as the baseline. Only enable and claim
an optimization after its quality and latency checks pass.
