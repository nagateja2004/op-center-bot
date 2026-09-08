"""Read-only full retrieval-path embedding-reuse check on three labeled queries."""
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter


def main():
    from src.config import settings
    from src.cache import close_cache
    from src.retrieval import configure_chroma_client, retrieve_multiple_queries
    from src.retrieval_metrics import profile_retrieval
    from src.vector_store import standalone_client
    close_cache()
    rows = []
    queries = [('What is a Factory?', 'Modeling'), ('What is a Resource?', 'Modeling'),
               ('What is a configurable data object?', 'Designer')]
    baseline = {}
    for reuse in (False, True):
        config = replace(settings, vector_store='pgvector', pgvector_search_mode='hnsw', parallel_hybrid=True, reuse_query_embeddings=reuse)
        with standalone_client(config) as client:
            configure_chroma_client(client)
            try:
                for query, manual in queries:
                    retrieve_multiple_queries(query, preferred_manuals=[manual], config=config)
                for query, manual in queries:
                    with profile_retrieval() as metrics:
                        start = perf_counter()
                        results = retrieve_multiple_queries(query, preferred_manuals=[manual], config=config)
                        elapsed = (perf_counter() - start) * 1000
                    ids = [r['chunk_id'] for r in results]
                    if not reuse:
                        baseline[query] = ids
                    else:
                        assert ids == baseline[query], 'Reuse changed retrieval ranking'
                        assert len(metrics.get('embedding_ms', [])) == 1, 'Repeated embedding'
                    rows.append({'query': query, 'reuse': reuse, 'embedding_calls': len(metrics.get('embedding_ms', [])),
                                 'embedding_ms': sum(metrics.get('embedding_ms', [])), 'retrieval_ms': elapsed, 'ids': ids})
            finally:
                configure_chroma_client(None)
    output = Path('benchmarks/results/query-embedding-reuse.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != 'ids'} for r in rows], indent=2))


if __name__ == '__main__':
    main()
