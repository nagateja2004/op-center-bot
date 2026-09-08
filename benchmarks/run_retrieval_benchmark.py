"""Repeated warm Chroma vs exact pgvector comparison; no LLM or Redis cache.

Run: python -m benchmarks.run_retrieval_benchmark --rounds 6
Uses the existing simplified retrieval adapter, not the full query planner.
Does not create indexes, migrate data or change the application's selector.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import random
from statistics import mean
import subprocess

from retrieval_benchmark import _percentile, attach_ground_truth, run_benchmark


def latency_summary(values):
    values = sorted(values)
    if not values:
        raise ValueError('No measured samples')
    return {'mean_ms': mean(values), **{f'p{p}_ms': _percentile(values, p / 100) for p in (50, 95, 99)}}


def reduction(old, new):
    return (old - new) / old * 100 if old > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--optimized', action='store_true', help='Compare HNSW, parallel, and shared-embedding increments too')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmarks/results'))
    args = parser.parse_args()
    if args.rounds < 2 or args.k < 1:
        parser.error('Use at least two rounds and positive k')
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['LANGSMITH_TRACING'] = 'false'
    from src.config import settings
    from src.cache import close_cache
    from src.retrieval import configure_chroma_client, create_chroma_client
    from src.vector_store import standalone_client
    close_cache()
    paths = [Path('tests/evaluation_questions.json'), Path('tests/retrieval_ground_truth.json')]
    cases = attach_ground_truth(*(json.loads(p.read_text()) for p in paths))
    configs = {name: replace(settings, vector_store=name, chroma_mode='local', pgvector_search_mode='exact',
                             parallel_hybrid=False, reuse_query_embeddings=False) for name in ('chroma', 'pgvector')}
    if args.optimized:
        configs.update({
            'pgvector_hnsw': replace(configs['pgvector'], pgvector_search_mode='hnsw'),
            'pgvector_parallel': replace(configs['pgvector'], pgvector_search_mode='hnsw', parallel_hybrid=True),
            'pgvector_final': replace(configs['pgvector'], pgvector_search_mode='hnsw', parallel_hybrid=True, reuse_query_embeddings=True),
        })
    report = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'protocol': {'rounds': args.rounds, 'k': args.k, 'seed': 20260907,
                     'cache': 'Redis disabled; warm models and OS/database pages',
                     'order': 'Alternating backend blocks (rotating/reversed for optimized variants); identical seeded question shuffle within each round',
                     'scope': 'Existing simplified dense/BM25/RRF/neighbors/cross-encoder adapter; no LLM, planner, or total-request timing',
                     'concurrency': 1, 'hnsw': args.optimized},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                        'ram_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
                        'embedding_model': settings.embedding_model, 'embedding_revision': settings.embedding_model_revision,
                        'reranker_model': settings.reranker_model, 'reranker_revision': settings.reranker_model_revision,
                        'embedding_device': settings.embedding_device,
                        'pool_min': settings.db_pool_min_size, 'pool_max': settings.db_pool_max_size,
                        'pool_timeout_s': settings.db_pool_timeout,
                        'dense_top_k': settings.vector_top_k, 'bm25_top_k': settings.bm25_top_k,
                        'fused_top_k': settings.fused_top_k, 'hnsw_ef_search': settings.hnsw_ef_search},
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'runs': [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir / f'storage-comparison-{stamp}.json'
    with ExitStack() as stack:
        pg = stack.enter_context(standalone_client(configs['pgvector']))
        assert pg.call(pg.repository.health_check())
        async def database_info():
            async with pg.repository.pool.connection() as c:
                cur = await c.execute("SELECT version() AS postgres, (SELECT extversion FROM pg_extension WHERE extname='vector') AS pgvector")
                return await cur.fetchone()
        report['environment'].update(pg.call(database_info()))
        report['environment']['dimension'] = pg.repository.dimension
        clients = {'pgvector': pg, 'chroma': create_chroma_client(configs['chroma'])}
        body = settings.chroma_collection
        ids = {name: set(client.get_collection(body).get()['ids']) for name, client in clients.items()}
        if ids['chroma'] != ids['pgvector']:
            raise ValueError('Body corpus IDs differ; comparison invalid')
        report['environment']['body_chunks'] = len(ids['pgvector'])
        report['input_sha256']['body_ids'] = hashlib.sha256('\n'.join(sorted(ids['pgvector'])).encode()).hexdigest()
        if args.optimized:
            from src.embeddings import create_embedding_model
            from retrieval_benchmark import retrieval_question
            model = create_embedding_model(configs['pgvector'])
            audit = []
            for case in cases:
                if not case.get('expected_evidence'):
                    continue
                vector = model.embed_query(retrieval_question(case))
                exact = pg.call(pg.repository.search(body, vector, 20, mode='exact'))
                ann = pg.call(pg.repository.search(body, vector, 20, mode='hnsw'))
                audit.append({'case': case['id'], 'exact_ids': [r['id'] for r in exact], 'ann_ids': [r['id'] for r in ann],
                              'overlap_at_20': len({r['id'] for r in exact} & {r['id'] for r in ann}) / len(exact)})
            report['ann_audit'] = audit
            report['ann_overlap_at_20'] = mean(r['overlap_at_20'] for r in audit)
            report['ann_explain'] = pg.call(pg.repository.search(body, vector, 20, mode='hnsw', explain=True))
            if 'document_chunks_embedding_hnsw' not in json.dumps(report['ann_explain']):
                raise RuntimeError('Representative EXPLAIN did not use HNSW; investigate before benchmarking')
            for name in configs:
                if name not in clients:
                    clients[name] = pg
        for n in range(args.rounds):
            shuffled = list(cases)
            random.Random(20260907 + n).shuffle(shuffled)
            order = list(configs)
            if args.optimized:
                offset = n % len(order)
                order = order[offset:] + order[:offset]
            if n % 2:
                order.reverse()
            for name in order:
                pg.repository.search_mode = configs[name].pgvector_search_mode
                configure_chroma_client(clients[name])
                measured = run_benchmark(shuffled, k=args.k, config=configs[name])
                # The legacy adapter labels its storage Chroma; correct the labels explicitly.
                measured['configuration']['semantic_pipeline'] = f'{name} dense retrieval'
                measured['configuration']['hybrid_pipeline'] = f'{name} + BM25 + weighted RRF + neighbors + cross-encoder'
                report['runs'].append({'round': n + 1, 'backend': name, **measured})
                output.write_text(json.dumps(report, indent=2))
                print(f'Round {n + 1}/{args.rounds} {name}: {measured["comparison"]["hybrid"]["p50_latency_seconds"] * 1000:.2f} ms hybrid p50', flush=True)
        report['pool_stats'] = pg.repository.pool.get_stats()
        configure_chroma_client(None)
    summaries = {}
    for name in configs:
        summaries[name] = {}
        for pipeline in ('semantic_only', 'hybrid'):
            rows = [case[pipeline] for run in report['runs'] if run['backend'] == name for case in run['cases'] if case['scored']]
            summaries[name][pipeline] = {**latency_summary([r['latency_seconds'] * 1000 for r in rows]),
                                        'samples': len(rows),
                                        'recall_at_k': mean(r['metrics']['recall_at_k'] for r in rows),
                                        'mrr': mean(r['metrics']['reciprocal_rank'] for r in rows),
                                        'stage_mean_ms': {stage: mean(r.get('stages_ms', {}).get(stage, 0) for r in rows)
                                                          for stage in {s for r in rows for s in r.get('stages_ms', {})}}}
    report['summary'] = summaries
    report['hybrid_latency_reduction_percent'] = {metric: reduction(summaries['chroma']['hybrid'][metric], summaries['pgvector']['hybrid'][metric]) for metric in ('p50_ms', 'p95_ms', 'p99_ms', 'mean_ms')}
    if args.optimized:
        report['optimized_reduction_percent'] = {baseline: {metric: reduction(summaries[baseline]['hybrid'][metric], summaries['pgvector_final']['hybrid'][metric])
                                                           for metric in ('p50_ms', 'p95_ms', 'p99_ms', 'mean_ms')}
                                                for baseline in ('chroma', 'pgvector')}
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'result_file': str(output), 'summary': summaries, 'reduction_percent': report['hybrid_latency_reduction_percent']}, indent=2))


if __name__ == '__main__':
    main()
