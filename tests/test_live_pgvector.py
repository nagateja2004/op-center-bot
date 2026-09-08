"""Opt-in, read-only query validation against an already imported local corpus.

PGVECTOR_CORPUS_DATABASE_URL is separate from the disposable storage-test URL.
No migration, insertion, deletion, or remote LLM request is performed here.
"""
from dataclasses import replace
import os

import pytest

from src.config import Settings
from src.retrieval import configure_chroma_client, vector_search
from src.vector_store import standalone_client
from retrieval_benchmark import hybrid_retrieve


@pytest.fixture(scope="module", params=['exact', 'hnsw'])
def corpus(request):
    url = os.getenv("PGVECTOR_CORPUS_DATABASE_URL")
    if not url:
        pytest.skip("Set PGVECTOR_CORPUS_DATABASE_URL to a populated pgvector corpus")
    config = replace(Settings(), vector_store="pgvector", database_url=url,
                     pgvector_search_mode=request.param, parallel_hybrid=request.param == 'hnsw',
                     reuse_query_embeddings=request.param == 'hnsw', hnsw_ef_search=200)
    with standalone_client(config) as client:
        assert client.call(client.repository.health_check())
        configure_chroma_client(client)
        try:
            yield config
        finally:
            configure_chroma_client(None)


@pytest.mark.parametrize("query,expected_evidence", [
    ("What is a Factory?", "e_4c479bc97ce91830dee94824"),
    ("What is a Resource?", "e_1491bf94daca483e04483560"),
    ("What is a configurable data object?", "e_811c57bcb340ab5855811529"),
    ('List the Resource page field definitions.', 'e_7f3d0519ae557b106f9fcf99'),
])
def test_live_pgvector_hybrid_retains_labeled_evidence_and_citations(corpus, query, expected_evidence):
    dense = vector_search(query, 20, config=corpus)
    assert len(dense) == 20
    distances = [doc["retrieval_scores"]["vector_distance"] for doc in dense]
    assert distances == sorted(distances)
    hybrid = hybrid_retrieve(query, k=5, config=corpus)
    assert 1 <= len(hybrid) <= 5
    assert expected_evidence in {doc["metadata"]["evidence_id"] for doc in hybrid}
    for doc in hybrid:
        assert doc["metadata"]["source_file"]
        assert doc["metadata"]["manual"]
        assert doc["metadata"]["pdf_page"] > 0
