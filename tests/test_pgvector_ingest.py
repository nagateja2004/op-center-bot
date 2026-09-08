import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src import pgvector_migrate as ingestion
from src.vector_store import PgVectorRepository


def records(count=1):
    metadata = {"manual": "Guide", "source_file": "guide.pdf", "chapter": "C",
                "section": "S", "heading_path": ["C", "S"], "pdf_page": 3,
                "printed_page": "2-1", "table_rows": [["a", "b"]]}
    return ([{"segment_id": f"s{i}", "evidence_id": "e1", "segment_index": i,
              "searchable_text": f"text {i}", "metadata": metadata} for i in range(count)],
            [{"representation_id": "r1", "evidence_id": "e1", "text": "heading", "metadata": metadata}])


class MemoryClient:
    timeout = 30
    def __init__(self):
        self.rows = {}
        self.publications = 0
        self.repository = SimpleNamespace(dimension=3, health_check=lambda: True,
                                          replace_collections=self.publish)
    def call(self, result):
        return result
    def publish(self, collections, chunks):
        self.publications += 1
        self.rows = {(c.collection, c.id): c for c in chunks}


def test_repeat_ingestion_is_deterministic_and_batched(monkeypatch):
    batches, loads = [], []
    def embed(texts):
        batches.append(len(texts))
        return [[1, 0, 0] for _ in texts]
    monkeypatch.setattr(ingestion, "create_embedding_model", lambda _: loads.append(1) or SimpleNamespace(embed_documents=embed))
    client = MemoryClient()
    config = SimpleNamespace(chroma_collection="body")
    body, reps = records(513)
    ingestion.build_pgvector_indexes(body, reps, config, client)
    first = dict(client.rows)
    ingestion.build_pgvector_indexes(body, reps, config, client)
    assert client.rows == first
    assert len(client.rows) == 514
    assert batches == [256, 256, 1, 1] * 2
    assert len(loads) == 2  # One cached factory call per job, never per chunk/batch.
    assert client.publications == 2
    row = client.rows[("body", "s512")]
    assert row.chunk_index == 512
    assert row.metadata["printed_page"] == "2-1"
    assert row.metadata["table_rows"] == [["a", "b"]]
    assert row.document_id == "guide.pdf" and row.parent_section_id


@pytest.mark.parametrize("problem", ["dimension", "page", "duplicate", "document"])
def test_invalid_input_never_publishes(monkeypatch, problem):
    body, reps = records()
    if problem == "page":
        body[0]["metadata"]["pdf_page"] = None
    if problem == "document":
        body[0]["metadata"].update(source_file="", manual="")
    if problem == "duplicate":
        body.append(body[0])
    vector = [1, 0] if problem == "dimension" else [1, 0, 0]
    monkeypatch.setattr(ingestion, "create_embedding_model", lambda _: SimpleNamespace(embed_documents=lambda texts: [vector for _ in texts]))
    client = MemoryClient()
    with pytest.raises(ValueError):
        ingestion.build_pgvector_indexes(body, reps, SimpleNamespace(chroma_collection="body"), client)
    assert client.publications == 0


def test_database_writes_use_one_transaction_and_batched_executemany():
    from src.vector_store import make_chunk
    body, _ = records(513)
    chunks = [make_chunk(r, "body", [1, 0, 0]) for r in body]
    batches, transactions = [], []
    class Connection:
        @asynccontextmanager
        async def transaction(self):
            transactions.append(1)
            yield
        @asynccontextmanager
        async def cursor(self):
            yield self
        async def execute(self, *args):
            pass
        async def executemany(self, statement, rows):
            batches.append(len(rows))
    @asynccontextmanager
    async def connection():
        yield Connection()
    repo = PgVectorRepository(SimpleNamespace(connection=connection), 3, "fixture", "revision")
    asyncio.run(repo.replace_collections(["body"], chunks))
    assert transactions == [1]
    assert batches == [256, 256, 1]


def test_summary_rejects_invalid_database_report():
    report = {"total_documents": 1, "total_chunks": 2, "missing_embeddings": 0,
              "missing_page_numbers": 1, "missing_document_ids": 0,
              "missing_parent_ids": 0, "duplicate_chunk_ids": 0,
              "embedding_dimensions": [3]}
    with pytest.raises(ValueError, match="missing_page_numbers"):
        ingestion.validate_report(report, 3)


def test_summary_prints_persisted_validation_counts(monkeypatch, capsys):
    from src import ingest
    report = {"total_documents": 1, "total_chunks": 2, "missing_embeddings": 0,
              "missing_page_numbers": 0, "missing_document_ids": 0,
              "missing_parent_ids": 0, "duplicate_chunk_ids": 0,
              "embedding_dimensions": [3]}
    monkeypatch.setattr(ingest, "validate_indexes", lambda *a, **kw: 1)
    client = MemoryClient()
    client.repository.ingestion_report = lambda collections: report
    assert ingestion.ingestion_summary(SimpleNamespace(chroma_collection="body"), client) == report
    assert '"missing_embeddings": 0' in capsys.readouterr().out


def test_manuals_command_selects_pgvector_and_reuses_job_client(monkeypatch, capsys):
    from contextlib import contextmanager
    from dataclasses import replace
    from src import ingest, retrieval
    client = MemoryClient()
    client.repository.migrate = lambda: None
    @contextmanager
    def opened(config):
        assert config.vector_store == "pgvector"
        yield client
    def run(config):
        assert config.vector_store == "pgvector"
        assert retrieval._chroma_client is client
        return {"total_manuals": 1, "processed": ["guide.pdf"], "skipped_unchanged": []}
    monkeypatch.setattr(ingestion, "settings", replace(ingestion.settings, database_url="postgresql://example/test", vector_store="chroma"))
    monkeypatch.setattr(ingestion, "standalone_client", opened)
    monkeypatch.setattr(ingest, "ingest_manuals", run)
    monkeypatch.setattr("sys.argv", ["pgvector_migrate", "--manuals"])
    ingestion.main()
    assert "PDF ingestion complete: 1 manuals" in capsys.readouterr().out
    assert retrieval._chroma_client is None
    assert ingestion.settings.vector_store == "chroma"
