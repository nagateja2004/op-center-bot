"""Explicit schema creation and optional backfill from canonical local JSON.

python -m src.pgvector_migrate [--backfill | --manuals]
Backfill reads existing JSON; --manuals runs the unchanged PDF ingestion pipeline.
Neither mode writes Chroma. Run offline with a matching corpus backup.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json

from src.config import settings
from src.embeddings import create_embedding_model
from src.vector_store import make_chunk, standalone_client, vector_literal


@contextmanager
def storage_client(config):
    from src import retrieval
    if retrieval._chroma_client is not None:
        from src.vector_store import PgVectorClient
        if not isinstance(retrieval._chroma_client, PgVectorClient):
            raise RuntimeError("Configured vector client is not pgvector")
        yield retrieval._chroma_client
    else:
        with standalone_client(config) as client:
            yield client


def build_pgvector_indexes(segments, representations, config, client=None):
    from src.ingest import REPRESENTATION_COLLECTION
    if client is None:
        with storage_client(config) as opened:
            return build_pgvector_indexes(segments, representations, config, opened)
    if not client.call(client.repository.health_check()):
        raise RuntimeError("Run python -m src.pgvector_migrate first")
    if {r["segment_id"] for r in segments} & {r["representation_id"] for r in representations}:
        raise ValueError("Body and representation chunk IDs overlap")
    model = create_embedding_model(config)
    chunks = []
    for records, collection, representation in (
        (segments, config.chroma_collection, False),
        (representations, REPRESENTATION_COLLECTION, True),
    ):
        if not records:
            raise ValueError("Refusing to publish an empty retrieval collection")
        ids = [r["representation_id" if representation else "segment_id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs")
        for start in range(0, len(records), 256):
            batch = records[start:start + 256]
            for record in batch:
                page = record["metadata"].get("pdf_page")
                if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                    raise ValueError("missing_page_numbers: every PDF chunk requires a positive pdf_page")
            vectors = model.embed_documents([r["text" if representation else "searchable_text"] for r in batch])
            if len(vectors) != len(batch):
                raise ValueError("Embedding batch size mismatch")
            for vector in vectors:
                vector_literal(vector, client.repository.dimension)
            chunks.extend(make_chunk(r, collection, v, representation) for r, v in zip(batch, vectors))
    # All inference completes before opening the publication transaction.
    # Writes can legitimately exceed a request's timeout during corpus backfill.
    old_timeout = client.timeout
    try:
        client.timeout = max(old_timeout, 3600)
        client.call(client.repository.replace_collections(
            [config.chroma_collection, REPRESENTATION_COLLECTION], chunks))
    finally:
        client.timeout = old_timeout
    return len(segments)


def validate_report(report, dimension):
    errors = {key: report[key] for key in (
        "missing_embeddings", "missing_page_numbers", "missing_document_ids",
        "missing_parent_ids", "duplicate_chunk_ids",
    ) if report[key]}
    if report["embedding_dimensions"] != [dimension]:
        errors["embedding_dimensions"] = report["embedding_dimensions"]
    if not report["total_documents"] or not report["total_chunks"]:
        errors["empty_corpus"] = True
    if errors:
        raise ValueError(f"pgvector ingestion validation failed: {json.dumps(errors)}")


def ingestion_summary(config, client=None):
    from src.ingest import REPRESENTATION_COLLECTION, validate_indexes
    if client is None:
        with storage_client(config) as opened:
            return ingestion_summary(config, opened)
    report = client.call(client.repository.ingestion_report(
        [config.chroma_collection, REPRESENTATION_COLLECTION]))
    print("pgvector ingestion summary:\n" + json.dumps(report, indent=2, sort_keys=True))
    validate_report(report, client.repository.dimension)
    validate_indexes(config, chroma_client=client)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hnsw', action='store_true', help='Create the L2 HNSW index explicitly after schema/import')
    parser.add_argument('--hnsw-m', type=int, default=16)
    parser.add_argument('--hnsw-ef-construction', type=int, default=64)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--backfill", action="store_true", help="Embed current JSON and atomically replace the two PostgreSQL collections")
    modes.add_argument("--manuals", action="store_true", help="Parse configured manuals with the existing pipeline, then ingest into pgvector")
    args = parser.parse_args()
    if not settings.database_url:
        parser.error("DATABASE_URL is required")
    config = replace(settings, vector_store="pgvector", pgvector_search_mode='exact')
    with standalone_client(config) as client:
        client.call(client.repository.migrate())
        print(f"pgvector schema ready: dimension={client.repository.dimension}, model={settings.embedding_model}")
        if args.manuals:
            from src.ingest import ingest_manuals
            from src.retrieval import configure_chroma_client
            # Reuse this job's pool through index existence checks, publication and validation.
            configure_chroma_client(client)
            try:
                manifest = ingest_manuals(config)
                print(f"PDF ingestion complete: {manifest['total_manuals']} manuals, "
                      f"{len(manifest['processed'])} processed, "
                      f"{len(manifest['skipped_unchanged'])} unchanged")
            finally:
                configure_chroma_client(None)
        if args.backfill:
            from src.ingest import _require_index_schema, _indexable_segments, _indexable_representations
            _require_index_schema(settings)
            evidence = json.loads(settings.evidence_units_path.read_text())
            evidence_ids = {r["evidence_id"] for r in evidence}
            segments = _indexable_segments(json.loads(settings.retrieval_segments_path.read_text()), settings)
            representations = _indexable_representations(json.loads(settings.search_representations_path.read_text()), evidence_ids, settings)
            if any(r["evidence_id"] not in evidence_ids for r in segments):
                raise ValueError("Orphan retrieval segment")
            count = build_pgvector_indexes(segments, representations, config, client)
            ingestion_summary(config, client)
            print(f"Backfill verified: {count} body chunks, {len(representations)} representations")
        if args.hnsw:
            client.timeout = 3600
            client.call(client.repository.create_hnsw_index(args.hnsw_m, args.hnsw_ef_construction))
            print('HNSW index ready (vector_l2_ops); exact search remains available')


if __name__ == "__main__":
    main()
