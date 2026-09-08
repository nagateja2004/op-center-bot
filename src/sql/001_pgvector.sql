-- __DIMENSION__ is replaced with the probed model dimension by PgVectorRepository.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS vector_model (
    id integer PRIMARY KEY CHECK (id = 1),
    model text NOT NULL,
    revision text NOT NULL,
    dimension integer NOT NULL CHECK (dimension > 0)
);
CREATE TABLE IF NOT EXISTS document_chunks (
    collection text NOT NULL,
    id text NOT NULL,
    document_id text NOT NULL,
    chunk_id text NOT NULL,
    content text NOT NULL,
    embedding vector(__DIMENSION__) NOT NULL,
    manual_name text NOT NULL,
    chapter text,
    section text,
    page_number integer,
    parent_section_id text NOT NULL,
    chunk_index integer NOT NULL,
    content_type text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS document_chunks_document ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS document_chunks_parent ON document_chunks (collection, parent_section_id, chunk_index);
CREATE INDEX IF NOT EXISTS document_chunks_metadata ON document_chunks USING gin (metadata jsonb_path_ops);
