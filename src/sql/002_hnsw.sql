CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
ON document_chunks USING hnsw (embedding vector_l2_ops)
WITH (m = __M__, ef_construction = __EF_CONSTRUCTION__);
