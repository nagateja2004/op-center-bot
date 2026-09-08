import os


os.environ.setdefault("CHECKPOINT_BACKEND", "sqlite")
os.environ["CHROMA_MODE"] = "local"
os.environ["CHAT_MEMORY_PATH"] = "/tmp/opcenter-chatbot-pytest.sqlite"
# Unit/legacy corpus tests must not inherit the live deployment or its credentials.
# PostgreSQL integration tests explicitly opt in with their dedicated DB variables.
os.environ["VECTOR_STORE"] = "chroma"
os.environ['PGVECTOR_SEARCH_MODE'] = 'exact'
os.environ['PARALLEL_HYBRID'] = 'false'
os.environ['REUSE_QUERY_EMBEDDINGS'] = 'false'
os.environ["DATABASE_URL"] = ""
os.environ["GROQ_API_KEY"] = "gsk_test"
os.environ["EVALUATION_API_TOKEN"] = ""
