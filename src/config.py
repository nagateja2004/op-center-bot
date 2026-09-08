"""Application configuration loaded from environment variables."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Self

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().casefold() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default))).expanduser()
    return value if value.is_absolute() else ROOT_DIR / value


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""), repr=False)
    deployment_environment: str = field(
        default_factory=lambda: os.getenv("DEPLOYMENT_ENVIRONMENT", "local").strip() or "local"
    )
    evaluation_api_token: str = field(
        default_factory=lambda: os.getenv("EVALUATION_API_TOKEN", "").strip(), repr=False
    )
    checkpoint_backend: str = field(
        default_factory=lambda: os.getenv("CHECKPOINT_BACKEND", "postgres").strip().casefold()
    )
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "").strip(), repr=False)
    vector_store: str = field(default_factory=lambda: os.getenv("VECTOR_STORE", "chroma").strip().lower())
    pgvector_search_mode: str = field(default_factory=lambda: os.getenv("PGVECTOR_SEARCH_MODE", "exact").strip().lower())
    hnsw_ef_search: int = field(default_factory=lambda: _env_int("HNSW_EF_SEARCH", 40))
    parallel_hybrid: bool = field(default_factory=lambda: _env_bool("PARALLEL_HYBRID"))
    reuse_query_embeddings: bool = field(default_factory=lambda: _env_bool("REUSE_QUERY_EMBEDDINGS"))
    dense_timeout: float = field(default_factory=lambda: float(os.getenv("DENSE_TIMEOUT", "10")))
    bm25_timeout: float = field(default_factory=lambda: float(os.getenv("BM25_TIMEOUT", "10")))
    db_pool_min_size: int = field(default_factory=lambda: _env_int("DB_POOL_MIN_SIZE", 1))
    db_pool_max_size: int = field(default_factory=lambda: _env_int("DB_POOL_MAX_SIZE", 10))
    db_pool_timeout: int = field(default_factory=lambda: _env_int("DB_POOL_TIMEOUT", 30))
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0").strip(), repr=False
    )
    groq_model_max_concurrency: int = field(
        default_factory=lambda: _env_int("GROQ_MODEL_MAX_CONCURRENCY", 4)
    )
    groq_model_requests_per_minute: int = field(
        default_factory=lambda: _env_int("GROQ_MODEL_REQUESTS_PER_MINUTE", 30)
    )
    groq_model_tokens_per_minute: int = field(
        default_factory=lambda: _env_int("GROQ_MODEL_TOKENS_PER_MINUTE", 60_000)
    )
    groq_max_queue_depth: int = field(
        default_factory=lambda: _env_int("GROQ_MAX_QUEUE_DEPTH", 20)
    )
    groq_max_queue_wait_seconds: int = field(
        default_factory=lambda: _env_int("GROQ_MAX_QUEUE_WAIT_SECONDS", 30)
    )
    groq_request_status_ttl_seconds: int = field(
        default_factory=lambda: _env_int("GROQ_REQUEST_STATUS_TTL_SECONDS", 900)
    )
    groq_request_timeout: float = field(
        default_factory=lambda: float(os.getenv("GROQ_REQUEST_TIMEOUT", "90"))
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(value.strip() for value in os.getenv("CORS_ORIGINS", "").split(",") if value.strip())
    )
    max_request_bytes: int = field(default_factory=lambda: _env_int("MAX_REQUEST_BYTES", 16_384))
    chat_request_ttl_seconds: int = field(
        default_factory=lambda: _env_int("CHAT_REQUEST_TTL_SECONDS", 300)
    )
    thread_ownership_ttl_seconds: int = field(
        default_factory=lambda: _env_int("THREAD_OWNERSHIP_TTL_SECONDS", 2_592_000)
    )
    manuals_dir: Path = field(default_factory=lambda: _env_path("MANUALS_DIRECTORY", ROOT_DIR / "manuals"))
    indexes_dir: Path = field(default_factory=lambda: _env_path("INDEXES_DIRECTORY", ROOT_DIR / "indexes"))
    chroma_dir: Path = field(default_factory=lambda: _env_path("CHROMA_DIRECTORY", ROOT_DIR / "indexes" / "chroma"))
    chroma_mode: str = field(default_factory=lambda: os.getenv("CHROMA_MODE", "server").strip().casefold())
    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "").strip())
    chroma_port: int = field(default_factory=lambda: _env_int("CHROMA_PORT", 8000))
    chroma_ssl: bool = field(default_factory=lambda: _env_bool("CHROMA_SSL"))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "opcenter_manuals").strip())
    sqlite_path: Path = field(default_factory=lambda: _env_path("CHAT_MEMORY_PATH", ROOT_DIR / "data" / "chat_memory.sqlite"))
    document_parser: str = field(default_factory=lambda: os.getenv("DOCUMENT_PARSER", "pymupdf").strip())
    ocr_enabled: bool = field(default_factory=lambda: _env_bool("OCR_ENABLED", True))
    ocr_language: str = field(
        default_factory=lambda: os.getenv("OCR_LANGUAGE", "eng").strip() or "eng"
    )
    ocr_dpi: int = field(default_factory=lambda: _env_int("OCR_DPI", 300))
    ocr_min_native_chars: int = field(
        default_factory=lambda: _env_int("OCR_MIN_NATIVE_CHARS", 40)
    )
    tessdata_prefix: str = field(
        default_factory=lambda: os.getenv("TESSDATA_PREFIX", "").strip()
    )
    paddle_ocr_enabled: bool = field(
        default_factory=lambda: _env_bool("PADDLE_OCR_ENABLED", True)
    )
    paddle_ocr_device: str = field(
        default_factory=lambda: os.getenv("PADDLE_OCR_DEVICE", "cpu").strip() or "cpu"
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu")
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "").strip()
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_model_revision: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_REVISION", "").strip()
        or "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    embedding_safety_limit: int = field(
        default_factory=lambda: _env_int("EMBEDDING_SAFETY_LIMIT", 512)
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "").strip()
        or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_model_revision: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL_REVISION", "").strip()
        or "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    )
    inference_max_concurrency: int = field(
        default_factory=lambda: _env_int("INFERENCE_MAX_CONCURRENCY", 4)
    )
    inference_max_queue_depth: int = field(
        default_factory=lambda: _env_int("INFERENCE_MAX_QUEUE_DEPTH", 32)
    )
    vector_top_k: int = 12
    bm25_top_k: int = 12
    fused_top_k: int = 18
    rerank_top_k: int = 8
    max_search_queries: int = 4
    max_retries: int = 1
    planner_input_token_budget: int = field(
        default_factory=lambda: _env_int("GROQ_PLANNER_INPUT_TOKEN_BUDGET", 600)
    )
    query_broadening_input_token_budget: int = field(
        default_factory=lambda: _env_int("GROQ_QUERY_BROADENING_INPUT_TOKEN_BUDGET", 400)
    )
    grader_input_token_budget: int = field(
        default_factory=lambda: _env_int("GROQ_GRADER_INPUT_TOKEN_BUDGET", 650)
    )
    answer_input_token_budget: int = field(
        default_factory=lambda: _env_int("GROQ_ANSWER_INPUT_TOKEN_BUDGET", 5_000)
    )
    verifier_input_token_budget: int = field(
        default_factory=lambda: _env_int("GROQ_VERIFIER_INPUT_TOKEN_BUDGET", 3_000)
    )
    diagram_input_token_budget: int = field(
        default_factory=lambda: _env_int("GROQ_DIAGRAM_INPUT_TOKEN_BUDGET", 1_600)
    )

    @property
    def evidence_units_path(self) -> Path:
        return _env_path("EVIDENCE_UNITS_PATH", self.indexes_dir / "evidence_units.json")

    @property
    def retrieval_segments_path(self) -> Path:
        return _env_path("RETRIEVAL_SEGMENTS_PATH", self.indexes_dir / "retrieval_segments.json")

    @property
    def search_representations_path(self) -> Path:
        return _env_path(
            "SEARCH_REPRESENTATIONS_PATH",
            self.indexes_dir / "search_representations.json",
        )

    @property
    def heading_index_path(self) -> Path:
        return self.indexes_dir / "heading_index.json"

    @property
    def concept_index_path(self) -> Path:
        return self.indexes_dir / "concept_index.json"

    @property
    def ingestion_audit_path(self) -> Path:
        return self.indexes_dir / "ingestion_audit.json"

    @property
    def manual_figures_path(self) -> Path:
        return self.indexes_dir / "manual_figures.json"

    @property
    def manual_figures_dir(self) -> Path:
        return self.indexes_dir / "manual_figures"

    @property
    def alias_config_path(self) -> Path:
        return ROOT_DIR / "config" / "opcenter_aliases.json"

    def validate(self) -> Self:
        """Raise a clear error when required environment variables are absent."""
        if self.document_parser.casefold() != "pymupdf":
            raise ValueError(
                f"Unsupported DOCUMENT_PARSER {self.document_parser!r}; "
                "only 'pymupdf' is supported."
            )
        required = {
            "GROQ_API_KEY": self.groq_api_key,
            "EMBEDDING_MODEL": self.embedding_model,
            "RERANKER_MODEL": self.reranker_model,
        }
        missing = [
            name
            for name, value in required.items()
            if not value.strip()
        ]
        if missing:
            names = ", ".join(missing)
            raise EnvironmentError(
                f"Missing required environment variable(s): {names}. "
                "Create .env and provide the missing value(s)."
            )
        if not self.groq_api_key.strip().startswith("gsk_"):
            raise EnvironmentError(
                "GROQ_API_KEY is not a Groq API key. Copy a key beginning with "
                "'gsk_' from the GroqCloud API Keys page into .env."
            )
        if self.checkpoint_backend not in {"postgres", "sqlite"}:
            raise EnvironmentError("CHECKPOINT_BACKEND must be 'postgres' or 'sqlite'.")
        if self.vector_store not in {"chroma", "pgvector"}:
            raise EnvironmentError("VECTOR_STORE must be chroma or pgvector")
        if self.pgvector_search_mode not in {"exact", "hnsw"}:
            raise ValueError("PGVECTOR_SEARCH_MODE must be exact or hnsw")
        if not 1 <= self.hnsw_ef_search <= 1000 or not 0 < self.dense_timeout < self.db_pool_timeout or not 0 < self.bm25_timeout < self.db_pool_timeout:
            raise ValueError("Invalid HNSW ef_search or branch timeouts (must be below DB_POOL_TIMEOUT)")
        if not 0 <= self.db_pool_min_size <= self.db_pool_max_size or self.db_pool_max_size < 1 or self.db_pool_timeout <= 0:
            raise ValueError("Invalid PostgreSQL pool sizes or timeout")
        if (self.checkpoint_backend == "postgres" or self.vector_store == "pgvector") and not self.database_url:
            raise EnvironmentError("DATABASE_URL is required for PostgreSQL checkpoints.")
        if self.database_url and not self.database_url.startswith(("postgresql://", "postgres://")):
            raise EnvironmentError("DATABASE_URL must use a PostgreSQL connection URL.")
        if not self.redis_url.startswith(("redis://", "rediss://")):
            raise EnvironmentError("REDIS_URL must use a Redis connection URL.")
        if self.chroma_mode not in {"server", "local"}:
            raise EnvironmentError("CHROMA_MODE must be 'server' or 'local'.")
        if self.vector_store == "chroma" and self.chroma_mode == "server" and not self.chroma_host:
            raise EnvironmentError("CHROMA_HOST is required when CHROMA_MODE=server.")
        if not self.chroma_collection:
            raise EnvironmentError("CHROMA_COLLECTION is required.")
        if self.ocr_dpi <= 0 or self.ocr_min_native_chars < 0:
            raise ValueError("OCR_DPI must be positive and OCR_MIN_NATIVE_CHARS cannot be negative.")
        if any(
            len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision.casefold())
            for revision in (self.embedding_model_revision, self.reranker_model_revision)
        ):
            raise EnvironmentError("Hugging Face model revisions must be full commit SHAs.")
        if min(
            self.groq_model_max_concurrency,
            self.groq_model_requests_per_minute,
            self.groq_model_tokens_per_minute,
            self.groq_max_queue_depth,
            self.groq_max_queue_wait_seconds,
            self.groq_request_status_ttl_seconds,
            self.groq_request_timeout,
        ) <= 0:
            raise ValueError("Groq Redis limiter settings must be positive.")
        if min(
            self.inference_max_concurrency,
            self.inference_max_queue_depth,
            self.chat_request_ttl_seconds,
            self.thread_ownership_ttl_seconds,
        ) <= 0:
            raise ValueError("Inference and chat queue settings must be positive.")
        return self


settings = Settings()
