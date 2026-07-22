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
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    checkpoint_backend: str = field(
        default_factory=lambda: os.getenv("CHECKPOINT_BACKEND", "postgres").strip().casefold()
    )
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "").strip())
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
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
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu")
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "").strip()
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_safety_limit: int = field(
        default_factory=lambda: _env_int("EMBEDDING_SAFETY_LIMIT", 512)
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "").strip()
        or "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
    def bm25_path(self) -> Path:
        return _env_path("BM25_INDEX_PATH", self.indexes_dir / "bm25.pkl")

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
        if self.checkpoint_backend == "postgres" and not self.database_url:
            raise EnvironmentError("DATABASE_URL is required for PostgreSQL checkpoints.")
        if self.database_url and not self.database_url.startswith(("postgresql://", "postgres://")):
            raise EnvironmentError("DATABASE_URL must use a PostgreSQL connection URL.")
        if not self.redis_url.startswith(("redis://", "rediss://")):
            raise EnvironmentError("REDIS_URL must use a Redis connection URL.")
        if self.chroma_mode not in {"server", "local"}:
            raise EnvironmentError("CHROMA_MODE must be 'server' or 'local'.")
        if self.chroma_mode == "server" and not self.chroma_host:
            raise EnvironmentError("CHROMA_HOST is required when CHROMA_MODE=server.")
        if not self.chroma_collection:
            raise EnvironmentError("CHROMA_COLLECTION is required.")
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
