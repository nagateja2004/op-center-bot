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


def _env_path(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default))).expanduser()
    return value if value.is_absolute() else ROOT_DIR / value


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    manuals_dir: Path = field(default_factory=lambda: _env_path("MANUALS_DIRECTORY", ROOT_DIR / "manuals"))
    indexes_dir: Path = field(default_factory=lambda: _env_path("INDEXES_DIRECTORY", ROOT_DIR / "indexes"))
    chroma_dir: Path = field(default_factory=lambda: _env_path("CHROMA_DIRECTORY", ROOT_DIR / "indexes" / "chroma"))
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
        default_factory=lambda: _env_int("GROQ_DIAGRAM_INPUT_TOKEN_BUDGET", 400)
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
                "Copy .env.example to .env and provide the missing value(s)."
            )
        return self


settings = Settings()
