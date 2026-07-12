"""Cached local embedding and reranking models."""

from __future__ import annotations

from functools import lru_cache
import math
import os
import sys
from typing import TYPE_CHECKING, Any

# Keep Transformers on the required local PyTorch path. Optional TensorFlow
# installations can otherwise be imported even though this project never uses them.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from src.config import Settings, settings

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder


def _embedding_class() -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def create_embedding_model(
    config: Settings = settings,
    device: str | None = None,
) -> HuggingFaceEmbeddings:
    """Return normalized local Hugging Face embeddings."""
    return _embedding_class()(
        model_name=config.embedding_model,
        model_kwargs={"device": device or config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def create_reranker(
    config: Settings = settings,
    device: str | None = None,
) -> "CrossEncoder":
    """Return the cached local cross-encoder reranker."""
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "The sentence-transformers cross-encoder is unstable on Python 3.13; "
            "use Python 3.11 or 3.12"
        )
    import torch
    from sentence_transformers import CrossEncoder

    selected_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = CrossEncoder(config.reranker_model, device=selected_device)
    finite_scores(
        model.predict(
            [("Opcenter evidence", "Opcenter manual evidence")],
            show_progress_bar=False,
        ),
        1,
    )
    return model


def finite_scores(values: Any, expected: int) -> list[float]:
    """Normalize scalar model outputs and reject missing or non-finite scores."""
    try:
        raw_scores = list(values)
    except TypeError as exc:
        raise ValueError("cross-encoder returned invalid scores") from exc
    scores: list[float] = []
    for score in raw_scores:
        if hasattr(score, "item"):
            try:
                score = score.item()
            except ValueError as exc:
                raise ValueError("cross-encoder returned invalid scores") from exc
        try:
            scores.append(float(score))
        except (TypeError, ValueError) as exc:
            raise ValueError("cross-encoder returned invalid scores") from exc
    if len(scores) != expected or not all(math.isfinite(score) for score in scores):
        raise ValueError("cross-encoder returned invalid scores")
    return scores
