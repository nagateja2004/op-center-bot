"""Role-based Groq clients with one explicit alternate-model attempt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
import re
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from groq import APIConnectionError, APIError, APIStatusError, APITimeoutError
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from src.config import Settings, settings

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_groq import ChatGroq


def _chat_groq_class():
    from langchain_groq import ChatGroq

    return ChatGroq


GroqRole = Literal[
    "planner", "query_broadening", "grader", "answer", "verifier", "diagram"
]
GroqErrorKind = Literal[
    "rate_limit",
    "function_call",
    "timeout",
    "invalid_structured",
    "unavailable_model",
    "bad_request",
    "connection",
    "provider",
]
ALLOWED_GROQ_ROLES = frozenset(
    {"planner", "query_broadening", "grader", "answer", "verifier", "diagram"}
)
ResultT = TypeVar("ResultT")
ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)
THINK_END_RE = re.compile(r"</think>\s*", re.I)


@dataclass(frozen=True, slots=True)
class RoleConfig:
    primary_model: str
    fallback_model: str
    temperature: float
    max_output_tokens: int
    timeout: float
    allow_structured_output: bool


ROLE_DEFAULTS: dict[GroqRole, RoleConfig] = {
    "planner": RoleConfig("openai/gpt-oss-20b", "meta-llama/llama-4-scout-17b-16e-instruct", 0.0, 1024, 30, True),
    "query_broadening": RoleConfig("openai/gpt-oss-20b", "", 0.1, 512, 20, False),
    "grader": RoleConfig("openai/gpt-oss-20b", "meta-llama/llama-4-scout-17b-16e-instruct", 0.0, 1024, 40, True),
    "answer": RoleConfig("openai/gpt-oss-120b", "qwen/qwen3.6-27b", 0.1, 4096, 90, False),
    "verifier": RoleConfig("qwen/qwen3.6-27b", "openai/gpt-oss-20b", 0.0, 4096, 60, False),
    "diagram": RoleConfig("openai/gpt-oss-20b", "", 0.0, 2048, 30, False),
}

PRIMARY_MODEL_ALIASES: dict[GroqRole, str] = {
    "planner": "GROQ_PLANNER_MODEL",
    "query_broadening": "GROQ_QUERY_MODEL",
    "grader": "GROQ_GRADER_MODEL",
    "answer": "GROQ_ANSWER_MODEL",
    "verifier": "GROQ_VERIFY_MODEL",
    "diagram": "GROQ_DIAGRAM_MODEL",
}
FALLBACK_MODEL_ALIASES: dict[GroqRole, str] = {
    "verifier": "GROQ_VERIFY_FALLBACK_MODEL",
}


class GroqRequestError(RuntimeError):
    """Sanitized provider failure safe to handle outside the LLM boundary."""

    def __init__(
        self,
        kind: GroqErrorKind,
        role: GroqRole,
        status_code: int | None = None,
    ) -> None:
        self.kind = kind
        self.role = role
        self.status_code = status_code
        super().__init__(f"LLM request unavailable for role={role} kind={kind}")


def role_config(role: GroqRole) -> RoleConfig:
    """Load one role's model and request policy from environment variables."""
    if role not in ALLOWED_GROQ_ROLES:
        raise ValueError(f"Groq is not allowed for role: {role}")
    default = ROLE_DEFAULTS[role]
    prefix = f"GROQ_{role.upper()}"
    primary = os.getenv(
        f"{prefix}_PRIMARY_MODEL",
        os.getenv(PRIMARY_MODEL_ALIASES[role], default.primary_model),
    ).strip()
    if not primary:
        raise EnvironmentError(f"Missing required environment variable: {prefix}_PRIMARY_MODEL")
    return RoleConfig(
        primary_model=primary,
        fallback_model=os.getenv(
            f"{prefix}_FALLBACK_MODEL",
            os.getenv(FALLBACK_MODEL_ALIASES.get(role, ""), default.fallback_model),
        ).strip(),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", default.temperature)),
        max_output_tokens=int(
            os.getenv(f"{prefix}_MAX_OUTPUT_TOKENS", default.max_output_tokens)
        ),
        timeout=float(os.getenv(f"{prefix}_TIMEOUT", default.timeout)),
        allow_structured_output=_env_bool(
            f"{prefix}_ALLOW_STRUCTURED_OUTPUT", default.allow_structured_output
        ),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@lru_cache(maxsize=16)
def create_llm(
    role: GroqRole,
    model: str,
    config: Settings = settings,
) -> ChatGroq:
    """Return a cached client configured for one role and model."""
    config.validate()
    policy = role_config(role)
    return _chat_groq_class()(
        api_key=config.groq_api_key,
        model=model,
        temperature=policy.temperature,
        max_tokens=policy.max_output_tokens,
        timeout=policy.timeout,
        max_retries=0,
    )


def call_llm(
    prompt: LanguageModelInput, *, task: GroqRole, evidence_count: int = 0
) -> AIMessage:
    """Run plain-text generation with at most one distinct fallback model."""
    policy = role_config(task)
    models = _distinct_models(policy)
    for attempt, model in enumerate(models):
        try:
            message = cast(
                AIMessage,
                _call_groq(task, lambda: create_llm(task, model).invoke(prompt)),
            )
            if THINK_END_RE.search(str(message.content)):
                message = message.model_copy(
                    update={"content": THINK_END_RE.split(str(message.content))[-1].strip()}
                )
            return message
        except GroqRequestError as exc:
            recovered = attempt == 0 and len(models) == 2
            _log_failure(
                task, model, None, prompt, evidence_count,
                "primary" if attempt == 0 else "fallback", error=exc,
                recovered=recovered,
            )
            if not recovered:
                raise
    raise GroqRequestError("provider", task)


def call_structured(
    prompt: LanguageModelInput,
    schema: type[ModelT],
    *,
    task: GroqRole,
    evidence_count: int = 0,
) -> ModelT:
    """Run flat JSON-schema output with at most one distinct fallback model."""
    policy = role_config(task)
    if not policy.allow_structured_output:
        raise ValueError(f"Structured output is not allowed for role: {task}")
    models = _distinct_models(policy)
    last_error: GroqRequestError | None = None
    for attempt, model in enumerate(models):
        try:
            result = _call_groq(
                task,
                lambda: create_llm(task, model)
                .with_structured_output(
                    schema,
                    method="json_schema",
                    strict=model.startswith("openai/gpt-oss-"),
                )
                .invoke(prompt),
            )
            return result if isinstance(result, schema) else schema.model_validate(result)
        except GroqRequestError as exc:
            last_error = exc
            recovered = attempt == 0 and len(models) == 2
            _log_failure(
                task, model, schema, prompt, evidence_count,
                "primary" if attempt == 0 else "fallback", error=exc,
                recovered=recovered,
            )
            if not recovered:
                break
        except Exception:
            last_error = GroqRequestError("invalid_structured", task)
            recovered = attempt == 0 and len(models) == 2
            _log_failure(
                task, model, schema, prompt, evidence_count,
                "primary" if attempt == 0 else "fallback", error=last_error,
                recovered=recovered,
            )
            if not recovered:
                break
    raise last_error or GroqRequestError("invalid_structured", task)


def _distinct_models(policy: RoleConfig) -> list[str]:
    return list(
        dict.fromkeys(
            model for model in (policy.primary_model, policy.fallback_model) if model
        )
    )


def _log_failure(
    role: GroqRole,
    model: str,
    schema: type[BaseModel] | None,
    prompt: object,
    evidence_count: int,
    stage: str,
    *,
    error: GroqRequestError,
    recovered: bool,
) -> None:
    (logger.info if recovered else logger.error)(
        "Groq role attempt failed role=%s model=%s schema=%s kind=%s status=%s "
        "prompt_length=%d evidence_count=%d stage=%s recovered=%s",
        role,
        model,
        schema.__name__ if schema else "none",
        error.kind,
        error.status_code,
        len(str(prompt)),
        evidence_count,
        stage,
        recovered,
    )


def _call_groq(role: GroqRole, operation: Callable[[], ResultT]) -> ResultT:
    if role not in ALLOWED_GROQ_ROLES:
        raise ValueError(f"Groq is not allowed for role: {role}")
    try:
        return operation()
    except GroqRequestError:
        raise
    except APITimeoutError as exc:
        raise GroqRequestError("timeout", role) from exc
    except APIStatusError as exc:
        raise GroqRequestError(
            _status_error_kind(exc.status_code, _status_error_detail(exc)),
            role,
            exc.status_code,
        ) from exc
    except APIConnectionError as exc:
        raise GroqRequestError("connection", role) from exc
    except APIError as exc:
        raise GroqRequestError("provider", role) from exc


def _status_error_kind(status_code: int, detail: str) -> GroqErrorKind:
    normalized = detail.casefold()
    if status_code == 429:
        return "rate_limit"
    if status_code == 404 or (
        "model" in normalized
        and any(
            term in normalized
            for term in ("not found", "unavailable", "decommissioned", "does not exist")
        )
    ):
        return "unavailable_model"
    if status_code == 400 and any(
        term in normalized
        for term in ("failed to call a function", "tool_use_failed", "tool call validation")
    ):
        return "function_call"
    return "bad_request" if status_code == 400 else "provider"


def _status_error_detail(exc: APIStatusError) -> str:
    """Return a short provider message without echoing generated payloads."""
    try:
        payload = exc.response.json()
    except (AttributeError, TypeError, ValueError):
        return ""
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message", "") if isinstance(error, dict) else ""
    if not isinstance(message, str):
        return ""
    return message.split("failed_generation", 1)[0].strip().rstrip("{, ")[:500]
