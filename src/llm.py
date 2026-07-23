"""Role-based asynchronous Groq clients with bounded retries and Redis limits."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
import random
import re
import time
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from groq import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncGroq
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from src.config import Settings, settings
from src.groq_limits import GroqLimitError, current_groq_request_id, get_groq_limiter
from src.observability import increment, observe


if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_groq import ChatGroq


ResultT = TypeVar("ResultT")
ModelT = TypeVar("ModelT", bound=BaseModel)
GroqRole = Literal[
    "planner", "query_broadening", "grader", "answer", "verifier", "diagram"
]
GroqErrorKind = Literal[
    "rate_limit", "function_call", "timeout", "invalid_structured",
    "unavailable_model", "bad_request", "connection", "provider",
]
ALLOWED_GROQ_ROLES = frozenset(
    {"planner", "query_broadening", "grader", "answer", "verifier", "diagram"}
)
logger = logging.getLogger(__name__)
THINK_END_RE = re.compile(r"</think>\s*", re.I)
_async_client: AsyncGroq | None = None


def _chat_groq_class():
    from langchain_groq import ChatGroq

    return ChatGroq


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
    "diagram": RoleConfig("openai/gpt-oss-20b", "openai/gpt-oss-120b", 0.0, 2048, 30, False),
}
PRIMARY_MODEL_ALIASES: dict[GroqRole, str] = {
    "planner": "GROQ_PLANNER_MODEL", "query_broadening": "GROQ_QUERY_MODEL",
    "grader": "GROQ_GRADER_MODEL", "answer": "GROQ_ANSWER_MODEL",
    "verifier": "GROQ_VERIFY_MODEL", "diagram": "GROQ_DIAGRAM_MODEL",
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
        retry_after: float | None = None,
    ) -> None:
        self.kind = kind
        self.role = role
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"LLM request unavailable for role={role} kind={kind}")


def initialize_async_groq_client(config: Settings = settings) -> AsyncGroq:
    """Create the one reusable async HTTP client for this backend process."""
    global _async_client
    if _async_client is None:
        config.validate()
        _async_client = AsyncGroq(
            api_key=config.groq_api_key,
            timeout=config.groq_request_timeout,
            max_retries=0,
        )
    return _async_client


async def close_async_groq_client() -> None:
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
        create_llm.cache_clear()


def role_config(role: GroqRole) -> RoleConfig:
    if role not in ALLOWED_GROQ_ROLES:
        raise ValueError(f"Groq is not allowed for role: {role}")
    default = ROLE_DEFAULTS[role]
    prefix = f"GROQ_{role.upper()}"
    primary = os.getenv(
        f"{prefix}_PRIMARY_MODEL", os.getenv(PRIMARY_MODEL_ALIASES[role], default.primary_model)
    ).strip()
    if not primary:
        raise EnvironmentError(f"Missing required environment variable: {prefix}_PRIMARY_MODEL")
    return RoleConfig(
        primary_model=primary,
        fallback_model=os.getenv(
            f"{prefix}_FALLBACK_MODEL", os.getenv(FALLBACK_MODEL_ALIASES.get(role, ""), default.fallback_model)
        ).strip(),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", default.temperature)),
        max_output_tokens=int(os.getenv(f"{prefix}_MAX_OUTPUT_TOKENS", default.max_output_tokens)),
        timeout=float(os.getenv(f"{prefix}_TIMEOUT", default.timeout)),
        allow_structured_output=_env_bool(f"{prefix}_ALLOW_STRUCTURED_OUTPUT", default.allow_structured_output),
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
def create_llm(role: GroqRole, model: str, config: Settings = settings) -> ChatGroq:
    """Return a cached role/model adapter backed by the shared async Groq client."""
    policy = role_config(role)
    completions = initialize_async_groq_client(config).chat.completions
    return _chat_groq_class()(
        api_key=config.groq_api_key,
        model=model,
        temperature=policy.temperature,
        max_tokens=policy.max_output_tokens,
        timeout=policy.timeout,
        max_retries=0,
        client=completions,
        async_client=completions,
    )


async def call_llm(
    prompt: LanguageModelInput, *, task: GroqRole, evidence_count: int = 0
) -> AIMessage:
    """Run plain-text generation with one primary and at most one fallback model."""
    policy = role_config(task)
    for attempt, model in enumerate(_distinct_models(policy)):
        try:
            message = cast(
                AIMessage,
                await _invoke_model(
                    task, model, policy, prompt,
                    lambda: create_llm(task, model).ainvoke(prompt),
                ),
            )
            if THINK_END_RE.search(str(message.content)):
                message = message.model_copy(
                    update={"content": THINK_END_RE.split(str(message.content))[-1].strip()}
                )
            return message
        except GroqRequestError as exc:
            recovered = attempt == 0 and _can_fallback(exc) and len(_distinct_models(policy)) == 2
            _log_failure(task, model, None, prompt, evidence_count, "primary" if attempt == 0 else "fallback", exc, recovered)
            if not recovered:
                raise
    raise GroqRequestError("provider", task)


async def call_structured(
    prompt: LanguageModelInput,
    schema: type[ModelT],
    *,
    task: GroqRole,
    evidence_count: int = 0,
) -> ModelT:
    """Run planner/grader structured output with one primary and one fallback."""
    policy = role_config(task)
    if not policy.allow_structured_output:
        raise ValueError(f"Structured output is not allowed for role: {task}")
    models = _distinct_models(policy)
    last_error: GroqRequestError | None = None
    for attempt, model in enumerate(models):
        try:
            result = await _invoke_model(
                task, model, policy, prompt,
                lambda: create_llm(task, model)
                .with_structured_output(schema, method="json_schema", strict=model.startswith("openai/gpt-oss-"))
                .ainvoke(prompt),
            )
            return result if isinstance(result, schema) else schema.model_validate(result)
        except GroqRequestError as exc:
            last_error = exc
            recovered = attempt == 0 and _can_fallback(exc) and len(models) == 2
            _log_failure(task, model, schema, prompt, evidence_count, "primary" if attempt == 0 else "fallback", exc, recovered)
            if not recovered:
                break
        except Exception:
            last_error = GroqRequestError("invalid_structured", task)
            recovered = attempt == 0 and len(models) == 2
            _log_failure(task, model, schema, prompt, evidence_count, "primary" if attempt == 0 else "fallback", last_error, recovered)
            if not recovered:
                break
    raise last_error or GroqRequestError("invalid_structured", task)


async def _invoke_model(
    role: GroqRole,
    model: str,
    policy: RoleConfig,
    prompt: LanguageModelInput,
    operation: Callable[[], Awaitable[ResultT]],
) -> ResultT:
    token_cost = max(1, len(str(prompt)) // 4 + policy.max_output_tokens)
    for retry in range(2):
        started = time.perf_counter()
        try:
            result = await get_groq_limiter().run(
                model, token_cost, lambda: _call_groq(role, operation)
            )
            _log_request(role, model, "success", started, result, token_cost)
            return result
        except GroqLimitError as exc:
            _log_request(role, model, exc.kind, started, None, token_cost)
            raise GroqRequestError("rate_limit", role, 429) from exc
        except GroqRequestError as exc:
            _log_request(role, model, exc.kind, started, None, token_cost)
            if retry or not _should_retry(exc):
                raise
            await get_groq_limiter().set_status(
                current_groq_request_id(), "retrying", model=model, reason=exc.kind
            )
            delay = min(30.0, exc.retry_after if exc.kind == "rate_limit" and exc.retry_after is not None else 0.25)
            await asyncio.sleep(delay + random.uniform(0.05, 0.25))
    raise GroqRequestError("provider", role)


async def _call_groq(role: GroqRole, operation: Callable[[], Awaitable[ResultT]]) -> ResultT:
    if role not in ALLOWED_GROQ_ROLES:
        raise ValueError(f"Groq is not allowed for role: {role}")
    try:
        return await operation()
    except GroqRequestError:
        raise
    except APITimeoutError as exc:
        raise GroqRequestError("timeout", role) from exc
    except APIStatusError as exc:
        raise GroqRequestError(
            _status_error_kind(exc.status_code, _status_error_detail(exc)),
            role,
            exc.status_code,
            _retry_after(exc),
        ) from exc
    except APIConnectionError as exc:
        raise GroqRequestError("connection", role) from exc
    except APIError as exc:
        raise GroqRequestError("provider", role) from exc


def _distinct_models(policy: RoleConfig) -> list[str]:
    return list(dict.fromkeys(model for model in (policy.primary_model, policy.fallback_model) if model))


def _should_retry(error: GroqRequestError) -> bool:
    return error.kind in {"rate_limit", "timeout"} or bool(error.status_code and error.status_code >= 500)


def _can_fallback(error: GroqRequestError) -> bool:
    return error.status_code not in {400, 401, 403}


def _log_request(role: GroqRole, model: str, status: str, started: float, result: object | None, estimated_tokens: int) -> None:
    usage = _token_usage(result, estimated_tokens)
    duration = time.perf_counter() - started
    increment("opcenter_groq_requests_total", role=role, model=model, status=status)
    observe("opcenter_groq_request_duration_seconds", duration, role=role, model=model, status=status)
    increment("opcenter_groq_tokens_total", usage[2], role=role, model=model)
    logger.info(
        "groq_request role=%s model=%s status=%s duration_ms=%.1f input_tokens=%s output_tokens=%s total_tokens=%s",
        role, model, status, duration * 1000,
        usage[0], usage[1], usage[2],
    )


def _token_usage(result: object | None, estimated_tokens: int) -> tuple[int, int, int]:
    metadata = getattr(result, "usage_metadata", None) or {}
    if not metadata:
        metadata = getattr(result, "response_metadata", {}).get("token_usage", {}) if result else {}
    input_tokens = int(metadata.get("input_tokens", metadata.get("prompt_tokens", estimated_tokens)))
    output_tokens = int(metadata.get("output_tokens", metadata.get("completion_tokens", 0)))
    total_tokens = int(metadata.get("total_tokens", input_tokens + output_tokens))
    return input_tokens, output_tokens, total_tokens


def _log_failure(role: GroqRole, model: str, schema: type[BaseModel] | None, prompt: object, evidence_count: int, stage: str, error: GroqRequestError, recovered: bool) -> None:
    (logger.info if recovered else logger.error)(
        "Groq role attempt failed role=%s model=%s schema=%s kind=%s status=%s prompt_length=%d evidence_count=%d stage=%s recovered=%s",
        role, model, schema.__name__ if schema else "none", error.kind, error.status_code,
        len(str(prompt)), evidence_count, stage, recovered,
    )


def _status_error_kind(status_code: int, detail: str) -> GroqErrorKind:
    normalized = detail.casefold()
    if status_code == 429:
        return "rate_limit"
    if status_code == 404 or ("model" in normalized and any(term in normalized for term in ("not found", "unavailable", "decommissioned", "does not exist"))):
        return "unavailable_model"
    if status_code == 400 and any(term in normalized for term in ("failed to call a function", "tool_use_failed", "tool call validation")):
        return "function_call"
    return "bad_request" if status_code == 400 else "provider"


def _status_error_detail(exc: APIStatusError) -> str:
    try:
        payload = exc.response.json()
    except (AttributeError, TypeError, ValueError):
        return ""
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message", "") if isinstance(error, dict) else ""
    return message.split("failed_generation", 1)[0].strip().rstrip("{, ")[:500] if isinstance(message, str) else ""


def _retry_after(exc: APIStatusError) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", {})
    try:
        return max(0.0, float(headers.get("retry-after", "")))
    except (AttributeError, TypeError, ValueError):
        return None
