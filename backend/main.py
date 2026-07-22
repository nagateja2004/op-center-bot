"""FastAPI entrypoint for the Opcenter RAG backend."""

import logging
from uuid import uuid4
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.dependencies import lifespan, readiness_checks
from backend.routes.chat import router as chat_router
from src.config import settings
from src.observability import increment, observe, render_prometheus


logger = logging.getLogger(__name__)


app = FastAPI(title="Opcenter RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Session-ID"])
app.include_router(chat_router)


@app.middleware("http")
async def request_safety(request: Request, call_next):
    started = perf_counter()
    request.state.request_id = str(uuid4())
    try:
        content_length = int(request.headers.get("content-length", "0") or 0)
    except ValueError:
        content_length = settings.max_request_bytes + 1
    if request.method == "POST" and content_length > settings.max_request_bytes:
        response = JSONResponse(status_code=413, content={"detail": "Request is too large."})
    else:
        body = await request.body() if request.method == "POST" else b""
        if len(body) > settings.max_request_bytes:
            response = JSONResponse(status_code=413, content={"detail": "Request is too large."})
        else:
            response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    route = getattr(request.scope.get("route"), "path", request.url.path)
    increment(
        "opcenter_http_requests_total",
        method=request.method,
        path=route,
        status=response.status_code,
    )
    observe(
        "opcenter_http_request_duration_seconds",
        perf_counter() - started,
        method=request.method,
        path=route,
    )
    return response


@app.exception_handler(Exception)
async def safe_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    logger.error(
        "unhandled_request_error request_id=%s error=%s",
        request_id,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "The request could not be completed.", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    checks = await readiness_checks(request.app)
    if not all(checks.values()):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "checks": checks},
        )
    return {"status": "ready", "checks": checks}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(render_prometheus(), media_type="text/plain; version=0.0.4")
