from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from aninamer.config import AppConfig
from aninamer.store import RuntimeStore
from aninamer.worker import AninamerWorker

from .schemas import (
    ErrorResponse,
    HealthResponse,
    HealthWorkerResponse,
    JobListResponse,
    JobRequestCreate,
    JobRequestResponse,
    JobResponse,
    RuntimeResponse,
    StatusResponse,
)
from .service import ApiService


@dataclass(frozen=True)
class ApiRuntime:
    config: AppConfig
    store: RuntimeStore
    worker: AninamerWorker
    service: ApiService


def create_app(
    config: AppConfig,
    *,
    store: RuntimeStore,
    worker: AninamerWorker,
) -> FastAPI:
    runtime = ApiRuntime(
        config=config,
        store=store,
        worker=worker,
        service=ApiService(store=store, worker=worker),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        app.state.runtime = runtime
        yield

    app = FastAPI(
        title="aninamer",
        version="0.1.3",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    _register_exception_handlers(app)

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        response_model_exclude_none=True,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
        operation_id="getHealth",
    )
    def get_health(response: Response) -> HealthResponse:
        health = runtime.worker.health_status()
        if not health.healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status="unhealthy",
                reason=health.reason,
                worker=HealthWorkerResponse(
                    running=health.running,
                    last_scan_at=health.last_scan_at,
                    consecutive_failures=health.consecutive_failures,
                    stale_after_seconds=health.stale_after_seconds,
                    unavailable_watch_root_count=len(
                        health.unavailable_watch_root_keys
                    ),
                ),
            )
        return HealthResponse(status="ok")

    def _api_service(request: Request) -> ApiService:
        return request.app.state.runtime.service

    ApiServiceDep = Depends(_api_service)
    AuthDep = Depends(require_api_token)

    @app.get(
        "/api/v1/runtime",
        dependencies=[AuthDep],
        response_model=RuntimeResponse,
        operation_id="getRuntime",
    )
    def get_runtime(service: ApiService = ApiServiceDep) -> RuntimeResponse:
        return service.runtime()

    @app.get(
        "/api/v1/jobs",
        dependencies=[AuthDep],
        response_model=JobListResponse,
        operation_id="listJobs",
    )
    def list_jobs(service: ApiService = ApiServiceDep) -> JobListResponse:
        items = service.list_jobs()
        return JobListResponse(items=items, total=len(items))

    @app.get(
        "/api/v1/jobs/{job_id}",
        dependencies=[AuthDep],
        response_model=JobResponse,
        operation_id="getJob",
    )
    def get_job(job_id: int, service: ApiService = ApiServiceDep) -> JobResponse:
        item = service.get_job(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/job-requests",
        dependencies=[AuthDep],
        response_model=JobRequestResponse,
        status_code=status.HTTP_202_ACCEPTED,
        operation_id="createJobRequest",
    )
    def create_job_request(
        payload: JobRequestCreate,
        service: ApiService = ApiServiceDep,
    ) -> JobRequestResponse:
        return service.create_job_request(payload.action, payload.job_id)

    @app.get(
        "/api/v1/job-requests/{request_id}",
        dependencies=[AuthDep],
        response_model=JobRequestResponse,
        operation_id="getJobRequest",
    )
    def get_job_request(
        request_id: int,
        service: ApiService = ApiServiceDep,
    ) -> JobRequestResponse:
        item = service.get_job_request(request_id)
        if item is None:
            raise HTTPException(status_code=404, detail="job request not found")
        return item

    @app.get(
        "/api/v1/status",
        dependencies=[AuthDep],
        response_model=StatusResponse,
        operation_id="getStatus",
    )
    def get_status(service: ApiService = ApiServiceDep) -> StatusResponse:
        return service.status()

    return app


def require_api_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    token = _extract_bearer_token(authorization)
    expected = request.app.state.runtime.config.api.token
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="missing_authorization",
            headers={"WWW-Authenticate": 'Bearer realm="aninamer-api"'},
        )
    if token != expected:
        raise HTTPException(status_code=403, detail="invalid_token")


def _extract_bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    normalized = token.strip()
    if scheme.casefold() != "bearer" or not normalized:
        return None
    return normalized


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _json_error(
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
            details=exc.errors(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        error_code = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            405: "method_not_allowed",
        }.get(exc.status_code, "http_error")
        return _json_error(
            status_code=exc.status_code,
            code=error_code,
            message=str(exc.detail),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        return _json_error(
            status_code=500,
            code="internal_server_error",
            message="Internal server error.",
        )


def _json_error(
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error={"code": code, "message": message, "details": details}
    ).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=payload, headers=headers)
