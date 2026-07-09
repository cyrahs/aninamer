from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from aninamer.config import AppConfig
from aninamer.store import IdempotencyConflictError, RuntimeStore
from aninamer.worker import AninamerWorker

from .schemas import (
    ApplyRequest,
    ApplyResponse,
    ApplyValidationRequest,
    ApplyValidationResponse,
    CorrectionRequest,
    CorrectionResponse,
    ErrorResponse,
    HealthResponse,
    HealthWorkerResponse,
    JobListResponse,
    JobRequestCreate,
    JobRequestResponse,
    JobResponse,
    OperationItemResultListResponse,
    OperationListResponse,
    OperationResponse,
    ReviewPayload,
    ReviewSafetyCheckRequest,
    ReviewSafetyCheckResponse,
    ReopenFromReviewRequest,
    ReopenFromReviewResponse,
    RematchCommitRequest,
    RematchCommitResponse,
    RematchPreviewRequest,
    RematchPreviewResponse,
    RollbackEligibilityRequest,
    RollbackEligibilityResponse,
    RuntimeResponse,
    SafetyResponse,
    StatusResponse,
    WithdrawRequest,
    WithdrawResponse,
)
from .service import ApiService, ApiServiceError


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

    app = FastAPI(
        title="aninamer",
        version="0.1.3",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.runtime = runtime
    _register_exception_handlers(app)
    static_dir = Path(__file__).parent / "static"
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

    @app.get("/ui", include_in_schema=False)
    def get_ui() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def get_favicon() -> FileResponse:
        return FileResponse(static_dir / "favicon.ico")

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
        "/api/v1/jobs/{job_id}/reopen-from-review",
        dependencies=[AuthDep],
        response_model=ReopenFromReviewResponse,
        operation_id="reopenJobFromReview",
    )
    def reopen_job_from_review(
        job_id: int,
        payload: ReopenFromReviewRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> ReopenFromReviewResponse:
        try:
            item = service.reopen_from_review(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.get(
        "/api/v1/jobs/{job_id}/review",
        dependencies=[AuthDep],
        response_model=ReviewPayload,
        operation_id="getJobReview",
    )
    def get_job_review(
        job_id: int,
        service: ApiService = ApiServiceDep,
    ) -> ReviewPayload:
        item = service.review(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.get(
        "/api/v1/jobs/{job_id}/operations",
        dependencies=[AuthDep],
        response_model=OperationListResponse,
        operation_id="listJobOperations",
    )
    def list_job_operations(
        job_id: int,
        service: ApiService = ApiServiceDep,
    ) -> OperationListResponse:
        item = service.list_operations(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.get(
        "/api/v1/jobs/{job_id}/operations/{operation_id}",
        dependencies=[AuthDep],
        response_model=OperationResponse,
        operation_id="getJobOperation",
    )
    def get_job_operation(
        job_id: int,
        operation_id: int,
        service: ApiService = ApiServiceDep,
    ) -> OperationResponse:
        item = service.get_operation(job_id, operation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="operation not found")
        return item

    @app.get(
        "/api/v1/jobs/{job_id}/operations/{operation_id}/items",
        dependencies=[AuthDep],
        response_model=OperationItemResultListResponse,
        operation_id="listJobOperationItems",
    )
    def list_job_operation_items(
        job_id: int,
        operation_id: int,
        service: ApiService = ApiServiceDep,
    ) -> OperationItemResultListResponse:
        item = service.list_operation_items(job_id, operation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="operation not found")
        return item

    @app.get(
        "/api/v1/jobs/{job_id}/safety",
        dependencies=[AuthDep],
        response_model=SafetyResponse,
        operation_id="getJobSafety",
    )
    def get_job_safety(
        job_id: int,
        service: ApiService = ApiServiceDep,
    ) -> SafetyResponse:
        item = service.safety(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/jobs/{job_id}/rematch-preview",
        dependencies=[AuthDep],
        response_model=RematchPreviewResponse,
        operation_id="previewJobRematch",
    )
    def preview_job_rematch(
        job_id: int,
        payload: RematchPreviewRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> RematchPreviewResponse:
        try:
            item = service.rematch_preview(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/jobs/{job_id}/rematch-commit",
        dependencies=[AuthDep],
        response_model=RematchCommitResponse,
        operation_id="commitJobRematch",
    )
    def commit_job_rematch(
        job_id: int,
        payload: RematchCommitRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> RematchCommitResponse:
        try:
            item = service.rematch_commit(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/jobs/{job_id}/corrections",
        dependencies=[AuthDep],
        response_model=CorrectionResponse,
        operation_id="commitJobCorrections",
    )
    def commit_job_corrections(
        job_id: int,
        payload: CorrectionRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> CorrectionResponse:
        try:
            item = service.corrections(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/jobs/{job_id}/validate",
        dependencies=[AuthDep],
        response_model=ApplyValidationResponse,
        operation_id="validateJobApply",
    )
    def validate_job_apply(
        job_id: int,
        payload: ApplyValidationRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> ApplyValidationResponse:
        try:
            item = service.validate_apply(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/jobs/{job_id}/apply",
        dependencies=[AuthDep],
        response_model=ApplyResponse,
        operation_id="applyJob",
    )
    def apply_job(
        job_id: int,
        payload: ApplyRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> ApplyResponse:
        try:
            item = service.apply_job(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/jobs/{job_id}/review-safety-check",
        dependencies=[AuthDep],
        response_model=ReviewSafetyCheckResponse,
        operation_id="checkJobReviewSafety",
    )
    def check_job_review_safety(
        job_id: int,
        payload: ReviewSafetyCheckRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> ReviewSafetyCheckResponse:
        try:
            item = service.review_safety_check(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/jobs/{job_id}/rollback-eligibility",
        dependencies=[AuthDep],
        response_model=RollbackEligibilityResponse,
        operation_id="checkRollbackEligibility",
    )
    def check_rollback_eligibility(
        job_id: int,
        payload: RollbackEligibilityRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> RollbackEligibilityResponse:
        try:
            item = service.rollback_eligibility(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post(
        "/api/v1/jobs/{job_id}/withdraw",
        dependencies=[AuthDep],
        response_model=WithdrawResponse,
        operation_id="withdrawJob",
    )
    def withdraw_job(
        job_id: int,
        payload: WithdrawRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        service: ApiService = ApiServiceDep,
    ) -> WithdrawResponse:
        try:
            item = service.withdraw_job(
                job_id,
                payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApiServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            ) from exc
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
        item = service.create_job_request(payload.action, payload.job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="job not found")
        return item

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
        detail = exc.detail
        error_code = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            405: "method_not_allowed",
        }.get(exc.status_code, "http_error")
        message = str(detail)
        details = None
        if isinstance(detail, dict):
            code = detail.get("code")
            if isinstance(code, str):
                error_code = code
            detail_message = detail.get("message")
            if isinstance(detail_message, str):
                message = detail_message
            details = detail.get("details")
        return _json_error(
            status_code=exc.status_code,
            code=error_code,
            message=message,
            details=details,
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
