from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(ApiSchema):
    code: str
    message: str
    details: object | None = None


class ErrorResponse(ApiSchema):
    error: ErrorDetail


class HealthWorkerResponse(ApiSchema):
    running: bool
    last_scan_at: str | None = None
    consecutive_failures: int = Field(ge=0)
    stale_after_seconds: int = Field(gt=0)
    unavailable_watch_root_count: int = Field(ge=0)


class HealthResponse(ApiSchema):
    status: Literal["ok", "unhealthy"]
    reason: str | None = None
    worker: HealthWorkerResponse | None = None


class RuntimeResponse(ApiSchema):
    auto_apply: bool
    settle_seconds: int
    scan_interval_seconds: int
    watch_root_keys: list[str]
    last_scan_at: str | None = None


class JobResponse(ApiSchema):
    id: int
    series_name: str
    watch_root_key: str
    source_kind: Literal["monitor", "api"]
    status: Literal[
        "pending",
        "planning",
        "planned",
        "apply_requested",
        "applying",
        "succeeded",
        "failed",
        "cleared",
    ]
    tmdb_id: int | None = None
    video_moves_count: int
    subtitle_moves_count: int
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_stage: str | None = None
    error_message: str | None = None


class JobListResponse(ApiSchema):
    items: list[JobResponse]
    total: int


class JobRequestCreate(ApiSchema):
    action: Literal["scan_now", "apply_job"]
    job_id: int | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "JobRequestCreate":
        if self.action == "apply_job" and self.job_id is None:
            raise ValueError("job_id is required for apply_job")
        if self.action == "scan_now" and self.job_id is not None:
            raise ValueError("job_id is not allowed for scan_now")
        return self


class JobRequestResponse(ApiSchema):
    id: int
    action: Literal["scan_now", "apply_job"]
    status: Literal["pending", "running", "succeeded", "failed", "rejected"]
    job_id: int | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None


class StatusSummary(ApiSchema):
    pending_count: int = Field(ge=0)
    planning_count: int = Field(ge=0)
    planned_count: int = Field(ge=0)
    apply_requested_count: int = Field(ge=0)
    applying_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)


class StatusItem(ApiSchema):
    job_id: int
    series_name: str
    watch_root_key: str
    status: Literal["pending", "planning", "planned", "apply_requested", "applying", "failed"]
    updated_at: str
    tmdb_id: int | None = None
    video_moves_count: int
    subtitle_moves_count: int
    error_stage: str | None = None
    error_message: str | None = None


class StatusResponse(ApiSchema):
    summary: StatusSummary
    pending_items: list[StatusItem]
    failed_items: list[StatusItem]
