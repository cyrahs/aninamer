from __future__ import annotations

from typing import Any, Literal

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


class PathRef(ApiSchema):
    root_kind: str
    root_id: str
    rel_path: str
    display_path: str
    normalized_abs_path: str | None = None
    realpath: str | None = None
    casefold_key: str | None = None
    redacted: bool = False


class JobVersions(ApiSchema):
    mapping_version: int = Field(ge=1)
    move_plan_version: int = Field(ge=1)
    move_state_revision: int = Field(ge=1)
    review_revision: int = Field(ge=1)


class RollbackCapabilityHints(ApiSchema):
    has_moved_items: bool
    has_archive_records: bool
    has_review_records: bool
    needs_safety_check: bool


class LastRollbackCheckSummary(ApiSchema):
    status: Literal["unchecked", "checked_no_token", "stale", "expired", "failed"]
    last_checked_at: str | None = None
    mode: str | None = None


class SelectionScope(ApiSchema):
    kind: Literal["all", "episode_targets", "move_items", "source_files"]
    ids: list[int] | None = None

    @model_validator(mode="after")
    def validate_ids(self) -> "SelectionScope":
        if self.kind == "all":
            if self.ids is not None:
                raise ValueError("ids is not allowed for all scope")
            return self
        if not self.ids:
            raise ValueError("ids is required for selected scope")
        if any(item <= 0 for item in self.ids):
            raise ValueError("ids must be positive")
        return self


class ReviewSourceFile(ApiSchema):
    id: int
    kind: Literal["video", "subtitle", "ignored"]
    original_path: PathRef
    current_path: PathRef
    current_path_kind: str
    ext: str
    file_identity: dict[str, Any]
    detected_subtitle_variant: Literal["chs", "cht", "chi", "unknown"] | None = None
    final_disposition: str
    ignored_reason: str | None = None


class ReviewEpisodeTarget(ApiSchema):
    id: int
    video_source_id: int
    season: int
    episode_start: int
    episode_end: int
    status: str
    manual: bool
    created_by: str
    move_item_ids: list[int]
    subtitle_link_ids: list[int]


class ReviewSubtitleLink(ApiSchema):
    id: int
    episode_target_id: int
    subtitle_source_id: int
    variant_detected: Literal["chs", "cht", "chi", "unknown"]
    variant_override: Literal["chs", "cht", "chi"] | None = None
    status: str
    created_by: str
    move_item_id: int | None = None


class TmdbReviewInfo(ApiSchema):
    selected_tmdb_id: int | None = None
    title_zh_cn: str | None = None
    original_name: str | None = None
    year: int | None = None
    candidates: list[dict[str, Any]]
    seasons: list[dict[str, Any]]
    specials: list[dict[str, Any]]


class ReviewMoveItem(ApiSchema):
    id: int
    source_id: int
    kind: Literal["video", "subtitle"]
    source_path: PathRef
    planned_destination_path: PathRef
    current_path: PathRef
    status: str
    error_message: str | None = None
    applied_at: str | None = None


class ReviewPayload(ApiSchema):
    job: JobResponse
    versions: JobVersions
    source_files: list[ReviewSourceFile]
    selected_tmdb_info: TmdbReviewInfo
    episode_targets: list[ReviewEpisodeTarget]
    subtitle_links: list[ReviewSubtitleLink]
    move_items: list[ReviewMoveItem]
    mapping_attempts: list[dict[str, Any]]
    mapping_evidence: list[dict[str, Any]]
    path_event_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    readiness_summary: dict[str, Any]
    readiness_acknowledgements: list[dict[str, Any]]
    finalize_status: str
    archive_path_summary: dict[str, Any]
    review_exclusion_proof: dict[str, Any]
    rollback_capability_hints: RollbackCapabilityHints
    last_rollback_check_summary: LastRollbackCheckSummary
    llm_failure_summary: dict[str, Any]
    review_revision: int


class OperationResponse(ApiSchema):
    id: int
    type: str
    job_id: int
    status: Literal["queued", "running", "succeeded", "partial", "failed", "canceled"]
    request_payload_hash: str
    base_versions: dict[str, Any]
    progress: dict[str, Any]
    result_summary: dict[str, Any]
    error: dict[str, Any] | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    poll_after_ms: int


class OperationListResponse(ApiSchema):
    items: list[OperationResponse]
    total: int


class OperationItemResultResponse(ApiSchema):
    id: int
    operation_id: int
    subject_kind: str
    subject_id: int | None = None
    status: str
    path_event_id: int | None = None
    reservation_id: int | None = None
    from_path: dict[str, Any] | None = None
    to_path: dict[str, Any] | None = None
    identity_check: str
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool
    retry_block_reason: str | None = None
    result_revision: int
    created_at: str


class OperationItemResultListResponse(ApiSchema):
    items: list[OperationItemResultResponse]
    total: int


class SafetyResponse(ApiSchema):
    roots: dict[str, Any]
    watcher_status: dict[str, Any]
    consumer_chain_digest: str | None = None
    review_exclusion_proof: dict[str, Any]
    last_source_gate_check_summary: dict[str, Any]
    safety_token_summary: dict[str, Any]
    active_operation_leases: list[dict[str, Any]]


class ReviewSafetyCheckRequest(ApiSchema):
    base_review_revision: int = Field(ge=1)


class ReviewSafetyCheckResponse(ApiSchema):
    status: Literal["verified", "failed"]
    checked_at: str
    ttl_seconds: int = Field(gt=0)
    review_safety_check_token: str | None = None
    token_status: Literal["current", "not_current"]
    consumer_chain_digest: str
    review_exclusion_proof: dict[str, Any]


class RollbackEligibilityRequest(ApiSchema):
    mode: Literal[
        "withdraw_to_review",
        "restore_moved_to_source",
        "restore_original_layout_to_source",
        "restore_original_layout_to_review",
    ]
    scope: SelectionScope
    base_review_revision: int = Field(ge=1)
    base_move_state_revision: int = Field(ge=1)
    review_safety_check_token: str | None = None
    source_gate_check_token: str | None = None

    @model_validator(mode="after")
    def validate_supported_scope(self) -> "RollbackEligibilityRequest":
        if self.scope.kind not in {"all", "move_items"}:
            raise ValueError(
                "rollback eligibility currently supports all or move_items scope"
            )
        return self


class RollbackEligibilityResponse(ApiSchema):
    mode: str
    status: Literal["eligible", "partial", "ineligible"]
    checked_at: str
    ttl_seconds: int = Field(gt=0)
    check_token: str | None = None
    checked_move_state_revision: int
    token_status: Literal["current", "expired", "stale", "not_current"]
    destination_preview: list[dict[str, Any]]
    item_result_summary: dict[str, Any]
    missing_files: list[dict[str, Any]]
    incomplete_records: list[dict[str, Any]]
    target_conflicts: list[dict[str, Any]]
    identity_mismatches: list[dict[str, Any]]
    retryable_items: list[dict[str, Any]]
    non_retryable_items: list[dict[str, Any]]
    affected_move_items: list[int]
    archive_restore_summary: dict[str, Any]


class WithdrawRequest(ApiSchema):
    mode: Literal["withdraw_to_review"]
    scope: SelectionScope
    base_review_revision: int = Field(ge=1)
    base_move_state_revision: int = Field(ge=1)
    check_token: str
    review_safety_check_token: str

    @model_validator(mode="after")
    def validate_supported_scope(self) -> "WithdrawRequest":
        if self.scope.kind not in {"all", "move_items"}:
            raise ValueError("withdraw currently supports all or move_items scope")
        if not self.check_token:
            raise ValueError("check_token is required")
        if not self.review_safety_check_token:
            raise ValueError("review_safety_check_token is required")
        return self


class WithdrawResponse(ApiSchema):
    operation_id: int
    status: Literal["succeeded", "failed"]
    mode: Literal["withdraw_to_review"]
    withdrawn_move_item_count: int = Field(ge=0)
    item_result_summary: dict[str, Any]
    withdrawn_items: list[dict[str, Any]]
    error: dict[str, Any] | None = None


class ReopenFromReviewRequest(ApiSchema):
    scope: SelectionScope
    base_review_revision: int = Field(ge=1)
    base_move_state_revision: int = Field(ge=1)
    review_safety_check_token: str

    @model_validator(mode="after")
    def validate_supported_scope(self) -> "ReopenFromReviewRequest":
        if self.scope.kind != "all":
            raise ValueError("reopen from review currently supports all scope")
        if not self.review_safety_check_token:
            raise ValueError("review_safety_check_token is required")
        return self


class ReopenFromReviewResponse(ApiSchema):
    operation_id: int
    status: Literal["succeeded", "failed"]
    reopened_move_item_count: int = Field(ge=0)
    item_result_summary: dict[str, Any]
    reopened_items: list[dict[str, Any]]
    error: dict[str, Any] | None = None


class RematchPreviewRequest(ApiSchema):
    tmdb_id: int = Field(gt=0)
    base_review_revision: int = Field(ge=1)


class RematchPreviewResponse(ApiSchema):
    operation_id: int
    status: Literal["preview_ready", "failed"]
    tmdb_id: int
    base_review_revision: int
    selected_tmdb_info: TmdbReviewInfo
    episode_targets: list[ReviewEpisodeTarget]
    subtitle_links: list[ReviewSubtitleLink]
    move_items: list[dict[str, Any]]
    mapping_evidence: list[dict[str, Any]]
    validation_summary: dict[str, Any]
    diff_summary: dict[str, Any]
    diff_items: list[dict[str, Any]]


class RematchCommitRequest(ApiSchema):
    preview_operation_id: int = Field(gt=0)
    tmdb_id: int = Field(gt=0)
    base_review_revision: int = Field(ge=1)


class RematchCommitResponse(ApiSchema):
    operation_id: int
    status: Literal["committed"]
    preview_operation_id: int
    tmdb_id: int
    base_review_revision: int
    selected_tmdb_info: TmdbReviewInfo
    diff_summary: dict[str, Any]
    committed_move_item_count: int = Field(ge=0)


class CorrectionOp(ApiSchema):
    op: Literal["update_episode", "override_subtitle_variant"]
    episode_target_id: int | None = Field(default=None, gt=0)
    subtitle_link_id: int | None = Field(default=None, gt=0)
    season: int | None = Field(default=None, ge=0)
    episode_start: int | None = Field(default=None, ge=1)
    episode_end: int | None = Field(default=None, ge=1)
    variant: Literal["detected", "chs", "cht", "chi"] | None = None

    @model_validator(mode="after")
    def validate_op_fields(self) -> "CorrectionOp":
        if self.op == "update_episode":
            if self.episode_target_id is None:
                raise ValueError("episode_target_id is required")
            if self.season is None:
                raise ValueError("season is required")
            if self.episode_start is None:
                raise ValueError("episode_start is required")
            if self.episode_end is None:
                raise ValueError("episode_end is required")
            if self.episode_end < self.episode_start:
                raise ValueError("episode_end must be >= episode_start")
            return self
        if self.subtitle_link_id is None:
            raise ValueError("subtitle_link_id is required")
        if self.variant is None:
            raise ValueError("variant is required")
        return self


class CorrectionRequest(ApiSchema):
    base_review_revision: int = Field(ge=1)
    atomic: bool = True
    ops: list[CorrectionOp] = Field(min_length=1)


class CorrectionResponse(ApiSchema):
    operation_id: int
    status: Literal["committed"]
    base_review_revision: int
    op_results: list[dict[str, Any]]
    diff_summary: dict[str, Any]
    committed_move_item_count: int = Field(ge=0)


class ApplyValidationRequest(ApiSchema):
    scope: Literal["all_for_apply", "selected_check", "retry_apply"]
    base_mapping_version: int = Field(ge=1)
    base_move_plan_version: int = Field(ge=1)
    base_move_state_revision: int = Field(ge=1)
    base_review_revision: int = Field(ge=1)
    selection: SelectionScope | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "ApplyValidationRequest":
        if self.scope == "selected_check" and self.selection is None:
            raise ValueError("selection is required for selected_check")
        if self.scope != "selected_check" and self.selection is not None:
            raise ValueError("selection is only allowed for selected_check")
        return self


class ApplyValidationResponse(ApiSchema):
    operation_id: int
    validation_id: int
    status: Literal["valid", "invalid"]
    scope: Literal["all_for_apply", "selected_check", "retry_apply"]
    checked_at: str
    ttl_seconds: int = Field(gt=0)
    validated_mapping_version: int = Field(ge=1)
    validated_move_plan_version: int = Field(ge=1)
    checked_move_state_revision: int = Field(ge=1)
    checked_review_revision: int = Field(ge=1)
    item_result_summary: dict[str, Any]
    blocking_items: list[dict[str, Any]]
    warning_items: list[dict[str, Any]]
    destination_preview: list[dict[str, Any]]
    affected_move_items: list[int]


class ApplyRequest(ApiSchema):
    scope: Literal["all_validated"]
    validation_id: int = Field(gt=0)
    base_mapping_version: int = Field(ge=1)
    base_move_plan_version: int = Field(ge=1)
    base_move_state_revision: int = Field(ge=1)
    base_review_revision: int = Field(ge=1)


class ApplyResponse(ApiSchema):
    operation_id: int
    status: Literal["succeeded", "failed"]
    validation_id: int
    applied_move_item_count: int = Field(ge=0)
    item_result_summary: dict[str, Any]
    moved_items: list[dict[str, Any]]
    error: dict[str, Any] | None = None


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
