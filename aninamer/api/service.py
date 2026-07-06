from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import shutil

from aninamer.monitoring import MONITOR_EXCLUDED_DIR_NAMES, MONITOR_REVIEW_DIR
from aninamer.store import (
    DestinationReservationConflictError,
    DestinationReservationSpec,
    JobMoveRecord,
    JobRecord,
    JobRequestRecord,
    IdempotencyConflictError,
    OperationItemResultRecord,
    OperationRecord,
    ReopenMoveItemUpdate,
    ReviewSnapshotRecord,
    RuntimeStore,
)
from aninamer.worker import AninamerWorker
from aninamer.pipeline import PlanBuildResult
from aninamer.plan import (
    PlannedMove,
    format_episode_base,
    format_season_folder,
    format_series_root_folder,
)
from aninamer.scanner import is_env_file_name

from .schemas import (
    ApplyRequest,
    ApplyResponse,
    ApplyValidationRequest,
    ApplyValidationResponse,
    CorrectionRequest,
    CorrectionResponse,
    JobRequestResponse,
    JobResponse,
    LastRollbackCheckSummary,
    OperationItemResultListResponse,
    OperationItemResultResponse,
    OperationListResponse,
    OperationResponse,
    PathRef,
    ReviewSafetyCheckRequest,
    ReviewSafetyCheckResponse,
    ReopenFromReviewRequest,
    ReopenFromReviewResponse,
    ReviewEpisodeTarget,
    ReviewMoveItem,
    ReviewPayload,
    ReviewSourceFile,
    ReviewSubtitleLink,
    RematchCommitRequest,
    RematchCommitResponse,
    RematchPreviewRequest,
    RematchPreviewResponse,
    RollbackCapabilityHints,
    RollbackEligibilityRequest,
    RollbackEligibilityResponse,
    RuntimeResponse,
    SelectionScope,
    SafetyResponse,
    StatusItem,
    StatusResponse,
    StatusSummary,
    TmdbReviewInfo,
    WithdrawRequest,
    WithdrawResponse,
)

_EPISODE_TOKEN_RE = re.compile(
    r"\bS(?P<season>\d{2})E(?P<start>\d{2})(?:-E(?P<end>\d{2}))?\b"
)
_SUBTITLE_VARIANT_SUFFIXES = (".chs", ".cht", ".chi")
_SAFETY_CHECK_TTL_SECONDS = 300
LEGACY_APPLY_DISABLED_MESSAGE = (
    "legacy apply is disabled until Operation/PathEvent/"
    "DestinationReservation safety chain is implemented"
)


@dataclass(frozen=True)
class ReopenReviewMoveSpec:
    move: JobMoveRecord
    review_path: Path
    from_path: dict[str, object]
    to_path: dict[str, object]


class ApiServiceError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class ApiService:
    store: RuntimeStore
    worker: AninamerWorker

    def runtime(self) -> RuntimeResponse:
        summary = self.worker.runtime_summary()
        return RuntimeResponse(
            auto_apply=summary.auto_apply,
            settle_seconds=summary.settle_seconds,
            scan_interval_seconds=summary.scan_interval_seconds,
            watch_root_keys=list(summary.watch_root_keys),
            last_scan_at=summary.last_scan_at,
        )

    def list_jobs(self) -> list[JobResponse]:
        return [self.model_job(item) for item in self.store.list_jobs()]

    def get_job(self, job_id: int) -> JobResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        return self.model_job(job)

    def review(self, job_id: int) -> ReviewPayload | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = _validation_source_files(job, moves, snapshot)
        snapshot_missing_targets = (
            snapshot is not None and not snapshot.episode_targets and bool(moves)
        )
        if snapshot is None or snapshot_missing_targets:
            episode_targets = _build_episode_targets(moves)
            subtitle_links = _build_subtitle_links(moves, episode_targets)
        else:
            episode_targets = [
                ReviewEpisodeTarget.model_validate(item)
                for item in snapshot.episode_targets
            ]
            subtitle_links = [
                ReviewSubtitleLink.model_validate(item)
                for item in snapshot.subtitle_links
            ]
        rollback_summary = self._last_rollback_check_summary_from_operations(
            operations
        )
        versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        if snapshot is None:
            selected_tmdb_info = TmdbReviewInfo(
                selected_tmdb_id=job.tmdb_id,
                title_zh_cn=job.series_name,
                original_name=None,
                year=None,
                candidates=[],
                seasons=[],
                specials=[],
            )
            mapping_attempts: list[dict[str, object]] = []
            mapping_evidence: list[dict[str, object]] = []
            path_event_summary: dict[str, object] = {
                "pending_count": 0,
                "incomplete_count": 0,
            }
            validation_summary: dict[str, object] = {
                "status": "snapshot_missing",
                "message": "review snapshot is not recorded for this job",
            }
        else:
            selected_tmdb_info = TmdbReviewInfo.model_validate(
                snapshot.selected_tmdb_info
            )
            mapping_attempts = snapshot.mapping_attempts
            mapping_evidence = snapshot.mapping_evidence
            path_event_summary = snapshot.path_event_summary
            validation_summary = snapshot.validation_summary
            if snapshot_missing_targets:
                validation_summary = {
                    **validation_summary,
                    "target_link_status": "derived_from_moves_for_legacy_snapshot",
                }
        return ReviewPayload(
            job=self.model_job(job),
            versions=versions,
            source_files=source_files,
            selected_tmdb_info=selected_tmdb_info,
            episode_targets=episode_targets,
            subtitle_links=subtitle_links,
            move_items=[self.model_review_move_item(job, move) for move in moves],
            mapping_attempts=mapping_attempts,
            mapping_evidence=mapping_evidence,
            path_event_summary=path_event_summary,
            validation_summary=validation_summary,
            readiness_summary=_build_readiness_summary(
                source_files,
                episode_targets,
                subtitle_links,
            ),
            readiness_acknowledgements=[],
            finalize_status="archived" if job.archive_path else "none",
            archive_path_summary={
                "present": job.archive_path is not None,
                "path": job.archive_path,
            },
            review_exclusion_proof={"status": "unknown"},
            rollback_capability_hints=self._rollback_capability_hints(job, moves),
            last_rollback_check_summary=rollback_summary,
            llm_failure_summary={
                "count": len(self.store.list_llm_failures(job.id)),
            },
            review_revision=versions["review_revision"],
        )

    def create_job_request(
        self,
        action: str,
        job_id: int | None,
    ) -> JobRequestResponse | None:
        if action == "apply_job" and (
            job_id is None or self.store.get_job(job_id) is None
        ):
            return None
        request = self.store.create_job_request(kind=action, target_job_id=job_id)
        if action == "apply_job":
            request = self.store.update_job_request(
                request.id,
                status="rejected",
                finished_at=request.created_at,
                error_message=LEGACY_APPLY_DISABLED_MESSAGE,
            )
        return self.model_job_request(request)

    def get_job_request(self, request_id: int) -> JobRequestResponse | None:
        record = self.store.get_job_request(request_id)
        if record is None:
            return None
        return self.model_job_request(record)

    def status(self) -> StatusResponse:
        jobs = self.store.list_jobs()
        pending_statuses = {"pending", "planning", "planned", "apply_requested", "applying"}
        pending_items = [
            self.model_status_item(job)
            for job in jobs
            if job.status in pending_statuses
        ]
        failed_items = [
            self.model_status_item(job) for job in jobs if job.status == "failed"
        ]
        pending_items.sort(key=lambda item: item.updated_at)
        failed_items.sort(key=lambda item: item.updated_at, reverse=True)
        summary = StatusSummary(
            pending_count=sum(1 for job in jobs if job.status == "pending"),
            planning_count=sum(1 for job in jobs if job.status == "planning"),
            planned_count=sum(1 for job in jobs if job.status == "planned"),
            apply_requested_count=sum(
                1 for job in jobs if job.status == "apply_requested"
            ),
            applying_count=sum(1 for job in jobs if job.status == "applying"),
            failed_count=sum(1 for job in jobs if job.status == "failed"),
        )
        return StatusResponse(
            summary=summary,
            pending_items=pending_items,
            failed_items=failed_items,
        )

    def list_operations(self, job_id: int) -> OperationListResponse | None:
        if self.store.get_job(job_id) is None:
            return None
        items = [self.model_operation(item) for item in self.store.list_operations(job_id)]
        return OperationListResponse(items=items, total=len(items))

    def get_operation(self, job_id: int, operation_id: int) -> OperationResponse | None:
        operation = self.store.get_operation(job_id, operation_id)
        if operation is None:
            return None
        return self.model_operation(operation)

    def list_operation_items(
        self,
        job_id: int,
        operation_id: int,
    ) -> OperationItemResultListResponse | None:
        if self.store.get_operation(job_id, operation_id) is None:
            return None
        items = [
            self.model_operation_item_result(item)
            for item in self.store.list_operation_item_results(operation_id)
        ]
        return OperationItemResultListResponse(items=items, total=len(items))

    def safety(self, job_id: int) -> SafetyResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        review_root = _review_root(job)
        roots = {
            "source": _validation_path_ref(
                job.series_dir,
                root_kind="source",
                root=Path(job.series_dir).parent,
            ).model_dump(),
            "output": _validation_path_ref(
                job.output_root,
                root_kind="output",
                root=Path(job.output_root),
            ).model_dump(),
            "archive": (
                _validation_path_ref(
                    job.archive_path,
                    root_kind="archive",
                    root=Path(job.archive_path).parent if job.archive_path else None,
                ).model_dump()
                if job.archive_path
                else None
            ),
            "review": _validation_path_ref(
                review_root,
                root_kind="review",
                root=review_root.parent,
            ).model_dump(),
            "fail": (
                _validation_path_ref(
                    job.fail_path,
                    root_kind="fail",
                    root=Path(job.fail_path).parent if job.fail_path else None,
                ).model_dump()
                if job.fail_path
                else None
            ),
        }
        consumer_digest = _consumer_chain_digest(self.worker)
        proof = _review_exclusion_proof(
            job=job,
            worker=self.worker,
            review_root=review_root,
            consumer_chain_digest=consumer_digest,
            checked_at=_now_iso(),
        )
        active_leases = [
            _active_review_safety_lease_summary(operation)
            for operation in self.store.list_operations(job.id)
            if _operation_has_active_review_safety_lease(operation)
        ]
        return SafetyResponse(
            roots=roots,
            watcher_status={
                "status": "configured",
                "watch_root_key": job.watch_root_key,
                "watch_root_count": len(self.worker.config.watch_roots),
            },
            consumer_chain_digest=consumer_digest,
            review_exclusion_proof=proof,
            last_source_gate_check_summary={"status": "unchecked"},
            safety_token_summary={
                "review_safety_check_token": "not_returned_by_get",
                "source_gate_check_token": "not_returned_by_get",
            },
            active_operation_leases=active_leases,
        )

    def review_safety_check(
        self,
        job_id: int,
        payload: ReviewSafetyCheckRequest,
        *,
        idempotency_key: str,
    ) -> ReviewSafetyCheckResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="review_safety_check",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return ReviewSafetyCheckResponse.model_validate(stored_payload)
            if existing_operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message=(
                        "review safety check operation with this idempotency key "
                        "is still running"
                    ),
                    details={
                        "operation_id": existing_operation.id,
                        "status": existing_operation.status,
                    },
                )
            raise ApiServiceError(
                status_code=409,
                code="idempotency_result_unavailable",
                message=(
                    "review safety check operation result is unavailable for "
                    "this idempotency key"
                ),
                details={
                    "operation_id": existing_operation.id,
                    "status": existing_operation.status,
                },
            )
        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = self.model_review_source_files(job, moves, snapshot)
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        if payload.base_review_revision != current_versions["review_revision"]:
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="review safety check base review revision is stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": {
                        "review_revision": payload.base_review_revision,
                    },
                },
            )
        claim_token = secrets.token_urlsafe(16)
        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="review_safety_check",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions={"review_revision": payload.base_review_revision},
            result_summary={"status": "running", "claim_token": claim_token},
            status="running",
        )
        if operation.result_summary.get("claim_token") != claim_token:
            stored_payload = operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return ReviewSafetyCheckResponse.model_validate(stored_payload)
            if operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message=(
                        "review safety check operation with this idempotency key "
                        "is still running"
                    ),
                    details={"operation_id": operation.id, "status": operation.status},
                )
            raise ApiServiceError(
                status_code=409,
                code="idempotency_result_unavailable",
                message=(
                    "review safety check operation result is unavailable for "
                    "this idempotency key"
                ),
                details={"operation_id": operation.id, "status": operation.status},
            )
        checked_at = _now_iso()
        consumer_digest = _consumer_chain_digest(self.worker)
        proof = _review_exclusion_proof(
            job=job,
            worker=self.worker,
            review_root=_review_root(job),
            consumer_chain_digest=consumer_digest,
            checked_at=checked_at,
        )
        verified = proof.get("status") == "verified"
        response_payload = {
            "status": "verified" if verified else "failed",
            "checked_at": checked_at,
            "ttl_seconds": _SAFETY_CHECK_TTL_SECONDS,
            "review_safety_check_token": (
                secrets.token_urlsafe(24) if verified else None
            ),
            "token_status": "current" if verified else "not_current",
            "consumer_chain_digest": consumer_digest,
            "review_exclusion_proof": proof,
        }
        operation = self.store.update_operation(
            operation.id,
            status="succeeded",
            finished_at=_now_iso(),
            result_summary={
                "status": response_payload["status"],
                "token_persisted": (
                    response_payload["review_safety_check_token"] is not None
                ),
                "consumer_chain_digest": consumer_digest,
                "review_exclusion_status": proof.get("status"),
                "response_payload": response_payload,
            },
        )
        stored_payload = operation.result_summary.get("response_payload")
        if isinstance(stored_payload, dict):
            response_payload = stored_payload
        return ReviewSafetyCheckResponse.model_validate(response_payload)

    def rematch_preview(
        self,
        job_id: int,
        payload: RematchPreviewRequest,
        *,
        idempotency_key: str,
    ) -> RematchPreviewResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="rematch_preview",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return RematchPreviewResponse.model_validate(stored_payload)
            if existing_operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message="rematch preview operation with this idempotency key is still running",
                    details={
                        "operation_id": existing_operation.id,
                        "status": existing_operation.status,
                    },
                )
            raise ApiServiceError(
                status_code=409,
                code="idempotency_result_unavailable",
                message=(
                    "rematch preview operation result is unavailable for "
                    "this idempotency key"
                ),
                details={
                    "operation_id": existing_operation.id,
                    "status": existing_operation.status,
                },
            )

        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = _validation_source_files(job, moves, snapshot)
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        if payload.base_review_revision != current_versions["review_revision"]:
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="rematch preview base review revision is stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": {
                        "review_revision": payload.base_review_revision,
                    },
                },
            )

        claim_token = secrets.token_urlsafe(16)
        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="rematch_preview",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions={"review_revision": payload.base_review_revision},
            result_summary={
                "tmdb_id": payload.tmdb_id,
                "status": "running_preview",
                "claim_token": claim_token,
            },
            status="running",
        )
        if operation.result_summary.get("claim_token") != claim_token:
            stored_payload = operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return RematchPreviewResponse.model_validate(stored_payload)
            if operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message="rematch preview operation with this idempotency key is still running",
                    details={"operation_id": operation.id, "status": operation.status},
                )
            raise ApiServiceError(
                status_code=409,
                code="idempotency_result_unavailable",
                message=(
                    "rematch preview operation result is unavailable for "
                    "this idempotency key"
                ),
                details={"operation_id": operation.id, "status": operation.status},
            )
        try:
            preview = self.worker.build_rematch_preview(job, tmdb_id=payload.tmdb_id)
            response_payload = _rematch_preview_payload(
                operation.id,
                payload,
                current_moves=moves,
                preview=preview,
            )
            operation = self.store.update_operation(
                operation.id,
                status="succeeded",
                finished_at=_now_iso(),
                result_summary={
                    "tmdb_id": payload.tmdb_id,
                    "status": "preview_ready",
                    "response_payload": response_payload,
                },
            )
        except Exception as exc:
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": type(exc).__name__,
                    "message": str(exc),
                },
                result_summary={
                    "tmdb_id": payload.tmdb_id,
                    "status": "failed",
                },
            )
            raise
        stored_payload = operation.result_summary.get("response_payload")
        if isinstance(stored_payload, dict):
            response_payload = stored_payload
        return RematchPreviewResponse.model_validate(response_payload)

    def rematch_commit(
        self,
        job_id: int,
        payload: RematchCommitRequest,
        *,
        idempotency_key: str,
    ) -> RematchCommitResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="rematch_commit",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return RematchCommitResponse.model_validate(stored_payload)

        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = _validation_source_files(job, moves, snapshot)
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        if payload.base_review_revision != current_versions["review_revision"]:
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="rematch commit base review revision is stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": {
                        "review_revision": payload.base_review_revision,
                    },
                },
            )
        invalid_moves = [move.id for move in moves if move.status != "proposed"]
        if invalid_moves:
            raise ApiServiceError(
                status_code=422,
                code="invalid_rematch_commit_state",
                message="rematch commit only supports proposed move items",
                details={"non_proposed_move_item_ids": invalid_moves},
            )

        preview_operation = self.store.get_operation(
            job.id,
            payload.preview_operation_id,
        )
        preview_payload = _validated_preview_payload_or_raise(
            preview_operation,
            requested_tmdb_id=payload.tmdb_id,
        )
        planned_moves = _planned_moves_from_preview_payload(preview_payload)
        if not planned_moves:
            raise ApiServiceError(
                status_code=422,
                code="invalid_rematch_preview",
                message="rematch preview contains no planned moves",
                details={"preview_operation_id": payload.preview_operation_id},
            )

        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="rematch_commit",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions={
                "review_revision": payload.base_review_revision,
                "preview_operation_id": payload.preview_operation_id,
            },
            result_summary={
                "tmdb_id": payload.tmdb_id,
                "preview_operation_id": payload.preview_operation_id,
                "status": "committing",
            },
            status="running",
        )

        def rematch_precondition(
            current_job: JobRecord,
            current_moves: list[JobMoveRecord],
            current_snapshot: ReviewSnapshotRecord | None,
            current_operations: list[OperationRecord],
        ) -> None:
            _assert_review_revision_current(
                expected_revision=payload.base_review_revision,
                job=current_job,
                moves=current_moves,
                operations=current_operations,
                snapshot=current_snapshot,
                excluded_operation_ids={operation.id},
                message="rematch commit base review revision is stale",
            )
            _assert_review_revision_current(
                expected_revision=int(preview_payload["base_review_revision"]),
                job=current_job,
                moves=current_moves,
                operations=current_operations,
                snapshot=current_snapshot,
                excluded_operation_ids={
                    operation.id,
                    payload.preview_operation_id,
                },
                message="rematch preview base review revision is stale",
            )

        try:
            selected_info = TmdbReviewInfo.model_validate(
                preview_payload["selected_tmdb_info"]
            )
            _job, _committed_moves, _snapshot, operation = (
                self.store.commit_rematch_plan(
                    job_id=job.id,
                    operation_id=operation.id,
                    series_name=selected_info.title_zh_cn or job.series_name,
                    tmdb_id=payload.tmdb_id,
                    moves=planned_moves,
                    snapshot_payload_factory=(
                        lambda committed_moves: _rematch_commit_snapshot_payload(
                            source_files=source_files,
                            committed_moves=committed_moves,
                            selected_info=selected_info,
                            preview_payload=preview_payload,
                            existing_mapping_attempts=(
                                snapshot.mapping_attempts
                                if snapshot is not None
                                else []
                            ),
                            existing_mapping_evidence=(
                                snapshot.mapping_evidence
                                if snapshot is not None
                                else []
                            ),
                            preview_operation_id=payload.preview_operation_id,
                            tmdb_id=payload.tmdb_id,
                        )
                    ),
                    operation_result_summary_factory=(
                        lambda committed_moves: _rematch_commit_result_summary(
                            operation_id=operation.id,
                            payload=payload,
                            selected_info=selected_info,
                            preview_payload=preview_payload,
                            committed_move_count=len(committed_moves),
                        )
                    ),
                    precondition_check=rematch_precondition,
                )
            )
        except Exception as exc:
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": type(exc).__name__,
                    "message": str(exc),
                },
                result_summary={
                    "tmdb_id": payload.tmdb_id,
                    "preview_operation_id": payload.preview_operation_id,
                    "status": "failed",
                },
            )
            raise
        stored_payload = operation.result_summary.get("response_payload")
        if isinstance(stored_payload, dict):
            response_payload = stored_payload
        return RematchCommitResponse.model_validate(response_payload)

    def corrections(
        self,
        job_id: int,
        payload: CorrectionRequest,
        *,
        idempotency_key: str,
    ) -> CorrectionResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        if not payload.atomic:
            raise ApiServiceError(
                status_code=422,
                code="unsupported_correction_mode",
                message="corrections currently require atomic=true",
            )
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="corrections",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return CorrectionResponse.model_validate(stored_payload)

        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        if snapshot is None:
            raise ApiServiceError(
                status_code=422,
                code="correction_snapshot_missing",
                message="manual correction requires a review snapshot",
            )
        source_files = _validation_source_files(job, moves, snapshot)
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        if payload.base_review_revision != current_versions["review_revision"]:
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="correction base review revision is stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": {
                        "review_revision": payload.base_review_revision,
                    },
                },
            )
        invalid_moves = [move.id for move in moves if move.status != "proposed"]
        if invalid_moves:
            raise ApiServiceError(
                status_code=422,
                code="invalid_correction_state",
                message="corrections only support proposed move items",
                details={"non_proposed_move_item_ids": invalid_moves},
            )

        selected_info = TmdbReviewInfo.model_validate(snapshot.selected_tmdb_info)
        if selected_info.selected_tmdb_id is None:
            raise ApiServiceError(
                status_code=422,
                code="correction_tmdb_missing",
                message="manual correction requires selected TMDB metadata",
            )
        corrected_targets, corrected_links, op_results = _apply_correction_ops(
            payload,
            selected_info=selected_info,
            episode_targets=snapshot.episode_targets,
            subtitle_links=snapshot.subtitle_links,
        )
        planned_moves = _planned_moves_from_corrected_snapshot(
            job,
            selected_info=selected_info,
            episode_targets=corrected_targets,
            subtitle_links=corrected_links,
            current_moves=moves,
        )
        if not planned_moves:
            raise ApiServiceError(
                status_code=422,
                code="invalid_correction",
                message="correction produced no planned moves",
            )

        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="corrections",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions={"review_revision": payload.base_review_revision},
            result_summary={
                "status": "committing",
                "op_count": len(payload.ops),
            },
            status="running",
        )

        def correction_precondition(
            current_job: JobRecord,
            current_moves: list[JobMoveRecord],
            current_snapshot: ReviewSnapshotRecord | None,
            current_operations: list[OperationRecord],
        ) -> None:
            _assert_review_revision_current(
                expected_revision=payload.base_review_revision,
                job=current_job,
                moves=current_moves,
                operations=current_operations,
                snapshot=current_snapshot,
                excluded_operation_ids={operation.id},
                message="correction base review revision is stale",
            )

        try:
            _job, _committed_moves, _snapshot, operation = (
                self.store.commit_rematch_plan(
                    job_id=job.id,
                    operation_id=operation.id,
                    series_name=selected_info.title_zh_cn or job.series_name,
                    tmdb_id=selected_info.selected_tmdb_id,
                    moves=planned_moves,
                    snapshot_payload_factory=(
                        lambda committed_moves: _correction_snapshot_payload(
                            source_files=source_files,
                            committed_moves=committed_moves,
                            selected_info=selected_info,
                            corrected_targets=corrected_targets,
                            corrected_links=corrected_links,
                            existing_mapping_attempts=snapshot.mapping_attempts,
                            existing_mapping_evidence=snapshot.mapping_evidence,
                            op_results=op_results,
                            operation_id=operation.id,
                        )
                    ),
                    operation_result_summary_factory=(
                        lambda committed_moves: _correction_result_summary(
                            operation_id=operation.id,
                            payload=payload,
                            op_results=op_results,
                            committed_move_count=len(committed_moves),
                            diff_summary=_correction_diff_summary(
                                moves,
                                committed_moves,
                            ),
                        )
                    ),
                    precondition_check=correction_precondition,
                    action_label="corrections",
                )
            )
        except Exception as exc:
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": type(exc).__name__,
                    "message": str(exc),
                },
                result_summary={
                    "status": "failed",
                    "op_count": len(payload.ops),
                },
            )
            raise
        stored_payload = operation.result_summary.get("response_payload")
        if isinstance(stored_payload, dict):
            response_payload = stored_payload
        return CorrectionResponse.model_validate(response_payload)

    def validate_apply(
        self,
        job_id: int,
        payload: ApplyValidationRequest,
        *,
        idempotency_key: str,
    ) -> ApplyValidationResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        if payload.scope == "retry_apply":
            raise ApiServiceError(
                status_code=422,
                code="unsupported_validation_scope",
                message="retry_apply validation is not implemented yet",
            )
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="validate",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return ApplyValidationResponse.model_validate(stored_payload)

        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = _validation_source_files(job, moves, snapshot)
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        requested_versions = {
            "mapping_version": payload.base_mapping_version,
            "move_plan_version": payload.base_move_plan_version,
            "move_state_revision": payload.base_move_state_revision,
            "review_revision": payload.base_review_revision,
        }
        stale_versions = {
            key: {
                "current": current_versions[key],
                "requested": requested,
            }
            for key, requested in requested_versions.items()
            if current_versions[key] != requested
        }
        if stale_versions:
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="apply validation base versions are stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": requested_versions,
                    "stale_versions": stale_versions,
                },
            )

        affected = _resolve_apply_validation_scope_or_raise(payload, moves)
        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="validate",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions=requested_versions,
            result_summary={
                "scope": payload.scope,
                "status": "checking",
                "affected_move_item_count": len(affected),
            },
            status="running",
        )
        try:
            response_payload = _apply_validation_response_payload(
                operation=operation,
                job=job,
                moves=affected,
                scope=payload.scope,
                versions=current_versions,
            )
            operation = self.store.update_operation(
                operation.id,
                status="succeeded",
                finished_at=_now_iso(),
                result_summary={
                    "scope": payload.scope,
                    "validation_status": response_payload["status"],
                    "affected_move_item_count": len(affected),
                    "blocking_count": len(response_payload["blocking_items"]),
                    "warning_count": len(response_payload["warning_items"]),
                    "response_payload": response_payload,
                },
            )
        except Exception as exc:
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": type(exc).__name__,
                    "message": str(exc),
                },
                result_summary={
                    "scope": payload.scope,
                    "status": "failed",
                    "affected_move_item_count": len(affected),
                },
            )
            raise
        stored_payload = operation.result_summary.get("response_payload")
        if isinstance(stored_payload, dict):
            response_payload = stored_payload
        return ApplyValidationResponse.model_validate(response_payload)

    def apply_job(
        self,
        job_id: int,
        payload: ApplyRequest,
        *,
        idempotency_key: str,
    ) -> ApplyResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="apply",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return ApplyResponse.model_validate(stored_payload)
            if existing_operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message="apply operation with this idempotency key is still running",
                    details={
                        "operation_id": existing_operation.id,
                        "status": existing_operation.status,
                    },
                )
            return ApplyResponse.model_validate(
                _apply_response_payload_from_operation(existing_operation)
            )

        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = _validation_source_files(job, moves, snapshot)
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        requested_versions = {
            "mapping_version": payload.base_mapping_version,
            "move_plan_version": payload.base_move_plan_version,
            "move_state_revision": payload.base_move_state_revision,
            "review_revision": payload.base_review_revision,
        }
        stale_versions = {
            key: {
                "current": current_versions[key],
                "requested": requested,
            }
            for key, requested in requested_versions.items()
            if current_versions[key] != requested
        }
        if stale_versions:
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="apply base versions are stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": requested_versions,
                    "stale_versions": stale_versions,
                },
            )
        validation_operation = self.store.get_operation(job.id, payload.validation_id)
        validation = _validated_apply_validation_or_raise(validation_operation)
        _assert_validation_matches_apply_request(validation, payload, current_versions)
        affected_ids = set(validation.affected_move_items)
        affected = [move for move in moves if move.id in affected_ids]
        if {move.id for move in affected} != affected_ids:
            raise ApiServiceError(
                status_code=422,
                code="invalid_apply_validation",
                message="validation references unknown move items",
                details={"affected_move_items": sorted(affected_ids)},
            )
        final_validation_payload = _apply_validation_response_payload(
            operation=validation_operation,
            job=job,
            moves=affected,
            scope="all_for_apply",
            versions=current_versions,
        )
        final_validation = ApplyValidationResponse.model_validate(
            final_validation_payload
        )
        if final_validation.status != "valid":
            raise ApiServiceError(
                status_code=422,
                code="apply_final_validation_failed",
                message="apply final validation failed before moving files",
                details=final_validation.model_dump(mode="json"),
            )

        claim_token = secrets.token_urlsafe(16)
        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="apply",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions=requested_versions | {"validation_id": payload.validation_id},
            result_summary={
                "status": "running",
                "validation_id": payload.validation_id,
                "affected_move_item_count": len(affected),
                "claim_token": claim_token,
            },
            status="running",
        )
        if operation.result_summary.get("claim_token") != claim_token:
            stored_payload = operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return ApplyResponse.model_validate(stored_payload)
            if operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message="apply operation with this idempotency key is still running",
                    details={
                        "operation_id": operation.id,
                        "status": operation.status,
                    },
                )
            return ApplyResponse.model_validate(
                _apply_response_payload_from_operation(operation)
            )
        try:
            reservations = self.store.reserve_destinations(
                operation_id=operation.id,
                job_id=job.id,
                reservations=[
                    _destination_reservation_spec(job, move) for move in affected
                ],
            )
            response_payload = _execute_apply_moves(
                store=self.store,
                job=job,
                operation=operation,
                moves=affected,
                reservations=reservations,
                validation_id=payload.validation_id,
            )
        except DestinationReservationConflictError as exc:
            response_payload = _apply_failure_response_payload(
                operation=operation,
                validation_id=payload.validation_id,
                error_code="destination_reserved",
                error_message=str(exc),
                item_result_summary={
                    "succeeded": 0,
                    "failed": len(affected),
                    "total": len(affected),
                },
            )
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": "destination_reserved",
                    "message": str(exc),
                },
                result_summary={
                    "status": "failed",
                    "validation_id": payload.validation_id,
                    "error": str(exc),
                    "response_payload": response_payload,
                },
            )
            raise ApiServiceError(
                status_code=409,
                code="destination_reserved",
                message="destination path is already reserved",
                details={
                    "conflict_code": exc.conflict_code,
                    "conflict_ref": exc.conflict_ref,
                },
            ) from exc
        except Exception as exc:
            self.store.update_destination_reservations_state(
                operation_id=operation.id,
                state="released",
            )
            response_payload = _apply_failure_response_payload(
                operation=operation,
                validation_id=payload.validation_id,
                error_code=(
                    exc.code if isinstance(exc, ApiServiceError) else type(exc).__name__
                ),
                error_message=str(exc),
                item_result_summary={
                    "succeeded": 0,
                    "failed": len(affected),
                    "total": len(affected),
                },
            )
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": (
                        exc.code
                        if isinstance(exc, ApiServiceError)
                        else type(exc).__name__
                    ),
                    "message": str(exc),
                },
                result_summary={
                    "status": "failed",
                    "validation_id": payload.validation_id,
                    "error": str(exc),
                    "response_payload": response_payload,
                },
            )
            raise
        return ApplyResponse.model_validate(response_payload)

    def rollback_eligibility(
        self,
        job_id: int,
        payload: RollbackEligibilityRequest,
        *,
        idempotency_key: str,
    ) -> RollbackEligibilityResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="rollback_eligibility",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return RollbackEligibilityResponse.model_validate(stored_payload)
            if existing_operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message=(
                        "rollback eligibility operation with this idempotency key "
                        "is still running"
                    ),
                    details={
                        "operation_id": existing_operation.id,
                        "status": existing_operation.status,
                    },
                )
            raise ApiServiceError(
                status_code=409,
                code="idempotency_result_unavailable",
                message=(
                    "rollback eligibility operation result is unavailable for "
                    "this idempotency key"
                ),
                details={
                    "operation_id": existing_operation.id,
                    "status": existing_operation.status,
                },
            )
        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = self.model_review_source_files(
            job,
            moves,
            snapshot,
        )
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        if (
            payload.base_review_revision != current_versions["review_revision"]
            or payload.base_move_state_revision
            != current_versions["move_state_revision"]
        ):
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="rollback eligibility base versions are stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": {
                        "review_revision": payload.base_review_revision,
                        "move_state_revision": payload.base_move_state_revision,
                    },
                },
            )
        affected = _resolve_scope_or_raise(payload.scope, moves)
        claim_token = secrets.token_urlsafe(16)
        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="rollback_eligibility",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions={
                "review_revision": payload.base_review_revision,
                "move_state_revision": payload.base_move_state_revision,
            },
            result_summary={
                "mode": payload.mode,
                "affected_move_item_count": 0,
                "eligible_move_item_count": 0,
                "token_persisted": False,
                "claim_token": claim_token,
            },
            status="running",
        )
        if operation.result_summary.get("claim_token") != claim_token:
            stored_payload = operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return RollbackEligibilityResponse.model_validate(stored_payload)
            if operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message=(
                        "rollback eligibility operation with this idempotency key "
                        "is still running"
                    ),
                    details={"operation_id": operation.id, "status": operation.status},
                )
            raise ApiServiceError(
                status_code=409,
                code="idempotency_result_unavailable",
                message=(
                    "rollback eligibility operation result is unavailable for "
                    "this idempotency key"
                ),
                details={"operation_id": operation.id, "status": operation.status},
            )
        try:
            checked_at = _now_iso()
            if payload.mode == "withdraw_to_review":
                response_payload = _withdraw_eligibility_response_payload(
                    job=job,
                    moves=affected,
                    operation_id=operation.id,
                    checked_at=checked_at,
                    checked_move_state_revision=payload.base_move_state_revision,
                    check_token=secrets.token_urlsafe(24),
                )
            else:
                response_payload = _unsupported_restore_eligibility_response_payload(
                    job=job,
                    moves=affected,
                    mode=payload.mode,
                    operation_id=operation.id,
                    checked_at=checked_at,
                    checked_move_state_revision=payload.base_move_state_revision,
                )
            operation = self.store.update_operation(
                operation.id,
                status="succeeded",
                finished_at=_now_iso(),
                result_summary={
                    "mode": payload.mode,
                    "affected_move_item_count": len(affected),
                    "eligible_move_item_count": response_payload[
                        "item_result_summary"
                    ]["eligible"],
                    "token_persisted": response_payload["check_token"] is not None,
                    "response_payload": response_payload,
                },
            )
        except Exception as exc:
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": type(exc).__name__,
                    "message": str(exc),
                },
                result_summary={
                    "mode": payload.mode,
                    "affected_move_item_count": len(affected),
                    "eligible_move_item_count": 0,
                    "token_persisted": False,
                    "error": str(exc),
                },
            )
            raise
        stored_payload = operation.result_summary.get("response_payload")
        if isinstance(stored_payload, dict):
            response_payload = stored_payload
        return RollbackEligibilityResponse.model_validate(response_payload)

    def withdraw_job(
        self,
        job_id: int,
        payload: WithdrawRequest,
        *,
        idempotency_key: str,
    ) -> WithdrawResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="withdraw",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return WithdrawResponse.model_validate(stored_payload)
            if existing_operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message="withdraw operation with this idempotency key is still running",
                    details={
                        "operation_id": existing_operation.id,
                        "status": existing_operation.status,
                    },
                )
            return WithdrawResponse.model_validate(
                _withdraw_response_payload_from_operation(existing_operation)
            )

        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = self.model_review_source_files(job, moves, snapshot)
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        requested_versions = {
            "review_revision": payload.base_review_revision,
            "move_state_revision": payload.base_move_state_revision,
        }
        stale_versions = {
            key: {"current": current_versions[key], "requested": requested}
            for key, requested in requested_versions.items()
            if current_versions[key] != requested
        }
        if stale_versions:
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="withdraw base versions are stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": requested_versions,
                    "stale_versions": stale_versions,
                },
            )
        affected = _resolve_scope_or_raise(payload.scope, moves)
        check_operation, check_payload = _validated_withdraw_check_or_raise(
            operations,
            payload=payload,
            current_versions=current_versions,
        )
        safety_operation, _safety_check = _validated_review_safety_check_or_raise(
            operations,
            token=payload.review_safety_check_token,
            job=job,
            worker=self.worker,
        )
        affected_ids = {move.id for move in affected}
        checked_ids = set(check_payload.affected_move_items)
        if affected_ids != checked_ids:
            raise ApiServiceError(
                status_code=412,
                code="stale_rollback_check",
                message="withdraw scope does not match rollback eligibility check",
                details={
                    "requested_move_item_ids": sorted(affected_ids),
                    "checked_move_item_ids": sorted(checked_ids),
                },
            )
        final_check = _withdraw_eligibility_response_payload(
            job=job,
            moves=affected,
            operation_id=check_operation.id,
            checked_at=_now_iso(),
            checked_move_state_revision=current_versions["move_state_revision"],
            check_token=payload.check_token,
        )
        if final_check["status"] != "eligible":
            raise ApiServiceError(
                status_code=422,
                code="withdraw_final_check_failed",
                message="withdraw final eligibility check failed before moving files",
                details=final_check,
            )

        claim_token = secrets.token_urlsafe(16)
        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="withdraw",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions=requested_versions
            | {
                "rollback_check_operation_id": check_operation.id,
                "review_safety_check_operation_id": safety_operation.id,
            },
            result_summary={
                "status": "running",
                "mode": payload.mode,
                "affected_move_item_count": len(affected),
                "claim_token": claim_token,
                "review_safety_check_operation_id": safety_operation.id,
            },
            status="running",
        )
        if operation.result_summary.get("claim_token") != claim_token:
            stored_payload = operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return WithdrawResponse.model_validate(stored_payload)
            if operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message="withdraw operation with this idempotency key is still running",
                    details={"operation_id": operation.id, "status": operation.status},
                )
            return WithdrawResponse.model_validate(
                _withdraw_response_payload_from_operation(operation)
            )
        destinations = {
            move.id: Path(_review_destination(job, move, check_operation.id))
            for move in affected
        }
        try:
            reservations = self.store.reserve_destinations(
                operation_id=operation.id,
                job_id=job.id,
                reservations=[
                    _withdraw_destination_reservation_spec(
                        job,
                        move,
                        destinations[move.id],
                    )
                    for move in affected
                ],
            )
            response_payload = _execute_withdraw_moves(
                store=self.store,
                job=job,
                operation=operation,
                moves=affected,
                destinations=destinations,
                reservations=reservations,
            )
        except DestinationReservationConflictError as exc:
            response_payload = _withdraw_failure_response_payload(
                operation=operation,
                error_code="destination_reserved",
                error_message=str(exc),
                total_count=len(affected),
            )
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={"code": "destination_reserved", "message": str(exc)},
                result_summary={
                    "status": "failed",
                    "mode": payload.mode,
                    "error": str(exc),
                    "response_payload": response_payload,
                },
            )
            raise ApiServiceError(
                status_code=409,
                code="destination_reserved",
                message="review destination path is already reserved",
                details={
                    "conflict_code": exc.conflict_code,
                    "conflict_ref": exc.conflict_ref,
                },
            ) from exc
        except Exception as exc:
            self.store.update_destination_reservations_state(
                operation_id=operation.id,
                state="released",
            )
            response_payload = _withdraw_failure_response_payload(
                operation=operation,
                error_code=(
                    exc.code if isinstance(exc, ApiServiceError) else type(exc).__name__
                ),
                error_message=str(exc),
                total_count=len(affected),
            )
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": (
                        exc.code
                        if isinstance(exc, ApiServiceError)
                        else type(exc).__name__
                    ),
                    "message": str(exc),
                },
                result_summary={
                    "status": "failed",
                    "mode": payload.mode,
                    "error": str(exc),
                    "response_payload": response_payload,
                },
            )
            raise
        return WithdrawResponse.model_validate(response_payload)

    def reopen_from_review(
        self,
        job_id: int,
        payload: ReopenFromReviewRequest,
        *,
        idempotency_key: str,
    ) -> ReopenFromReviewResponse | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        request_hash = _payload_hash(payload.model_dump(mode="json"))
        existing_operation = self.store.get_operation_by_idempotency(
            job_id=job.id,
            operation_type="reopen_from_review",
            idempotency_key=idempotency_key,
        )
        if existing_operation is not None:
            if existing_operation.request_payload_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with a different payload"
                )
            stored_payload = existing_operation.result_summary.get(
                "response_payload"
            )
            if isinstance(stored_payload, dict):
                return ReopenFromReviewResponse.model_validate(stored_payload)
            if existing_operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message=(
                        "reopen from review operation with this idempotency key "
                        "is still running"
                    ),
                    details={
                        "operation_id": existing_operation.id,
                        "status": existing_operation.status,
                    },
                )
            return ReopenFromReviewResponse.model_validate(
                _reopen_response_payload_from_operation(existing_operation)
            )

        moves = self.store.list_job_moves(job.id)
        operations = self.store.list_operations(job.id)
        snapshot = self.store.get_review_snapshot(job.id)
        source_files = _validation_source_files(job, moves, snapshot)
        current_versions = _job_versions(
            job,
            moves,
            operations,
            source_files,
            snapshot=_snapshot_revision_payload(snapshot),
        )
        requested_versions = {
            "review_revision": payload.base_review_revision,
            "move_state_revision": payload.base_move_state_revision,
        }
        stale_versions = {
            key: {"current": current_versions[key], "requested": requested}
            for key, requested in requested_versions.items()
            if current_versions[key] != requested
        }
        if stale_versions:
            raise ApiServiceError(
                status_code=412,
                code="stale_versions",
                message="reopen from review base versions are stale",
                details={
                    "current_versions": current_versions,
                    "requested_versions": requested_versions,
                    "stale_versions": stale_versions,
                },
            )
        affected = _resolve_reopen_scope_or_raise(payload.scope, moves)
        if not affected:
            raise ApiServiceError(
                status_code=422,
                code="no_reopenable_items",
                message="no withdrawn move items were selected",
                details={"scope": payload.scope.model_dump(mode="json")},
            )
        withdraw_results = self.store.list_latest_successful_withdraw_item_results(
            job_id=job.id,
            move_item_ids=[move.id for move in affected],
        )
        _reopen_review_specs_or_raise(job, affected, withdraw_results)
        safety_operation, _safety_check = _validated_review_safety_check_or_raise(
            operations,
            token=payload.review_safety_check_token,
            job=job,
            worker=self.worker,
        )

        claim_token = secrets.token_urlsafe(16)
        operation = self.store.create_operation(
            job_id=job.id,
            operation_type="reopen_from_review",
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            base_versions=requested_versions
            | {"review_safety_check_operation_id": safety_operation.id},
            result_summary={
                "status": "running",
                "affected_move_item_count": len(affected),
                "claim_token": claim_token,
                "review_safety_check_operation_id": safety_operation.id,
            },
            status="running",
        )
        if operation.result_summary.get("claim_token") != claim_token:
            stored_payload = operation.result_summary.get("response_payload")
            if isinstance(stored_payload, dict):
                return ReopenFromReviewResponse.model_validate(stored_payload)
            if operation.status in {"queued", "running"}:
                raise ApiServiceError(
                    status_code=409,
                    code="operation_in_progress",
                    message=(
                        "reopen from review operation with this idempotency key "
                        "is still running"
                    ),
                    details={"operation_id": operation.id, "status": operation.status},
                )
            return ReopenFromReviewResponse.model_validate(
                _reopen_response_payload_from_operation(operation)
            )
        try:
            final_moves = self.store.list_job_moves(job.id)
            final_moves_by_id = {move.id: move for move in final_moves}
            final_affected = [
                final_moves_by_id[move.id]
                for move in affected
                if move.id in final_moves_by_id
            ]
            final_withdraw_results = (
                self.store.list_latest_successful_withdraw_item_results(
                    job_id=job.id,
                    move_item_ids=[move.id for move in affected],
                )
            )
            final_specs = _reopen_review_specs_or_raise(
                job,
                final_affected,
                final_withdraw_results,
                expected_move_item_ids=[move.id for move in affected],
            )
            response_payload = _execute_reopen_from_review(
                store=self.store,
                job=job,
                operation=operation,
                specs=final_specs,
            )
        except Exception as exc:
            response_payload = _reopen_failure_response_payload(
                operation=operation,
                error_code=(
                    exc.code if isinstance(exc, ApiServiceError) else type(exc).__name__
                ),
                error_message=str(exc),
                total_count=len(affected),
            )
            self.store.update_operation(
                operation.id,
                status="failed",
                finished_at=_now_iso(),
                error={
                    "code": (
                        exc.code
                        if isinstance(exc, ApiServiceError)
                        else type(exc).__name__
                    ),
                    "message": str(exc),
                },
                result_summary={
                    "status": "failed",
                    "error": str(exc),
                    "response_payload": response_payload,
                },
            )
            raise
        return ReopenFromReviewResponse.model_validate(response_payload)

    @staticmethod
    def model_job(job: JobRecord) -> JobResponse:
        return JobResponse(
            id=job.id,
            series_name=job.series_name,
            watch_root_key=job.watch_root_key,
            source_kind=job.source_kind,
            status=job.status,
            tmdb_id=job.tmdb_id,
            video_moves_count=job.video_moves_count,
            subtitle_moves_count=job.subtitle_moves_count,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error_stage=job.error_stage,
            error_message=job.error_message,
        )

    @staticmethod
    def model_job_request(item: JobRequestRecord) -> JobRequestResponse:
        return JobRequestResponse(
            id=item.id,
            action=item.kind,
            status=item.status,
            job_id=item.target_job_id,
            created_at=item.created_at,
            updated_at=item.updated_at,
            started_at=item.started_at,
            finished_at=item.finished_at,
            error_message=item.error_message,
        )

    @staticmethod
    def model_status_item(job: JobRecord) -> StatusItem:
        return StatusItem(
            job_id=job.id,
            series_name=job.series_name,
            watch_root_key=job.watch_root_key,
            status=job.status,
            updated_at=job.updated_at,
            tmdb_id=job.tmdb_id,
            video_moves_count=job.video_moves_count,
            subtitle_moves_count=job.subtitle_moves_count,
            error_stage=job.error_stage,
            error_message=job.error_message,
        )

    @staticmethod
    def model_review_move_item(job: JobRecord, move: JobMoveRecord) -> ReviewMoveItem:
        current_path = move.dst if move.status == "moved" else move.src
        current_path_kind = "output" if move.status == "moved" else "source"
        return ReviewMoveItem(
            id=move.id,
            source_id=move.src_id,
            kind=move.kind,
            source_path=_validation_path_ref(
                move.src,
                root_kind="source",
                root=Path(job.series_dir),
            ),
            planned_destination_path=_validation_path_ref(
                move.dst,
                root_kind="output",
                root=Path(job.output_root),
            ),
            current_path=_validation_path_ref(
                current_path,
                root_kind=current_path_kind,
                root=(
                    Path(job.output_root)
                    if current_path_kind == "output"
                    else Path(job.series_dir)
                ),
            ),
            status=move.status,
            error_message=move.error_message,
            applied_at=move.applied_at,
        )

    @staticmethod
    def model_review_source_files(
        job: JobRecord,
        moves: list[JobMoveRecord],
        snapshot: ReviewSnapshotRecord | None = None,
    ) -> list[ReviewSourceFile]:
        if snapshot is not None:
            return [
                ReviewSourceFile.model_validate(item)
                for item in snapshot.source_files
            ]
        items = _validation_source_files(job, moves, None)
        items.sort(key=lambda item: (item.original_path.display_path, item.id))
        return items

    @staticmethod
    def model_operation(item: OperationRecord) -> OperationResponse:
        return OperationResponse(
            id=item.id,
            type=item.operation_type,
            job_id=item.job_id,
            status=item.status,
            request_payload_hash=item.request_payload_hash,
            base_versions=item.base_versions,
            progress=item.progress,
            result_summary=_public_operation_result_summary(item.result_summary),
            error=item.error,
            created_at=item.created_at,
            started_at=item.started_at,
            finished_at=item.finished_at,
            poll_after_ms=item.poll_after_ms,
        )

    @staticmethod
    def model_operation_item_result(
        item: OperationItemResultRecord,
    ) -> OperationItemResultResponse:
        return OperationItemResultResponse(
            id=item.id,
            operation_id=item.operation_id,
            subject_kind=item.subject_kind,
            subject_id=item.subject_id,
            status=item.status,
            path_event_id=item.path_event_id,
            reservation_id=item.reservation_id,
            from_path=item.from_path,
            to_path=item.to_path,
            identity_check=item.identity_check,
            error_code=item.error_code,
            error_message=item.error_message,
            retryable=item.retryable,
            retry_block_reason=item.retry_block_reason,
            result_revision=item.result_revision,
            created_at=item.created_at,
        )

    def _rollback_capability_hints(
        self,
        job: JobRecord,
        moves: list[JobMoveRecord],
    ) -> RollbackCapabilityHints:
        return RollbackCapabilityHints(
            has_moved_items=any(move.status == "moved" for move in moves),
            has_archive_records=job.archive_path is not None,
            has_review_records=any(move.status == "withdrawn" for move in moves),
            needs_safety_check=True,
        )

    def _last_rollback_check_summary(self, job_id: int) -> LastRollbackCheckSummary:
        return self._last_rollback_check_summary_from_operations(
            self.store.list_operations(job_id)
        )

    @staticmethod
    def _last_rollback_check_summary_from_operations(
        operations: list[OperationRecord],
    ) -> LastRollbackCheckSummary:
        for operation in operations:
            if operation.operation_type == "rollback_eligibility":
                mode = operation.result_summary.get("mode")
                return LastRollbackCheckSummary(
                    status="checked_no_token",
                    last_checked_at=operation.finished_at or operation.created_at,
                    mode=mode if isinstance(mode, str) else None,
                )
        return LastRollbackCheckSummary(status="unchecked")


def _job_versions(
    job: JobRecord,
    moves: list[JobMoveRecord],
    operations: list[OperationRecord],
    source_files: list[ReviewSourceFile] | None = None,
    snapshot: dict[str, object] | None = None,
) -> dict[str, int]:
    mapping_version = _stable_revision(
        {
            "tmdb_id": job.tmdb_id,
            "series_name": job.series_name,
            "mapped_sources": [
                {
                    "source_id": move.src_id,
                    "kind": move.kind,
                    "destination": move.dst,
                }
                for move in moves
            ],
        }
    )
    move_plan_version = _stable_revision(
        [
            {
                "id": move.id,
                "source_id": move.src_id,
                "kind": move.kind,
                "source": move.src,
                "destination": move.dst,
            }
            for move in moves
        ]
    )
    move_state_revision = _stable_revision(
        [
            {
                "id": move.id,
                "status": move.status,
                "current_path": move.dst if move.status == "moved" else move.src,
                "error_message": move.error_message,
                "applied_at": move.applied_at,
            }
            for move in moves
        ]
    )
    review_revision = _stable_revision(
        {
            "job": {
                "id": job.id,
                "status": job.status,
                "updated_at": job.updated_at,
                "archive_path": job.archive_path,
                "fail_path": job.fail_path,
            },
            "mapping_version": mapping_version,
            "move_plan_version": move_plan_version,
            "move_state_revision": move_state_revision,
            "operations": [
                {
                    "id": operation.id,
                    "type": operation.operation_type,
                    "status": operation.status,
                    "finished_at": operation.finished_at,
                    "result_summary": operation.result_summary,
                }
                for operation in operations
            ],
            "source_files": (
                [item.model_dump(mode="json") for item in source_files]
                if source_files is not None
                else []
            ),
            "snapshot": snapshot or {},
        }
    )
    return {
        "mapping_version": mapping_version,
        "move_plan_version": move_plan_version,
        "move_state_revision": move_state_revision,
        "review_revision": review_revision,
    }


def _consumer_chain_digest(worker: AninamerWorker) -> str:
    return _payload_hash(
        {
            "watch_roots": [
                {
                    "key": item.key,
                    "input_root": _safe_path_string(item.input_root),
                    "output_root": _safe_path_string(item.output_root),
                }
                for item in worker.config.watch_roots
            ],
            "excluded_dir_names": sorted(MONITOR_EXCLUDED_DIR_NAMES),
            "consumer_kinds": ["monitor_discovery", "monitor_snapshot"],
        }
    )


def _review_exclusion_proof(
    *,
    job: JobRecord,
    worker: AninamerWorker,
    review_root: Path,
    consumer_chain_digest: str,
    checked_at: str,
) -> dict[str, object]:
    canonical_ref = _validation_path_ref(
        review_root,
        root_kind="review",
        root=review_root.parent,
    ).model_dump(mode="json")
    base: dict[str, object] = {
        "status": "failed",
        "checked_at": checked_at,
        "ttl_seconds": _SAFETY_CHECK_TTL_SECONDS,
        "canonical_review_root": canonical_ref,
        "canonical_review_root_key": _safe_path_key(review_root),
        "consumer_chain_digest": consumer_chain_digest,
        "matched_exclusion_rule": f"dirname:{MONITOR_REVIEW_DIR}",
        "required_excluded_dir_names": sorted(MONITOR_EXCLUDED_DIR_NAMES),
        "covered_consumers": [],
    }
    if _path_has_env_component(review_root):
        return {
            **base,
            "reason": "sensitive_env_path",
            "message": "review root contains a sensitive .env path component",
        }
    excluded_names = {name.casefold() for name in MONITOR_EXCLUDED_DIR_NAMES}
    if MONITOR_REVIEW_DIR.casefold() not in excluded_names:
        return {
            **base,
            "reason": "review_rule_not_configured",
            "message": "monitor exclusion rules do not include review",
        }
    if review_root.name.casefold() != MONITOR_REVIEW_DIR.casefold():
        return {
            **base,
            "reason": "unexpected_review_root_name",
            "message": "review root must be named review",
        }
    try:
        review_parent = review_root.parent.resolve(strict=False)
    except OSError as exc:
        return {
            **base,
            "reason": "review_parent_unavailable",
            "message": f"{type(exc).__name__}: {exc}",
        }
    matching_roots: list[dict[str, object]] = []
    expected_watch_root: dict[str, object] | None = None
    review_parent_key = _safe_path_key(review_parent)
    for item in worker.config.watch_roots:
        input_root = _safe_path_string(item.input_root)
        record = {
            "kind": "monitor",
            "watch_root_key": item.key,
            "input_root": input_root,
            "exclusion_rule": f"dirname:{MONITOR_REVIEW_DIR}",
        }
        if item.key == job.watch_root_key:
            expected_watch_root = record
        if _safe_path_key(item.input_root) == review_parent_key:
            matching_roots.append(record)
    if expected_watch_root is None:
        return {
            **base,
            "reason": "watch_root_not_configured",
            "message": "job watch root is not configured",
            "watch_root_key": job.watch_root_key,
        }
    if not any(item["watch_root_key"] == job.watch_root_key for item in matching_roots):
        return {
            **base,
            "reason": "review_root_not_sibling_of_watch_root",
            "message": "review root must be directly under the job watch input root",
            "watch_root_key": job.watch_root_key,
            "expected_watch_root": expected_watch_root,
            "review_parent": str(review_parent),
        }
    identity: dict[str, object] = {"exists": False}
    try:
        stat_result = review_root.stat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        return {
            **base,
            "reason": "review_root_unavailable",
            "message": f"{type(exc).__name__}: {exc}",
        }
    else:
        identity = {
            "exists": True,
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
            "mtime_ns": stat_result.st_mtime_ns,
        }
    return {
        **base,
        "status": "verified",
        "reason": None,
        "message": "review root is covered by monitor dirname exclusion",
        "covered_consumers": matching_roots,
        "review_root_identity": identity,
    }


def _operation_has_active_review_safety_lease(operation: OperationRecord) -> bool:
    if operation.status not in {"queued", "running"}:
        return False
    lease_id = operation.base_versions.get("review_safety_check_operation_id")
    return isinstance(lease_id, int)


def _active_review_safety_lease_summary(
    operation: OperationRecord,
) -> dict[str, object]:
    return {
        "operation_id": operation.id,
        "operation_type": operation.operation_type,
        "status": operation.status,
        "review_safety_check_operation_id": operation.base_versions.get(
            "review_safety_check_operation_id"
        ),
    }


def _validated_review_safety_check_or_raise(
    operations: list[OperationRecord],
    *,
    token: str,
    job: JobRecord,
    worker: AninamerWorker,
) -> tuple[OperationRecord, ReviewSafetyCheckResponse]:
    current_digest = _consumer_chain_digest(worker)
    current_proof = _review_exclusion_proof(
        job=job,
        worker=worker,
        review_root=_review_root(job),
        consumer_chain_digest=current_digest,
        checked_at=_now_iso(),
    )
    if current_proof.get("status") != "verified":
        raise ApiServiceError(
            status_code=422,
            code="review_safety_not_verified",
            message="review safety proof is not currently verified",
            details={"review_exclusion_proof": current_proof},
        )
    current_root_key = current_proof.get("canonical_review_root_key")
    for operation in operations:
        if operation.operation_type != "review_safety_check":
            continue
        if operation.status != "succeeded":
            continue
        stored_payload = operation.result_summary.get("response_payload")
        if not isinstance(stored_payload, dict):
            continue
        check = ReviewSafetyCheckResponse.model_validate(stored_payload)
        if check.review_safety_check_token != token:
            continue
        if check.status != "verified" or check.token_status != "current":
            raise ApiServiceError(
                status_code=422,
                code="invalid_review_safety_check",
                message="review safety check is not currently verified",
                details=check.model_dump(mode="json"),
            )
        if not _review_safety_check_token_time_current(check):
            raise ApiServiceError(
                status_code=412,
                code="stale_review_safety_check",
                message="review safety check token has expired",
                details=check.model_dump(mode="json"),
            )
        checked_root_key = check.review_exclusion_proof.get(
            "canonical_review_root_key"
        )
        if (
            check.consumer_chain_digest != current_digest
            or checked_root_key != current_root_key
        ):
            raise ApiServiceError(
                status_code=412,
                code="stale_review_safety_check",
                message="review safety check proof is stale",
                details={
                    "checked_consumer_chain_digest": check.consumer_chain_digest,
                    "current_consumer_chain_digest": current_digest,
                    "checked_review_root_key": checked_root_key,
                    "current_review_root_key": current_root_key,
                },
            )
        return operation, check
    raise ApiServiceError(
        status_code=422,
        code="invalid_review_safety_check",
        message="review safety check token was not found",
    )


def _review_safety_check_token_time_current(
    check: ReviewSafetyCheckResponse,
) -> bool:
    try:
        checked_at = datetime.fromisoformat(check.checked_at)
    except ValueError:
        return False
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) <= checked_at + timedelta(
        seconds=check.ttl_seconds
    )


def _public_operation_result_summary(value: object) -> dict[str, object]:
    redacted = _redact_operation_secret_fields(value)
    return redacted if isinstance(redacted, dict) else {}


def _redact_operation_secret_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: None
            if key
            in {
                "check_token",
                "claim_token",
                "review_safety_check_token",
                "source_gate_check_token",
            }
            else _redact_operation_secret_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_operation_secret_fields(item) for item in value]
    return value


def _stable_revision(payload: object) -> int:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode()
    return int(hashlib.sha256(encoded).hexdigest()[:8], 16) + 1


def _snapshot_revision_payload(
    snapshot: ReviewSnapshotRecord | None,
) -> dict[str, object]:
    if snapshot is None:
        return {}
    return {
        "selected_tmdb_info": snapshot.selected_tmdb_info,
        "episode_targets": snapshot.episode_targets,
        "subtitle_links": snapshot.subtitle_links,
        "mapping_attempts": snapshot.mapping_attempts,
        "mapping_evidence": snapshot.mapping_evidence,
        "path_event_summary": snapshot.path_event_summary,
        "validation_summary": snapshot.validation_summary,
    }


def _assert_review_revision_current(
    *,
    expected_revision: int,
    job: JobRecord,
    moves: list[JobMoveRecord],
    operations: list[OperationRecord],
    snapshot: ReviewSnapshotRecord | None,
    excluded_operation_ids: set[int],
    message: str,
) -> None:
    revision_operations = [
        operation
        for operation in operations
        if operation.id not in excluded_operation_ids
    ]
    source_files = _validation_source_files(job, moves, snapshot)
    current_versions = _job_versions(
        job,
        moves,
        revision_operations,
        source_files,
        snapshot=_snapshot_revision_payload(snapshot),
    )
    if expected_revision != current_versions["review_revision"]:
        raise ApiServiceError(
            status_code=412,
            code="stale_versions",
            message=message,
            details={
                "current_versions": current_versions,
                "requested_versions": {
                    "review_revision": expected_revision,
                },
            },
        )


def _rematch_preview_payload(
    operation_id: int,
    request: RematchPreviewRequest,
    *,
    current_moves: list[JobMoveRecord],
    preview: PlanBuildResult,
) -> dict[str, object]:
    episode_targets, subtitle_links = _preview_targets_and_links(preview)
    diff_summary, diff_items = _rematch_diff(current_moves, list(preview.plan.moves))
    return {
        "operation_id": operation_id,
        "status": "preview_ready",
        "tmdb_id": request.tmdb_id,
        "base_review_revision": request.base_review_revision,
        "selected_tmdb_info": _preview_tmdb_info(preview).model_dump(mode="json"),
        "episode_targets": [
            target.model_dump(mode="json") for target in episode_targets
        ],
        "subtitle_links": [link.model_dump(mode="json") for link in subtitle_links],
        "move_items": _preview_move_items(current_moves, list(preview.plan.moves)),
        "mapping_evidence": _preview_mapping_evidence(preview),
        "validation_summary": {
            "status": "preview_only",
            "message": "rematch preview does not modify moves or files",
            "season_episode_counts": preview.season_episode_counts,
        },
        "diff_summary": diff_summary,
        "diff_items": diff_items,
    }


def _validated_preview_payload_or_raise(
    operation: OperationRecord | None,
    *,
    requested_tmdb_id: int,
) -> dict[str, object]:
    if operation is None:
        raise ApiServiceError(
            status_code=422,
            code="invalid_rematch_preview",
            message="rematch preview operation was not found",
        )
    if operation.operation_type != "rematch_preview" or operation.status != "succeeded":
        raise ApiServiceError(
            status_code=422,
            code="invalid_rematch_preview",
            message="rematch preview operation is not a succeeded preview",
            details={
                "operation_id": operation.id,
                "operation_type": operation.operation_type,
                "status": operation.status,
            },
        )
    stored_payload = operation.result_summary.get("response_payload")
    if not isinstance(stored_payload, dict):
        raise ApiServiceError(
            status_code=422,
            code="invalid_rematch_preview",
            message="rematch preview operation does not contain a response payload",
            details={"operation_id": operation.id},
        )
    preview = RematchPreviewResponse.model_validate(stored_payload)
    if preview.tmdb_id != requested_tmdb_id:
        raise ApiServiceError(
            status_code=422,
            code="invalid_rematch_preview",
            message="rematch preview tmdb id does not match the commit request",
            details={
                "preview_operation_id": operation.id,
                "preview_tmdb_id": preview.tmdb_id,
                "requested_tmdb_id": requested_tmdb_id,
            },
        )
    return preview.model_dump(mode="json")


def _rematch_commit_snapshot_payload(
    *,
    source_files: list[ReviewSourceFile],
    committed_moves: list[JobMoveRecord],
    selected_info: TmdbReviewInfo,
    preview_payload: dict[str, object],
    existing_mapping_attempts: list[dict[str, object]],
    existing_mapping_evidence: list[dict[str, object]],
    preview_operation_id: int,
    tmdb_id: int,
) -> dict[str, object]:
    committed_source_files = _committed_source_files(
        source_files,
        committed_moves,
    )
    committed_targets, committed_links = _committed_targets_and_links(
        preview_payload,
        committed_moves,
    )
    validation_summary = preview_payload.get("validation_summary", {})
    if not isinstance(validation_summary, dict):
        validation_summary = {}
    return {
        "source_files": [
            item.model_dump(mode="json") for item in committed_source_files
        ],
        "selected_tmdb_info": selected_info.model_dump(mode="json"),
        "episode_targets": committed_targets,
        "subtitle_links": committed_links,
        "mapping_attempts": _append_rematch_attempt(
            existing_mapping_attempts,
            tmdb_id=tmdb_id,
            mapped_video_count=len(committed_targets),
            mapped_subtitle_count=len(committed_links),
        ),
        "mapping_evidence": _append_rematch_evidence(
            existing_mapping_evidence,
            preview_payload,
        ),
        "path_event_summary": {
            "status": "not_started",
            "pending_count": 0,
            "incomplete_count": 0,
        },
        "validation_summary": {
            "status": "planned",
            "rematched_from_operation_id": preview_operation_id,
            "season_episode_counts": validation_summary.get(
                "season_episode_counts",
                {},
            ),
        },
    }


def _rematch_commit_result_summary(
    *,
    operation_id: int,
    payload: RematchCommitRequest,
    selected_info: TmdbReviewInfo,
    preview_payload: dict[str, object],
    committed_move_count: int,
) -> dict[str, object]:
    response_payload = {
        "operation_id": operation_id,
        "status": "committed",
        "preview_operation_id": payload.preview_operation_id,
        "tmdb_id": payload.tmdb_id,
        "base_review_revision": payload.base_review_revision,
        "selected_tmdb_info": selected_info.model_dump(mode="json"),
        "diff_summary": preview_payload.get("diff_summary", {}),
        "committed_move_item_count": committed_move_count,
    }
    return {
        "tmdb_id": payload.tmdb_id,
        "preview_operation_id": payload.preview_operation_id,
        "status": "committed",
        "committed_move_item_count": committed_move_count,
        "response_payload": response_payload,
    }


def _apply_correction_ops(
    payload: CorrectionRequest,
    *,
    selected_info: TmdbReviewInfo,
    episode_targets: list[dict[str, object]],
    subtitle_links: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    targets = [
        ReviewEpisodeTarget.model_validate(item).model_dump(mode="json")
        for item in episode_targets
    ]
    links = [
        ReviewSubtitleLink.model_validate(item).model_dump(mode="json")
        for item in subtitle_links
    ]
    targets_by_id = {int(item["id"]): item for item in targets}
    links_by_id = {int(item["id"]): item for item in links}
    season_counts = _season_episode_counts_from_tmdb_info(selected_info)
    op_results: list[dict[str, object]] = []

    for index, op in enumerate(payload.ops, start=1):
        if op.op == "update_episode":
            target_id = int(op.episode_target_id or 0)
            target = targets_by_id.get(target_id)
            if target is None:
                raise ApiServiceError(
                    status_code=422,
                    code="invalid_correction",
                    message="episode target was not found",
                    details={"episode_target_id": target_id},
                )
            if target.get("status") != "proposed":
                raise ApiServiceError(
                    status_code=422,
                    code="invalid_correction_state",
                    message="only proposed episode targets can be corrected",
                    details={
                        "episode_target_id": target_id,
                        "status": target.get("status"),
                    },
                )
            season = int(op.season or 0)
            episode_start = int(op.episode_start or 0)
            episode_end = int(op.episode_end or 0)
            _validate_correction_episode_range(
                season=season,
                episode_start=episode_start,
                episode_end=episode_end,
                season_counts=season_counts,
            )
            before = {
                "season": target["season"],
                "episode_start": target["episode_start"],
                "episode_end": target["episode_end"],
            }
            target.update(
                {
                    "season": season,
                    "episode_start": episode_start,
                    "episode_end": episode_end,
                    "status": "proposed",
                    "manual": True,
                    "created_by": "user",
                }
            )
            op_results.append(
                {
                    "index": index,
                    "op": op.op,
                    "status": "applied",
                    "episode_target_id": target_id,
                    "before": before,
                    "after": {
                        "season": season,
                        "episode_start": episode_start,
                        "episode_end": episode_end,
                    },
                }
            )
            continue

        link_id = int(op.subtitle_link_id or 0)
        link = links_by_id.get(link_id)
        if link is None:
            raise ApiServiceError(
                status_code=422,
                code="invalid_correction",
                message="subtitle link was not found",
                details={"subtitle_link_id": link_id},
            )
        if link.get("status") != "proposed":
            raise ApiServiceError(
                status_code=422,
                code="invalid_correction_state",
                message="only proposed subtitle links can be corrected",
                details={"subtitle_link_id": link_id, "status": link.get("status")},
            )
        before_variant = _subtitle_variant_for_corrected_link(link)
        variant_override = None if op.variant == "detected" else op.variant
        link.update(
            {
                "variant_override": variant_override,
                "status": "proposed",
                "created_by": "user",
            }
        )
        op_results.append(
            {
                "index": index,
                "op": op.op,
                "status": "applied",
                "subtitle_link_id": link_id,
                "before": {"variant": before_variant},
                "after": {
                    "variant": _subtitle_variant_for_corrected_link(link),
                    "variant_override": variant_override,
                },
            }
        )

    return targets, links, op_results


def _season_episode_counts_from_tmdb_info(
    selected_info: TmdbReviewInfo,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in selected_info.seasons:
        season_number = item.get("season_number")
        episode_count = item.get("episode_count")
        if isinstance(season_number, int) and isinstance(episode_count, int):
            counts[season_number] = max(0, episode_count)
    if selected_info.specials:
        specials_count = 0
        for item in selected_info.specials:
            episode_number = item.get("episode_number")
            if isinstance(episode_number, int):
                specials_count = max(specials_count, episode_number)
        counts[0] = max(counts.get(0, 0), specials_count, len(selected_info.specials))
    return counts


def _validate_correction_episode_range(
    *,
    season: int,
    episode_start: int,
    episode_end: int,
    season_counts: dict[int, int],
) -> None:
    max_episode = season_counts.get(season)
    if max_episode is None or max_episode <= 0:
        raise ApiServiceError(
            status_code=422,
            code="invalid_correction",
            message="season is not available in selected TMDB metadata",
            details={"season": season, "available_seasons": sorted(season_counts)},
        )
    if episode_start < 1 or episode_end < episode_start or episode_end > max_episode:
        raise ApiServiceError(
            status_code=422,
            code="invalid_correction",
            message="episode range is outside selected TMDB metadata bounds",
            details={
                "season": season,
                "episode_start": episode_start,
                "episode_end": episode_end,
                "max_episode": max_episode,
            },
        )


def _planned_moves_from_corrected_snapshot(
    job: JobRecord,
    *,
    selected_info: TmdbReviewInfo,
    episode_targets: list[dict[str, object]],
    subtitle_links: list[dict[str, object]],
    current_moves: list[JobMoveRecord],
) -> list[PlannedMove]:
    tmdb_id = selected_info.selected_tmdb_id
    if tmdb_id is None:
        raise ApiServiceError(
            status_code=422,
            code="correction_tmdb_missing",
            message="manual correction requires selected TMDB metadata",
        )
    series_name = selected_info.title_zh_cn or job.series_name
    output_root = Path(job.output_root).resolve(strict=False)
    series_folder = format_series_root_folder(series_name, selected_info.year, tmdb_id)
    moves_by_key = {(move.src_id, move.kind): move for move in current_moves}
    links_by_target: dict[int, list[dict[str, object]]] = {}
    for link in subtitle_links:
        target_id = link.get("episode_target_id")
        if isinstance(target_id, int):
            links_by_target.setdefault(target_id, []).append(link)

    planned_moves: list[PlannedMove] = []
    used_dests: set[str] = set()
    for target in sorted(episode_targets, key=lambda item: int(item["id"])):
        target_id = int(target["id"])
        video_source_id = int(target["video_source_id"])
        video_move = moves_by_key.get((video_source_id, "video"))
        if video_move is None:
            raise ApiServiceError(
                status_code=422,
                code="invalid_correction",
                message="episode target does not have a proposed video move item",
                details={
                    "episode_target_id": target_id,
                    "video_source_id": video_source_id,
                },
            )
        season = int(target["season"])
        episode_start = int(target["episode_start"])
        episode_end = int(target["episode_end"])
        season_folder = format_season_folder(season)
        episode_base = format_episode_base(
            series_name,
            season,
            episode_start,
            episode_end,
        )
        raw_video_dst = (
            output_root
            / series_folder
            / season_folder
            / f"{episode_base}{Path(video_move.src).suffix}"
        )
        video_dst = _ensure_correction_destination(raw_video_dst, output_root)
        video_key = _path_key(video_dst)
        if video_key in used_dests:
            raise ApiServiceError(
                status_code=422,
                code="invalid_correction",
                message="manual correction produces a video destination collision",
                details={
                    "episode_target_id": target_id,
                    "destination": str(video_dst),
                },
            )
        used_dests.add(video_key)
        planned_moves.append(
            PlannedMove(
                src=Path(video_move.src),
                dst=video_dst,
                kind="video",
                src_id=video_source_id,
            )
        )

        for link in sorted(
            links_by_target.get(target_id, []),
            key=lambda item: int(item["id"]),
        ):
            subtitle_source_id = int(link["subtitle_source_id"])
            subtitle_move = moves_by_key.get((subtitle_source_id, "subtitle"))
            if subtitle_move is None:
                raise ApiServiceError(
                    status_code=422,
                    code="invalid_correction",
                    message="subtitle link does not have a proposed subtitle move item",
                    details={
                        "subtitle_link_id": link.get("id"),
                        "subtitle_source_id": subtitle_source_id,
                    },
                )
            variant = _subtitle_variant_for_corrected_link(link)
            subtitle_dst = (
                output_root
                / series_folder
                / season_folder
                / f"{episode_base}.{variant}{Path(subtitle_move.src).suffix}"
            ).resolve(strict=False)
            subtitle_dst = _disambiguate_correction_destination(
                subtitle_dst,
                used_dests,
            )
            subtitle_dst = _ensure_correction_destination(
                subtitle_dst,
                output_root,
            )
            used_dests.add(_path_key(subtitle_dst))
            planned_moves.append(
                PlannedMove(
                    src=Path(subtitle_move.src),
                    dst=subtitle_dst,
                    kind="subtitle",
                    src_id=subtitle_source_id,
                )
            )

    planned_moves.sort(key=lambda move: (move.kind, move.src_id, str(move.dst)))
    return planned_moves


def _subtitle_variant_for_corrected_link(link: dict[str, object]) -> str:
    override = link.get("variant_override")
    if override in {"chs", "cht", "chi"}:
        return str(override)
    detected = link.get("variant_detected")
    if detected in {"chs", "cht", "chi"}:
        return str(detected)
    return "chi"


def _disambiguate_correction_destination(
    dst: Path,
    used_dests: set[str],
) -> Path:
    if _path_key(dst) not in used_dests:
        return dst
    suffix = dst.suffix
    stem = dst.name[: -len(suffix)] if suffix else dst.name
    counter = 1
    while True:
        if suffix:
            candidate = dst.with_name(f"{stem}.{counter}{suffix}")
        else:
            candidate = dst.with_name(f"{stem}.{counter}")
        if _path_key(candidate) not in used_dests:
            return candidate
        counter += 1


def _ensure_correction_destination(dst: Path, output_root: Path) -> Path:
    output_root_resolved = output_root.resolve(strict=False)
    dst_resolved = dst.resolve(strict=False)
    if not dst_resolved.is_relative_to(output_root_resolved):
        raise ApiServiceError(
            status_code=422,
            code="invalid_correction_destination",
            message="manual correction destination is outside output root",
            details={
                "destination": str(dst_resolved),
                "output_root": str(output_root_resolved),
            },
        )
    if dst_resolved.exists():
        raise ApiServiceError(
            status_code=422,
            code="destination_conflict",
            message="manual correction destination already exists",
            details={"destination": str(dst_resolved)},
        )
    return dst_resolved


def _correction_snapshot_payload(
    *,
    source_files: list[ReviewSourceFile],
    committed_moves: list[JobMoveRecord],
    selected_info: TmdbReviewInfo,
    corrected_targets: list[dict[str, object]],
    corrected_links: list[dict[str, object]],
    existing_mapping_attempts: list[dict[str, object]],
    existing_mapping_evidence: list[dict[str, object]],
    op_results: list[dict[str, object]],
    operation_id: int,
) -> dict[str, object]:
    committed_targets, committed_links = _committed_corrected_targets_and_links(
        corrected_targets,
        corrected_links,
        committed_moves,
    )
    return {
        "source_files": [
            item.model_dump(mode="json")
            for item in _committed_source_files(source_files, committed_moves)
        ],
        "selected_tmdb_info": selected_info.model_dump(mode="json"),
        "episode_targets": committed_targets,
        "subtitle_links": committed_links,
        "mapping_attempts": _append_correction_attempt(
            existing_mapping_attempts,
            tmdb_id=selected_info.selected_tmdb_id,
            op_count=len(op_results),
        ),
        "mapping_evidence": _append_correction_evidence(
            existing_mapping_evidence,
            committed_targets,
            committed_links,
            op_results,
        ),
        "path_event_summary": {
            "status": "not_started",
            "pending_count": 0,
            "incomplete_count": 0,
        },
        "validation_summary": {
            "status": "planned",
            "corrected_by_operation_id": operation_id,
            "season_episode_counts": _season_episode_counts_from_tmdb_info(
                selected_info
            ),
        },
    }


def _committed_corrected_targets_and_links(
    corrected_targets: list[dict[str, object]],
    corrected_links: list[dict[str, object]],
    committed_moves: list[JobMoveRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    moves_by_key = {(move.src_id, move.kind): move for move in committed_moves}
    links_by_target: dict[int, list[dict[str, object]]] = {}
    for link in corrected_links:
        target_id = link.get("episode_target_id")
        if isinstance(target_id, int):
            links_by_target.setdefault(target_id, []).append(link)

    committed_links: list[dict[str, object]] = []
    committed_targets: list[dict[str, object]] = []
    for target in sorted(corrected_targets, key=lambda item: int(item["id"])):
        target_id = int(target["id"])
        video_source_id = int(target["video_source_id"])
        video_move = moves_by_key.get((video_source_id, "video"))
        link_ids: list[int] = []
        for link in sorted(
            links_by_target.get(target_id, []),
            key=lambda item: int(item["id"]),
        ):
            subtitle_source_id = int(link["subtitle_source_id"])
            subtitle_move = moves_by_key.get((subtitle_source_id, "subtitle"))
            link_id = int(link["id"])
            link_ids.append(link_id)
            committed_links.append(
                {
                    **link,
                    "id": link_id,
                    "episode_target_id": target_id,
                    "status": "proposed",
                    "move_item_id": (
                        subtitle_move.id if subtitle_move is not None else None
                    ),
                }
            )
        committed_targets.append(
            {
                **target,
                "id": target_id,
                "status": "proposed",
                "move_item_ids": [video_move.id] if video_move is not None else [],
                "subtitle_link_ids": link_ids,
            }
        )
    committed_links.sort(key=lambda item: int(item["id"]))
    return committed_targets, committed_links


def _append_correction_attempt(
    existing_attempts: list[dict[str, object]],
    *,
    tmdb_id: int | None,
    op_count: int,
) -> list[dict[str, object]]:
    attempts = list(existing_attempts)
    attempts.append(
        {
            "id": len(attempts) + 1,
            "stage": "correction",
            "created_by": "user",
            "tmdb_id": tmdb_id,
            "status": "succeeded",
            "op_count": op_count,
        }
    )
    return attempts


def _append_correction_evidence(
    existing_evidence: list[dict[str, object]],
    committed_targets: list[dict[str, object]],
    committed_links: list[dict[str, object]],
    op_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    evidence = list(existing_evidence)
    targets_by_id = {int(item["id"]): item for item in committed_targets}
    links_by_id = {int(item["id"]): item for item in committed_links}
    for result in op_results:
        if result.get("op") == "update_episode":
            target_id = result.get("episode_target_id")
            target = targets_by_id.get(target_id) if isinstance(target_id, int) else None
            if target is None:
                continue
            evidence.append(
                {
                    "id": len(evidence) + 1,
                    "episode_target_id": target_id,
                    "source_file_id": target.get("video_source_id"),
                    "evidence_kind": "manual_override",
                    "summary": (
                        "Manual correction set episode target "
                        f"{target_id} to "
                        f"{_episode_label(int(target['season']), int(target['episode_start']), int(target['episode_end']))}."
                    ),
                    "confidence": "user",
                    "candidate_refs": [
                        {"kind": "episode_target", "id": target_id},
                        {"kind": "video_source", "id": target.get("video_source_id")},
                    ],
                }
            )
            continue
        link_id = result.get("subtitle_link_id")
        link = links_by_id.get(link_id) if isinstance(link_id, int) else None
        if link is None:
            continue
        evidence.append(
            {
                "id": len(evidence) + 1,
                "episode_target_id": link.get("episode_target_id"),
                "source_file_id": link.get("subtitle_source_id"),
                "evidence_kind": "manual_override",
                "summary": (
                    "Manual correction set subtitle link "
                    f"{link_id} variant to "
                    f"{_subtitle_variant_for_corrected_link(link)}."
                ),
                "confidence": "user",
                "candidate_refs": [
                    {"kind": "subtitle_link", "id": link_id},
                    {"kind": "subtitle_source", "id": link.get("subtitle_source_id")},
                ],
            }
        )
    return evidence


def _correction_result_summary(
    *,
    operation_id: int,
    payload: CorrectionRequest,
    op_results: list[dict[str, object]],
    committed_move_count: int,
    diff_summary: dict[str, object],
) -> dict[str, object]:
    response_payload = {
        "operation_id": operation_id,
        "status": "committed",
        "base_review_revision": payload.base_review_revision,
        "op_results": op_results,
        "diff_summary": diff_summary,
        "committed_move_item_count": committed_move_count,
    }
    return {
        "status": "committed",
        "op_count": len(payload.ops),
        "committed_move_item_count": committed_move_count,
        "diff_summary": diff_summary,
        "response_payload": response_payload,
    }


def _correction_diff_summary(
    current_moves: list[JobMoveRecord],
    committed_moves: list[JobMoveRecord],
) -> dict[str, object]:
    planned = [
        PlannedMove(
            src=Path(move.src),
            dst=Path(move.dst),
            kind=move.kind,
            src_id=move.src_id,
        )
        for move in committed_moves
    ]
    summary, _items = _rematch_diff(current_moves, planned)
    return summary


def _planned_moves_from_preview_payload(
    preview_payload: dict[str, object],
) -> list[PlannedMove]:
    raw_items = preview_payload.get("move_items", [])
    if not isinstance(raw_items, list):
        return []
    planned_moves: list[PlannedMove] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        preview_destination = item.get("preview_destination_path")
        source_path = item.get("source_path")
        if not isinstance(preview_destination, dict) or not isinstance(source_path, dict):
            continue
        kind = item.get("kind")
        if kind not in {"video", "subtitle"}:
            continue
        source_id = item.get("source_id")
        if not isinstance(source_id, int):
            continue
        src = _path_ref_to_path(source_path)
        dst = _path_ref_to_path(preview_destination)
        if src is None or dst is None:
            continue
        planned_moves.append(
            PlannedMove(
                src=src,
                dst=dst,
                kind=kind,
                src_id=source_id,
            )
        )
    planned_moves.sort(key=lambda move: (move.kind, move.src_id, str(move.dst)))
    return planned_moves


def _path_ref_to_path(value: dict[str, object]) -> Path | None:
    for key in ("display_path", "normalized_abs_path", "rel_path"):
        raw = value.get(key)
        if isinstance(raw, str) and raw:
            return Path(raw)
    return None


def _committed_source_files(
    source_files: list[ReviewSourceFile],
    committed_moves: list[JobMoveRecord],
) -> list[ReviewSourceFile]:
    planned_ids = {move.src_id for move in committed_moves}
    updated: list[ReviewSourceFile] = []
    for item in source_files:
        final_disposition = item.final_disposition
        if item.kind in {"video", "subtitle"}:
            final_disposition = (
                "planned_output" if item.id in planned_ids else "original"
            )
        updated.append(
            item.model_copy(
                update={
                    "current_path": item.original_path,
                    "current_path_kind": "source",
                    "final_disposition": final_disposition,
                }
            )
        )
    return updated


def _committed_targets_and_links(
    preview_payload: dict[str, object],
    committed_moves: list[JobMoveRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    moves_by_key = {(move.src_id, move.kind): move for move in committed_moves}
    raw_links = preview_payload.get("subtitle_links", [])
    if not isinstance(raw_links, list):
        raw_links = []
    preview_links_by_target: dict[int, list[dict[str, object]]] = {}
    for raw_link in raw_links:
        if not isinstance(raw_link, dict):
            continue
        target_id = raw_link.get("episode_target_id")
        if isinstance(target_id, int):
            preview_links_by_target.setdefault(target_id, []).append(raw_link)

    targets: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    raw_targets = preview_payload.get("episode_targets", [])
    if not isinstance(raw_targets, list):
        raw_targets = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            continue
        target_id = raw_target.get("id")
        video_source_id = raw_target.get("video_source_id")
        if not isinstance(target_id, int) or not isinstance(video_source_id, int):
            continue
        video_move = moves_by_key.get((video_source_id, "video"))
        link_ids: list[int] = []
        for raw_link in preview_links_by_target.get(target_id, []):
            subtitle_source_id = raw_link.get("subtitle_source_id")
            if not isinstance(subtitle_source_id, int):
                continue
            subtitle_move = moves_by_key.get((subtitle_source_id, "subtitle"))
            link_id = len(links) + 1
            link_ids.append(link_id)
            links.append(
                {
                    "id": link_id,
                    "episode_target_id": len(targets) + 1,
                    "subtitle_source_id": subtitle_source_id,
                    "variant_detected": raw_link.get("variant_detected", "unknown"),
                    "variant_override": raw_link.get("variant_override"),
                    "status": "proposed",
                    "created_by": "rematch",
                    "move_item_id": (
                        subtitle_move.id if subtitle_move is not None else None
                    ),
                }
            )
        targets.append(
            {
                "id": len(targets) + 1,
                "video_source_id": video_source_id,
                "season": raw_target.get("season"),
                "episode_start": raw_target.get("episode_start"),
                "episode_end": raw_target.get("episode_end"),
                "status": "proposed",
                "manual": False,
                "created_by": "rematch",
                "move_item_ids": [video_move.id] if video_move is not None else [],
                "subtitle_link_ids": link_ids,
            }
        )
    return targets, links


def _append_rematch_attempt(
    existing_attempts: list[dict[str, object]],
    *,
    tmdb_id: int,
    mapped_video_count: int,
    mapped_subtitle_count: int,
) -> list[dict[str, object]]:
    attempts = list(existing_attempts)
    attempts.append(
        {
            "id": len(attempts) + 1,
            "stage": "rematch",
            "created_by": "rematch",
            "tmdb_id": tmdb_id,
            "status": "succeeded",
            "mapped_video_count": mapped_video_count,
            "mapped_subtitle_count": mapped_subtitle_count,
        }
    )
    return attempts


def _append_rematch_evidence(
    existing_evidence: list[dict[str, object]],
    preview_payload: dict[str, object],
) -> list[dict[str, object]]:
    evidence = list(existing_evidence)
    raw_items = preview_payload.get("mapping_evidence", [])
    if not isinstance(raw_items, list):
        return evidence
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item["id"] = len(evidence) + 1
        item["evidence_kind"] = f"committed_{item.get('evidence_kind', 'rematch')}"
        evidence.append(item)
    return evidence


def _preview_tmdb_info(preview: PlanBuildResult) -> TmdbReviewInfo:
    return TmdbReviewInfo(
        selected_tmdb_id=preview.plan.tmdb_id,
        title_zh_cn=preview.plan.series_name_zh_cn,
        original_name=preview.tmdb_details.original_name,
        year=preview.tmdb_details.year,
        candidates=[
            {
                "id": candidate.id,
                "name": candidate.name,
                "original_name": candidate.original_name,
                "first_air_date": candidate.first_air_date,
                "year": candidate.year,
                "origin_country": list(candidate.origin_country or ()),
                "popularity": candidate.popularity,
                "vote_count": candidate.vote_count,
            }
            for candidate in preview.tmdb_candidates
        ],
        seasons=[
            {
                "season_number": season.season_number,
                "episode_count": season.episode_count,
            }
            for season in preview.tmdb_details.seasons
        ],
        specials=(
            [
                {
                    "season_number": preview.specials_zh.season_number,
                    "episode_number": episode.episode_number,
                    "name": episode.name,
                    "overview": episode.overview,
                    "air_date": episode.air_date,
                }
                for episode in preview.specials_zh.episodes
            ]
            if preview.specials_zh is not None
            else []
        ),
    )


def _preview_targets_and_links(
    preview: PlanBuildResult,
) -> tuple[list[ReviewEpisodeTarget], list[ReviewSubtitleLink]]:
    preview_moves = {(move.src_id, move.kind): move for move in preview.plan.moves}
    targets: list[ReviewEpisodeTarget] = []
    links: list[ReviewSubtitleLink] = []
    for target_id, item in enumerate(preview.mapping.items, start=1):
        link_ids: list[int] = []
        for subtitle_id in item.subtitle_ids:
            subtitle_move = preview_moves.get((subtitle_id, "subtitle"))
            link_id = len(links) + 1
            link_ids.append(link_id)
            links.append(
                ReviewSubtitleLink(
                    id=link_id,
                    episode_target_id=target_id,
                    subtitle_source_id=subtitle_id,
                    variant_detected=(
                        _subtitle_variant_from_dst(str(subtitle_move.dst))
                        if subtitle_move is not None
                        else "unknown"
                    ),
                    variant_override=None,
                    status="preview",
                    created_by="rematch",
                    move_item_id=None,
                )
            )
        targets.append(
            ReviewEpisodeTarget(
                id=target_id,
                video_source_id=item.video_id,
                season=item.season,
                episode_start=item.episode_start,
                episode_end=item.episode_end,
                status="preview",
                manual=False,
                created_by="rematch",
                move_item_ids=[],
                subtitle_link_ids=link_ids,
            )
        )
    return targets, links


def _preview_mapping_evidence(preview: PlanBuildResult) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for idx, item in enumerate(preview.mapping.items, start=1):
        target = _episode_label(item.season, item.episode_start, item.episode_end)
        evidence.append(
            {
                "id": len(evidence) + 1,
                "episode_target_id": idx,
                "source_file_id": item.video_id,
                "evidence_kind": "rematch_preview",
                "summary": f"Preview maps video source {item.video_id} to {target}.",
                "confidence": "unknown",
                "candidate_refs": [
                    {"kind": "tmdb", "id": preview.plan.tmdb_id},
                    {"kind": "video_source", "id": item.video_id},
                ],
            }
        )
        for subtitle_id in item.subtitle_ids:
            evidence.append(
                {
                    "id": len(evidence) + 1,
                    "episode_target_id": idx,
                    "source_file_id": subtitle_id,
                    "evidence_kind": "rematch_subtitle_preview",
                    "summary": (
                        f"Preview links subtitle source {subtitle_id} to {target}."
                    ),
                    "confidence": "unknown",
                    "candidate_refs": [
                        {"kind": "video_source", "id": item.video_id},
                        {"kind": "subtitle_source", "id": subtitle_id},
                    ],
                }
            )
    return evidence


def _preview_move_items(
    current_moves: list[JobMoveRecord],
    preview_moves: list[PlannedMove],
) -> list[dict[str, object]]:
    current_by_key = {(move.src_id, move.kind): move for move in current_moves}
    items: list[dict[str, object]] = []
    for move in preview_moves:
        current = current_by_key.get((move.src_id, move.kind))
        change_type = "added"
        if current is not None:
            change_type = (
                "unchanged"
                if _safe_path_key(current.dst) == _safe_path_key(move.dst)
                else "changed"
            )
        items.append(
            {
                "source_id": move.src_id,
                "kind": move.kind,
                "change_type": change_type,
                "source_path": _validation_path_ref(
                    move.src,
                    root_kind="source",
                ).model_dump(mode="json"),
                "current_planned_destination_path": (
                    _validation_path_ref(
                        current.dst,
                        root_kind="output",
                    ).model_dump(mode="json")
                    if current is not None
                    else None
                ),
                "preview_destination_path": _validation_path_ref(
                    move.dst,
                    root_kind="output",
                ).model_dump(mode="json"),
            }
        )
    preview_keys = {(move.src_id, move.kind) for move in preview_moves}
    for move in current_moves:
        if (move.src_id, move.kind) in preview_keys:
            continue
        items.append(
            {
                "source_id": move.src_id,
                "kind": move.kind,
                "change_type": "removed",
                "source_path": _validation_path_ref(
                    move.src,
                    root_kind="source",
                ).model_dump(mode="json"),
                "current_planned_destination_path": _validation_path_ref(
                    move.dst,
                    root_kind="output",
                ).model_dump(mode="json"),
                "preview_destination_path": None,
            }
        )
    items.sort(key=lambda item: (str(item["kind"]), int(item["source_id"])))
    return items


def _rematch_diff(
    current_moves: list[JobMoveRecord],
    preview_moves: list[PlannedMove],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    items = _preview_move_items(current_moves, preview_moves)
    counts = {
        "added": 0,
        "changed": 0,
        "removed": 0,
        "unchanged": 0,
    }
    for item in items:
        change_type = str(item["change_type"])
        counts[change_type] = counts.get(change_type, 0) + 1
    return {**counts, "total": len(items)}, items


def _episode_label(season: int, start: int, end: int) -> str:
    label = f"S{season:02d}E{start:02d}"
    if end != start:
        label = f"{label}-E{end:02d}"
    return label


def _path_key(path: str | Path) -> str:
    return str(Path(path).resolve(strict=False)).casefold()


def _safe_path_key(path: str | Path) -> str:
    return _safe_path_string(path).casefold()


def _safe_path_string(path: str | Path) -> str:
    raw_path = Path(path)
    if _path_has_env_component(raw_path):
        return str(raw_path)
    try:
        return str(raw_path.resolve(strict=False))
    except OSError:
        return str(raw_path)


def _build_readiness_summary(
    source_files: list[ReviewSourceFile],
    episode_targets: list[ReviewEpisodeTarget],
    subtitle_links: list[ReviewSubtitleLink],
) -> dict[str, object]:
    mapped_video_ids = {target.video_source_id for target in episode_targets}
    linked_subtitle_ids = {link.subtitle_source_id for link in subtitle_links}
    unmapped_videos = [
        item
        for item in source_files
        if item.kind == "video" and item.id not in mapped_video_ids
    ]
    unassociated_subtitles = [
        item
        for item in source_files
        if item.kind == "subtitle" and item.id not in linked_subtitle_ids
    ]
    ignored_files = [item for item in source_files if item.kind == "ignored"]
    blocking_count = len(unmapped_videos)
    warning_count = len(unassociated_subtitles) + len(ignored_files)
    status = "ready"
    if blocking_count:
        status = "blocked"
    elif warning_count:
        status = "warning"
    acknowledgements_required: list[dict[str, object]] = []
    if unmapped_videos:
        acknowledgements_required.append(
            {
                "kind": "unmapped_videos",
                "count": len(unmapped_videos),
                "severity": "blocking",
            }
        )
    if unassociated_subtitles:
        acknowledgements_required.append(
            {
                "kind": "unassociated_subtitles",
                "count": len(unassociated_subtitles),
                "severity": "warning",
            }
        )
    if ignored_files:
        acknowledgements_required.append(
            {
                "kind": "ignored_files",
                "count": len(ignored_files),
                "severity": "warning",
            }
        )
    return {
        "status": status,
        "blocking_count": blocking_count,
        "warning_count": warning_count,
        "mapped_video_count": len(mapped_video_ids),
        "episode_target_count": len(episode_targets),
        "subtitle_link_count": len(subtitle_links),
        "unmapped_video_count": len(unmapped_videos),
        "unassociated_subtitle_count": len(unassociated_subtitles),
        "ignored_file_count": len(ignored_files),
        "will_stay_original_count": (
            len(unmapped_videos) + len(unassociated_subtitles) + len(ignored_files)
        ),
        "will_archive_count": 0,
        "will_move_to_review_count": 0,
        "acknowledgements_required": acknowledgements_required,
        "affected_refs": {
            "unmapped_videos": [_source_file_summary(item) for item in unmapped_videos],
            "unassociated_subtitles": [
                _source_file_summary(item) for item in unassociated_subtitles
            ],
            "ignored_files": [_source_file_summary(item) for item in ignored_files],
        },
    }


def _source_file_summary(item: ReviewSourceFile) -> dict[str, object]:
    return {
        "source_file_id": item.id,
        "path": item.original_path.model_dump(mode="json"),
        "ignored_reason": item.ignored_reason,
    }


def _path_ref(path: str | Path, *, root_kind: str, root: Path | None = None) -> PathRef:
    raw_path = Path(path)
    display_path = str(raw_path)
    normalized = str(raw_path.resolve(strict=False))
    rel_path = raw_path.name
    if root is not None:
        try:
            rel_path = str(raw_path.resolve(strict=False).relative_to(root.resolve(strict=False)))
        except ValueError:
            rel_path = raw_path.name
    return PathRef(
        root_kind=root_kind,
        root_id=f"{root_kind}:default:1",
        rel_path=rel_path,
        display_path=display_path,
        normalized_abs_path=normalized,
        realpath=normalized,
        casefold_key=normalized.casefold(),
        redacted=False,
    )


def _source_file_from_move(job: JobRecord, move: JobMoveRecord) -> ReviewSourceFile:
    current_path = move.dst if move.status == "moved" else move.src
    current_path_kind = "output" if move.status == "moved" else "source"
    final_disposition = "planned_output"
    if move.status == "moved":
        final_disposition = "output"
    elif move.status == "failed":
        final_disposition = "failed"
    elif move.status == "skipped":
        final_disposition = "original"
    elif move.status == "rolled_back":
        final_disposition = "source"
    return ReviewSourceFile(
        id=move.src_id,
        kind=move.kind,
        original_path=_validation_path_ref(
            move.src,
            root_kind="source",
            root=Path(job.series_dir),
        ),
        current_path=_validation_path_ref(
            current_path,
            root_kind=current_path_kind,
        ),
        current_path_kind=current_path_kind,
        ext=Path(move.src).suffix,
        file_identity={
            "identity_strength": "missing",
            "size_bytes": None,
            "mtime_ns": None,
            "content_hash": None,
        },
        detected_subtitle_variant="unknown" if move.kind == "subtitle" else None,
        final_disposition=final_disposition,
        ignored_reason=None,
    )

def _payload_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_has_env_component(path: str | Path) -> bool:
    return any(is_env_file_name(part) for part in Path(path).parts)


def _validation_path_ref(
    path: str | Path,
    *,
    root_kind: str,
    root: Path | None = None,
) -> PathRef:
    raw_path = Path(path)
    if not _path_has_env_component(raw_path) and not (
        root is not None and _path_has_env_component(root)
    ):
        return _path_ref(raw_path, root_kind=root_kind, root=root)
    rel_path = raw_path.name
    if root is not None:
        try:
            rel_path = str(raw_path.relative_to(root))
        except ValueError:
            rel_path = raw_path.name
    return PathRef(
        root_kind=root_kind,
        root_id=f"{root_kind}:default:1",
        rel_path=rel_path,
        display_path=str(raw_path),
        normalized_abs_path=None,
        realpath=None,
        casefold_key=None,
        redacted=True,
    )


def _validation_source_files(
    job: JobRecord,
    moves: list[JobMoveRecord],
    snapshot: ReviewSnapshotRecord | None,
) -> list[ReviewSourceFile]:
    if snapshot is not None:
        return [
            ReviewSourceFile.model_validate(item)
            for item in snapshot.source_files
        ]
    items: list[ReviewSourceFile] = []
    for move in moves:
        if not (
            _path_has_env_component(move.src)
            or _path_has_env_component(move.dst)
        ):
            items.append(_source_file_from_move(job, move))
            continue
        current_path = move.dst if move.status == "moved" else move.src
        current_path_kind = "output" if move.status == "moved" else "source"
        final_disposition = "planned_output"
        if move.status == "moved":
            final_disposition = "output"
        elif move.status == "failed":
            final_disposition = "failed"
        elif move.status == "skipped":
            final_disposition = "original"
        elif move.status == "rolled_back":
            final_disposition = "source"
        items.append(
            ReviewSourceFile(
                id=move.src_id,
                kind=move.kind,
                original_path=_validation_path_ref(
                    move.src,
                    root_kind="source",
                    root=Path(job.series_dir),
                ),
                current_path=_validation_path_ref(
                    current_path,
                    root_kind=current_path_kind,
                ),
                current_path_kind=current_path_kind,
                ext=Path(move.src).suffix,
                file_identity={
                    "identity_strength": "missing",
                    "size_bytes": None,
                    "mtime_ns": None,
                    "content_hash": None,
                },
                detected_subtitle_variant=(
                    "unknown" if move.kind == "subtitle" else None
                ),
                final_disposition=final_disposition,
                ignored_reason=None,
            )
        )
    items.sort(key=lambda item: (item.original_path.display_path, item.id))
    return items


def _resolve_apply_validation_scope_or_raise(
    payload: ApplyValidationRequest,
    moves: list[JobMoveRecord],
) -> list[JobMoveRecord]:
    if payload.scope == "all_for_apply":
        return [move for move in moves if move.status == "proposed"]
    if payload.scope == "selected_check":
        assert payload.selection is not None
        if payload.selection.kind == "all":
            return [move for move in moves if move.status == "proposed"]
        if payload.selection.kind != "move_items":
            raise ApiServiceError(
                status_code=422,
                code="unsupported_validation_selection",
                message="apply validation currently supports all or move_items selection",
                details={"selection_kind": payload.selection.kind},
            )
        selected = set(payload.selection.ids or [])
        moves_by_id = {move.id: move for move in moves}
        unknown_ids = sorted(selected - set(moves_by_id))
        if unknown_ids:
            raise ApiServiceError(
                status_code=422,
                code="invalid_validation_selection",
                message="selected move items were not found",
                details={"unknown_move_item_ids": unknown_ids},
            )
        not_proposed_ids = sorted(
            move.id
            for move in moves
            if move.id in selected and move.status != "proposed"
        )
        if not_proposed_ids:
            raise ApiServiceError(
                status_code=422,
                code="invalid_validation_selection",
                message="selected move items are not proposed",
                details={"not_proposed_move_item_ids": not_proposed_ids},
            )
        return [move for move in moves if move.id in selected]
    raise ApiServiceError(
        status_code=422,
        code="unsupported_validation_scope",
        message="validation scope is not implemented yet",
        details={"scope": payload.scope},
    )


def _apply_validation_response_payload(
    *,
    operation: OperationRecord,
    job: JobRecord,
    moves: list[JobMoveRecord],
    scope: str,
    versions: dict[str, int],
) -> dict[str, object]:
    checked_at = _now_iso()
    output_root = Path(job.output_root).resolve(strict=False)
    destination_keys: dict[str, list[int]] = {}
    item_results = [
        _apply_validation_item(
            move,
            output_root=output_root,
            destination_keys=destination_keys,
        )
        for move in moves
    ]
    for result in item_results:
        if result["status"] != "ok":
            continue
        duplicate_ids = destination_keys.get(str(result["destination_key"]), [])
        if len(duplicate_ids) > 1:
            result["status"] = "blocking"
            result["code"] = "destination_collision"
            result["message"] = "destination conflicts with another selected move"
            result["conflicting_move_item_ids"] = duplicate_ids

    blocking_items = [
        _validation_problem_summary(result)
        for result in item_results
        if result["status"] == "blocking"
    ]
    warning_items = [
        _validation_problem_summary(result)
        for result in item_results
        if result["status"] == "warning"
    ]
    if not moves:
        blocking_items.append(
            {
                "move_item_id": None,
                "source_id": None,
                "kind": "job",
                "code": "no_proposed_move_items",
                "message": "no proposed move items are available for apply validation",
            }
        )
    valid_count = sum(1 for result in item_results if result["status"] == "ok")
    invalid_count = len(blocking_items)
    response_status = "valid" if invalid_count == 0 else "invalid"
    return {
        "operation_id": operation.id,
        "validation_id": operation.id,
        "status": response_status,
        "scope": scope,
        "checked_at": checked_at,
        "ttl_seconds": 300,
        "validated_mapping_version": versions["mapping_version"],
        "validated_move_plan_version": versions["move_plan_version"],
        "checked_move_state_revision": versions["move_state_revision"],
        "checked_review_revision": versions["review_revision"],
        "item_result_summary": {
            "checked": len(moves),
            "valid": valid_count,
            "invalid": invalid_count,
            "warning": len(warning_items),
        },
        "blocking_items": blocking_items,
        "warning_items": warning_items,
        "destination_preview": [
            {
                "move_item_id": move.id,
                "source_id": move.src_id,
                "kind": move.kind,
                "from": _validation_path_ref(
                    move.src,
                    root_kind="source",
                    root=Path(job.series_dir),
                ).model_dump(mode="json"),
                "to": _validation_path_ref(
                    move.dst,
                    root_kind="output",
                    root=Path(job.output_root),
                ).model_dump(mode="json"),
            }
            for move in moves
        ],
        "affected_move_items": [move.id for move in moves],
    }


def _apply_validation_item(
    move: JobMoveRecord,
    *,
    output_root: Path,
    destination_keys: dict[str, list[int]],
) -> dict[str, object]:
    src = Path(move.src)
    dst = Path(move.dst)
    base = {
        "move_item_id": move.id,
        "source_id": move.src_id,
        "kind": move.kind,
        "status": "ok",
        "code": "ok",
        "message": "ready",
        "destination": str(dst),
    }
    if _path_has_env_component(src) or _path_has_env_component(dst):
        return {
            **base,
            "status": "blocking",
            "code": "sensitive_env_path",
            "message": "sensitive .env paths cannot be validated",
        }
    if move.status != "proposed":
        return {
            **base,
            "status": "blocking",
            "code": "invalid_move_state",
            "message": "only proposed move items can be applied",
            "move_status": move.status,
        }
    dst_resolved = dst.resolve(strict=False)
    dst_key = _path_key(dst_resolved)
    destination_keys.setdefault(dst_key, []).append(move.id)
    base = {
        **base,
        "destination": str(dst_resolved),
        "destination_key": dst_key,
    }
    if not src.is_file():
        return {
            **base,
            "status": "blocking",
            "code": "source_missing",
            "message": "source file is missing",
            "source": str(src),
        }
    if not dst_resolved.is_relative_to(output_root):
        return {
            **base,
            "status": "blocking",
            "code": "outside_output_root",
            "message": "destination is outside output root",
            "output_root": str(output_root),
        }
    if dst_resolved.exists():
        return {
            **base,
            "status": "blocking",
            "code": "destination_exists",
            "message": "destination already exists",
        }
    parent = dst_resolved.parent
    if parent.exists() and not parent.is_dir():
        return {
            **base,
            "status": "blocking",
            "code": "parent_not_directory",
            "message": "destination parent exists but is not a directory",
            "parent": str(parent),
        }
    return base


def _validated_apply_validation_or_raise(
    operation: OperationRecord | None,
) -> ApplyValidationResponse:
    if operation is None:
        raise ApiServiceError(
            status_code=422,
            code="invalid_apply_validation",
            message="apply validation operation was not found",
        )
    if operation.operation_type != "validate" or operation.status != "succeeded":
        raise ApiServiceError(
            status_code=422,
            code="invalid_apply_validation",
            message="apply validation operation is not a succeeded validation",
            details={
                "operation_id": operation.id,
                "operation_type": operation.operation_type,
                "status": operation.status,
            },
        )
    stored_payload = operation.result_summary.get("response_payload")
    if not isinstance(stored_payload, dict):
        raise ApiServiceError(
            status_code=422,
            code="invalid_apply_validation",
            message="apply validation operation does not contain a response payload",
            details={"operation_id": operation.id},
        )
    validation = ApplyValidationResponse.model_validate(stored_payload)
    if validation.scope != "all_for_apply" or validation.status != "valid":
        raise ApiServiceError(
            status_code=422,
            code="invalid_apply_validation",
            message="apply requires a valid all_for_apply validation",
            details=validation.model_dump(mode="json"),
        )
    return validation


def _assert_validation_matches_apply_request(
    validation: ApplyValidationResponse,
    payload: ApplyRequest,
    current_versions: dict[str, int],
) -> None:
    mismatches: dict[str, dict[str, int]] = {}
    expected = {
        "mapping_version": validation.validated_mapping_version,
        "move_plan_version": validation.validated_move_plan_version,
        "move_state_revision": validation.checked_move_state_revision,
    }
    current = {
        "mapping_version": current_versions["mapping_version"],
        "move_plan_version": current_versions["move_plan_version"],
        "move_state_revision": current_versions["move_state_revision"],
    }
    requested = {
        "mapping_version": payload.base_mapping_version,
        "move_plan_version": payload.base_move_plan_version,
        "move_state_revision": payload.base_move_state_revision,
    }
    for key, expected_value in expected.items():
        if current[key] != expected_value or requested[key] != expected_value:
            mismatches[key] = {
                "validation": expected_value,
                "current": current[key],
                "requested": requested[key],
            }
    if mismatches:
        raise ApiServiceError(
            status_code=412,
            code="stale_versions",
            message="apply validation versions are stale",
            details={"stale_versions": mismatches},
        )


def _apply_response_payload_from_operation(
    operation: OperationRecord,
) -> dict[str, object]:
    stored_payload = operation.result_summary.get("response_payload")
    if isinstance(stored_payload, dict):
        return stored_payload
    status = "succeeded" if operation.status == "succeeded" else "failed"
    validation_id = operation.result_summary.get("validation_id")
    if not isinstance(validation_id, int):
        validation_id = operation.base_versions.get("validation_id", 0)
    if not isinstance(validation_id, int):
        validation_id = 0
    applied_count = operation.result_summary.get("applied_move_item_count", 0)
    if not isinstance(applied_count, int):
        applied_count = 0
    item_result_summary = operation.result_summary.get("item_result_summary")
    if not isinstance(item_result_summary, dict):
        item_result_summary = {
            "succeeded": applied_count if status == "succeeded" else 0,
            "failed": 0 if status == "succeeded" else 1,
            "total": applied_count if status == "succeeded" else 1,
        }
    moved_items = operation.result_summary.get("moved_items")
    if not isinstance(moved_items, list):
        moved_items = []
    return {
        "operation_id": operation.id,
        "status": status,
        "validation_id": validation_id,
        "applied_move_item_count": applied_count,
        "item_result_summary": item_result_summary,
        "moved_items": moved_items,
    }


def _apply_failure_response_payload(
    *,
    operation: OperationRecord,
    validation_id: int,
    error_code: str,
    error_message: str,
    item_result_summary: dict[str, object],
    moved_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "operation_id": operation.id,
        "status": "failed",
        "validation_id": validation_id,
        "applied_move_item_count": 0,
        "item_result_summary": item_result_summary,
        "moved_items": moved_items or [],
        "error": {
            "code": error_code,
            "message": error_message,
        },
    }


def _withdraw_response_payload_from_operation(
    operation: OperationRecord,
) -> dict[str, object]:
    stored_payload = operation.result_summary.get("response_payload")
    if isinstance(stored_payload, dict):
        return stored_payload
    status = "succeeded" if operation.status == "succeeded" else "failed"
    withdrawn_count = operation.result_summary.get("withdrawn_move_item_count", 0)
    if not isinstance(withdrawn_count, int):
        withdrawn_count = 0
    item_result_summary = operation.result_summary.get("item_result_summary")
    if not isinstance(item_result_summary, dict):
        item_result_summary = {
            "succeeded": withdrawn_count if status == "succeeded" else 0,
            "failed": 0 if status == "succeeded" else 1,
            "total": withdrawn_count if status == "succeeded" else 1,
        }
    withdrawn_items = operation.result_summary.get("withdrawn_items")
    if not isinstance(withdrawn_items, list):
        withdrawn_items = []
    return {
        "operation_id": operation.id,
        "status": status,
        "mode": "withdraw_to_review",
        "withdrawn_move_item_count": withdrawn_count,
        "item_result_summary": item_result_summary,
        "withdrawn_items": withdrawn_items,
    }


def _withdraw_failure_response_payload(
    *,
    operation: OperationRecord,
    error_code: str,
    error_message: str,
    total_count: int,
    withdrawn_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "operation_id": operation.id,
        "status": "failed",
        "mode": "withdraw_to_review",
        "withdrawn_move_item_count": 0,
        "item_result_summary": {
            "succeeded": 0,
            "failed": total_count,
            "total": total_count,
        },
        "withdrawn_items": withdrawn_items or [],
        "error": {
            "code": error_code,
            "message": error_message,
        },
    }


def _reopen_response_payload_from_operation(
    operation: OperationRecord,
) -> dict[str, object]:
    stored_payload = operation.result_summary.get("response_payload")
    if isinstance(stored_payload, dict):
        return stored_payload
    status = "succeeded" if operation.status == "succeeded" else "failed"
    reopened_count = operation.result_summary.get("reopened_move_item_count", 0)
    if not isinstance(reopened_count, int):
        reopened_count = 0
    item_result_summary = operation.result_summary.get("item_result_summary")
    if not isinstance(item_result_summary, dict):
        item_result_summary = {
            "succeeded": reopened_count if status == "succeeded" else 0,
            "failed": 0 if status == "succeeded" else 1,
            "total": reopened_count if status == "succeeded" else 1,
        }
    reopened_items = operation.result_summary.get("reopened_items")
    if not isinstance(reopened_items, list):
        reopened_items = []
    return {
        "operation_id": operation.id,
        "status": status,
        "reopened_move_item_count": reopened_count,
        "item_result_summary": item_result_summary,
        "reopened_items": reopened_items,
    }


def _reopen_failure_response_payload(
    *,
    operation: OperationRecord,
    error_code: str,
    error_message: str,
    total_count: int,
) -> dict[str, object]:
    return {
        "operation_id": operation.id,
        "status": "failed",
        "reopened_move_item_count": 0,
        "item_result_summary": {
            "succeeded": 0,
            "failed": total_count,
            "total": total_count,
        },
        "reopened_items": [],
        "error": {
            "code": error_code,
            "message": error_message,
        },
    }


def _resolve_reopen_scope_or_raise(
    scope: SelectionScope,
    moves: list[JobMoveRecord],
) -> list[JobMoveRecord]:
    if scope.kind == "all":
        return [move for move in moves if move.status == "withdrawn"]
    raise ApiServiceError(
        status_code=422,
        code="invalid_reopen_selection",
        message="reopen from review currently supports all scope",
        details={"scope_kind": scope.kind},
    )


def _reopen_review_specs_or_raise(
    job: JobRecord,
    moves: list[JobMoveRecord],
    withdraw_results: dict[int, OperationItemResultRecord],
    *,
    expected_move_item_ids: list[int] | None = None,
) -> list[ReopenReviewMoveSpec]:
    expected_ids = set(expected_move_item_ids or [move.id for move in moves])
    moves_by_id = {move.id: move for move in moves}
    problems: list[dict[str, object]] = [
        {
            "move_item_id": move_id,
            "code": "move_item_missing",
            "message": "selected move item no longer exists",
        }
        for move_id in sorted(expected_ids - set(moves_by_id))
    ]
    specs: list[ReopenReviewMoveSpec] = []
    for move in moves:
        result = withdraw_results.get(move.id)
        problem = _reopen_readiness_problem(job, move, result)
        if problem is not None:
            problems.append(problem)
            continue
        assert result is not None
        assert result.to_path is not None
        review_path = _path_ref_to_path(result.to_path)
        assert review_path is not None
        specs.append(
            ReopenReviewMoveSpec(
                move=move,
                review_path=review_path,
                from_path=_validation_path_ref(
                    review_path,
                    root_kind="review",
                    root=_review_root(job),
                ).model_dump(mode="json"),
                to_path=_validation_path_ref(
                    move.dst,
                    root_kind="output",
                    root=Path(job.output_root),
                ).model_dump(mode="json"),
            )
        )
    if problems:
        raise ApiServiceError(
            status_code=422,
            code="reopen_from_review_not_ready",
            message="one or more move items cannot be reopened from review",
            details={
                "problems": problems,
                "affected_move_item_ids": sorted(expected_ids),
            },
        )
    return specs


def _reopen_readiness_problem(
    job: JobRecord,
    move: JobMoveRecord,
    withdraw_result: OperationItemResultRecord | None,
) -> dict[str, object] | None:
    base = {
        "move_item_id": move.id,
        "source_id": move.src_id,
        "kind": move.kind,
        "planned_destination": move.dst,
    }
    if move.status != "withdrawn":
        return {
            **base,
            "code": "invalid_move_state",
            "message": "only withdrawn move items can be reopened from review",
            "move_status": move.status,
        }
    if withdraw_result is None or withdraw_result.to_path is None:
        return {
            **base,
            "code": "missing_withdraw_record",
            "message": "successful withdraw result for move item was not found",
        }
    review_path = _path_ref_to_path(withdraw_result.to_path)
    if review_path is None:
        return {
            **base,
            "code": "invalid_withdraw_record",
            "message": "withdraw result does not contain a usable review path",
        }
    return _reopen_review_path_problem(job, move, review_path, base=base)


def _reopen_review_path_problem(
    job: JobRecord,
    move: JobMoveRecord,
    review_path: Path,
    *,
    base: dict[str, object] | None = None,
) -> dict[str, object] | None:
    details = base or {
        "move_item_id": move.id,
        "source_id": move.src_id,
        "kind": move.kind,
        "planned_destination": move.dst,
    }
    review_root = _review_root(job)
    if (
        _path_has_env_component(review_path)
        or _path_has_env_component(review_root)
        or _path_has_env_component(move.dst)
    ):
        return {
            **details,
            "code": "sensitive_env_path",
            "message": "sensitive .env paths cannot be reopened from review",
            "review_path": str(review_path),
        }
    review_root_resolved = review_root.resolve(strict=False)
    review_path_resolved = review_path.resolve(strict=False)
    if not review_path_resolved.is_relative_to(review_root_resolved):
        return {
            **details,
            "code": "outside_review_root",
            "message": "review path is outside review root",
            "review_root": str(review_root_resolved),
            "review_path": str(review_path_resolved),
        }
    if not review_path.is_file():
        return {
            **details,
            "code": "review_source_missing",
            "message": "review source file is missing",
            "review_path": str(review_path),
        }
    return None


def _validated_withdraw_check_or_raise(
    operations: list[OperationRecord],
    *,
    payload: WithdrawRequest,
    current_versions: dict[str, int],
) -> tuple[OperationRecord, RollbackEligibilityResponse]:
    for operation in operations:
        if operation.operation_type != "rollback_eligibility":
            continue
        if operation.status != "succeeded":
            continue
        stored_payload = operation.result_summary.get("response_payload")
        if not isinstance(stored_payload, dict):
            continue
        check = RollbackEligibilityResponse.model_validate(stored_payload)
        if check.check_token != payload.check_token:
            continue
        if check.mode != payload.mode:
            raise ApiServiceError(
                status_code=422,
                code="invalid_rollback_check",
                message="rollback check mode does not match withdraw request",
            )
        if check.status != "eligible" or check.token_status != "current":
            raise ApiServiceError(
                status_code=422,
                code="invalid_rollback_check",
                message="rollback check is not currently eligible",
                details=check.model_dump(mode="json"),
            )
        if not _rollback_check_token_time_current(check):
            raise ApiServiceError(
                status_code=412,
                code="stale_rollback_check",
                message="rollback check token has expired",
                details=check.model_dump(mode="json"),
            )
        if check.checked_move_state_revision != current_versions["move_state_revision"]:
            raise ApiServiceError(
                status_code=412,
                code="stale_rollback_check",
                message="rollback check move state revision is stale",
                details={
                    "checked_move_state_revision": check.checked_move_state_revision,
                    "current_move_state_revision": current_versions[
                        "move_state_revision"
                    ],
                },
            )
        return operation, check
    raise ApiServiceError(
        status_code=422,
        code="invalid_rollback_check",
        message="rollback check token was not found",
    )


def _rollback_check_token_time_current(check: RollbackEligibilityResponse) -> bool:
    try:
        checked_at = datetime.fromisoformat(check.checked_at)
    except ValueError:
        return False
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) <= checked_at + timedelta(
        seconds=check.ttl_seconds
    )


def _unsupported_restore_eligibility_response_payload(
    *,
    job: JobRecord,
    moves: list[JobMoveRecord],
    mode: str,
    operation_id: int,
    checked_at: str,
    checked_move_state_revision: int,
) -> dict[str, object]:
    incomplete_records = [
        {
            "move_item_id": move.id,
            "reason": "restore_mode_not_implemented",
        }
        for move in moves
    ]
    return {
        "mode": mode,
        "status": "ineligible",
        "checked_at": checked_at,
        "ttl_seconds": 300,
        "check_token": None,
        "checked_move_state_revision": checked_move_state_revision,
        "token_status": "not_current",
        "destination_preview": [
            {
                "move_item_id": move.id,
                "from": _validation_path_ref(
                    move.dst,
                    root_kind="output",
                    root=Path(job.output_root),
                ).model_dump(mode="json"),
                "to": _validation_path_ref(
                    _review_destination(job, move, operation_id),
                    root_kind="review",
                    root=_review_root(job),
                ).model_dump(mode="json"),
            }
            for move in moves
        ],
        "item_result_summary": {
            "checked": len(moves),
            "eligible": 0,
            "ineligible": len(moves),
        },
        "missing_files": _missing_current_files(moves),
        "incomplete_records": incomplete_records,
        "target_conflicts": [],
        "identity_mismatches": incomplete_records,
        "retryable_items": [],
        "non_retryable_items": [
            {
                "move_item_id": move.id,
                "retry_block_reason": "restore_mode_not_implemented",
            }
            for move in moves
        ],
        "affected_move_items": [move.id for move in moves],
        "archive_restore_summary": _archive_restore_summary(job),
    }


def _withdraw_eligibility_response_payload(
    *,
    job: JobRecord,
    moves: list[JobMoveRecord],
    operation_id: int,
    checked_at: str,
    checked_move_state_revision: int,
    check_token: str,
) -> dict[str, object]:
    problems: list[dict[str, object]] = []
    missing_files: list[dict[str, object]] = []
    target_conflicts: list[dict[str, object]] = []
    destination_keys: dict[str, list[int]] = {}
    preview: list[dict[str, object]] = []
    for move in moves:
        destination = Path(_review_destination(job, move, operation_id))
        preview.append(
            {
                "move_item_id": move.id,
                "from": _validation_path_ref(
                    move.dst,
                    root_kind="output",
                    root=Path(job.output_root),
                ).model_dump(mode="json"),
                "to": _validation_path_ref(
                    destination,
                    root_kind="review",
                    root=_review_root(job),
                ).model_dump(mode="json"),
            }
        )
        problem = _withdraw_readiness_problem(job, move, destination)
        if problem is not None:
            problems.append(problem)
            if problem.get("code") == "source_missing":
                missing_files.append(problem)
            if problem.get("code") in {
                "review_destination_exists",
                "review_destination_collision",
                "review_parent_not_directory",
            }:
                target_conflicts.append(problem)
            continue
        key = _path_key(destination.resolve(strict=False))
        destination_keys.setdefault(key, []).append(move.id)
    for key, move_ids in destination_keys.items():
        if len(move_ids) <= 1:
            continue
        problem = {
            "move_item_id": None,
            "code": "review_destination_collision",
            "message": "multiple moved items would withdraw to the same review path",
            "destination": key,
            "conflicting_move_item_ids": move_ids,
        }
        problems.append(problem)
        target_conflicts.append(problem)
    eligible = len(moves) if moves and not problems else 0
    status = "eligible" if eligible else "ineligible"
    return {
        "mode": "withdraw_to_review",
        "status": status,
        "checked_at": checked_at,
        "ttl_seconds": 300,
        "check_token": check_token if status == "eligible" else None,
        "checked_move_state_revision": checked_move_state_revision,
        "token_status": "current" if status == "eligible" else "not_current",
        "destination_preview": preview,
        "item_result_summary": {
            "checked": len(moves),
            "eligible": eligible,
            "ineligible": 0 if status == "eligible" else len(moves),
        },
        "missing_files": missing_files,
        "incomplete_records": [],
        "target_conflicts": target_conflicts,
        "identity_mismatches": [],
        "retryable_items": [
            {"move_item_id": move.id}
            for move in moves
            if status == "eligible"
        ],
        "non_retryable_items": [
            {
                "move_item_id": problem.get("move_item_id"),
                "retry_block_reason": problem.get("code"),
            }
            for problem in problems
        ],
        "affected_move_items": [move.id for move in moves],
        "archive_restore_summary": _archive_restore_summary(job),
    }


def _archive_restore_summary(job: JobRecord) -> dict[str, object]:
    return {
        "archive_path": (
            _validation_path_ref(job.archive_path, root_kind="archive").model_dump(
                mode="json"
            )
            if job.archive_path
            else None
        ),
        "has_archive_records": job.archive_path is not None,
    }


def _review_root(job: JobRecord) -> Path:
    if job.fail_path:
        return Path(job.fail_path).parent.parent / MONITOR_REVIEW_DIR
    return Path(job.series_dir).parent / MONITOR_REVIEW_DIR


def _destination_reservation_spec(
    job: JobRecord,
    move: JobMoveRecord,
) -> DestinationReservationSpec:
    if _path_has_env_component(move.src) or _path_has_env_component(move.dst):
        raise ApiServiceError(
            status_code=422,
            code="sensitive_env_path",
            message="sensitive .env paths cannot be reserved for apply",
            details={"move_item_id": move.id, "source_id": move.src_id},
        )
    output_root = Path(job.output_root).resolve(strict=False)
    dst = Path(move.dst).resolve(strict=False)
    parent = dst.parent.resolve(strict=False)
    ancestors: list[str] = []
    current = parent
    while True:
        ancestors.append(str(current))
        if current == current.parent:
            break
        current = current.parent
    dst_key = str(dst)
    casefold_key = dst_key.casefold()
    return DestinationReservationSpec(
        root_id="output:default:1",
        root_generation=1,
        root_alias_group_id="output:default",
        path=_validation_path_ref(
            dst,
            root_kind="output",
            root=output_root,
        ).model_dump(mode="json"),
        normalized_path_key=dst_key,
        casefold_path_key=casefold_key,
        canonical_destination_key=dst_key,
        canonical_parent_key=str(parent),
        ancestor_keys=tuple(ancestors),
        descendant_prefix_key=f"{dst_key}/",
    )


def _withdraw_destination_reservation_spec(
    job: JobRecord,
    move: JobMoveRecord,
    destination: Path,
) -> DestinationReservationSpec:
    if _path_has_env_component(move.dst) or _path_has_env_component(destination):
        raise ApiServiceError(
            status_code=422,
            code="sensitive_env_path",
            message="sensitive .env paths cannot be reserved for withdraw",
            details={"move_item_id": move.id, "source_id": move.src_id},
        )
    review_root = _review_root(job).resolve(strict=False)
    dst = destination.resolve(strict=False)
    parent = dst.parent.resolve(strict=False)
    ancestors: list[str] = []
    current = parent
    while True:
        ancestors.append(str(current))
        if current == current.parent:
            break
        current = current.parent
    dst_key = str(dst)
    return DestinationReservationSpec(
        root_id="review:default:1",
        root_generation=1,
        root_alias_group_id="review:default",
        path=_validation_path_ref(
            dst,
            root_kind="review",
            root=review_root,
        ).model_dump(mode="json"),
        normalized_path_key=dst_key,
        casefold_path_key=dst_key.casefold(),
        canonical_destination_key=dst_key,
        canonical_parent_key=str(parent),
        ancestor_keys=tuple(ancestors),
        descendant_prefix_key=f"{dst_key}/",
    )


def _apply_move_readiness_problem(
    job: JobRecord,
    move: JobMoveRecord,
) -> dict[str, object] | None:
    result = _apply_validation_item(
        move,
        output_root=Path(job.output_root).resolve(strict=False),
        destination_keys={},
    )
    if result["status"] == "ok":
        return None
    return _validation_problem_summary(result)


def _withdraw_readiness_problem(
    job: JobRecord,
    move: JobMoveRecord,
    destination: Path,
) -> dict[str, object] | None:
    src = Path(move.dst)
    review_root = _review_root(job)
    output_root = Path(job.output_root)
    base = {
        "move_item_id": move.id,
        "source_id": move.src_id,
        "kind": move.kind,
        "source": str(src),
        "destination": str(destination),
    }
    if (
        _path_has_env_component(src)
        or _path_has_env_component(destination)
        or _path_has_env_component(review_root)
        or _path_has_env_component(output_root)
    ):
        return {
            **base,
            "code": "sensitive_env_path",
            "message": "sensitive .env paths cannot be withdrawn",
        }
    if move.status != "moved":
        return {
            **base,
            "code": "invalid_move_state",
            "message": "only moved items can be withdrawn to review",
            "move_status": move.status,
        }
    src_resolved = src.resolve(strict=False)
    output_root_resolved = output_root.resolve(strict=False)
    if not src_resolved.is_relative_to(output_root_resolved):
        return {
            **base,
            "code": "outside_output_root",
            "message": "current output path is outside output root",
            "output_root": str(output_root_resolved),
        }
    if not src.is_file():
        return {
            **base,
            "code": "source_missing",
            "message": "current output file is missing",
        }
    review_root_resolved = review_root.resolve(strict=False)
    dst_resolved = destination.resolve(strict=False)
    if not dst_resolved.is_relative_to(review_root_resolved):
        return {
            **base,
            "code": "outside_review_root",
            "message": "review destination is outside review root",
            "review_root": str(review_root_resolved),
        }
    if dst_resolved.exists():
        return {
            **base,
            "code": "review_destination_exists",
            "message": "review destination already exists",
        }
    parent = dst_resolved.parent
    if parent.exists() and not parent.is_dir():
        return {
            **base,
            "code": "review_parent_not_directory",
            "message": "review destination parent exists but is not a directory",
            "parent": str(parent),
        }
    return None


def _execute_apply_moves(
    *,
    store: RuntimeStore,
    job: JobRecord,
    operation: OperationRecord,
    moves: list[JobMoveRecord],
    reservations: list[object],
    validation_id: int,
) -> dict[str, object]:
    reservations_by_index = list(reservations)
    if len(reservations_by_index) != len(moves):
        raise ApiServiceError(
            status_code=500,
            code="apply_failed",
            message="destination reservation count did not match move count",
            details={
                "move_count": len(moves),
                "reservation_count": len(reservations_by_index),
                "validation_id": validation_id,
            },
        )
    moved: list[tuple[JobMoveRecord, Path, Path, int]] = []
    moved_items: list[dict[str, object]] = []
    response_payload: dict[str, object] | None = None
    claimed_move_ids: set[int] = set()
    try:
        try:
            claimed_moves = store.mark_apply_items_applying(
                job_id=job.id,
                move_item_ids=[move.id for move in moves],
            )
            claimed_move_ids = {move.id for move in claimed_moves}
        except Exception as exc:
            raise ApiServiceError(
                status_code=409,
                code="apply_move_not_ready",
                message="move items could not be claimed for apply",
                details={"error": str(exc), "validation_id": validation_id},
            ) from exc
        for index, move in enumerate(moves):
            src = Path(move.src)
            dst = Path(move.dst)
            reservation = reservations_by_index[index]
            reservation_id = getattr(reservation, "id", None)
            readiness_problem = _apply_move_readiness_problem(job, move)
            from_path = _validation_path_ref(
                src,
                root_kind="source",
                root=Path(job.series_dir),
            ).model_dump(mode="json")
            to_path = _validation_path_ref(
                dst,
                root_kind="output",
                root=Path(job.output_root),
            ).model_dump(mode="json")
            event = store.create_path_event(
                operation_id=operation.id,
                job_id=job.id,
                subject_type="move_item",
                move_item_id=move.id,
                source_file_id=move.src_id,
                from_path=from_path,
                to_path=to_path,
                reservation_id=reservation_id,
                pre_identity=(
                    {
                        "identity_strength": "not_checked",
                        "reason": "readiness_failed",
                    }
                    if readiness_problem is not None
                    else _file_identity(src)
                ),
            )
            event_id = int(event["id"])
            try:
                if readiness_problem is not None:
                    raise ApiServiceError(
                        status_code=409,
                        code="apply_move_not_ready",
                        message="move item is no longer ready for apply",
                        details=readiness_problem,
                    )
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            except Exception as move_exc:
                now = _now_iso()
                store.update_path_event(
                    event_id,
                    result="failed",
                    fs_done_at=now,
                    db_done_at=now,
                    error_code=(
                        move_exc.code
                        if isinstance(move_exc, ApiServiceError)
                        else type(move_exc).__name__
                    ),
                    error_message=str(move_exc),
                )
                store.create_operation_item_result(
                    operation_id=operation.id,
                    subject_kind="move_item",
                    subject_id=move.id,
                    status="failed",
                    path_event_id=event_id,
                    reservation_id=reservation_id,
                    from_path=from_path,
                    to_path=to_path,
                    identity_check="not_applicable",
                    error_code=(
                        move_exc.code
                        if isinstance(move_exc, ApiServiceError)
                        else type(move_exc).__name__
                    ),
                    error_message=str(move_exc),
                    retryable=True,
                )
                raise
            moved.append((move, src, dst, event_id))
            now = _now_iso()
            post_identity = _file_identity(dst)
            store.update_path_event(
                event_id,
                result="succeeded",
                fs_done_at=now,
                db_done_at=now,
                post_identity=post_identity,
            )
            store.update_job_move(
                move.id,
                status="moved",
                error_message=None,
                applied_at=now,
            )
            store.create_operation_item_result(
                operation_id=operation.id,
                subject_kind="move_item",
                subject_id=move.id,
                status="succeeded",
                path_event_id=int(event["id"]),
                reservation_id=reservation_id,
                from_path=from_path,
                to_path=to_path,
                identity_check="matched",
            )
            moved_items.append(
                {
                    "move_item_id": move.id,
                    "source_id": move.src_id,
                    "kind": move.kind,
                    "from": from_path,
                    "to": to_path,
                    "path_event_id": event_id,
                    "reservation_id": reservation_id,
                }
            )
        response_payload = {
            "operation_id": operation.id,
            "status": "succeeded",
            "validation_id": validation_id,
            "applied_move_item_count": len(moved_items),
            "item_result_summary": {
                "succeeded": len(moved_items),
                "failed": 0,
                "total": len(moves),
            },
            "moved_items": moved_items,
        }
        store.complete_apply_operation_success(
            operation_id=operation.id,
            job_id=job.id,
            result_summary={
                "status": "succeeded",
                "validation_id": validation_id,
                "applied_move_item_count": response_payload[
                    "applied_move_item_count"
                ],
                "item_result_summary": response_payload["item_result_summary"],
                "moved_items": moved_items,
                "response_payload": response_payload,
            },
        )
    except Exception as exc:
        if not claimed_move_ids:
            if isinstance(exc, ApiServiceError):
                raise exc
            raise ApiServiceError(
                status_code=500,
                code="apply_failed",
                message="apply failed before move items were claimed",
                details={"error": str(exc), "validation_id": validation_id},
            ) from exc
        rollback_errors: list[str] = []
        for move, src, dst, event_id in reversed(moved):
            try:
                if dst.exists():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dst), str(src))
                store.update_path_event(
                    event_id,
                    result="rolled_back",
                    fs_done_at=_now_iso(),
                    db_done_at=_now_iso(),
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
                store.update_job_move(
                    move.id,
                    status="proposed",
                    error_message=f"apply rolled back after {type(exc).__name__}: {exc}",
                    applied_at=None,
                )
                store.create_operation_item_result(
                    operation_id=operation.id,
                    subject_kind="move_item",
                    subject_id=move.id,
                    status="rolled_back",
                    path_event_id=event_id,
                    from_path=_validation_path_ref(
                        dst,
                        root_kind="output",
                        root=Path(job.output_root),
                    ).model_dump(mode="json"),
                    to_path=_validation_path_ref(
                        src,
                        root_kind="source",
                        root=Path(job.series_dir),
                    ).model_dump(mode="json"),
                    identity_check="not_applicable",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                    result_revision=2,
                )
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        moved_ids = {move.id for move, _src, _dst, _event_id in moved}
        for move in moves:
            if move.id in moved_ids or move.id not in claimed_move_ids:
                continue
            try:
                store.update_job_move(
                    move.id,
                    status="proposed",
                    error_message=f"apply aborted after {type(exc).__name__}: {exc}",
                    applied_at=None,
                )
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        store.update_job(
            job.id,
            status="failed",
            error_stage="apply",
            error_message=str(exc),
            finished_at=_now_iso(),
        )
        if isinstance(exc, ApiServiceError):
            details = dict(exc.details or {})
            details["rollback_errors"] = rollback_errors
            raise ApiServiceError(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                details=details,
            ) from exc
        raise ApiServiceError(
            status_code=500,
            code="apply_failed",
            message="apply failed and rollback was attempted",
            details={
                "error": str(exc),
                "rollback_errors": rollback_errors,
                "validation_id": validation_id,
            },
        ) from exc
    assert response_payload is not None
    return response_payload


def _execute_withdraw_moves(
    *,
    store: RuntimeStore,
    job: JobRecord,
    operation: OperationRecord,
    moves: list[JobMoveRecord],
    destinations: dict[int, Path],
    reservations: list[object],
) -> dict[str, object]:
    reservations_by_index = list(reservations)
    if len(reservations_by_index) != len(moves):
        raise ApiServiceError(
            status_code=500,
            code="withdraw_failed",
            message="destination reservation count did not match move count",
            details={
                "move_count": len(moves),
                "reservation_count": len(reservations_by_index),
            },
        )
    moved: list[tuple[JobMoveRecord, Path, Path, int]] = []
    withdrawn_items: list[dict[str, object]] = []
    claimed_move_ids: set[int] = set()
    response_payload: dict[str, object] | None = None
    try:
        try:
            claimed_moves = store.mark_withdraw_items_applying(
                job_id=job.id,
                move_item_ids=[move.id for move in moves],
            )
            claimed_move_ids = {move.id for move in claimed_moves}
        except Exception as exc:
            raise ApiServiceError(
                status_code=409,
                code="withdraw_move_not_ready",
                message="move items could not be claimed for withdraw",
                details={"error": str(exc)},
            ) from exc
        for index, move in enumerate(moves):
            src = Path(move.dst)
            dst = destinations[move.id]
            reservation = reservations_by_index[index]
            reservation_id = getattr(reservation, "id", None)
            readiness_problem = _withdraw_readiness_problem(job, move, dst)
            from_path = _validation_path_ref(
                src,
                root_kind="output",
                root=Path(job.output_root),
            ).model_dump(mode="json")
            to_path = _validation_path_ref(
                dst,
                root_kind="review",
                root=_review_root(job),
            ).model_dump(mode="json")
            event = store.create_path_event(
                operation_id=operation.id,
                job_id=job.id,
                subject_type="move_item",
                move_item_id=move.id,
                source_file_id=move.src_id,
                from_path=from_path,
                to_path=to_path,
                reservation_id=reservation_id,
                pre_identity=(
                    {
                        "identity_strength": "not_checked",
                        "reason": "readiness_failed",
                    }
                    if readiness_problem is not None
                    else _file_identity(src)
                ),
            )
            event_id = int(event["id"])
            try:
                if readiness_problem is not None:
                    raise ApiServiceError(
                        status_code=409,
                        code="withdraw_move_not_ready",
                        message="move item is no longer ready for withdraw",
                        details=readiness_problem,
                    )
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            except Exception as move_exc:
                now = _now_iso()
                store.update_path_event(
                    event_id,
                    result="failed",
                    fs_done_at=now,
                    db_done_at=now,
                    error_code=(
                        move_exc.code
                        if isinstance(move_exc, ApiServiceError)
                        else type(move_exc).__name__
                    ),
                    error_message=str(move_exc),
                )
                store.create_operation_item_result(
                    operation_id=operation.id,
                    subject_kind="move_item",
                    subject_id=move.id,
                    status="failed",
                    path_event_id=event_id,
                    reservation_id=reservation_id,
                    from_path=from_path,
                    to_path=to_path,
                    identity_check="not_applicable",
                    error_code=(
                        move_exc.code
                        if isinstance(move_exc, ApiServiceError)
                        else type(move_exc).__name__
                    ),
                    error_message=str(move_exc),
                    retryable=True,
                )
                raise
            moved.append((move, src, dst, event_id))
            now = _now_iso()
            post_identity = _file_identity(dst)
            store.update_path_event(
                event_id,
                result="succeeded",
                fs_done_at=now,
                db_done_at=now,
                post_identity=post_identity,
            )
            store.update_job_move(
                move.id,
                status="withdrawn",
                error_message=None,
                applied_at=now,
            )
            store.create_operation_item_result(
                operation_id=operation.id,
                subject_kind="move_item",
                subject_id=move.id,
                status="succeeded",
                path_event_id=event_id,
                reservation_id=reservation_id,
                from_path=from_path,
                to_path=to_path,
                identity_check="matched",
            )
            withdrawn_items.append(
                {
                    "move_item_id": move.id,
                    "source_id": move.src_id,
                    "kind": move.kind,
                    "from": from_path,
                    "to": to_path,
                    "path_event_id": event_id,
                    "reservation_id": reservation_id,
                }
            )
        response_payload = {
            "operation_id": operation.id,
            "status": "succeeded",
            "mode": "withdraw_to_review",
            "withdrawn_move_item_count": len(withdrawn_items),
            "item_result_summary": {
                "succeeded": len(withdrawn_items),
                "failed": 0,
                "total": len(moves),
            },
            "withdrawn_items": withdrawn_items,
        }
        store.complete_withdraw_operation_success(
            operation_id=operation.id,
            job_id=job.id,
            result_summary={
                "status": "succeeded",
                "mode": "withdraw_to_review",
                "withdrawn_move_item_count": response_payload[
                    "withdrawn_move_item_count"
                ],
                "item_result_summary": response_payload["item_result_summary"],
                "withdrawn_items": withdrawn_items,
                "response_payload": response_payload,
            },
        )
    except Exception as exc:
        if not claimed_move_ids:
            if isinstance(exc, ApiServiceError):
                raise exc
            raise ApiServiceError(
                status_code=500,
                code="withdraw_failed",
                message="withdraw failed before move items were claimed",
                details={"error": str(exc)},
            ) from exc
        rollback_errors: list[str] = []
        for move, src, dst, event_id in reversed(moved):
            try:
                if dst.exists():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dst), str(src))
                store.update_path_event(
                    event_id,
                    result="rolled_back",
                    fs_done_at=_now_iso(),
                    db_done_at=_now_iso(),
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
                store.update_job_move(
                    move.id,
                    status="moved",
                    error_message=f"withdraw rolled back after {type(exc).__name__}: {exc}",
                    applied_at=move.applied_at,
                )
                store.create_operation_item_result(
                    operation_id=operation.id,
                    subject_kind="move_item",
                    subject_id=move.id,
                    status="rolled_back",
                    path_event_id=event_id,
                    from_path=_validation_path_ref(
                        dst,
                        root_kind="review",
                        root=_review_root(job),
                    ).model_dump(mode="json"),
                    to_path=_validation_path_ref(
                        src,
                        root_kind="output",
                        root=Path(job.output_root),
                    ).model_dump(mode="json"),
                    identity_check="not_applicable",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                    result_revision=2,
                )
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        moved_ids = {move.id for move, _src, _dst, _event_id in moved}
        for move in moves:
            if move.id in moved_ids or move.id not in claimed_move_ids:
                continue
            try:
                store.update_job_move(
                    move.id,
                    status="moved",
                    error_message=f"withdraw aborted after {type(exc).__name__}: {exc}",
                    applied_at=move.applied_at,
                )
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        store.update_job(
            job.id,
            status="failed",
            error_stage="withdraw",
            error_message=str(exc),
            finished_at=_now_iso(),
        )
        if isinstance(exc, ApiServiceError):
            details = dict(exc.details or {})
            details["rollback_errors"] = rollback_errors
            raise ApiServiceError(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                details=details,
            ) from exc
        raise ApiServiceError(
            status_code=500,
            code="withdraw_failed",
            message="withdraw failed and rollback was attempted",
            details={"error": str(exc), "rollback_errors": rollback_errors},
        ) from exc
    assert response_payload is not None
    return response_payload


def _execute_reopen_from_review(
    *,
    store: RuntimeStore,
    job: JobRecord,
    operation: OperationRecord,
    specs: list[ReopenReviewMoveSpec],
) -> dict[str, object]:
    if not specs:
        raise ApiServiceError(
            status_code=422,
            code="no_reopenable_items",
            message="no withdrawn move items were selected",
        )
    updates: list[ReopenMoveItemUpdate] = []
    reopened_items: list[dict[str, object]] = []
    for spec in specs:
        problem = _reopen_review_path_problem(job, spec.move, spec.review_path)
        if problem is not None:
            raise ApiServiceError(
                status_code=422,
                code="reopen_from_review_not_ready",
                message="move item cannot be reopened from review",
                details=problem,
            )
        identity = _file_identity(spec.review_path)
        updates.append(
            ReopenMoveItemUpdate(
                move_item_id=spec.move.id,
                source_path=str(spec.review_path),
                from_path={
                    **spec.from_path,
                    "file_identity": identity,
                    "original_source_path": _validation_path_ref(
                        spec.move.src,
                        root_kind="source",
                        root=Path(job.series_dir),
                    ).model_dump(mode="json"),
                },
                to_path=spec.to_path,
            )
        )
        reopened_items.append(
            {
                "move_item_id": spec.move.id,
                "source_id": spec.move.src_id,
                "kind": spec.move.kind,
                "from": spec.from_path,
                "to": spec.to_path,
                "original_source": _validation_path_ref(
                    spec.move.src,
                    root_kind="source",
                    root=Path(job.series_dir),
                ).model_dump(mode="json"),
            }
        )
    response_payload = {
        "operation_id": operation.id,
        "status": "succeeded",
        "reopened_move_item_count": len(reopened_items),
        "item_result_summary": {
            "succeeded": len(reopened_items),
            "failed": 0,
            "total": len(specs),
        },
        "reopened_items": reopened_items,
    }
    store.complete_reopen_from_review_operation_success(
        operation_id=operation.id,
        job_id=job.id,
        updates=updates,
        result_summary={
            "status": "succeeded",
            "reopened_move_item_count": response_payload[
                "reopened_move_item_count"
            ],
            "item_result_summary": response_payload["item_result_summary"],
            "reopened_items": reopened_items,
            "response_payload": response_payload,
        },
    )
    return response_payload


def _file_identity(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "identity_strength": "missing",
            "error": str(exc),
        }
    return {
        "identity_strength": "weak",
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _validation_problem_summary(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key
        in {
            "move_item_id",
            "source_id",
            "kind",
            "code",
            "message",
            "source",
            "destination",
            "output_root",
            "parent",
            "move_status",
            "conflicting_move_item_ids",
        }
    }


def _resolve_scope_or_raise(
    scope: SelectionScope,
    moves: list[JobMoveRecord],
) -> list[JobMoveRecord]:
    if scope.kind == "all":
        return [move for move in moves if move.status == "moved"]
    if scope.kind == "move_items":
        selected = set(scope.ids or [])
        moves_by_id = {move.id: move for move in moves}
        unknown_ids = sorted(selected - set(moves_by_id))
        not_moved_ids = sorted(
            move_id
            for move_id in selected & set(moves_by_id)
            if moves_by_id[move_id].status != "moved"
        )
        if unknown_ids or not_moved_ids:
            raise ApiServiceError(
                status_code=422,
                code="invalid_rollback_selection",
                message="selected move items are not rollback eligible",
                details={
                    "unknown_move_item_ids": unknown_ids,
                    "not_moved_move_item_ids": not_moved_ids,
                },
            )
        return [move for move in moves if move.id in selected]
    return []


def _review_destination(
    job: JobRecord,
    move: JobMoveRecord,
    operation_id: int,
) -> str:
    return str(
        _review_root(job)
        / f"{job.id}-{operation_id}"
        / Path(move.dst).name
    )


def _build_episode_targets(moves: list[JobMoveRecord]) -> list[ReviewEpisodeTarget]:
    targets: list[ReviewEpisodeTarget] = []
    target_id = 1
    for move in moves:
        if move.kind != "video":
            continue
        parsed = _parse_episode_from_destination(move.dst)
        if parsed is None:
            status = "failed" if move.status == "failed" else "manual_needed"
            season = 0
            episode_start = 0
            episode_end = 0
        else:
            season, episode_start, episode_end = parsed
            status = "applied" if move.status == "moved" else move.status
        targets.append(
            ReviewEpisodeTarget(
                id=target_id,
                video_source_id=move.src_id,
                season=season,
                episode_start=episode_start,
                episode_end=episode_end,
                status=status,
                manual=False,
                created_by="db_snapshot",
                move_item_ids=[move.id],
                subtitle_link_ids=[],
            )
        )
        target_id += 1
    links = _build_subtitle_links(moves, targets)
    link_ids_by_target: dict[int, list[int]] = {}
    for link in links:
        link_ids_by_target.setdefault(link.episode_target_id, []).append(link.id)
    return [
        target.model_copy(
            update={"subtitle_link_ids": link_ids_by_target.get(target.id, [])}
        )
        for target in targets
    ]


def _build_subtitle_links(
    moves: list[JobMoveRecord],
    episode_targets: list[ReviewEpisodeTarget],
) -> list[ReviewSubtitleLink]:
    target_by_base = {
        _video_target_base(move.dst): target
        for move, target in zip(
            [item for item in moves if item.kind == "video"],
            episode_targets,
            strict=False,
        )
    }
    links: list[ReviewSubtitleLink] = []
    for move in moves:
        if move.kind != "subtitle":
            continue
        target = target_by_base.get(_subtitle_target_base(move.dst))
        if target is None:
            continue
        link_id = len(links) + 1
        links.append(
            ReviewSubtitleLink(
                id=link_id,
                episode_target_id=target.id,
                subtitle_source_id=move.src_id,
                variant_detected=_subtitle_variant_from_dst(move.dst),
                variant_override=None,
                status="applied" if move.status == "moved" else move.status,
                created_by="db_snapshot",
                move_item_id=move.id,
            )
        )
    return links


def _parse_episode_from_destination(path: str) -> tuple[int, int, int] | None:
    match = _EPISODE_TOKEN_RE.search(Path(path).stem)
    if match is None:
        return None
    season = int(match.group("season"))
    start = int(match.group("start"))
    end = int(match.group("end") or match.group("start"))
    return season, start, end


def _video_target_base(path: str) -> str:
    return Path(path).stem


def _subtitle_target_base(path: str) -> str:
    stem = Path(path).stem
    parts = stem.split(".")
    for idx, part in enumerate(parts):
        if f".{part.casefold()}" in _SUBTITLE_VARIANT_SUFFIXES:
            return ".".join(parts[:idx])
    return stem


def _subtitle_variant_from_dst(path: str) -> str:
    for part in Path(path).stem.split("."):
        normalized = part.casefold()
        if f".{normalized}" in _SUBTITLE_VARIANT_SUFFIXES:
            return normalized
    return "unknown"


def _missing_current_files(moves: list[JobMoveRecord]) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    for move in moves:
        current_path = Path(move.dst if move.status == "moved" else move.src)
        if _path_has_env_component(current_path):
            missing.append(
                {
                    "move_item_id": move.id,
                    "path": _validation_path_ref(
                        current_path,
                        root_kind="output" if move.status == "moved" else "source",
                    ).model_dump(mode="json"),
                    "reason": "sensitive_env_path",
                }
            )
            continue
        if not current_path.exists():
            missing.append(
                {
                    "move_item_id": move.id,
                    "path": str(current_path),
                    "reason": "current_path_missing",
                }
            )
    return missing
