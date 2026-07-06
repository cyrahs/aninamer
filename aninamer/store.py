from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from aninamer.plan import PlannedMove


JobSourceKind = Literal["monitor", "api"]
JobStatus = Literal[
    "pending",
    "planning",
    "planned",
    "apply_requested",
    "applying",
    "succeeded",
    "failed",
    "cleared",
]
JobRequestKind = Literal["scan_now", "apply_job"]
JobRequestStatus = Literal["pending", "running", "succeeded", "failed", "rejected"]
ArtifactKind = Literal["result"]
OperationStatus = Literal["queued", "running", "succeeded", "partial", "failed", "canceled"]
JobMoveStatus = Literal[
    "proposed",
    "applying",
    "moved",
    "failed",
    "rolled_back",
    "withdrawn",
    "skipped",
]
NotificationSeverity = Literal["info", "success", "warning", "error"]
NotificationDeliveryStatus = Literal["pending", "delivered", "retry", "disabled"]

PENDING_JOB_STATUSES = frozenset(
    {"pending", "planning", "planned", "apply_requested", "applying"}
)

_JOB_COLUMNS = {
    "series_name",
    "watch_root_key",
    "source_kind",
    "status",
    "tmdb_id",
    "video_moves_count",
    "subtitle_moves_count",
    "started_at",
    "finished_at",
    "error_stage",
    "error_message",
    "series_dir",
    "output_root",
    "archive_path",
    "fail_path",
}
_JOB_REQUEST_COLUMNS = {
    "kind",
    "status",
    "target_job_id",
    "started_at",
    "finished_at",
    "error_message",
}
_JOB_MOVE_COLUMNS = {
    "src",
    "status",
    "error_message",
    "applied_at",
}
_OPERATION_COLUMNS = {
    "status",
    "progress",
    "result_summary",
    "error",
    "started_at",
    "finished_at",
}
_TIMESTAMP_FIELDS = {
    "started_at",
    "finished_at",
    "created_at",
    "updated_at",
    "last_scan_at",
    "next_attempt_at",
    "last_attempt_at",
    "delivered_at",
    "applied_at",
}


@dataclass(frozen=True)
class JobRecord:
    id: int
    series_name: str
    watch_root_key: str
    source_kind: JobSourceKind
    status: JobStatus
    tmdb_id: int | None
    video_moves_count: int
    subtitle_moves_count: int
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    error_stage: str | None
    error_message: str | None
    series_dir: str
    output_root: str
    archive_path: str | None
    fail_path: str | None


@dataclass(frozen=True)
class JobRequestRecord:
    id: int
    kind: JobRequestKind
    status: JobRequestStatus
    target_job_id: int | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    error_message: str | None


@dataclass(frozen=True)
class JobMoveRecord:
    id: int
    job_id: int
    src_id: int
    kind: Literal["video", "subtitle"]
    src: str
    dst: str
    status: JobMoveStatus
    error_message: str | None
    applied_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class NotificationRecord:
    id: int
    event_kind: str
    severity: NotificationSeverity
    title: str
    message: str
    job_id: int | None
    job_request_id: int | None
    payload: dict[str, Any]
    markdown: str
    image_url: str
    disable_web_page_preview: bool
    disable_notification: bool
    delivery_status: NotificationDeliveryStatus
    attempt_count: int
    next_attempt_at: str | None
    last_attempt_at: str | None
    delivered_at: str | None
    last_error: str | None
    created_at: str


@dataclass(frozen=True)
class LLMFailureRecord:
    id: int
    job_id: int
    stage: str
    attempt: int
    raw_reply: str
    error_message: str
    created_at: str


class IdempotencyConflictError(ValueError):
    pass


class DestinationReservationConflictError(ValueError):
    def __init__(self, conflict_code: str, conflict_ref: str) -> None:
        super().__init__(f"{conflict_code}: {conflict_ref}")
        self.conflict_code = conflict_code
        self.conflict_ref = conflict_ref


@dataclass(frozen=True)
class OperationRecord:
    id: int
    job_id: int
    operation_type: str
    status: OperationStatus
    idempotency_key: str | None
    request_payload_hash: str
    base_versions: dict[str, Any]
    progress: dict[str, Any]
    result_summary: dict[str, Any]
    error: dict[str, Any] | None
    poll_after_ms: int
    created_at: str
    started_at: str | None
    finished_at: str | None


@dataclass(frozen=True)
class OperationItemResultRecord:
    id: int
    operation_id: int
    subject_kind: str
    subject_id: int | None
    status: str
    path_event_id: int | None
    reservation_id: int | None
    from_path: dict[str, Any] | None
    to_path: dict[str, Any] | None
    identity_check: str
    error_code: str | None
    error_message: str | None
    retryable: bool
    retry_block_reason: str | None
    result_revision: int
    created_at: str


@dataclass(frozen=True)
class ReopenMoveItemUpdate:
    move_item_id: int
    source_path: str
    from_path: dict[str, Any]
    to_path: dict[str, Any]


@dataclass(frozen=True)
class ReviewSnapshotRecord:
    job_id: int
    source_files: list[dict[str, Any]]
    selected_tmdb_info: dict[str, Any]
    episode_targets: list[dict[str, Any]]
    subtitle_links: list[dict[str, Any]]
    mapping_attempts: list[dict[str, Any]]
    mapping_evidence: list[dict[str, Any]]
    path_event_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class DestinationReservationSpec:
    root_id: str
    root_generation: int
    root_alias_group_id: str
    path: dict[str, Any]
    normalized_path_key: str
    casefold_path_key: str
    canonical_destination_key: str
    canonical_parent_key: str
    ancestor_keys: tuple[str, ...]
    descendant_prefix_key: str


@dataclass(frozen=True)
class DestinationReservationRecord:
    id: int
    operation_id: int
    job_id: int
    root_id: str
    root_generation: int
    root_alias_group_id: str
    path: dict[str, Any]
    normalized_path_key: str
    casefold_path_key: str
    canonical_destination_key: str
    canonical_parent_key: str
    ancestor_keys: tuple[str, ...]
    descendant_prefix_key: str
    state: str
    conflict_code: str | None
    conflict_ref: str | None
    created_at: str
    expires_at: str | None
    released_at: str | None
    committed_at: str | None


@dataclass(frozen=True)
class RuntimeSnapshot:
    last_scan_at: str | None
    jobs: tuple[JobRecord, ...]
    job_requests: tuple[JobRequestRecord, ...]


class RuntimeStore:
    def __init__(self, postgres_dsn: str) -> None:
        self._postgres_dsn = postgres_dsn
        self.bootstrap()

    @property
    def postgres_dsn(self) -> str:
        return self._postgres_dsn

    def bootstrap(self) -> None:
        with self._connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_state (
                        id SMALLINT PRIMARY KEY CHECK (id = 1),
                        last_scan_at TIMESTAMPTZ NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        id BIGSERIAL PRIMARY KEY,
                        series_name TEXT NOT NULL,
                        watch_root_key TEXT NOT NULL,
                        source_kind TEXT NOT NULL
                            CHECK (source_kind IN ('monitor', 'api')),
                        status TEXT NOT NULL
                            CHECK (
                                status IN (
                                    'pending',
                                    'planning',
                                    'planned',
                                    'apply_requested',
                                    'applying',
                                    'succeeded',
                                    'failed',
                                    'cleared'
                                )
                            ),
                        tmdb_id INTEGER NULL,
                        video_moves_count INTEGER NOT NULL DEFAULT 0
                            CHECK (video_moves_count >= 0),
                        subtitle_moves_count INTEGER NOT NULL DEFAULT 0
                            CHECK (subtitle_moves_count >= 0),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        started_at TIMESTAMPTZ NULL,
                        finished_at TIMESTAMPTZ NULL,
                        error_stage TEXT NULL,
                        error_message TEXT NULL,
                        series_dir TEXT NOT NULL,
                        output_root TEXT NOT NULL,
                        archive_path TEXT NULL,
                        fail_path TEXT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    DO $$
                    DECLARE
                        constraint_name TEXT;
                    BEGIN
                        FOR constraint_name IN
                            SELECT conname
                            FROM pg_constraint
                            WHERE conrelid = 'jobs'::regclass
                              AND contype = 'c'
                              AND pg_get_constraintdef(oid) ILIKE '%status%'
                        LOOP
                            EXECUTE format(
                                'ALTER TABLE jobs DROP CONSTRAINT %I',
                                constraint_name
                            );
                        END LOOP;
                        EXECUTE $sql$
                            ALTER TABLE jobs
                            ADD CONSTRAINT jobs_status_check
                            CHECK (
                                status IN (
                                    'pending',
                                    'planning',
                                    'planned',
                                    'apply_requested',
                                    'applying',
                                    'succeeded',
                                    'failed',
                                    'cleared'
                                )
                            )
                        $sql$;
                    END $$;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS job_requests (
                        id BIGSERIAL PRIMARY KEY,
                        kind TEXT NOT NULL
                            CHECK (kind IN ('scan_now', 'apply_job')),
                        status TEXT NOT NULL
                            CHECK (
                                status IN (
                                    'pending',
                                    'running',
                                    'succeeded',
                                    'failed',
                                    'rejected'
                                )
                            ),
                        target_job_id BIGINT NULL REFERENCES jobs(id),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        started_at TIMESTAMPTZ NULL,
                        finished_at TIMESTAMPTZ NULL,
                        error_message TEXT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS job_artifacts (
                        job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        artifact_kind TEXT NOT NULL
                            CHECK (artifact_kind IN ('result', 'plan', 'rollback')),
                        payload JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (job_id, artifact_kind)
                    )
                    """
                )
                cur.execute(
                    """
                    DO $$
                    DECLARE
                        constraint_name TEXT;
                    BEGIN
                        FOR constraint_name IN
                            SELECT conname
                            FROM pg_constraint
                            WHERE conrelid = 'job_artifacts'::regclass
                              AND contype = 'c'
                              AND pg_get_constraintdef(oid) ILIKE '%artifact_kind%'
                        LOOP
                            EXECUTE format(
                                'ALTER TABLE job_artifacts DROP CONSTRAINT %I',
                                constraint_name
                            );
                        END LOOP;
                        EXECUTE $sql$
                            ALTER TABLE job_artifacts
                            ADD CONSTRAINT job_artifacts_kind_check
                            CHECK (artifact_kind IN ('result', 'plan', 'rollback'))
                        $sql$;
                    END $$;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS job_move_items (
                        id BIGSERIAL PRIMARY KEY,
                        job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        src_id INTEGER NOT NULL,
                        kind TEXT NOT NULL
                            CHECK (kind IN ('video', 'subtitle')),
                        src TEXT NOT NULL,
                        dst TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'proposed'
                            CHECK (
                                status IN (
                                    'proposed',
                                    'applying',
                                    'moved',
                                    'failed',
                                    'rolled_back',
                                    'withdrawn',
                                    'skipped'
                                )
                            ),
                        error_message TEXT NULL,
                        applied_at TIMESTAMPTZ NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE job_move_items
                    DROP CONSTRAINT IF EXISTS job_move_items_status_check
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE job_move_items
                    ADD CONSTRAINT job_move_items_status_check
                    CHECK (
                        status IN (
                            'proposed',
                            'applying',
                            'moved',
                            'failed',
                            'rolled_back',
                            'withdrawn',
                            'skipped'
                        )
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS notifications (
                        id BIGSERIAL PRIMARY KEY,
                        event_kind TEXT NOT NULL,
                        severity TEXT NOT NULL
                            CHECK (severity IN ('info', 'success', 'warning', 'error')),
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        job_id BIGINT NULL REFERENCES jobs(id) ON DELETE SET NULL,
                        job_request_id BIGINT NULL REFERENCES job_requests(id) ON DELETE SET NULL,
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        markdown TEXT NOT NULL DEFAULT '',
                        image_url TEXT NOT NULL DEFAULT '',
                        disable_web_page_preview BOOLEAN NOT NULL DEFAULT TRUE,
                        disable_notification BOOLEAN NOT NULL DEFAULT FALSE,
                        delivery_status TEXT NOT NULL DEFAULT 'pending'
                            CHECK (
                                delivery_status IN (
                                    'pending',
                                    'delivered',
                                    'retry',
                                    'disabled'
                                )
                            ),
                        attempt_count INTEGER NOT NULL DEFAULT 0
                            CHECK (attempt_count >= 0),
                        next_attempt_at TIMESTAMPTZ NULL,
                        last_attempt_at TIMESTAMPTZ NULL,
                        delivered_at TIMESTAMPTZ NULL,
                        last_error TEXT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_failures (
                        id BIGSERIAL PRIMARY KEY,
                        job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        stage TEXT NOT NULL,
                        attempt INTEGER NOT NULL CHECK (attempt > 0),
                        raw_reply TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operations (
                        id BIGSERIAL PRIMARY KEY,
                        job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        operation_type TEXT NOT NULL,
                        status TEXT NOT NULL
                            CHECK (
                                status IN (
                                    'queued',
                                    'running',
                                    'succeeded',
                                    'partial',
                                    'failed',
                                    'canceled'
                                )
                            ),
                        idempotency_key TEXT NULL,
                        request_payload_hash TEXT NOT NULL,
                        base_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
                        progress JSONB NOT NULL DEFAULT '{}'::jsonb,
                        result_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                        error JSONB NULL,
                        poll_after_ms INTEGER NOT NULL DEFAULT 1000
                            CHECK (poll_after_ms > 0),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        started_at TIMESTAMPTZ NULL,
                        finished_at TIMESTAMPTZ NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_operations_idempotency
                    ON operations (job_id, operation_type, idempotency_key)
                    WHERE idempotency_key IS NOT NULL
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_operations_job_id
                    ON operations (job_id, id DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operation_item_results (
                        id BIGSERIAL PRIMARY KEY,
                        operation_id BIGINT NOT NULL
                            REFERENCES operations(id) ON DELETE CASCADE,
                        subject_kind TEXT NOT NULL,
                        subject_id BIGINT NULL,
                        status TEXT NOT NULL,
                        path_event_id BIGINT NULL,
                        reservation_id BIGINT NULL,
                        from_path JSONB NULL,
                        to_path JSONB NULL,
                        identity_check TEXT NOT NULL DEFAULT 'not_applicable',
                        error_code TEXT NULL,
                        error_message TEXT NULL,
                        retryable BOOLEAN NOT NULL DEFAULT FALSE,
                        retry_block_reason TEXT NULL,
                        result_revision INTEGER NOT NULL DEFAULT 1,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_operation_item_results_operation
                    ON operation_item_results (operation_id, id)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS path_events (
                        id BIGSERIAL PRIMARY KEY,
                        event_sequence BIGINT NOT NULL,
                        operation_id BIGINT NOT NULL
                            REFERENCES operations(id) ON DELETE CASCADE,
                        job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        subject_type TEXT NOT NULL,
                        move_item_id BIGINT NULL REFERENCES job_move_items(id)
                            ON DELETE SET NULL,
                        source_file_id BIGINT NULL,
                        finalize_event_id BIGINT NULL,
                        from_path JSONB NOT NULL,
                        to_path JSONB NOT NULL,
                        staging_path JSONB NULL,
                        reservation_id BIGINT NULL,
                        move_stage TEXT NOT NULL DEFAULT 'direct_rename',
                        pre_identity JSONB NULL,
                        post_identity JSONB NULL,
                        fs_done_at TIMESTAMPTZ NULL,
                        db_done_at TIMESTAMPTZ NULL,
                        result TEXT NOT NULL DEFAULT 'pending',
                        error_code TEXT NULL,
                        error_message TEXT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_path_events_job_id
                    ON path_events (job_id, event_sequence, id)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS destination_reservations (
                        id BIGSERIAL PRIMARY KEY,
                        operation_id BIGINT NOT NULL
                            REFERENCES operations(id) ON DELETE CASCADE,
                        job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        root_id TEXT NOT NULL,
                        root_generation INTEGER NOT NULL,
                        root_alias_group_id TEXT NOT NULL,
                        path JSONB NOT NULL,
                        normalized_path_key TEXT NOT NULL,
                        casefold_path_key TEXT NOT NULL,
                        canonical_destination_key TEXT NOT NULL,
                        canonical_parent_key TEXT NOT NULL,
                        ancestor_keys TEXT[] NOT NULL,
                        descendant_prefix_key TEXT NOT NULL,
                        state TEXT NOT NULL
                            CHECK (
                                state IN (
                                    'reserved',
                                    'committed',
                                    'released',
                                    'expired',
                                    'conflict'
                                )
                            ),
                        conflict_code TEXT NULL,
                        conflict_ref TEXT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        expires_at TIMESTAMPTZ NULL,
                        released_at TIMESTAMPTZ NULL,
                        committed_at TIMESTAMPTZ NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_destination_reserved_key
                    ON destination_reservations (canonical_destination_key)
                    WHERE state = 'reserved'
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_destination_reserved_casefold_key
                    ON destination_reservations (casefold_path_key)
                    WHERE state = 'reserved'
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_destination_reservations_operation
                    ON destination_reservations (operation_id, id)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mapping_attempts (
                        id BIGSERIAL PRIMARY KEY,
                        job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        stage TEXT NOT NULL,
                        created_by TEXT NOT NULL,
                        tmdb_id INTEGER NULL,
                        model TEXT NULL,
                        prompt_digest TEXT NULL,
                        input_digest TEXT NULL,
                        attempt_no INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mapping_evidence (
                        id BIGSERIAL PRIMARY KEY,
                        attempt_id BIGINT NOT NULL
                            REFERENCES mapping_attempts(id) ON DELETE CASCADE,
                        episode_target_id BIGINT NULL,
                        source_file_id BIGINT NULL,
                        evidence_kind TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        confidence TEXT NOT NULL DEFAULT 'unknown',
                        uncertainty_reason TEXT NULL,
                        candidate_refs JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subtitle_link_evidence (
                        id BIGSERIAL PRIMARY KEY,
                        attempt_id BIGINT NOT NULL
                            REFERENCES mapping_attempts(id) ON DELETE CASCADE,
                        subtitle_link_id BIGINT NULL,
                        subtitle_source_id BIGINT NULL,
                        evidence_kind TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        confidence TEXT NOT NULL DEFAULT 'unknown'
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS job_review_snapshots (
                        job_id BIGINT PRIMARY KEY
                            REFERENCES jobs(id) ON DELETE CASCADE,
                        source_files JSONB NOT NULL DEFAULT '[]'::jsonb,
                        selected_tmdb_info JSONB NOT NULL DEFAULT '{}'::jsonb,
                        episode_targets JSONB NOT NULL DEFAULT '[]'::jsonb,
                        subtitle_links JSONB NOT NULL DEFAULT '[]'::jsonb,
                        mapping_attempts JSONB NOT NULL DEFAULT '[]'::jsonb,
                        mapping_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
                        path_event_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                        validation_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE job_review_snapshots
                    ADD COLUMN IF NOT EXISTS episode_targets JSONB NOT NULL
                        DEFAULT '[]'::jsonb
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE job_review_snapshots
                    ADD COLUMN IF NOT EXISTS subtitle_links JSONB NOT NULL
                        DEFAULT '[]'::jsonb
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS markdown TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS image_url TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS disable_web_page_preview BOOLEAN NOT NULL DEFAULT TRUE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS disable_notification BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS delivery_status TEXT NOT NULL DEFAULT 'pending'
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ NULL
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS last_attempt_at TIMESTAMPTZ NULL
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS delivered_at TIMESTAMPTZ NULL
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS last_error TEXT NULL
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_updated_at
                    ON jobs (updated_at DESC, id DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_series_dir_status
                    ON jobs (series_dir, status)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_job_requests_status_id
                    ON job_requests (status, id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_job_move_items_job_id
                    ON job_move_items (job_id, id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_notifications_id
                    ON notifications (id ASC)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_notifications_delivery
                    ON notifications (delivery_status, next_attempt_at, id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_llm_failures_job_id
                    ON llm_failures (job_id, id)
                    """
                )
                cur.execute(
                    """
                    INSERT INTO runtime_state (id, last_scan_at)
                    VALUES (1, NULL)
                    ON CONFLICT (id) DO NOTHING
                    """
                )

    def snapshot(self) -> RuntimeSnapshot:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT last_scan_at FROM runtime_state WHERE id = 1")
                last_scan_row = cur.fetchone()
                cur.execute("SELECT * FROM jobs ORDER BY updated_at DESC, id DESC")
                jobs = tuple(self._job_from_row(row) for row in cur.fetchall())
                cur.execute(
                    "SELECT * FROM job_requests ORDER BY updated_at DESC, id DESC"
                )
                job_requests = tuple(
                    self._job_request_from_row(row) for row in cur.fetchall()
                )
        return RuntimeSnapshot(
            last_scan_at=_isoformat(last_scan_row["last_scan_at"]) if last_scan_row else None,
            jobs=jobs,
            job_requests=job_requests,
        )

    def list_jobs(self) -> list[JobRecord]:
        return list(self.snapshot().jobs)

    def get_job(self, job_id: int) -> JobRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
                row = cur.fetchone()
        if row is None:
            return None
        return self._job_from_row(row)

    def get_job_request(self, request_id: int) -> JobRequestRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM job_requests WHERE id = %s", (request_id,))
                row = cur.fetchone()
        if row is None:
            return None
        return self._job_request_from_row(row)

    def create_job(
        self,
        *,
        series_name: str,
        watch_root_key: str,
        source_kind: JobSourceKind,
        series_dir: Path,
        output_root: Path,
    ) -> JobRecord:
        params = {
            "series_name": series_name,
            "watch_root_key": watch_root_key,
            "source_kind": source_kind,
            "status": "pending",
            "tmdb_id": None,
            "video_moves_count": 0,
            "subtitle_moves_count": 0,
            "started_at": None,
            "finished_at": None,
            "error_stage": None,
            "error_message": None,
            "series_dir": str(series_dir.resolve(strict=False)),
            "output_root": str(output_root.resolve(strict=False)),
            "archive_path": None,
            "fail_path": None,
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (
                        series_name,
                        watch_root_key,
                        source_kind,
                        status,
                        tmdb_id,
                        video_moves_count,
                        subtitle_moves_count,
                        started_at,
                        finished_at,
                        error_stage,
                        error_message,
                        series_dir,
                        output_root,
                        archive_path,
                        fail_path
                    )
                    VALUES (
                        %(series_name)s,
                        %(watch_root_key)s,
                        %(source_kind)s,
                        %(status)s,
                        %(tmdb_id)s,
                        %(video_moves_count)s,
                        %(subtitle_moves_count)s,
                        %(started_at)s,
                        %(finished_at)s,
                        %(error_stage)s,
                        %(error_message)s,
                        %(series_dir)s,
                        %(output_root)s,
                        %(archive_path)s,
                        %(fail_path)s
                    )
                    RETURNING *
                    """,
                    params,
                )
                row = cur.fetchone()
        assert row is not None
        return self._job_from_row(row)

    def update_job(self, job_id: int, **changes: object) -> JobRecord:
        if not changes:
            current = self.get_job(job_id)
            if current is None:
                raise KeyError(job_id)
            return current
        unknown = sorted(set(changes) - _JOB_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported job update fields: {unknown}")
        params = {"job_id": job_id}
        assignments: list[sql.Composable] = []
        for key, value in changes.items():
            assignments.append(
                sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder(key))
            )
            params[key] = _normalize_db_value(key, value)
        query = sql.SQL(
            "UPDATE jobs SET {assignments}, updated_at = NOW() "
            "WHERE id = %(job_id)s RETURNING *"
        ).format(assignments=sql.SQL(", ").join(assignments))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_from_row(row)

    def find_active_job_by_series_dir(self, series_dir: Path) -> JobRecord | None:
        resolved = str(series_dir.resolve(strict=False))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM jobs
                    WHERE series_dir = %s
                      AND status = ANY(%s)
                    ORDER BY updated_at DESC, id DESC
                    LIMIT 1
                    """,
                    (resolved, list(PENDING_JOB_STATUSES)),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return self._job_from_row(row)

    def create_job_request(
        self,
        *,
        kind: JobRequestKind,
        target_job_id: int | None = None,
    ) -> JobRequestRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO job_requests (
                        kind,
                        status,
                        target_job_id,
                        started_at,
                        finished_at,
                        error_message
                    )
                    VALUES (%s, 'pending', %s, NULL, NULL, NULL)
                    RETURNING *
                    """,
                    (kind, target_job_id),
                )
                row = cur.fetchone()
        assert row is not None
        return self._job_request_from_row(row)

    def update_job_request(self, request_id: int, **changes: object) -> JobRequestRecord:
        if not changes:
            current = self.get_job_request(request_id)
            if current is None:
                raise KeyError(request_id)
            return current
        unknown = sorted(set(changes) - _JOB_REQUEST_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported job request update fields: {unknown}")
        params = {"request_id": request_id}
        assignments: list[sql.Composable] = []
        for key, value in changes.items():
            assignments.append(
                sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder(key))
            )
            params[key] = _normalize_db_value(key, value)
        query = sql.SQL(
            "UPDATE job_requests SET {assignments}, updated_at = NOW() "
            "WHERE id = %(request_id)s RETURNING *"
        ).format(assignments=sql.SQL(", ").join(assignments))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._job_request_from_row(row)

    def list_pending_job_requests(self) -> list[JobRequestRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM job_requests
                    WHERE status = 'pending'
                    ORDER BY id ASC
                    """
                )
                rows = cur.fetchall()
        return [self._job_request_from_row(row) for row in rows]

    def replace_job_moves(
        self,
        job_id: int,
        moves: Sequence[PlannedMove],
    ) -> list[JobMoveRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM path_events WHERE job_id = %s LIMIT 1",
                    (job_id,),
                )
                if cur.fetchone() is not None:
                    raise RuntimeError(
                        "cannot replace job moves after path events exist"
                    )
                cur.execute(
                    """
                    SELECT id, status
                    FROM job_move_items
                    WHERE job_id = %s
                    FOR UPDATE
                    """,
                    (job_id,),
                )
                non_proposed = [
                    row["id"]
                    for row in cur.fetchall()
                    if row["status"] != "proposed"
                ]
                if non_proposed:
                    raise RuntimeError(
                        "cannot replace non-proposed job move items: "
                        + ", ".join(str(item) for item in non_proposed)
                    )
                cur.execute("DELETE FROM job_move_items WHERE job_id = %s", (job_id,))
                rows: list[dict[str, Any]] = []
                for move in moves:
                    cur.execute(
                        """
                        INSERT INTO job_move_items (
                            job_id,
                            src_id,
                            kind,
                            src,
                            dst,
                            status,
                            error_message,
                            applied_at
                        )
                        VALUES (%s, %s, %s, %s, %s, 'proposed', NULL, NULL)
                        RETURNING *
                        """,
                        (
                            job_id,
                            move.src_id,
                            move.kind,
                            str(move.src),
                            str(move.dst),
                        ),
                    )
                    row = cur.fetchone()
                    assert row is not None
                    rows.append(row)
        return [self._job_move_from_row(row) for row in rows]

    def commit_rematch_plan(
        self,
        *,
        job_id: int,
        operation_id: int,
        series_name: str,
        tmdb_id: int,
        moves: Sequence[PlannedMove],
        snapshot_payload_factory: Callable[
            [list[JobMoveRecord]], dict[str, Any]
        ],
        operation_result_summary_factory: Callable[
            [list[JobMoveRecord]], dict[str, Any]
        ],
        precondition_check: Callable[
            [
                JobRecord,
                list[JobMoveRecord],
                ReviewSnapshotRecord | None,
                list[OperationRecord],
            ],
            None,
        ]
        | None = None,
        action_label: str = "rematch",
    ) -> tuple[JobRecord, list[JobMoveRecord], ReviewSnapshotRecord, OperationRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM operations
                    WHERE id = %s AND job_id = %s
                    FOR UPDATE
                    """,
                    (operation_id, job_id),
                )
                operation_row = cur.fetchone()
                if operation_row is None:
                    raise ValueError(
                        f"operation {operation_id} does not belong to job {job_id}"
                    )
                locked_operation = self._operation_from_row(operation_row)
                if (
                    locked_operation.status == "succeeded"
                    and isinstance(
                        locked_operation.result_summary.get("response_payload"),
                        dict,
                    )
                ):
                    cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
                    job_row = cur.fetchone()
                    if job_row is None:
                        raise KeyError(job_id)
                    cur.execute(
                        """
                        SELECT *
                        FROM job_move_items
                        WHERE job_id = %s
                        ORDER BY id ASC
                        """,
                        (job_id,),
                    )
                    committed_moves = [
                        self._job_move_from_row(row) for row in cur.fetchall()
                    ]
                    cur.execute(
                        "SELECT * FROM job_review_snapshots WHERE job_id = %s",
                        (job_id,),
                    )
                    snapshot_row = cur.fetchone()
                    if snapshot_row is None:
                        raise KeyError(job_id)
                    return (
                        self._job_from_row(job_row),
                        committed_moves,
                        self._review_snapshot_from_row(snapshot_row),
                        locked_operation,
                    )
                if locked_operation.status != "running":
                    raise RuntimeError(
                        f"cannot commit {action_label} for operation in "
                        f"{locked_operation.status} state"
                    )
                cur.execute(
                    """
                    SELECT *
                    FROM jobs
                    WHERE id = %s
                    FOR UPDATE
                    """,
                    (job_id,),
                )
                locked_job_row = cur.fetchone()
                if locked_job_row is None:
                    raise KeyError(job_id)
                locked_job = self._job_from_row(locked_job_row)
                cur.execute(
                    "SELECT 1 FROM path_events WHERE job_id = %s LIMIT 1",
                    (job_id,),
                )
                if cur.fetchone() is not None:
                    raise RuntimeError(
                        "cannot replace job moves after path events exist"
                    )
                cur.execute(
                    """
                    SELECT id, status
                    FROM job_move_items
                    WHERE job_id = %s
                    FOR UPDATE
                    """,
                    (job_id,),
                )
                locked_move_rows = cur.fetchall()
                locked_moves = [
                    self._job_move_from_row(row) for row in locked_move_rows
                ]
                non_proposed = [
                    move.id for move in locked_moves if move.status != "proposed"
                ]
                if non_proposed:
                    raise RuntimeError(
                        f"cannot commit {action_label} with non-proposed move items: "
                        + ", ".join(str(item) for item in non_proposed)
                    )
                cur.execute(
                    """
                    SELECT *
                    FROM job_review_snapshots
                    WHERE job_id = %s
                    FOR UPDATE
                    """,
                    (job_id,),
                )
                locked_snapshot_row = cur.fetchone()
                locked_snapshot = (
                    self._review_snapshot_from_row(locked_snapshot_row)
                    if locked_snapshot_row is not None
                    else None
                )
                cur.execute(
                    """
                    SELECT *
                    FROM operations
                    WHERE job_id = %s
                    ORDER BY id DESC
                    """,
                    (job_id,),
                )
                locked_operations = [
                    self._operation_from_row(row) for row in cur.fetchall()
                ]
                if precondition_check is not None:
                    precondition_check(
                        locked_job,
                        locked_moves,
                        locked_snapshot,
                        locked_operations,
                    )
                cur.execute("DELETE FROM job_move_items WHERE job_id = %s", (job_id,))
                move_rows: list[dict[str, Any]] = []
                for move in moves:
                    cur.execute(
                        """
                        INSERT INTO job_move_items (
                            job_id,
                            src_id,
                            kind,
                            src,
                            dst,
                            status,
                            error_message,
                            applied_at
                        )
                        VALUES (%s, %s, %s, %s, %s, 'proposed', NULL, NULL)
                        RETURNING *
                        """,
                        (
                            job_id,
                            move.src_id,
                            move.kind,
                            str(move.src),
                            str(move.dst),
                        ),
                    )
                    row = cur.fetchone()
                    assert row is not None
                    move_rows.append(row)
                committed_moves = [
                    self._job_move_from_row(row) for row in move_rows
                ]
                snapshot_payload = snapshot_payload_factory(committed_moves)
                result_summary = operation_result_summary_factory(committed_moves)
                video_count = sum(
                    1 for move in committed_moves if move.kind == "video"
                )
                subtitle_count = sum(
                    1 for move in committed_moves if move.kind == "subtitle"
                )
                cur.execute(
                    """
                    UPDATE jobs
                    SET
                        series_name = %s,
                        status = 'planned',
                        tmdb_id = %s,
                        video_moves_count = %s,
                        subtitle_moves_count = %s,
                        error_stage = NULL,
                        error_message = NULL,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (
                        series_name,
                        tmdb_id,
                        video_count,
                        subtitle_count,
                        job_id,
                    ),
                )
                job_row = cur.fetchone()
                if job_row is None:
                    raise KeyError(job_id)
                cur.execute(
                    """
                    INSERT INTO job_review_snapshots (
                        job_id,
                        source_files,
                        selected_tmdb_info,
                        episode_targets,
                        subtitle_links,
                        mapping_attempts,
                        mapping_evidence,
                        path_event_summary,
                        validation_summary
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id)
                    DO UPDATE SET
                        source_files = EXCLUDED.source_files,
                        selected_tmdb_info = EXCLUDED.selected_tmdb_info,
                        episode_targets = EXCLUDED.episode_targets,
                        subtitle_links = EXCLUDED.subtitle_links,
                        mapping_attempts = EXCLUDED.mapping_attempts,
                        mapping_evidence = EXCLUDED.mapping_evidence,
                        path_event_summary = EXCLUDED.path_event_summary,
                        validation_summary = EXCLUDED.validation_summary,
                        updated_at = NOW()
                    RETURNING *
                    """,
                    (
                        job_id,
                        Jsonb(snapshot_payload["source_files"]),
                        Jsonb(snapshot_payload["selected_tmdb_info"]),
                        Jsonb(snapshot_payload.get("episode_targets", [])),
                        Jsonb(snapshot_payload.get("subtitle_links", [])),
                        Jsonb(snapshot_payload.get("mapping_attempts", [])),
                        Jsonb(snapshot_payload.get("mapping_evidence", [])),
                        Jsonb(snapshot_payload.get("path_event_summary", {})),
                        Jsonb(snapshot_payload.get("validation_summary", {})),
                    ),
                )
                snapshot_row = cur.fetchone()
                assert snapshot_row is not None
                cur.execute(
                    """
                    UPDATE operations
                    SET
                        status = 'succeeded',
                        result_summary = %s,
                        finished_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (Jsonb(result_summary), operation_id),
                )
                updated_operation_row = cur.fetchone()
                if updated_operation_row is None:
                    raise KeyError(operation_id)
        return (
            self._job_from_row(job_row),
            committed_moves,
            self._review_snapshot_from_row(snapshot_row),
            self._operation_from_row(updated_operation_row),
        )

    def list_job_moves(self, job_id: int) -> list[JobMoveRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM job_move_items
                    WHERE job_id = %s
                    ORDER BY id ASC
                    """,
                    (job_id,),
                )
                rows = cur.fetchall()
        return [self._job_move_from_row(row) for row in rows]

    def update_job_move(self, move_id: int, **changes: object) -> JobMoveRecord:
        if not changes:
            current = self._get_job_move(move_id)
            if current is None:
                raise KeyError(move_id)
            return current
        unknown = sorted(set(changes) - _JOB_MOVE_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported job move update fields: {unknown}")
        params = {"move_id": move_id}
        assignments: list[sql.Composable] = []
        for key, value in changes.items():
            assignments.append(
                sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder(key))
            )
            params[key] = _normalize_db_value(key, value)
        query = sql.SQL(
            "UPDATE job_move_items SET {assignments}, updated_at = NOW() "
            "WHERE id = %(move_id)s RETURNING *"
        ).format(assignments=sql.SQL(", ").join(assignments))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        if row is None:
            raise KeyError(move_id)
        return self._job_move_from_row(row)

    def update_job_moves_for_job(self, job_id: int, **changes: object) -> list[JobMoveRecord]:
        unknown = sorted(set(changes) - _JOB_MOVE_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported job move update fields: {unknown}")
        if not changes:
            return self.list_job_moves(job_id)
        params = {"job_id": job_id}
        assignments: list[sql.Composable] = []
        for key, value in changes.items():
            assignments.append(
                sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder(key))
            )
            params[key] = _normalize_db_value(key, value)
        query = sql.SQL(
            "UPDATE job_move_items SET {assignments}, updated_at = NOW() "
            "WHERE job_id = %(job_id)s RETURNING *"
        ).format(assignments=sql.SQL(", ").join(assignments))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [self._job_move_from_row(row) for row in rows]

    def mark_apply_items_applying(
        self,
        *,
        job_id: int,
        move_item_ids: Sequence[int],
    ) -> list[JobMoveRecord]:
        move_ids = list(dict.fromkeys(move_item_ids))
        if not move_ids:
            raise ValueError("apply requires at least one move item")
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'applying',
                        started_at = COALESCE(started_at, %s),
                        finished_at = NULL,
                        error_stage = NULL,
                        error_message = NULL,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (now, now, job_id),
                )
                job_row = cur.fetchone()
                if job_row is None:
                    raise KeyError(job_id)
                cur.execute(
                    """
                    UPDATE job_move_items
                    SET status = 'applying',
                        error_message = NULL,
                        applied_at = NULL,
                        updated_at = %s
                    WHERE job_id = %s
                      AND id = ANY(%s)
                      AND status = 'proposed'
                    RETURNING *
                    """,
                    (now, job_id, move_ids),
                )
                rows = cur.fetchall()
                claimed_ids = {int(row["id"]) for row in rows}
                missing_ids = sorted(set(move_ids) - claimed_ids)
                if missing_ids:
                    raise ValueError(
                        "move items are not proposed or do not belong to job: "
                        f"{missing_ids}"
                    )
        return [self._job_move_from_row(row) for row in rows]

    def mark_withdraw_items_applying(
        self,
        *,
        job_id: int,
        move_item_ids: Sequence[int],
    ) -> list[JobMoveRecord]:
        move_ids = list(dict.fromkeys(move_item_ids))
        if not move_ids:
            raise ValueError("withdraw requires at least one move item")
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'applying',
                        started_at = COALESCE(started_at, %s),
                        finished_at = NULL,
                        error_stage = NULL,
                        error_message = NULL,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (now, now, job_id),
                )
                job_row = cur.fetchone()
                if job_row is None:
                    raise KeyError(job_id)
                cur.execute(
                    """
                    UPDATE job_move_items
                    SET status = 'applying',
                        error_message = NULL,
                        updated_at = %s
                    WHERE job_id = %s
                      AND id = ANY(%s)
                      AND status = 'moved'
                    RETURNING *
                    """,
                    (now, job_id, move_ids),
                )
                rows = cur.fetchall()
                claimed_ids = {int(row["id"]) for row in rows}
                missing_ids = sorted(set(move_ids) - claimed_ids)
                if missing_ids:
                    raise ValueError(
                        "move items are not moved or do not belong to job: "
                        f"{missing_ids}"
                    )
        return [self._job_move_from_row(row) for row in rows]

    def _get_job_move(self, move_id: int) -> JobMoveRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM job_move_items WHERE id = %s", (move_id,))
                row = cur.fetchone()
        if row is None:
            return None
        return self._job_move_from_row(row)

    def create_operation(
        self,
        *,
        job_id: int,
        operation_type: str,
        idempotency_key: str | None,
        request_payload_hash: str,
        base_versions: dict[str, Any] | None = None,
        status: OperationStatus = "succeeded",
        result_summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> OperationRecord:
        base_versions = base_versions or {}
        result_summary = result_summary or {}
        now = datetime.now(timezone.utc)
        started_at = None if status == "queued" else now
        finished_at = (
            now
            if status in {"succeeded", "partial", "failed", "canceled"}
            else None
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO operations (
                        job_id,
                        operation_type,
                        status,
                        idempotency_key,
                        request_payload_hash,
                        base_versions,
                        progress,
                        result_summary,
                        error,
                        started_at,
                        finished_at
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        '{}'::jsonb,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    ON CONFLICT (job_id, operation_type, idempotency_key)
                    WHERE idempotency_key IS NOT NULL
                    DO NOTHING
                    RETURNING *
                    """,
                    (
                        job_id,
                        operation_type,
                        status,
                        idempotency_key,
                        request_payload_hash,
                        Jsonb(base_versions),
                        Jsonb(result_summary),
                        Jsonb(error) if error is not None else None,
                        started_at,
                        finished_at,
                    ),
                )
                row = cur.fetchone()
                if row is not None:
                    return self._operation_from_row(row)
                if idempotency_key is None:
                    raise RuntimeError("operation insert unexpectedly returned no row")
                cur.execute(
                    """
                    SELECT *
                    FROM operations
                    WHERE job_id = %s
                      AND operation_type = %s
                      AND idempotency_key = %s
                    """,
                    (job_id, operation_type, idempotency_key),
                )
                existing = cur.fetchone()
        if existing is None:
            raise RuntimeError("idempotent operation conflict was not readable")
        operation = self._operation_from_row(existing)
        if operation.request_payload_hash != request_payload_hash:
            raise IdempotencyConflictError(
                "idempotency key was already used with a different payload"
            )
        return operation

    def get_operation_by_idempotency(
        self,
        *,
        job_id: int,
        operation_type: str,
        idempotency_key: str,
    ) -> OperationRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM operations
                    WHERE job_id = %s
                      AND operation_type = %s
                      AND idempotency_key = %s
                    """,
                    (job_id, operation_type, idempotency_key),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return self._operation_from_row(row)

    def update_operation_result_summary(
        self,
        operation_id: int,
        result_summary: dict[str, Any],
    ) -> OperationRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE operations
                    SET result_summary = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (Jsonb(result_summary), operation_id),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(operation_id)
        return self._operation_from_row(row)

    def update_operation(
        self,
        operation_id: int,
        **changes: object,
    ) -> OperationRecord:
        unknown = sorted(set(changes) - _OPERATION_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported operation update fields: {unknown}")
        if not changes:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM operations WHERE id = %s", (operation_id,))
                    row = cur.fetchone()
            if row is None:
                raise KeyError(operation_id)
            return self._operation_from_row(row)
        params = {"operation_id": operation_id}
        assignments: list[sql.Composable] = []
        for key, value in changes.items():
            assignments.append(
                sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder(key))
            )
            if key in {"progress", "result_summary", "error"}:
                params[key] = Jsonb(value) if value is not None else None
            else:
                params[key] = _normalize_db_value(key, value)
        query = sql.SQL(
            "UPDATE operations SET {assignments} "
            "WHERE id = %(operation_id)s RETURNING *"
        ).format(assignments=sql.SQL(", ").join(assignments))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        if row is None:
            raise KeyError(operation_id)
        return self._operation_from_row(row)

    def save_review_snapshot(
        self,
        job_id: int,
        *,
        source_files: list[dict[str, Any]],
        selected_tmdb_info: dict[str, Any],
        episode_targets: list[dict[str, Any]] | None = None,
        subtitle_links: list[dict[str, Any]] | None = None,
        mapping_attempts: list[dict[str, Any]] | None = None,
        mapping_evidence: list[dict[str, Any]] | None = None,
        path_event_summary: dict[str, Any] | None = None,
        validation_summary: dict[str, Any] | None = None,
    ) -> ReviewSnapshotRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO job_review_snapshots (
                        job_id,
                        source_files,
                        selected_tmdb_info,
                        episode_targets,
                        subtitle_links,
                        mapping_attempts,
                        mapping_evidence,
                        path_event_summary,
                        validation_summary
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id)
                    DO UPDATE SET
                        source_files = EXCLUDED.source_files,
                        selected_tmdb_info = EXCLUDED.selected_tmdb_info,
                        episode_targets = EXCLUDED.episode_targets,
                        subtitle_links = EXCLUDED.subtitle_links,
                        mapping_attempts = EXCLUDED.mapping_attempts,
                        mapping_evidence = EXCLUDED.mapping_evidence,
                        path_event_summary = EXCLUDED.path_event_summary,
                        validation_summary = EXCLUDED.validation_summary,
                        updated_at = NOW()
                    RETURNING *
                    """,
                    (
                        job_id,
                        Jsonb(source_files),
                        Jsonb(selected_tmdb_info),
                        Jsonb(episode_targets or []),
                        Jsonb(subtitle_links or []),
                        Jsonb(mapping_attempts or []),
                        Jsonb(mapping_evidence or []),
                        Jsonb(path_event_summary or {}),
                        Jsonb(validation_summary or {}),
                    ),
                )
                row = cur.fetchone()
        assert row is not None
        return self._review_snapshot_from_row(row)

    def get_review_snapshot(self, job_id: int) -> ReviewSnapshotRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM job_review_snapshots WHERE job_id = %s",
                    (job_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return self._review_snapshot_from_row(row)

    def list_operations(self, job_id: int) -> list[OperationRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM operations
                    WHERE job_id = %s
                    ORDER BY id DESC
                    """,
                    (job_id,),
                )
                rows = cur.fetchall()
        return [self._operation_from_row(row) for row in rows]

    def get_operation(self, job_id: int, operation_id: int) -> OperationRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM operations
                    WHERE job_id = %s AND id = %s
                    """,
                    (job_id, operation_id),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return self._operation_from_row(row)

    def list_operation_item_results(
        self,
        operation_id: int,
        *,
        limit: int = 100,
    ) -> list[OperationItemResultRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM operation_item_results
                    WHERE operation_id = %s
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (operation_id, limit),
                )
                rows = cur.fetchall()
        return [self._operation_item_result_from_row(row) for row in rows]

    def list_latest_successful_withdraw_item_results(
        self,
        *,
        job_id: int,
        move_item_ids: Sequence[int],
    ) -> dict[int, OperationItemResultRecord]:
        move_ids = list(dict.fromkeys(move_item_ids))
        if not move_ids:
            return {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (operation_item_results.subject_id)
                        operation_item_results.*
                    FROM operation_item_results
                    JOIN operations
                      ON operations.id = operation_item_results.operation_id
                    WHERE operations.job_id = %s
                      AND operations.operation_type = 'withdraw'
                      AND operations.status = 'succeeded'
                      AND operation_item_results.subject_kind = 'move_item'
                      AND operation_item_results.subject_id = ANY(%s)
                      AND operation_item_results.status = 'succeeded'
                      AND operation_item_results.to_path IS NOT NULL
                    ORDER BY
                        operation_item_results.subject_id,
                        operations.finished_at DESC NULLS LAST,
                        operation_item_results.id DESC
                    """,
                    (job_id, move_ids),
                )
                rows = cur.fetchall()
        results = [self._operation_item_result_from_row(row) for row in rows]
        return {
            int(item.subject_id): item
            for item in results
            if item.subject_id is not None
        }

    def create_operation_item_result(
        self,
        *,
        operation_id: int,
        subject_kind: str,
        subject_id: int | None,
        status: str,
        path_event_id: int | None = None,
        reservation_id: int | None = None,
        from_path: dict[str, Any] | None = None,
        to_path: dict[str, Any] | None = None,
        identity_check: str = "not_applicable",
        error_code: str | None = None,
        error_message: str | None = None,
        retryable: bool = False,
        retry_block_reason: str | None = None,
        result_revision: int = 1,
    ) -> OperationItemResultRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT job_id FROM operations WHERE id = %s",
                    (operation_id,),
                )
                operation_row = cur.fetchone()
                if operation_row is None:
                    raise ValueError(f"operation {operation_id} does not exist")
                operation_job_id = operation_row["job_id"]
                if subject_kind == "move_item" and subject_id is not None:
                    cur.execute(
                        "SELECT job_id FROM job_move_items WHERE id = %s",
                        (subject_id,),
                    )
                    move_row = cur.fetchone()
                    if move_row is None:
                        raise ValueError(f"move item {subject_id} does not exist")
                    if move_row["job_id"] != operation_job_id:
                        raise ValueError(
                            f"move item {subject_id} does not belong to "
                            f"operation {operation_id}'s job"
                        )
                if path_event_id is not None:
                    cur.execute(
                        """
                        SELECT operation_id, job_id
                        FROM path_events
                        WHERE id = %s
                        """,
                        (path_event_id,),
                    )
                    event_row = cur.fetchone()
                    if event_row is None:
                        raise ValueError(f"path event {path_event_id} does not exist")
                    if (
                        event_row["operation_id"] != operation_id
                        or event_row["job_id"] != operation_job_id
                    ):
                        raise ValueError(
                            f"path event {path_event_id} does not belong to "
                            f"operation {operation_id}"
                        )
                if reservation_id is not None:
                    cur.execute(
                        """
                        SELECT operation_id, job_id
                        FROM destination_reservations
                        WHERE id = %s
                        """,
                        (reservation_id,),
                    )
                    reservation_row = cur.fetchone()
                    if reservation_row is None:
                        raise ValueError(
                            f"reservation {reservation_id} does not exist"
                        )
                    if (
                        reservation_row["operation_id"] != operation_id
                        or reservation_row["job_id"] != operation_job_id
                    ):
                        raise ValueError(
                            f"reservation {reservation_id} does not belong to "
                            f"operation {operation_id}"
                        )
                cur.execute(
                    """
                    INSERT INTO operation_item_results (
                        operation_id,
                        subject_kind,
                        subject_id,
                        status,
                        path_event_id,
                        reservation_id,
                        from_path,
                        to_path,
                        identity_check,
                        error_code,
                        error_message,
                        retryable,
                        retry_block_reason,
                        result_revision
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        operation_id,
                        subject_kind,
                        subject_id,
                        status,
                        path_event_id,
                        reservation_id,
                        Jsonb(from_path) if from_path is not None else None,
                        Jsonb(to_path) if to_path is not None else None,
                        identity_check,
                        error_code,
                        error_message,
                        retryable,
                        retry_block_reason,
                        result_revision,
                    ),
                )
                row = cur.fetchone()
        assert row is not None
        return self._operation_item_result_from_row(row)

    def complete_reopen_from_review_operation_success(
        self,
        *,
        operation_id: int,
        job_id: int,
        updates: Sequence[ReopenMoveItemUpdate],
        result_summary: dict[str, Any],
    ) -> OperationRecord:
        if not updates:
            raise ValueError("reopen requires at least one move item")
        update_by_id = {item.move_item_id: item for item in updates}
        if len(update_by_id) != len(updates):
            raise ValueError("duplicate reopen move item updates")
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM operations
                    WHERE id = %s AND job_id = %s
                    FOR UPDATE
                    """,
                    (operation_id, job_id),
                )
                operation_row = cur.fetchone()
                if operation_row is None:
                    raise KeyError(operation_id)
                operation = self._operation_from_row(operation_row)
                if operation.status != "running":
                    raise RuntimeError(
                        f"cannot complete reopen operation in {operation.status} state"
                    )
                cur.execute(
                    """
                    SELECT *
                    FROM jobs
                    WHERE id = %s
                    FOR UPDATE
                    """,
                    (job_id,),
                )
                job_row = cur.fetchone()
                if job_row is None:
                    raise KeyError(job_id)
                cur.execute(
                    """
                    SELECT *
                    FROM job_move_items
                    WHERE job_id = %s
                      AND id = ANY(%s)
                    FOR UPDATE
                    """,
                    (job_id, list(update_by_id)),
                )
                move_rows = cur.fetchall()
                moves_by_id = {
                    int(row["id"]): self._job_move_from_row(row)
                    for row in move_rows
                }
                missing_ids = sorted(set(update_by_id) - set(moves_by_id))
                not_withdrawn_ids = sorted(
                    move_id
                    for move_id, move in moves_by_id.items()
                    if move.status != "withdrawn"
                )
                if missing_ids or not_withdrawn_ids:
                    raise RuntimeError(
                        "cannot reopen move items that are missing or not withdrawn: "
                        f"missing={missing_ids}, not_withdrawn={not_withdrawn_ids}"
                    )
                for move_id, update in update_by_id.items():
                    cur.execute(
                        """
                        INSERT INTO operation_item_results (
                            operation_id,
                            subject_kind,
                            subject_id,
                            status,
                            path_event_id,
                            reservation_id,
                            from_path,
                            to_path,
                            identity_check,
                            error_code,
                            error_message,
                            retryable,
                            retry_block_reason,
                            result_revision
                        )
                        VALUES (
                            %s,
                            'move_item',
                            %s,
                            'succeeded',
                            NULL,
                            NULL,
                            %s,
                            %s,
                            'matched',
                            NULL,
                            NULL,
                            FALSE,
                            NULL,
                            1
                        )
                        """,
                        (
                            operation_id,
                            move_id,
                            Jsonb(update.from_path),
                            Jsonb(update.to_path),
                        ),
                    )
                    cur.execute(
                        """
                        UPDATE job_move_items
                        SET src = %s,
                            status = 'proposed',
                            error_message = NULL,
                            applied_at = NULL,
                            updated_at = %s
                        WHERE id = %s
                          AND job_id = %s
                        """,
                        (update.source_path, now, move_id, job_id),
                    )
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'planned',
                        finished_at = NULL,
                        error_stage = NULL,
                        error_message = NULL,
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (now, job_id),
                )
                cur.execute(
                    """
                    UPDATE operations
                    SET status = 'succeeded',
                        result_summary = %s,
                        error = NULL,
                        finished_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (Jsonb(result_summary), now, operation_id),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(operation_id)
        return self._operation_from_row(row)

    def create_path_event(
        self,
        *,
        operation_id: int,
        job_id: int,
        subject_type: str,
        move_item_id: int | None,
        source_file_id: int | None,
        from_path: dict[str, Any],
        to_path: dict[str, Any],
        reservation_id: int | None = None,
        pre_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT job_id FROM operations WHERE id = %s",
                    (operation_id,),
                )
                operation_row = cur.fetchone()
                if operation_row is None:
                    raise ValueError(f"operation {operation_id} does not exist")
                if operation_row["job_id"] != job_id:
                    raise ValueError(
                        f"operation {operation_id} does not belong to job {job_id}"
                    )
                if move_item_id is not None:
                    cur.execute(
                        "SELECT job_id FROM job_move_items WHERE id = %s",
                        (move_item_id,),
                    )
                    move_row = cur.fetchone()
                    if move_row is None:
                        raise ValueError(f"move item {move_item_id} does not exist")
                    if move_row["job_id"] != job_id:
                        raise ValueError(
                            f"move item {move_item_id} does not belong to job {job_id}"
                        )
                if reservation_id is not None:
                    cur.execute(
                        """
                        SELECT operation_id, job_id
                        FROM destination_reservations
                        WHERE id = %s
                        """,
                        (reservation_id,),
                    )
                    reservation_row = cur.fetchone()
                    if reservation_row is None:
                        raise ValueError(
                            f"reservation {reservation_id} does not exist"
                        )
                    if (
                        reservation_row["operation_id"] != operation_id
                        or reservation_row["job_id"] != job_id
                    ):
                        raise ValueError(
                            f"reservation {reservation_id} does not belong to "
                            f"operation {operation_id}"
                        )
                cur.execute(
                    """
                    SELECT COALESCE(MAX(event_sequence), 0) + 1 AS next_sequence
                    FROM path_events
                    WHERE job_id = %s
                    """,
                    (job_id,),
                )
                sequence_row = cur.fetchone()
                assert sequence_row is not None
                cur.execute(
                    """
                    INSERT INTO path_events (
                        event_sequence,
                        operation_id,
                        job_id,
                        subject_type,
                        move_item_id,
                        source_file_id,
                        from_path,
                        to_path,
                        reservation_id,
                        pre_identity
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        sequence_row["next_sequence"],
                        operation_id,
                        job_id,
                        subject_type,
                        move_item_id,
                        source_file_id,
                        Jsonb(from_path),
                        Jsonb(to_path),
                        reservation_id,
                        Jsonb(pre_identity) if pre_identity is not None else None,
                    ),
                )
                row = cur.fetchone()
        assert row is not None
        return dict(row)

    def update_path_event(
        self,
        event_id: int,
        *,
        result: str,
        fs_done_at: str | None = None,
        db_done_at: str | None = None,
        post_identity: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE path_events
                    SET
                        result = %s,
                        fs_done_at = %s,
                        db_done_at = %s,
                        post_identity = %s,
                        error_code = %s,
                        error_message = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (
                        result,
                        _normalize_db_value("finished_at", fs_done_at),
                        _normalize_db_value("finished_at", db_done_at),
                        Jsonb(post_identity) if post_identity is not None else None,
                        error_code,
                        error_message,
                        event_id,
                    ),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(event_id)
        return dict(row)

    def reserve_destinations(
        self,
        *,
        operation_id: int,
        job_id: int,
        reservations: Sequence[DestinationReservationSpec],
    ) -> list[DestinationReservationRecord]:
        if not reservations:
            return []
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT job_id FROM operations WHERE id = %s",
                    (operation_id,),
                )
                operation_row = cur.fetchone()
                if operation_row is None:
                    raise ValueError(f"operation {operation_id} does not exist")
                if operation_row["job_id"] != job_id:
                    raise ValueError(
                        f"operation {operation_id} does not belong to job {job_id}"
                    )
                cur.execute("LOCK TABLE destination_reservations IN EXCLUSIVE MODE")
                cur.execute(
                    """
                    SELECT *
                    FROM destination_reservations
                    WHERE state = 'reserved'
                    """
                )
                existing = [
                    self._destination_reservation_from_row(row)
                    for row in cur.fetchall()
                ]
                accepted: list[DestinationReservationSpec] = []
                for spec in reservations:
                    self._ensure_no_reservation_conflict(spec, existing, accepted)
                    accepted.append(spec)
                rows: list[dict[str, Any]] = []
                for spec in accepted:
                    cur.execute(
                        """
                        INSERT INTO destination_reservations (
                            operation_id,
                            job_id,
                            root_id,
                            root_generation,
                            root_alias_group_id,
                            path,
                            normalized_path_key,
                            casefold_path_key,
                            canonical_destination_key,
                            canonical_parent_key,
                            ancestor_keys,
                            descendant_prefix_key,
                            state,
                            created_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            'reserved',
                            %s
                        )
                        RETURNING *
                        """,
                        (
                            operation_id,
                            job_id,
                            spec.root_id,
                            spec.root_generation,
                            spec.root_alias_group_id,
                            Jsonb(spec.path),
                            spec.normalized_path_key,
                            spec.casefold_path_key,
                            spec.canonical_destination_key,
                            spec.canonical_parent_key,
                            list(spec.ancestor_keys),
                            spec.descendant_prefix_key,
                            now,
                        ),
                    )
                    row = cur.fetchone()
                    assert row is not None
                    rows.append(row)
        return [self._destination_reservation_from_row(row) for row in rows]

    def update_destination_reservations_state(
        self,
        *,
        operation_id: int,
        state: str,
    ) -> list[DestinationReservationRecord]:
        timestamp_column = {
            "committed": "committed_at",
            "released": "released_at",
        }.get(state)
        if timestamp_column is None:
            raise ValueError(f"unsupported reservation state update: {state}")
        query = sql.SQL(
            """
            UPDATE destination_reservations
            SET state = %(state)s,
                {timestamp_column} = NOW()
            WHERE operation_id = %(operation_id)s
              AND state = 'reserved'
            RETURNING *
            """
        ).format(timestamp_column=sql.Identifier(timestamp_column))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    {
                        "operation_id": operation_id,
                        "state": state,
                    },
                )
                rows = cur.fetchall()
        return [self._destination_reservation_from_row(row) for row in rows]

    def complete_apply_operation_success(
        self,
        *,
        operation_id: int,
        job_id: int,
        result_summary: dict[str, Any],
    ) -> OperationRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT job_id
                    FROM operations
                    WHERE id = %s
                    """,
                    (operation_id,),
                )
                operation_row = cur.fetchone()
                if operation_row is None:
                    raise KeyError(operation_id)
                if operation_row["job_id"] != job_id:
                    raise ValueError(
                        f"operation {operation_id} does not belong to job {job_id}"
                    )
                cur.execute(
                    """
                    UPDATE destination_reservations
                    SET state = 'committed',
                        committed_at = NOW()
                    WHERE operation_id = %s
                      AND state = 'reserved'
                    """,
                    (operation_id,),
                )
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'succeeded',
                        finished_at = NOW(),
                        error_stage = NULL,
                        error_message = NULL,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (job_id,),
                )
                cur.execute(
                    """
                    UPDATE operations
                    SET status = 'succeeded',
                        result_summary = %s,
                        error = NULL,
                        finished_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (Jsonb(result_summary), operation_id),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(operation_id)
        return self._operation_from_row(row)

    def complete_withdraw_operation_success(
        self,
        *,
        operation_id: int,
        job_id: int,
        result_summary: dict[str, Any],
    ) -> OperationRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT job_id
                    FROM operations
                    WHERE id = %s
                    """,
                    (operation_id,),
                )
                operation_row = cur.fetchone()
                if operation_row is None:
                    raise KeyError(operation_id)
                if operation_row["job_id"] != job_id:
                    raise ValueError(
                        f"operation {operation_id} does not belong to job {job_id}"
                    )
                cur.execute(
                    """
                    UPDATE destination_reservations
                    SET state = 'committed',
                        committed_at = NOW()
                    WHERE operation_id = %s
                      AND state = 'reserved'
                    """,
                    (operation_id,),
                )
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'succeeded',
                        finished_at = NOW(),
                        error_stage = NULL,
                        error_message = NULL,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (job_id,),
                )
                cur.execute(
                    """
                    UPDATE operations
                    SET status = 'succeeded',
                        result_summary = %s,
                        error = NULL,
                        finished_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (Jsonb(result_summary), operation_id),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(operation_id)
        return self._operation_from_row(row)

    @staticmethod
    def _ensure_no_reservation_conflict(
        candidate: DestinationReservationSpec,
        existing: Sequence[DestinationReservationRecord],
        accepted: Sequence[DestinationReservationSpec],
    ) -> None:
        for item in existing:
            if _reservation_keys_conflict(
                candidate.canonical_destination_key,
                candidate.ancestor_keys,
                item.canonical_destination_key,
                item.ancestor_keys,
                candidate.casefold_path_key,
                item.casefold_path_key,
            ):
                raise DestinationReservationConflictError(
                    "destination_reserved",
                    item.canonical_destination_key,
                )
        for item in accepted:
            if _reservation_keys_conflict(
                candidate.canonical_destination_key,
                candidate.ancestor_keys,
                item.canonical_destination_key,
                item.ancestor_keys,
                candidate.casefold_path_key,
                item.casefold_path_key,
            ):
                raise DestinationReservationConflictError(
                    "destination_reserved",
                    item.canonical_destination_key,
                )

    def set_last_scan_at(self, timestamp: str | None = None) -> None:
        resolved = _normalize_db_value("last_scan_at", timestamp)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE runtime_state SET last_scan_at = %s WHERE id = 1",
                    (resolved or datetime.now(timezone.utc),),
                )

    def recover_incomplete_jobs(self) -> list[JobRecord]:
        now = datetime.now(timezone.utc)
        recovered_failed_jobs: list[JobRecord] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending',
                        error_stage = NULL,
                        error_message = NULL,
                        updated_at = %s
                    WHERE status = 'planning'
                    """,
                    (now,),
                )
                cur.execute(
                    """
                    WITH recovered AS (
                        UPDATE jobs
                        SET status = 'failed',
                            error_stage = 'apply',
                            error_message = 'worker restarted during apply',
                            finished_at = %s,
                            updated_at = %s
                        WHERE status = 'applying'
                        RETURNING *
                    ),
                    failed_moves AS (
                        UPDATE job_move_items
                        SET status = 'failed',
                            error_message = 'worker restarted during apply',
                            updated_at = %s
                        WHERE job_id IN (SELECT id FROM recovered)
                          AND status = 'applying'
                        RETURNING id
                    )
                    SELECT *
                    FROM recovered
                    """,
                    (now, now, now),
                )
                recovered_failed_jobs = [
                    self._job_from_row(row) for row in cur.fetchall()
                ]
        return recovered_failed_jobs

    def create_notification(
        self,
        *,
        event_kind: str,
        severity: NotificationSeverity,
        title: str,
        message: str,
        markdown: str,
        image_url: str = "",
        job_id: int | None = None,
        job_request_id: int | None = None,
        payload: dict[str, Any] | None = None,
        disable_web_page_preview: bool = True,
        disable_notification: bool = False,
        delivery_status: NotificationDeliveryStatus = "pending",
    ) -> NotificationRecord:
        resolved_payload = payload or {}
        next_attempt_at = datetime.now(timezone.utc) if delivery_status == "pending" else None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO notifications (
                        event_kind,
                        severity,
                        title,
                        message,
                        job_id,
                        job_request_id,
                        payload,
                        markdown,
                        image_url,
                        disable_web_page_preview,
                        disable_notification,
                        delivery_status,
                        attempt_count,
                        next_attempt_at,
                        last_attempt_at,
                        delivered_at,
                        last_error
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, NULL, NULL, NULL)
                    RETURNING *
                    """,
                    (
                        event_kind,
                        severity,
                        title,
                        message,
                        job_id,
                        job_request_id,
                        Jsonb(resolved_payload),
                        markdown,
                        image_url,
                        disable_web_page_preview,
                        disable_notification,
                        delivery_status,
                        next_attempt_at,
                    ),
                )
                row = cur.fetchone()
        assert row is not None
        return self._notification_from_row(row)

    def latest_notification_id(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(id), 0) AS latest_id FROM notifications")
                row = cur.fetchone()
        if row is None:
            return 0
        return int(row["latest_id"])

    def list_notifications_after(
        self,
        after_id: int,
        *,
        limit: int = 100,
    ) -> list[NotificationRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM notifications
                    WHERE id > %s
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (after_id, limit),
                )
                rows = cur.fetchall()
        return [self._notification_from_row(row) for row in rows]

    def list_due_notifications(self, *, limit: int = 100) -> list[NotificationRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM notifications
                    WHERE delivery_status = ANY(%s)
                      AND next_attempt_at IS NOT NULL
                      AND next_attempt_at <= NOW()
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (["pending", "retry"], limit),
                )
                rows = cur.fetchall()
        return [self._notification_from_row(row) for row in rows]

    def record_llm_failure(
        self,
        *,
        job_id: int,
        stage: str,
        attempt: int,
        raw_reply: str,
        error_message: str,
    ) -> LLMFailureRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_failures (
                        job_id,
                        stage,
                        attempt,
                        raw_reply,
                        error_message
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (job_id, stage, attempt, raw_reply, error_message),
                )
                row = cur.fetchone()
        assert row is not None
        return self._llm_failure_from_row(row)

    def list_llm_failures(self, job_id: int) -> list[LLMFailureRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM llm_failures
                    WHERE job_id = %s
                    ORDER BY id ASC
                    """,
                    (job_id,),
                )
                rows = cur.fetchall()
        return [self._llm_failure_from_row(row) for row in rows]

    def mark_notification_delivered(
        self,
        notification_id: int,
        *,
        attempt_count: int,
    ) -> NotificationRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE notifications
                    SET delivery_status = 'delivered',
                        attempt_count = %s,
                        next_attempt_at = NULL,
                        last_attempt_at = NOW(),
                        delivered_at = NOW(),
                        last_error = NULL
                    WHERE id = %s
                    RETURNING *
                    """,
                    (attempt_count, notification_id),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(notification_id)
        return self._notification_from_row(row)

    def mark_notification_retry(
        self,
        notification_id: int,
        *,
        attempt_count: int,
        next_attempt_at: str,
        last_error: str,
    ) -> NotificationRecord:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE notifications
                    SET delivery_status = 'retry',
                        attempt_count = %s,
                        next_attempt_at = %s,
                        last_attempt_at = NOW(),
                        delivered_at = NULL,
                        last_error = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (
                        attempt_count,
                        _normalize_db_value("next_attempt_at", next_attempt_at),
                        last_error,
                        notification_id,
                    ),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(notification_id)
        return self._notification_from_row(row)

    def save_artifact(
        self,
        job_id: int,
        artifact_kind: ArtifactKind,
        payload: dict[str, Any],
    ) -> None:
        if artifact_kind != "result":
            raise ValueError("only result artifacts are supported")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO job_artifacts (job_id, artifact_kind, payload)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (job_id, artifact_kind)
                    DO UPDATE
                    SET payload = EXCLUDED.payload,
                        updated_at = NOW()
                    """,
                    (job_id, artifact_kind, Jsonb(payload)),
                )

    def load_artifact(
        self,
        job_id: int,
        artifact_kind: ArtifactKind,
    ) -> dict[str, Any] | None:
        if artifact_kind != "result":
            raise ValueError("only result artifacts are supported")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT payload
                    FROM job_artifacts
                    WHERE job_id = %s AND artifact_kind = %s
                    """,
                    (job_id, artifact_kind),
                )
                row = cur.fetchone()
        if row is None:
            return None
        payload = row["payload"]
        if not isinstance(payload, dict):
            raise ValueError("artifact payload must be an object")
        return payload

    def _connect(self, *, autocommit: bool = False) -> psycopg.Connection[dict[str, Any]]:
        return psycopg.connect(
            self._postgres_dsn,
            autocommit=autocommit,
            row_factory=dict_row,
        )

    @staticmethod
    def _job_from_row(row: dict[str, Any]) -> JobRecord:
        return JobRecord(
            id=row["id"],
            series_name=row["series_name"],
            watch_root_key=row["watch_root_key"],
            source_kind=row["source_kind"],
            status=row["status"],
            tmdb_id=row["tmdb_id"],
            video_moves_count=row["video_moves_count"],
            subtitle_moves_count=row["subtitle_moves_count"],
            created_at=_isoformat(row["created_at"]) or "",
            updated_at=_isoformat(row["updated_at"]) or "",
            started_at=_isoformat(row["started_at"]),
            finished_at=_isoformat(row["finished_at"]),
            error_stage=row["error_stage"],
            error_message=row["error_message"],
            series_dir=row["series_dir"],
            output_root=row["output_root"],
            archive_path=row["archive_path"],
            fail_path=row["fail_path"],
        )

    @staticmethod
    def _job_request_from_row(row: dict[str, Any]) -> JobRequestRecord:
        return JobRequestRecord(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            target_job_id=row["target_job_id"],
            created_at=_isoformat(row["created_at"]) or "",
            updated_at=_isoformat(row["updated_at"]) or "",
            started_at=_isoformat(row["started_at"]),
            finished_at=_isoformat(row["finished_at"]),
            error_message=row["error_message"],
        )

    @staticmethod
    def _job_move_from_row(row: dict[str, Any]) -> JobMoveRecord:
        return JobMoveRecord(
            id=row["id"],
            job_id=row["job_id"],
            src_id=row["src_id"],
            kind=row["kind"],
            src=row["src"],
            dst=row["dst"],
            status=row["status"],
            error_message=row["error_message"],
            applied_at=_isoformat(row["applied_at"]),
            created_at=_isoformat(row["created_at"]) or "",
            updated_at=_isoformat(row["updated_at"]) or "",
        )

    @staticmethod
    def _notification_from_row(row: dict[str, Any]) -> NotificationRecord:
        payload = row["payload"]
        if not isinstance(payload, dict):
            raise ValueError("notification payload must be an object")
        return NotificationRecord(
            id=row["id"],
            event_kind=row["event_kind"],
            severity=row["severity"],
            title=row["title"],
            message=row["message"],
            job_id=row["job_id"],
            job_request_id=row["job_request_id"],
            payload=payload,
            markdown=row["markdown"],
            image_url=row["image_url"],
            disable_web_page_preview=bool(row["disable_web_page_preview"]),
            disable_notification=bool(row["disable_notification"]),
            delivery_status=row["delivery_status"],
            attempt_count=row["attempt_count"],
            next_attempt_at=_isoformat(row["next_attempt_at"]),
            last_attempt_at=_isoformat(row["last_attempt_at"]),
            delivered_at=_isoformat(row["delivered_at"]),
            last_error=row["last_error"],
            created_at=_isoformat(row["created_at"]) or "",
        )

    @staticmethod
    def _llm_failure_from_row(row: dict[str, Any]) -> LLMFailureRecord:
        return LLMFailureRecord(
            id=row["id"],
            job_id=row["job_id"],
            stage=row["stage"],
            attempt=row["attempt"],
            raw_reply=row["raw_reply"],
            error_message=row["error_message"],
            created_at=_isoformat(row["created_at"]) or "",
        )

    @staticmethod
    def _operation_from_row(row: dict[str, Any]) -> OperationRecord:
        return OperationRecord(
            id=row["id"],
            job_id=row["job_id"],
            operation_type=row["operation_type"],
            status=row["status"],
            idempotency_key=row["idempotency_key"],
            request_payload_hash=row["request_payload_hash"],
            base_versions=_json_object(row["base_versions"]),
            progress=_json_object(row["progress"]),
            result_summary=_json_object(row["result_summary"]),
            error=_json_object(row["error"]) if row["error"] is not None else None,
            poll_after_ms=row["poll_after_ms"],
            created_at=_isoformat(row["created_at"]) or "",
            started_at=_isoformat(row["started_at"]),
            finished_at=_isoformat(row["finished_at"]),
        )

    @staticmethod
    def _operation_item_result_from_row(
        row: dict[str, Any],
    ) -> OperationItemResultRecord:
        return OperationItemResultRecord(
            id=row["id"],
            operation_id=row["operation_id"],
            subject_kind=row["subject_kind"],
            subject_id=row["subject_id"],
            status=row["status"],
            path_event_id=row["path_event_id"],
            reservation_id=row["reservation_id"],
            from_path=_json_object(row["from_path"]) if row["from_path"] is not None else None,
            to_path=_json_object(row["to_path"]) if row["to_path"] is not None else None,
            identity_check=row["identity_check"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            retryable=bool(row["retryable"]),
            retry_block_reason=row["retry_block_reason"],
            result_revision=row["result_revision"],
            created_at=_isoformat(row["created_at"]) or "",
        )

    @staticmethod
    def _review_snapshot_from_row(row: dict[str, Any]) -> ReviewSnapshotRecord:
        return ReviewSnapshotRecord(
            job_id=row["job_id"],
            source_files=_json_list(row["source_files"]),
            selected_tmdb_info=_json_object(row["selected_tmdb_info"]),
            episode_targets=_json_list(row["episode_targets"]),
            subtitle_links=_json_list(row["subtitle_links"]),
            mapping_attempts=_json_list(row["mapping_attempts"]),
            mapping_evidence=_json_list(row["mapping_evidence"]),
            path_event_summary=_json_object(row["path_event_summary"]),
            validation_summary=_json_object(row["validation_summary"]),
            created_at=_isoformat(row["created_at"]) or "",
            updated_at=_isoformat(row["updated_at"]) or "",
        )

    @staticmethod
    def _destination_reservation_from_row(
        row: dict[str, Any],
    ) -> DestinationReservationRecord:
        return DestinationReservationRecord(
            id=row["id"],
            operation_id=row["operation_id"],
            job_id=row["job_id"],
            root_id=row["root_id"],
            root_generation=row["root_generation"],
            root_alias_group_id=row["root_alias_group_id"],
            path=_json_object(row["path"]),
            normalized_path_key=row["normalized_path_key"],
            casefold_path_key=row["casefold_path_key"],
            canonical_destination_key=row["canonical_destination_key"],
            canonical_parent_key=row["canonical_parent_key"],
            ancestor_keys=tuple(row["ancestor_keys"]),
            descendant_prefix_key=row["descendant_prefix_key"],
            state=row["state"],
            conflict_code=row["conflict_code"],
            conflict_ref=row["conflict_ref"],
            created_at=_isoformat(row["created_at"]) or "",
            expires_at=_isoformat(row["expires_at"]),
            released_at=_isoformat(row["released_at"]),
            committed_at=_isoformat(row["committed_at"]),
        )


def _normalize_db_value(field: str, value: object) -> object:
    if field in _TIMESTAMP_FIELDS:
        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise ValueError(f"{field} must be datetime, iso string, or null")
    return value


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError("json value must be an object")


def _json_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("json value must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("json list items must be objects")
        items.append(item)
    return items


def _reservation_keys_conflict(
    left_key: str,
    left_ancestors: Sequence[str],
    right_key: str,
    right_ancestors: Sequence[str],
    left_casefold_key: str | None = None,
    right_casefold_key: str | None = None,
) -> bool:
    if left_key == right_key:
        return True
    if left_key in right_ancestors:
        return True
    if right_key in left_ancestors:
        return True
    if left_casefold_key is None or right_casefold_key is None:
        return False
    left_casefold_ancestors = {item.casefold() for item in left_ancestors}
    right_casefold_ancestors = {item.casefold() for item in right_ancestors}
    if left_casefold_key == right_casefold_key:
        return True
    if left_casefold_key in right_casefold_ancestors:
        return True
    return right_casefold_key in left_casefold_ancestors


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()
