from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


JobSourceKind = Literal["monitor", "api"]
JobStatus = Literal[
    "pending",
    "planning",
    "planned",
    "apply_requested",
    "applying",
    "succeeded",
    "failed",
]
JobRequestKind = Literal["scan_now", "apply_job"]
JobRequestStatus = Literal["pending", "running", "succeeded", "failed", "rejected"]
ArtifactKind = Literal["plan", "result", "rollback"]
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
_TIMESTAMP_FIELDS = {
    "started_at",
    "finished_at",
    "created_at",
    "updated_at",
    "last_scan_at",
    "next_attempt_at",
    "last_attempt_at",
    "delivered_at",
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
                                    'failed'
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
                            CHECK (artifact_kind IN ('plan', 'result', 'rollback')),
                        payload JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (job_id, artifact_kind)
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
                    ALTER TABLE notifications
                    ADD COLUMN IF NOT EXISTS markdown TEXT NOT NULL DEFAULT ''
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
                    UPDATE jobs
                    SET status = 'failed',
                        error_stage = 'apply',
                        error_message = 'worker restarted during apply',
                        finished_at = %s,
                        updated_at = %s
                    WHERE status = 'applying'
                    RETURNING *
                    """,
                    (now, now),
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
                        disable_web_page_preview,
                        disable_notification,
                        delivery_status,
                        attempt_count,
                        next_attempt_at,
                        last_attempt_at,
                        delivered_at,
                        last_error
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, NULL, NULL, NULL)
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


def _normalize_db_value(field: str, value: object) -> object:
    if field in _TIMESTAMP_FIELDS:
        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise ValueError(f"{field} must be datetime, iso string, or null")
    return value


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()
