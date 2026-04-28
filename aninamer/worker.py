from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Callable, Literal

from aninamer.artifacts import (
    rename_plan_from_payload,
    rename_plan_to_payload,
    rename_result_to_payload,
)
from aninamer.config import AppConfig, WatchRootConfig
from aninamer.errors import NotificationDeliveryError
from aninamer.llm_client import LLMClient
from aninamer.monitoring import (
    MonitorTarget,
    discover_series_dirs_status,
    finalize_series_dir_after_apply,
    is_settled,
    is_transient_filesystem_error,
    move_series_dir_to_fail,
    path_is_dir,
    snapshot_series_files,
)
from aninamer.openai_llm_client import (
    openai_llm_for_tmdb_id_from_settings,
    openai_llm_from_settings,
)
from aninamer.pipeline import (
    PlanBuildOptions,
    build_rename_plan_for_series,
    execute_apply,
    tmdb_client_from_settings,
)
from aninamer.plan import RenamePlan
from aninamer.store import JobRecord, JobRequestRecord, NotificationRecord, RuntimeStore
from aninamer.tmdb_client import TMDBClient, build_poster_image_url
from aninamer.webhook_delivery import (
    WebhookTransport,
    response_error_text,
    send_notification_webhook,
)

logger = logging.getLogger(__name__)
MAX_NOTIFICATION_BATCH = 100
MAX_NOTIFICATION_BACKOFF_SECONDS = 300
_TELEGRAM_MARKDOWN_V2_SPECIALS = frozenset("\\_*[]()~`>#+-=|{}.!")


@dataclass(frozen=True)
class WorkerRuntimeSummary:
    last_scan_at: str | None
    auto_apply: bool
    settle_seconds: int
    scan_interval_seconds: int
    watch_root_keys: tuple[str, ...]


@dataclass(frozen=True)
class WorkerHealthStatus:
    healthy: bool
    reason: str | None
    running: bool
    started_at: str | None
    stopped_at: str | None
    current_scan_started_at: str | None
    last_scan_at: str | None
    last_success_at: str | None
    last_error_at: str | None
    last_error_message: str | None
    consecutive_failures: int
    stale_after_seconds: int
    unavailable_watch_root_keys: tuple[str, ...]


@dataclass(frozen=True)
class NotificationPresentation:
    severity: str
    title: str
    message: str
    markdown: str


class AninamerWorker:
    def __init__(
        self,
        config: AppConfig,
        store: RuntimeStore,
        *,
        tmdb_client_factory: Callable[[], TMDBClient] | None = None,
        llm_for_tmdb_id_factory: Callable[[], LLMClient] | None = None,
        llm_for_mapping_factory: Callable[[], LLMClient] | None = None,
        webhook_transport: WebhookTransport | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._tmdb_client_factory = tmdb_client_factory or (
            lambda: tmdb_client_from_settings(config.tmdb)
        )
        self._llm_for_tmdb_id_factory = llm_for_tmdb_id_factory or (
            lambda: openai_llm_for_tmdb_id_from_settings(config.openai)
        )
        self._llm_for_mapping_factory = llm_for_mapping_factory or (
            lambda: openai_llm_from_settings(config.openai)
        )
        self._webhook_transport = webhook_transport
        self._targets = tuple(
            MonitorTarget(
                key=item.key,
                input_root=item.input_root,
                output_root=item.output_root,
            )
            for item in config.watch_roots
        )
        self._state_lock = threading.Lock()
        self._running = False
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._current_scan_started_at: str | None = None
        self._last_success_at: str | None = None
        self._last_error_at: str | None = None
        self._last_error_message: str | None = None
        self._consecutive_failures = 0
        self._watch_root_errors: dict[str, str] = {}
        if config.notifications is None and config.notifications_warning is not None:
            logger.warning(config.notifications_warning)

    def runtime_summary(self) -> WorkerRuntimeSummary:
        snapshot = self._store.snapshot()
        return WorkerRuntimeSummary(
            last_scan_at=snapshot.last_scan_at,
            auto_apply=self._config.worker.auto_apply,
            settle_seconds=self._config.worker.settle_seconds,
            scan_interval_seconds=self._config.worker.scan_interval_seconds,
            watch_root_keys=tuple(item.key for item in self._targets),
        )

    def health_status(self) -> WorkerHealthStatus:
        snapshot = self._store.snapshot()
        with self._state_lock:
            running = self._running
            started_at = self._started_at
            stopped_at = self._stopped_at
            current_scan_started_at = self._current_scan_started_at
            last_success_at = self._last_success_at
            last_error_at = self._last_error_at
            last_error_message = self._last_error_message
            consecutive_failures = self._consecutive_failures
            unavailable_watch_root_keys = tuple(sorted(self._watch_root_errors))

        stale_after_seconds = self._config.worker.health_stale_after_seconds
        reason: str | None = None
        if not running:
            reason = "worker has not started" if started_at is None else "worker is not running"
        elif unavailable_watch_root_keys and len(unavailable_watch_root_keys) == len(
            self._targets
        ):
            reason = "all watch roots are unavailable"
        elif _is_stale(
            snapshot.last_scan_at,
            stale_after_seconds=stale_after_seconds,
            now=datetime.now(timezone.utc),
            fallback_started_at=current_scan_started_at or started_at,
        ):
            reason = "worker scan is stale"

        return WorkerHealthStatus(
            healthy=reason is None,
            reason=reason,
            running=running,
            started_at=started_at,
            stopped_at=stopped_at,
            current_scan_started_at=current_scan_started_at,
            last_scan_at=snapshot.last_scan_at,
            last_success_at=last_success_at,
            last_error_at=last_error_at,
            last_error_message=last_error_message,
            consecutive_failures=consecutive_failures,
            stale_after_seconds=stale_after_seconds,
            unavailable_watch_root_keys=unavailable_watch_root_keys,
        )

    def recover(self) -> None:
        recovered_failed_jobs = self._store.recover_incomplete_jobs()
        for job in recovered_failed_jobs:
            self._notify_job_apply_failed(job)

    def scan_once(self) -> None:
        self._deliver_due_notifications()
        self._process_job_requests()
        self._discover_new_jobs()
        self._process_jobs()
        self._store.set_last_scan_at()

    def run_forever(self, shutdown_check: Callable[[], bool]) -> None:
        self._mark_worker_started()
        try:
            try:
                self.recover()
            except Exception as exc:
                self._record_scan_failure(exc)
                logger.exception("worker: recover_failed")

            while not shutdown_check():
                self._mark_scan_started()
                try:
                    self.scan_once()
                except Exception as exc:
                    self._record_scan_failure(exc)
                    logger.exception("worker: scan_loop_failed")
                else:
                    self._record_scan_success()
                finally:
                    self._mark_scan_finished()

                sleep_remaining = self._config.worker.scan_interval_seconds
                while sleep_remaining > 0 and not shutdown_check():
                    chunk = min(sleep_remaining, 1.0)
                    time.sleep(chunk)
                    sleep_remaining -= chunk
        finally:
            self._mark_worker_stopped()

    def _process_job_requests(self) -> None:
        for request in self._store.list_pending_job_requests():
            self._handle_job_request(request)

    def _mark_worker_started(self) -> None:
        now = _now_iso()
        with self._state_lock:
            self._running = True
            self._started_at = now
            self._stopped_at = None

    def _mark_worker_stopped(self) -> None:
        now = _now_iso()
        with self._state_lock:
            self._running = False
            self._stopped_at = now
            self._current_scan_started_at = None

    def _mark_scan_started(self) -> None:
        with self._state_lock:
            self._current_scan_started_at = _now_iso()

    def _mark_scan_finished(self) -> None:
        with self._state_lock:
            self._current_scan_started_at = None

    def _record_scan_success(self) -> None:
        now = _now_iso()
        with self._state_lock:
            self._last_success_at = now
            self._consecutive_failures = 0

    def _record_scan_failure(self, exc: Exception) -> None:
        now = _now_iso()
        with self._state_lock:
            self._last_error_at = now
            self._last_error_message = f"{type(exc).__name__}: {exc}"
            self._consecutive_failures += 1

    def _record_watch_root_available(self, key: str) -> None:
        with self._state_lock:
            self._watch_root_errors.pop(key, None)

    def _record_watch_root_unavailable(self, key: str, error_message: str | None) -> None:
        with self._state_lock:
            self._watch_root_errors[key] = error_message or "unavailable"

    def _handle_job_request(self, request: JobRequestRecord) -> None:
        started = self._store.update_job_request(
            request.id,
            status="running",
            started_at=_now_or_existing(request.started_at),
            error_message=None,
        )
        try:
            if started.kind == "scan_now":
                self._store.update_job_request(
                    started.id,
                    status="succeeded",
                    finished_at=_now_iso(),
                )
                return

            if started.kind != "apply_job":
                raise ValueError(f"unsupported request kind {started.kind}")
            if started.target_job_id is None:
                rejected = self._store.update_job_request(
                    started.id,
                    status="rejected",
                    finished_at=_now_iso(),
                    error_message="apply_job request requires target_job_id",
                )
                self._notify_job_request_rejected(rejected)
                return

            job = self._store.get_job(started.target_job_id)
            if job is None:
                rejected = self._store.update_job_request(
                    started.id,
                    status="rejected",
                    finished_at=_now_iso(),
                    error_message="job not found",
                )
                self._notify_job_request_rejected(rejected)
                return
            if job.status != "planned":
                rejected = self._store.update_job_request(
                    started.id,
                    status="rejected",
                    finished_at=_now_iso(),
                    error_message="job is not in planned status",
                )
                self._notify_job_request_rejected(rejected)
                return

            self._store.update_job(
                job.id,
                status="apply_requested",
                error_stage=None,
                error_message=None,
            )
            self._store.update_job_request(
                started.id,
                status="succeeded",
                finished_at=_now_iso(),
            )
        except Exception as exc:
            failed = self._store.update_job_request(
                started.id,
                status="failed",
                finished_at=_now_iso(),
                error_message=str(exc),
            )
            self._notify_job_request_failed(failed)

    def _discover_new_jobs(self) -> None:
        for target in self._targets:
            try:
                discovery = discover_series_dirs_status(target.input_root)
            except Exception:
                logger.exception(
                    "worker: discover_target_failed watch_root_key=%s input_root=%s",
                    target.key,
                    target.input_root,
                )
                self._record_watch_root_unavailable(target.key, "discovery failed")
                continue
            if discovery.unavailable:
                self._record_watch_root_unavailable(target.key, discovery.error_message)
                continue
            self._record_watch_root_available(target.key)
            series_dirs = discovery.series_dirs
            for series_dir in series_dirs:
                if self._store.find_active_job_by_series_dir(series_dir) is not None:
                    continue
                logger.info(
                    "worker: discovered series_dir=%s watch_root_key=%s",
                    series_dir,
                    target.key,
                )
                self._store.create_job(
                    series_name=series_dir.name,
                    watch_root_key=target.key,
                    source_kind="monitor",
                    series_dir=series_dir,
                    output_root=target.output_root,
                )

    def _process_jobs(self) -> None:
        for job in self._store.list_jobs():
            if job.status == "pending":
                self._maybe_plan_job(job)
            elif job.status == "apply_requested":
                self._apply_job(job)

    def _maybe_plan_job(self, job: JobRecord) -> None:
        series_dir = Path(job.series_dir)
        try:
            if not path_is_dir(series_dir):
                return
            if not is_settled(series_dir, self._config.worker.settle_seconds):
                return
        except OSError as exc:
            if is_transient_filesystem_error(exc):
                logger.warning(
                    "worker: source_unavailable_skip_plan job_id=%s series_dir=%s error=%s",
                    job.id,
                    series_dir,
                    exc,
                )
                return
            self._fail_job(job, stage="plan", exc=exc)
            return

        target = self._target_for_key(job.watch_root_key)
        self._store.update_job(
            job.id,
            status="planning",
            started_at=_now_or_existing(job.started_at),
            error_stage=None,
            error_message=None,
        )
        try:
            plan = build_rename_plan_for_series(
                series_dir=series_dir,
                output_root=target.output_root,
                options=PlanBuildOptions(
                    max_candidates=self._config.worker.max_candidates,
                    max_output_tokens=self._config.worker.max_output_tokens,
                    allow_existing_dest=self._config.worker.allow_existing_dest,
                ),
                tmdb_client_factory=self._tmdb_client_factory,
                llm_for_tmdb_id_factory=self._llm_for_tmdb_id_factory,
                llm_for_mapping_factory=self._llm_for_mapping_factory,
            )
            self._store.save_artifact(job.id, "plan", rename_plan_to_payload(plan))
            video_count, subtitle_count = _count_moves(plan)
            updated_job = self._store.update_job(
                job.id,
                status="apply_requested" if self._config.worker.auto_apply else "planned",
                tmdb_id=plan.tmdb_id,
                video_moves_count=video_count,
                subtitle_moves_count=subtitle_count,
            )
            if updated_job.status == "apply_requested":
                logger.info("worker: auto_apply_requested job_id=%s", updated_job.id)
        except Exception as exc:
            if _is_transient_exception(exc):
                self._retry_job_after_transient_error(
                    job,
                    retry_status="pending",
                    stage="plan",
                    series_dir=series_dir,
                    exc=exc,
                )
                return
            self._fail_job(job, stage="plan", exc=exc)

    def _apply_job(self, job: JobRecord) -> None:
        payload = self._store.load_artifact(job.id, "plan")
        if payload is None:
            self._fail_job(job, stage="apply", exc=ValueError("plan artifact missing"))
            return

        series_dir = Path(job.series_dir)
        self._store.update_job(
            job.id,
            status="applying",
            started_at=_now_or_existing(job.started_at),
            error_stage=None,
            error_message=None,
        )
        try:
            plan = rename_plan_from_payload(payload)
            before_files = snapshot_series_files(series_dir)
            execution = execute_apply(
                plan,
                dry_run=False,
                two_stage=self._config.worker.two_stage,
            )
            self._store.save_artifact(
                job.id,
                "rollback",
                rename_plan_to_payload(execution.rollback_plan),
            )
            finalize = finalize_series_dir_after_apply(
                series_dir,
                plan,
                before_files=before_files,
            )
            self._store.save_artifact(
                job.id,
                "result",
                rename_result_to_payload(
                    execution=execution,
                    finalize=finalize,
                ),
            )
            updated_job = self._store.update_job(
                job.id,
                status="succeeded",
                finished_at=_now_iso(),
                archive_path=(
                    str(finalize.archive_path)
                    if finalize.archive_path is not None
                    else None
                ),
                fail_path=None,
            )
            self._notify_job_apply_succeeded(updated_job, finalize.status)
        except Exception as exc:
            if _is_transient_exception(exc):
                self._retry_job_after_transient_error(
                    job,
                    retry_status="apply_requested",
                    stage="apply",
                    series_dir=series_dir,
                    exc=exc,
                )
                return
            self._fail_job(job, stage="apply", exc=exc)

    def _retry_job_after_transient_error(
        self,
        job: JobRecord,
        *,
        retry_status: Literal["pending", "apply_requested"],
        stage: Literal["plan", "apply"],
        series_dir: Path,
        exc: Exception,
    ) -> None:
        logger.warning(
            "worker: transient_%s_error_retry job_id=%s series_dir=%s error=%s",
            stage,
            job.id,
            series_dir,
            exc,
        )
        self._store.update_job(
            job.id,
            status=retry_status,
            error_stage=None,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    def _fail_job(self, job: JobRecord, *, stage: str, exc: Exception) -> None:
        series_dir = Path(job.series_dir)
        fail_path: str | None = None
        try:
            can_move_to_fail = path_is_dir(series_dir)
        except OSError as path_exc:
            can_move_to_fail = False
            logger.warning(
                "worker: failed_job_source_unavailable job_id=%s series_dir=%s error=%s",
                job.id,
                series_dir,
                path_exc,
            )
        if can_move_to_fail:
            try:
                fail_path = str(move_series_dir_to_fail(series_dir))
            except Exception as move_exc:
                logger.warning(
                    "worker: failed_move_to_fail series_dir=%s error=%s",
                    series_dir,
                    move_exc,
                )
        logger.exception("worker: job_failed id=%s stage=%s", job.id, stage, exc_info=exc)
        updated_job = self._store.update_job(
            job.id,
            status="failed",
            error_stage=stage,
            error_message=f"{type(exc).__name__}: {exc}",
            finished_at=_now_iso(),
            fail_path=fail_path,
        )
        if stage == "plan":
            self._notify_job_plan_failed(updated_job)
        elif stage == "apply":
            self._notify_job_apply_failed(updated_job)

    def _target_for_key(self, key: str) -> MonitorTarget:
        for item in self._targets:
            if item.key == key:
                return item
        raise KeyError(key)

    def _deliver_due_notifications(self) -> None:
        notifications = self._store.list_due_notifications(limit=MAX_NOTIFICATION_BATCH)
        if not notifications:
            return
        config = self._config.notifications
        if config is None:
            return
        for notification in notifications:
            attempt_count = notification.attempt_count + 1
            try:
                response = send_notification_webhook(
                    config,
                    markdown=notification.markdown,
                    image_url=notification.image_url,
                    disable_web_page_preview=notification.disable_web_page_preview,
                    disable_notification=notification.disable_notification,
                    transport=self._webhook_transport,
                )
                if 200 <= response.status < 300:
                    self._store.mark_notification_delivered(
                        notification.id,
                        attempt_count=attempt_count,
                    )
                    continue
                raise NotificationDeliveryError(
                    f"webhook returned {response.status}: {response_error_text(response)}"
                )
            except Exception as exc:
                logger.warning(
                    "worker: webhook_delivery_failed notification_id=%s attempt=%s error=%s",
                    notification.id,
                    attempt_count,
                    exc,
                )
                self._store.mark_notification_retry(
                    notification.id,
                    attempt_count=attempt_count,
                    next_attempt_at=_retry_at_iso(attempt_count),
                    last_error=str(exc),
                )

    def _notify_job_plan_failed(self, job: JobRecord) -> None:
        self._create_notification(
            event_kind="job_plan_failed",
            job_id=job.id,
            payload={
                "error_stage": job.error_stage,
                "error_message": job.error_message,
            },
        )

    def _notify_job_apply_succeeded(
        self,
        job: JobRecord,
        finalize_status: str,
    ) -> None:
        self._create_notification(
            event_kind="job_apply_succeeded",
            job_id=job.id,
            payload={"finalize_status": finalize_status},
        )

    def _notify_job_apply_failed(self, job: JobRecord) -> None:
        self._create_notification(
            event_kind="job_apply_failed",
            job_id=job.id,
            payload={
                "error_stage": job.error_stage,
                "error_message": job.error_message,
            },
        )

    def _notify_job_request_rejected(self, request: JobRequestRecord) -> None:
        self._create_notification(
            event_kind="job_request_rejected",
            job_id=request.target_job_id,
            job_request_id=request.id,
            payload={
                "request_action": request.kind,
                "job_id": request.target_job_id,
                "error_message": request.error_message,
            },
        )

    def _notify_job_request_failed(self, request: JobRequestRecord) -> None:
        self._create_notification(
            event_kind="job_request_failed",
            job_id=request.target_job_id,
            job_request_id=request.id,
            payload={
                "request_action": request.kind,
                "job_id": request.target_job_id,
                "error_message": request.error_message,
            },
        )

    def _create_notification(
        self,
        *,
        event_kind: str,
        job_id: int | None = None,
        job_request_id: int | None = None,
        payload: dict[str, object] | None = None,
    ) -> NotificationRecord:
        job = self._store.get_job(job_id) if job_id is not None else None
        job_request = (
            self._store.get_job_request(job_request_id)
            if job_request_id is not None
            else None
        )
        resolved_payload = payload or {}
        presentation = _build_notification_presentation(
            event_kind=event_kind,
            job=job,
            job_request=job_request,
            payload=resolved_payload,
        )
        image_url = self._notification_image_url(job)
        return self._store.create_notification(
            event_kind=event_kind,
            severity=presentation.severity,
            title=presentation.title,
            message=presentation.message,
            markdown=presentation.markdown,
            image_url=image_url,
            job_id=job_id,
            job_request_id=job_request_id,
            payload=payload,
            delivery_status=(
                "pending" if self._config.notifications is not None else "disabled"
            ),
        )

    def _notification_image_url(self, job: JobRecord | None) -> str:
        if job is None or job.tmdb_id is None:
            return ""
        try:
            _series_name, details = self._tmdb_client_factory().resolve_series_title(
                job.tmdb_id
            )
        except Exception as exc:
            logger.warning(
                "worker: notification_image_lookup_failed job_id=%s tmdb_id=%s error=%s",
                job.id,
                job.tmdb_id,
                exc,
            )
            return ""
        return build_poster_image_url(details.poster_path)


def _count_moves(plan: RenamePlan) -> tuple[int, int]:
    video_count = sum(1 for move in plan.moves if move.kind == "video")
    subtitle_count = sum(1 for move in plan.moves if move.kind == "subtitle")
    return video_count, subtitle_count


def _now_or_existing(value: str | None) -> str:
    if value is not None:
        return value
    return _now_iso()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_stale(
    last_scan_at: str | None,
    *,
    stale_after_seconds: int,
    now: datetime,
    fallback_started_at: str | None,
) -> bool:
    observed_candidates = [
        item
        for item in (
            _parse_iso_datetime(last_scan_at),
            _parse_iso_datetime(fallback_started_at),
        )
        if item is not None
    ]
    if not observed_candidates:
        return False
    observed_at = max(observed_candidates)
    return (now - observed_at).total_seconds() > stale_after_seconds


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_transient_exception(exc: Exception) -> bool:
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, OSError) and is_transient_filesystem_error(current):
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def _build_notification_presentation(
    *,
    event_kind: str,
    job: JobRecord | None,
    job_request: JobRequestRecord | None,
    payload: dict[str, object],
) -> NotificationPresentation:
    del job_request
    subject = _notification_subject(job)

    if event_kind == "job_apply_succeeded":
        finalize_status = payload.get("finalize_status")
        if finalize_status == "archived":
            title = "归档成功"
            message = "已归档完成"
            severity = "success"
        elif finalize_status == "deleted":
            title = "处理完成"
            message = "已完成处理，源目录已清理"
            severity = "warning"
        else:
            title = "处理完成"
            message = "已完成处理，未执行归档"
            severity = "warning"
    elif event_kind == "job_plan_failed":
        title = "归档失败"
        message = "生成归档计划失败"
        severity = "error"
    elif event_kind == "job_apply_failed":
        title = "归档失败"
        message = "执行归档失败"
        severity = "error"
    elif event_kind == "job_request_rejected":
        title = "归档失败"
        message = "归档请求被拒绝"
        severity = "error"
    elif event_kind == "job_request_failed":
        title = "归档失败"
        message = "处理归档请求失败"
        severity = "error"
    else:
        title = "归档失败"
        message = "归档任务失败"
        severity = "error"

    return NotificationPresentation(
        severity=severity,
        title=title,
        message=message,
        markdown=_render_notification_markdown(
            title=title,
            subject=subject,
            message=message,
        ),
    )


def _notification_subject(job: JobRecord | None) -> str:
    if job is None:
        return "归档任务"
    return f"《{job.series_name}》"


def _render_notification_markdown(
    *,
    title: str,
    subject: str,
    message: str,
) -> str:
    return "\n".join(
        [
            _escape_telegram_markdown_v2(title),
            "",
            _escape_telegram_markdown_v2(subject),
            _escape_telegram_markdown_v2(message),
        ]
    )


def _escape_telegram_markdown_v2(text: str) -> str:
    escaped: list[str] = []
    for char in text:
        if char in _TELEGRAM_MARKDOWN_V2_SPECIALS:
            escaped.append("\\")
        escaped.append(char)
    return "".join(escaped)


def _retry_at_iso(attempt_count: int) -> str:
    delay_seconds = min(10 * (2 ** max(attempt_count - 1, 0)), MAX_NOTIFICATION_BACKOFF_SECONDS)
    retry_at = datetime.now(timezone.utc).timestamp() + delay_seconds
    return datetime.fromtimestamp(retry_at, tz=timezone.utc).isoformat()
