from __future__ import annotations

from dataclasses import dataclass

from aninamer.store import JobRecord, JobRequestRecord, RuntimeStore
from aninamer.worker import AninamerWorker

from .schemas import (
    JobRequestResponse,
    JobResponse,
    RuntimeResponse,
    StatusItem,
    StatusResponse,
    StatusSummary,
)


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

    def create_job_request(self, action: str, job_id: int | None) -> JobRequestResponse:
        request = self.store.create_job_request(kind=action, target_job_id=job_id)
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
