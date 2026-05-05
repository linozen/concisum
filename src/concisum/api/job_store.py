from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StepProgress:
    """Per-step progress tracking within a pipeline job."""

    instance_id: str
    step_type: str
    label: str
    status: str = "pending"  # pending/running/completed/failed
    detail: str = ""


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress_stage: str = ""
    pipeline_id: str | None = None
    step_progress: list[StepProgress] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "status": self.status.value,
            "progress_stage": self.progress_stage,
            "pipeline_id": self.pipeline_id,
            "step_progress": [
                {
                    "instance_id": sp.instance_id,
                    "step_type": sp.step_type,
                    "label": sp.label,
                    "status": sp.status,
                    "detail": sp.detail,
                }
                for sp in self.step_progress
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(
        self,
        pipeline_id: str | None = None,
        step_progress: list[StepProgress] | None = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            pipeline_id=pipeline_id,
            step_progress=step_progress or [],
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, progress_stage: str = "") -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = status
        if progress_stage:
            job.progress_stage = progress_stage
        job.updated_at = datetime.now(timezone.utc).isoformat()

    def update_progress(self, job_id: str, stage: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.progress_stage = stage
        job.updated_at = datetime.now(timezone.utc).isoformat()

    def update_result(self, job_id: str, result: dict[str, Any]) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.COMPLETED
        job.result = result
        job.updated_at = datetime.now(timezone.utc).isoformat()

    def update_error(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.FAILED
        job.error = error
        job.updated_at = datetime.now(timezone.utc).isoformat()

    def update_step_status(
        self,
        job_id: str,
        instance_id: str,
        status: str | Enum,
        detail: str = "",
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        status_str = status.value if isinstance(status, Enum) else status
        for sp in job.step_progress:
            if sp.instance_id == instance_id:
                sp.status = status_str
                if detail:
                    sp.detail = detail
                break
        job.updated_at = datetime.now(timezone.utc).isoformat()

    def update_step_detail(self, job_id: str, instance_id: str, detail: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        for sp in job.step_progress:
            if sp.instance_id == instance_id:
                sp.detail = detail
                break
        job.updated_at = datetime.now(timezone.utc).isoformat()
