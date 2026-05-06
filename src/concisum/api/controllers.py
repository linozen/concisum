from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from litestar import Controller, Response, get, post

from concisum.api.job_store import JobStore, StepProgress
from concisum.api.pipeline_controller import _run_pipeline
from concisum.pipeline import StepRegistry

LOG = logging.getLogger(__name__)


@dataclass
class CreateJobRequest:
    utterances: list[dict[str, str]]
    diagnosis: bool = False
    tools: list[str] | None = None
    chunk_size: int = 50
    therapist_speaker_number: int = 1
    model: str | None = None


class JobController(Controller):
    path = "/jobs"

    @post("/")
    async def create_job(
        self,
        data: CreateJobRequest,
        job_store: JobStore,
    ) -> Response[dict[str, Any]]:
        """Backward-compatible job creation.

        Internally selects the appropriate pipeline template and delegates
        to PipelineExecutor.
        """
        template_id = "summary_with_diagnosis" if data.diagnosis else "summary"
        template = StepRegistry.get_template(template_id)

        step_progress = [
            StepProgress(
                instance_id=sc.instance_id,
                step_type=sc.step_type,
                label=StepRegistry.get_step(sc.step_type).definition().label,
            )
            for sc in template.steps
        ]

        job = job_store.create(
            pipeline_id=template_id,
            step_progress=step_progress,
        )

        # Build step overrides from the flat request params
        step_overrides: dict[str, dict[str, Any]] = {
            "chunk": {
                "chunk_size": data.chunk_size,
                "therapist_speaker": str(data.therapist_speaker_number),
            },
            "summarize": {
                "therapist_speaker": str(data.therapist_speaker_number),
            },
        }
        if data.diagnosis and data.tools:
            step_overrides["diagnose"] = {"tools": ",".join(data.tools)}
            step_overrides["symptoms"] = {
                "therapist_speaker": str(data.therapist_speaker_number),
            }

        input_data = {"utterances": data.utterances}

        asyncio.create_task(
            _run_pipeline(
                job.id,
                template_id,
                input_data,
                step_overrides,
                job_store,
                data.model,
            )
        )

        return Response(
            content={"job_id": job.id, "status": job.status.value},
            status_code=202,
        )

    @get("/{job_id:str}")
    async def get_job(
        self,
        job_id: str,
        job_store: JobStore,
    ) -> Response[dict[str, Any]]:
        job = job_store.get(job_id)
        if job is None:
            return Response(content={"error": "job not found"}, status_code=404)
        return Response(content=job.to_dict(), status_code=200)
