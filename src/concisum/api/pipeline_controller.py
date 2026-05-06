from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from litestar import Controller, Response, get, post

from concisum.api.job_store import JobStatus, JobStore, StepProgress
from concisum.config import set_active_model
from concisum.pipeline import StepRegistry
from concisum.pipeline.pipeline import PipelineExecutor

LOG = logging.getLogger(__name__)


@dataclass
class RunRequest:
    template_id: str
    step_overrides: dict[str, dict[str, Any]] | None = None
    input_data: dict[str, Any] | None = None
    model: str | None = None


async def _run_pipeline(
    job_id: str,
    template_id: str,
    input_data: dict[str, Any],
    step_overrides: dict[str, dict[str, Any]],
    job_store: JobStore,
    model: str | None = None,
) -> None:
    try:
        # Apply per-request model override; the contextvar is scoped to this
        # asyncio.Task so concurrent requests don't interfere.
        set_active_model(model)

        job_store.update_status(job_id, JobStatus.PROCESSING)
        template = StepRegistry.get_template(template_id)
        executor = PipelineExecutor(template, StepRegistry, job_store, job_id)
        outputs = await executor.execute(input_data, step_overrides)

        # Build result from pipeline outputs
        result = _build_result(template_id, outputs)
        job_store.update_result(job_id, result)
        LOG.info("Pipeline job %s completed", job_id)

    except Exception as exc:
        LOG.exception("Pipeline job %s failed", job_id)
        job_store.update_error(job_id, f"{type(exc).__name__}: {exc}")


def _build_result(template_id: str, outputs: dict[str, Any]) -> dict[str, Any]:
    """Convert pipeline step outputs into the API result format.

    Maintains backward compatibility with the existing result structure
    (content, diagnosis, symptoms) so thoth can consume the result
    without changes.
    """
    result: dict[str, Any] = {}

    # Summary content from combine step
    if "combine" in outputs:
        full_summary = outputs["combine"]
        result["content"] = full_summary.content
        result["diagnosis"] = None
        result["symptoms"] = None

    # Diagnosis from diagnose step
    if "diagnose" in outputs:
        diagnosis = outputs["diagnose"]
        result["diagnosis"] = diagnosis.model_dump() if hasattr(diagnosis, "model_dump") else diagnosis

    # Symptoms from symptoms step
    if "symptoms" in outputs:
        symptoms = outputs["symptoms"]
        result["symptoms"] = symptoms.model_dump() if hasattr(symptoms, "model_dump") else symptoms

    # Topics from aggregate_topics step
    if "aggregate_topics" in outputs:
        topic_report = outputs["aggregate_topics"]
        result["topics"] = topic_report.model_dump() if hasattr(topic_report, "model_dump") else topic_report

    # If no combine step (diagnosis_only, topic_modelling), set content to None
    if "content" not in result:
        result["content"] = None

    return result


class PipelineController(Controller):
    path = "/pipelines"

    @get("/steps")
    async def list_steps(self) -> list[dict[str, Any]]:
        return [s.model_dump() for s in StepRegistry.list_steps()]

    @get("/templates")
    async def list_templates(self) -> list[dict[str, Any]]:
        return [t.model_dump() for t in StepRegistry.list_templates()]

    @get("/templates/{template_id:str}")
    async def get_template(self, template_id: str) -> Response[dict[str, Any]]:
        try:
            t = StepRegistry.get_template(template_id)
            return Response(content=t.model_dump(), status_code=200)
        except KeyError:
            return Response(content={"error": "template not found"}, status_code=404)

    @post("/run")
    async def run_pipeline(
        self,
        data: RunRequest,
        job_store: JobStore,
    ) -> Response[dict[str, Any]]:
        try:
            template = StepRegistry.get_template(data.template_id)
        except KeyError:
            return Response(
                content={"error": f"unknown template: {data.template_id}"},
                status_code=400,
            )

        # Pre-populate step progress from template
        step_progress = [
            StepProgress(
                instance_id=sc.instance_id,
                step_type=sc.step_type,
                label=StepRegistry.get_step(sc.step_type).definition().label,
            )
            for sc in template.steps
        ]

        job = job_store.create(
            pipeline_id=template.id,
            step_progress=step_progress,
        )

        asyncio.create_task(
            _run_pipeline(
                job.id,
                data.template_id,
                data.input_data or {},
                data.step_overrides or {},
                job_store,
                data.model,
            )
        )

        return Response(
            content={"job_id": job.id, "status": job.status.value},
            status_code=202,
        )
