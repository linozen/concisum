from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from concisum.pipeline.step import StepStatus

if TYPE_CHECKING:
    from concisum.api.job_store import JobStore
    from concisum.pipeline.registry import StepRegistry as StepRegistryType

LOG = logging.getLogger(__name__)


class EdgeDef(BaseModel):
    """A directed edge in the pipeline DAG."""

    from_step: str  # instance_id of source step
    to_step: str  # instance_id of target step


class StepPosition(BaseModel):
    x: float = 0
    y: float = 0


class StepConfig(BaseModel):
    """Per-step configuration within a pipeline instance."""

    step_type: str  # registry key, e.g. "chunk_transcript"
    instance_id: str  # unique within this pipeline, e.g. "chunk"
    params: dict[str, Any] = {}
    position: StepPosition = StepPosition()


class PipelineTemplate(BaseModel):
    """A reusable pipeline definition (the 'recipe')."""

    id: str
    label: str
    description: str
    steps: list[StepConfig]
    edges: list[EdgeDef]


class PipelineRunRequest(BaseModel):
    """What the client sends to execute a pipeline."""

    template_id: str
    step_overrides: dict[str, dict[str, Any]] = {}
    input_data: dict[str, Any]


class PipelineExecutor:
    """Executes a pipeline DAG, tracking per-step status."""

    def __init__(
        self,
        template: PipelineTemplate,
        registry: type[StepRegistryType],
        job_store: JobStore,
        job_id: str,
    ):
        self.template = template
        self.registry = registry
        self.job_store = job_store
        self.job_id = job_id

    def _topo_sort(self) -> list[StepConfig]:
        """Topological sort of the pipeline DAG."""
        step_map = {s.instance_id: s for s in self.template.steps}
        in_degree: dict[str, int] = {s.instance_id: 0 for s in self.template.steps}
        adj: dict[str, list[str]] = defaultdict(list)

        for edge in self.template.edges:
            adj[edge.from_step].append(edge.to_step)
            in_degree[edge.to_step] += 1

        queue: deque[str] = deque(
            sid for sid, deg in in_degree.items() if deg == 0
        )
        ordered: list[StepConfig] = []

        while queue:
            sid = queue.popleft()
            ordered.append(step_map[sid])
            for neighbor in adj[sid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(self.template.steps):
            raise ValueError("Pipeline DAG contains a cycle")

        return ordered

    def _resolve_inputs(
        self,
        step_config: StepConfig,
        outputs: dict[str, Any],
        initial_input: Any,
    ) -> Any:
        """Resolve the input for a step from its predecessor outputs."""
        predecessors = [
            e.from_step
            for e in self.template.edges
            if e.to_step == step_config.instance_id
        ]
        if not predecessors:
            return initial_input
        if len(predecessors) == 1:
            return outputs[predecessors[0]]
        # Multiple inputs: return dict keyed by predecessor instance_id
        return {pid: outputs[pid] for pid in predecessors}

    async def execute(
        self,
        input_data: dict[str, Any],
        step_overrides: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute the pipeline DAG and return all step outputs."""
        ordered = self._topo_sort()
        outputs: dict[str, Any] = {}

        # The initial input is the raw input_data dict
        # Steps that have no predecessors receive this directly
        initial_input = input_data

        for step_config in ordered:
            step_cls = self.registry.get_step(step_config.step_type)
            step = step_cls()

            # Merge params: template defaults <- step_overrides
            merged_params = dict(step_config.params)
            if step_config.instance_id in step_overrides:
                merged_params.update(step_overrides[step_config.instance_id])

            # Resolve input from predecessors
            step_input = self._resolve_inputs(step_config, outputs, initial_input)

            # Update job store: step is running
            self.job_store.update_step_status(
                self.job_id, step_config.instance_id, StepStatus.RUNNING
            )
            self.job_store.update_progress(
                self.job_id, step_cls.definition().label
            )

            def make_progress_cb(instance_id: str):
                def cb(detail: str) -> None:
                    self.job_store.update_step_detail(
                        self.job_id, instance_id, detail
                    )
                    self.job_store.update_progress(self.job_id, detail)
                return cb

            try:
                result = await step.run(
                    step_input,
                    merged_params,
                    on_progress=make_progress_cb(step_config.instance_id),
                )
                outputs[step_config.instance_id] = result
                self.job_store.update_step_status(
                    self.job_id, step_config.instance_id, StepStatus.COMPLETED
                )
                LOG.info(
                    "Pipeline %s: step %s completed",
                    self.job_id,
                    step_config.instance_id,
                )
            except Exception as exc:
                self.job_store.update_step_status(
                    self.job_id,
                    step_config.instance_id,
                    StepStatus.FAILED,
                    detail=str(exc),
                )
                raise

        return outputs
