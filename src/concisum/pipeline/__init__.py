"""Pipeline framework for concisum.

Importing this module registers all built-in steps and pipeline templates.
"""

from concisum.pipeline.pipeline import (
    EdgeDef,
    PipelineTemplate,
    StepConfig,
    StepPosition,
)
from concisum.pipeline.registry import StepRegistry

# Import step modules to trigger @StepRegistry.register_step decorators
import concisum.summary.steps  # noqa: F401
import concisum.diagnosis.steps  # noqa: F401

# -- Built-in pipeline templates --

StepRegistry.register_template(
    PipelineTemplate(
        id="summary",
        label="Summary",
        description="Chunk, summarize, and combine a transcript",
        steps=[
            StepConfig(
                step_type="chunk_transcript",
                instance_id="chunk",
                position=StepPosition(x=0, y=100),
            ),
            StepConfig(
                step_type="summarize_chunks",
                instance_id="summarize",
                position=StepPosition(x=250, y=100),
            ),
            StepConfig(
                step_type="combine_summaries",
                instance_id="combine",
                position=StepPosition(x=500, y=100),
            ),
        ],
        edges=[
            EdgeDef(from_step="chunk", to_step="summarize"),
            EdgeDef(from_step="summarize", to_step="combine"),
        ],
    )
)

StepRegistry.register_template(
    PipelineTemplate(
        id="summary_with_diagnosis",
        label="Summary + Diagnosis",
        description="Full pipeline with symptom extraction and ICD-10 diagnosis",
        steps=[
            StepConfig(
                step_type="chunk_transcript",
                instance_id="chunk",
                position=StepPosition(x=0, y=130),
            ),
            StepConfig(
                step_type="summarize_chunks",
                instance_id="summarize",
                position=StepPosition(x=250, y=50),
            ),
            StepConfig(
                step_type="combine_summaries",
                instance_id="combine",
                position=StepPosition(x=500, y=50),
            ),
            StepConfig(
                step_type="extract_symptoms",
                instance_id="symptoms",
                position=StepPosition(x=250, y=210),
            ),
            StepConfig(
                step_type="diagnose",
                instance_id="diagnose",
                position=StepPosition(x=500, y=210),
            ),
        ],
        edges=[
            EdgeDef(from_step="chunk", to_step="summarize"),
            EdgeDef(from_step="summarize", to_step="combine"),
            EdgeDef(from_step="chunk", to_step="symptoms"),
            EdgeDef(from_step="symptoms", to_step="diagnose"),
        ],
    )
)

StepRegistry.register_template(
    PipelineTemplate(
        id="diagnosis_only",
        label="Diagnosis Only",
        description="Extract symptoms and generate ICD-10 diagnosis without summary",
        steps=[
            StepConfig(
                step_type="chunk_transcript",
                instance_id="chunk",
                position=StepPosition(x=0, y=100),
            ),
            StepConfig(
                step_type="extract_symptoms",
                instance_id="symptoms",
                position=StepPosition(x=250, y=100),
            ),
            StepConfig(
                step_type="diagnose",
                instance_id="diagnose",
                position=StepPosition(x=500, y=100),
            ),
        ],
        edges=[
            EdgeDef(from_step="chunk", to_step="symptoms"),
            EdgeDef(from_step="symptoms", to_step="diagnose"),
        ],
    )
)

__all__ = [
    "EdgeDef",
    "PipelineTemplate",
    "StepConfig",
    "StepPosition",
    "StepRegistry",
]
