from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concisum.pipeline.pipeline import PipelineTemplate
    from concisum.pipeline.step import Step, StepDef


class StepRegistry:
    """Global registry for step types and pipeline templates."""

    _steps: dict[str, type[Step]] = {}
    _templates: dict[str, PipelineTemplate] = {}

    @classmethod
    def register_step(cls, step_cls: type[Step]) -> type[Step]:
        """Decorator to register a step class."""
        defn = step_cls.definition()
        cls._steps[defn.id] = step_cls
        return step_cls

    @classmethod
    def register_template(cls, template: PipelineTemplate) -> None:
        cls._templates[template.id] = template

    @classmethod
    def get_step(cls, step_type: str) -> type[Step]:
        if step_type not in cls._steps:
            raise KeyError(f"Unknown step type: {step_type}")
        return cls._steps[step_type]

    @classmethod
    def list_steps(cls) -> list[StepDef]:
        return [s.definition() for s in cls._steps.values()]

    @classmethod
    def list_templates(cls) -> list[PipelineTemplate]:
        return list(cls._templates.values())

    @classmethod
    def get_template(cls, template_id: str) -> PipelineTemplate:
        if template_id not in cls._templates:
            raise KeyError(f"Unknown pipeline template: {template_id}")
        return cls._templates[template_id]
