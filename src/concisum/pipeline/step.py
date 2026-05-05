from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    SELECT = "select"


class StepParamDef(BaseModel):
    """Schema for a configurable parameter on a step."""

    name: str
    label: str
    type: ParamType
    default: Any
    description: str = ""
    options: list[str] | None = None  # for SELECT type
    min_value: float | None = None
    max_value: float | None = None


class StepDef(BaseModel):
    """Serializable definition of a step type (for API/frontend)."""

    id: str
    label: str
    description: str
    category: str  # "summary", "diagnosis", "topic", etc.
    input_type: str
    output_type: str
    params: list[StepParamDef]


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Step(ABC):
    """Base class for all pipeline steps."""

    @classmethod
    @abstractmethod
    def definition(cls) -> StepDef:
        """Return the static definition (params, types) for this step."""
        ...

    @abstractmethod
    async def run(
        self,
        input_data: Any,
        params: dict[str, Any],
        on_progress: Callable[[str], None] | None = None,
    ) -> Any:
        """Execute this step. Returns typed output."""
        ...
