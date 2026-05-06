import os
from contextvars import ContextVar

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

_DEFAULT_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
_DEFAULT_MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:26b")

# Per-execution model override. Set by the API entrypoints (e.g. _run_pipeline)
# before kicking off step work. Each asyncio.Task copies the parent context, so
# concurrent requests don't collide.
_active_model_name: ContextVar[str | None] = ContextVar(
    "concisum_active_model_name", default=None
)


def set_active_model(name: str | None) -> None:
    """Override the model name for the current execution context."""
    _active_model_name.set(name or None)


def get_active_model_name() -> str:
    """Return the active model name, falling back to the env default."""
    return _active_model_name.get() or _DEFAULT_MODEL_NAME


def build_model(name: str | None = None) -> OpenAIModel:
    """Build a fresh OpenAIModel pointed at the configured Ollama backend.

    Resolution order: explicit `name` > active contextvar > OLLAMA_MODEL env.
    Build a new instance per call so each request can use its own model.
    """
    return OpenAIModel(
        name or get_active_model_name(),
        provider=OpenAIProvider(base_url=_DEFAULT_BASE_URL, api_key="ollama"),
    )


# Backward-compatible module-level default. Existing imports of `model` still
# work; new code should prefer `build_model()` so per-request overrides apply.
model = build_model()
