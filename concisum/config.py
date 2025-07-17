import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    provider=OpenAIProvider(
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434/v1"), api_key="ollama"
    ),
)
