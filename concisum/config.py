from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    "qwen3:8b",
    provider=OpenAIProvider(
        base_url="http://klips80.osi.internal:11434/v1", api_key="ollama"
    ),
)
