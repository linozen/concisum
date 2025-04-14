from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    "llama-3.1-sauerkrautlm-8b-instruct",
    provider=OpenAIProvider(base_url="http://127.0.0.1:1234/v1", api_key="lm_studio"),
)
