from dataclasses import dataclass
from typing import Optional


@dataclass
class SummarizerConfig:
    """Configuration for the transcript summarizer.

    Attributes:
        model_path: Path to the GGUF model file
        use_gpu: Whether to use GPU acceleration
        context_length: Maximum context length for the model
        temperature: Temperature for generation (0-1)
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
    """

    # Based on https://openrouter.ai/mistralai/mistral-nemo/parameters
    model_path: str = "models/mistral-nemo-instruct-2407-q6_k.gguf"
    # model_path: str = "/Users/lino/.cache/lm-studio/models/bartowski/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF/Llama-3.1-SauerkrautLM-8b-Instruct-Q4_K_S.gguf"
    # model_path: str = "/Users/lino/.cache/lm-studio/models/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/mixtral-8x7b-instruct-v0.1.Q2_K.gguf"
    use_gpu: bool = False
    context_length: int = 8192 * 4

    temperature: float = 0.9
    top_p: float = 1
    top_k: int = 0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    min_p: float = 0.0
