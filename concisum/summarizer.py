import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from concisum.summary.config import SummarizerConfig
from concisum.summary.schemas import SUMMARY_SCHEMA, SYSTEM_PROMPT
from concisum.summary.utils import prepare_transcript_text, create_prompt

LOG = logging.getLogger(__name__)


@dataclass
class Utterance:
    utterance_id: str
    speaker: Optional[str]
    text: str
    words: List[str] = field(default_factory=list)


class TranscriptSummarizer:
    """Summarizes therapy transcripts and generates ICD-10 diagnoses using LLM."""

    def __init__(self, config: SummarizerConfig):
        """Initialize the summarizer.

        Args:
            config: Configuration object for the summarizer
        """
        self.config = config
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the LLM model with configured parameters."""
        try:
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_length,
                n_gpu_layers=-1 if self.config.use_gpu else 0,
                verbose=False,
            )
            LOG.info(f"Model at {self.config.model_path} was initialized successfully")
        except Exception as e:
            LOG.error(f"Failed to initialize model: {str(e)}")
            raise

    def summarize(self, utterances: List[Utterance]) -> Dict[str, str]:
        """Generate a summary with ICD-10 diagnosis for the transcript.

        Args:
            utterances: List of transcript utterances

        Returns:
            Dictionary containing summary and ICD-10 diagnosis

        Raises:
            ValueError: If utterances list is empty
        """
        if not utterances:
            raise ValueError("No utterances provided for summarization")

        try:
            # Prepare input text
            transcript_text = prepare_transcript_text(utterances)
            messages = create_prompt(transcript_text, SYSTEM_PROMPT)
            LOG.debug(f"Prompt: {messages}")

            # Count tokens in the input
            prompt_tokens = len(self.model.tokenize(str(messages).encode()))
            LOG.info(f"Input length: {prompt_tokens} tokens")

            if prompt_tokens > self.config.context_length:
                LOG.warning(
                    f"Input length ({prompt_tokens} tokens) exceeds model's context length "
                    f"({self.config.context_length} tokens). This may affect performance."
                )

            # Generate response
            response = self.model.create_chat_completion(
                messages=messages,
                # max_tokens=,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                response_format=SUMMARY_SCHEMA,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                min_p=self.config.min_p,
            )

            LOG.debug(f"Response: {response}")
            # Log completion tokens
            completion_tokens = response["usage"]["completion_tokens"]
            total_tokens = response["usage"]["total_tokens"]
            LOG.info(f"Completion length: {completion_tokens} tokens")
            LOG.info(f"Total tokens used: {total_tokens}")

            # Parse the JSON string from the response
            content = response["choices"][0]["message"]["content"]
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                LOG.error(f"Failed to parse model response as JSON: {str(e)}")
                LOG.debug(f"Raw content: {content}")
                return {
                    "summary": "Fehler bei der Zusammenfassung (JSON Parse Error)",
                    "icd-10-diagnosis": "Keine Diagnose verfügbar",
                    "icd-10-justification": "Keine Begründung verfügbar",
                }

            LOG.info("Successfully generated summary and diagnosis")
            return result

        except Exception as e:
            LOG.error(f"Error during summarization: {str(e)}")
            return {
                "summary": "Fehler bei der Zusammenfassung",
                "icd-10-diagnosis": "Keine Diagnose verfügbar",
                "icd-10-justification": "Keine Begründung verfügbar",
            }
