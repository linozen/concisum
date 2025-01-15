from typing import Dict, List
import logging

from pydantic import BaseModel, Field
from openai import OpenAI

from .utils import prepare_transcript_text
from .models import TherapySummary, Utterance

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist eine professionelle deutsche Psychotherapeutin und
Expertin für psychiche Diagnosen nach ICD-10 (Kapitel V, F00-F99). Deine Aufgabe
ist es, Therapiegespräche ausführlich zu analysieren. Erstelle:

1. Eine detaillierte Zusammenfassung, die die wichtigsten Aspekte des Gesprächs,
   die Hauptprobleme, Symptome, und den therapeutischen Verlauf zusammenfasst.

2. Eine präzise ICD-10-Diagnose aus Kapitel V (F00-F99) mit vollständigem Code,
   Bezeichnung und ggf. Schweregrad.

3. Eine ausführliche diagnostische Begründung, die alle relevanten
   Diagnosekriterien systematisch durchgeht und mit Beispielen aus dem Gespräch
   belegt.

Verwende eine klare Struktur und fachliche Sprache. Sei präzise und
evidenzbasiert in deiner Analyse. Wenn die Informationen aus dem Transkript
nicht ausreichen, um eine erste Verdachtsdiagnose zu stellen, dann darf auch
KEINE Diagnose gestellt werden. Antworte in diesem Fall mit 'F00.0 - Diagnose
nicht bestimmbar'.

Antworte IMMER in deutscher Sprache."""

USER_PROMPT = """Bitte fasse das untenstehende Therapiegespräch zusammen und erstelle
und begründe auf dieser Basis eine ICD-10 Diagnose aus Kapitel V (F00-F99).\n\n"""


class TranscriptSummarizer:
    """Summarizes therapy transcripts and generates ICD-10 diagnoses using LLM."""

    def __init__(self, temperature: float, model: str):
        """Initialize the summarizer."""
        # Initialize client
        self.client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1/")
        self.model = model
        self.temperature = temperature

    def summarize(self, utterances: List[Utterance]) -> Dict[str, str]:
        """Generate a summary with ICD-10 diagnosis for the transcript."""
        if not utterances:
            raise ValueError("No utterances provided for summarization")

        try:
            # Prepare input text
            transcript_text = prepare_transcript_text(utterances)

            # Generate response
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT + transcript_text},
                ],
                response_format=TherapySummary,
                temperature=self.temperature,
            )

            result = completion.choices[0].message.parsed

            # Convert to dictionary with expected keys
            return {
                "summary": result.zusammenfassung,
                "icd-10-diagnosis": result.icd_10_diagnose,
                "icd-10-justification": result.icd_10_begruendung,
            }

        except Exception as e:
            LOG.error(f"Error during summarization: {str(e)}")
            return {
                "summary": "Fehler bei der Zusammenfassung",
                "icd-10-diagnosis": "Keine Diagnose verfügbar",
                "icd-10-justification": "Keine Begründung verfügbar",
            }
