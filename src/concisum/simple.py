"""Non-agentic single-prompt baseline for transcript summarization and diagnosis.

Sends the entire transcript to the LLM in one prompt without chunking,
hierarchical processing, structured output validation, or retry logic.
Used as a comparison baseline against the agentic pipeline.
"""

import logging
import os

from openai import AsyncOpenAI

from concisum.summary.models import FullSummary, UtteranceList
from concisum.diagnosis.models import Diagnosis

LOG = logging.getLogger(__name__)

SUMMARY_PROMPT = """\
Du bist ein Experte für psychotherapeutische Sitzungen. Fasse das folgende \
Therapietranskript zwischen einer Therapeut*in und einer Klient*in zusammen. \
Konzentriere dich auf die wichtigsten Inhalte, Themen, therapeutischen \
Interventionen, sowie relevante Gefühle, Gedanken und Verhaltensweisen. \
Verfasse die Zusammenfassung auf Deutsch in der dritten Person, in einem \
professionellen Ton, und mit maximal 300 Wörtern.

Transkript:

{transcript}"""

DIAGNOSIS_PROMPT = """\
Du bist ein psychiatrischer Experte. Erstelle basierend auf dem folgenden \
Therapietranskript eine ICD-10-Diagnose (Kapitel V, F00-F99). Gib an:
1. Die vollständige ICD-10-Diagnose mit Code und Bezeichnung
2. Eine fachliche Begründung mit konkreten Belegen aus dem Gespräch
3. Eine Sicherheitsbewertung von 0 bis 1

Transkript:

{transcript}"""


def _format_transcript(utterance_list: UtteranceList) -> str:
    """Format utterances as plain text for a single prompt."""
    lines = []
    for utt in utterance_list.utterances:
        role = "[Therapeut*in]" if utt.speaker in ("0", "1") else "[Klient*in]"
        # Use speaker number to determine role (same heuristic as agentic pipeline)
        if utt.speaker == "1":
            role = "[Therapeut*in]"
        else:
            role = "[Klient*in]"
        lines.append(f"{role}: {utt.text}")
    return "\n".join(lines)


async def simple_summarize(
    utterance_list: UtteranceList,
    with_diagnosis: bool = False,
) -> FullSummary:
    """Summarize a transcript using a single LLM call (non-agentic baseline).

    No chunking, no structured output validation, no retry logic.
    The LLM receives the full transcript and produces free-text output.
    """
    client = AsyncOpenAI(
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    transcript = _format_transcript(utterance_list)

    # Summarization
    LOG.info("Simple mode: sending full transcript for summarization (%d utterances)", len(utterance_list.utterances))
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript)}],
    )
    summary_text = response.choices[0].message.content or ""

    diagnosis = None
    symptoms = None

    if with_diagnosis:
        LOG.info("Simple mode: sending full transcript for diagnosis")
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": DIAGNOSIS_PROMPT.format(transcript=transcript)}],
        )
        diagnosis_text = response.choices[0].message.content or ""

        # Wrap the free-text diagnosis in the Diagnosis model as best-effort
        diagnosis = Diagnosis(
            icd_10_diagnose=diagnosis_text,
            icd_10_begruendung="(Generiert im Einzelprompt-Modus ohne strukturierte Validierung)",
            icd_10_sicherheit=0.0,
        )

    return FullSummary(
        content=summary_text,
        diagnosis=diagnosis,
        symptoms=symptoms,
    )
