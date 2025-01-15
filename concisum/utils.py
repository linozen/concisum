import logging
import json
from pathlib import Path
from re import I
from typing import List, Dict, Optional

from .models import Utterance

LOG = logging.getLogger(__name__)


def load_json_transcript(file_path: str | Path) -> List[Utterance]:
    """Load and parse a JSON transcript file.

    Args:
        file_path: Path to JSON transcript file

    Returns:
        List of Utterance objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file isn't valid JSON
        KeyError: If the JSON structure isn't as expected
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse JSON from {file_path}: {str(e)}", e.doc, e.pos
            )

    if "utterances" not in data:
        raise KeyError(f"No 'utterances' key found in {file_path}")

    utterances = []
    for utt in data["utterances"]:
        try:
            # Get the most common speaker ID from hyp_spk
            speaker_ids = (utt.get("hyp_spk", "") or utt.get("ref_spk", "")).split()
            speaker = (
                max(set(speaker_ids), key=speaker_ids.count)
                if speaker_ids
                else "UNKNOWN"
            )

            utterance = Utterance(
                text=(utt.get("hyp_text", "") or utt.get("ref_text", "")),
                speaker=speaker,
            )
            utterances.append(utterance)
        except Exception as e:
            LOG.warning(f"Skipping malformed utterance: {utt}\nError: {str(e)}")
            continue

    return utterances


def prepare_transcript_text(utterances: List[Utterance]) -> str:
    """Convert utterances to a formatted text representation.

    Args:
        utterances: List of transcript utterances

    Returns:
        Formatted string representation of the transcript
    """
    transcript_lines = []
    for utterance in utterances:
        speaker = f"Sprecher {utterance.speaker}" if utterance.speaker else "Unbekannt"
        transcript_lines.append(f"{speaker}: {utterance.text}")

    return "\n".join(transcript_lines)


def save_as_markdown(summary: dict, output_path: Path) -> None:
    """Save the summary and diagnosis as a formatted markdown file.

    Args:
        summary: Dictionary containing summary and diagnosis
        output_path: Path to save the markdown file
    """
    markdown_content = f"""# Therapiegespräch Zusammenfassung

## Zusammenfassung
{summary['summary']}

## ICD-10 Diagnose
{summary['icd-10-diagnosis']}

## Begründung für die Verdachtsdiagnose
{summary['icd-10-justification']}

---
*Generiert mit Concisum*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)


def format_conversation_for_display(
    transcript_text: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Format the conversation elements for display."""
    prompt = user_prompt + transcript_text

    return f"""[bold cyan]System Prompt:[/bold cyan]
{system_prompt}

[bold cyan]User Prompt:[/bold cyan]
{prompt}"""
