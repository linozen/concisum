"""Tools available to agentic pipeline agents.

Each tool is a plain function registered with pydantic-ai agents.
Tools are opt-in via the CLI ``--tools`` flag.

Available tools:
    icd10       — Look up ICD-10 Chapter V diagnostic criteria by code or keyword.
    transcript  — Search the original transcript for terms or exact quotes.
"""

import json
import logging
import re
from pathlib import Path
from typing import Callable

LOG = logging.getLogger(__name__)

_SOURCES_DIR = Path(__file__).parent.parent.parent / "sources"
_ICD10_DIR = _SOURCES_DIR / "icd-10"
_DIAGNOSES_PATH = _SOURCES_DIR / "diagnoses_extracted.json"

# Lazy-loaded caches
_icd10_texts: dict[str, str] | None = None
_diagnoses: list[dict] | None = None


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_icd10_texts() -> dict[str, str]:
    """Load ICD-10 enhanced markdown files (official ICD-10-GM Chapter V)."""
    global _icd10_texts
    if _icd10_texts is not None:
        return _icd10_texts
    _icd10_texts = {}
    for f in sorted(_ICD10_DIR.glob("*_enhanced.md")):
        _icd10_texts[f.stem] = f.read_text()
    LOG.info("Loaded %d ICD-10 reference files", len(_icd10_texts))
    return _icd10_texts


def _load_diagnoses() -> list[dict]:
    """Load the structured diagnoses database (ICD-10 codes with criteria)."""
    global _diagnoses
    if _diagnoses is not None:
        return _diagnoses
    with open(_DIAGNOSES_PATH) as f:
        data = json.load(f)
    loaded: list[dict] = data.get("diagnoses", [])
    _diagnoses = loaded
    LOG.info("Loaded %d structured diagnoses", len(loaded))
    return loaded


# ---------------------------------------------------------------------------
# Tool: icd10
# ---------------------------------------------------------------------------

def lookup_icd10(query: str) -> str:
    """Search ICD-10 Chapter V (F00-F99) reference for diagnostic criteria.

    Args:
        query: An ICD-10 code (e.g. "F10"), diagnosis name
               (e.g. "Alkoholabhängigkeit"), or symptom keyword.

    Returns:
        Matching ICD-10 sections with codes, criteria, and descriptions.
        Up to 3 matches.
    """
    query_lower = query.lower()
    results: list[str] = []

    # Search structured diagnoses database first (precise matches)
    diagnoses = _load_diagnoses()
    for d in diagnoses:
        code = d.get("code", "")
        title = d.get("title", "")
        desc = d.get("description_short", "")
        keywords: list[str] = d.get("symptoms_keywords", [])
        criteria = d.get("criteria_text", "")

        searchable = f"{code} {title} {desc} {' '.join(keywords)}".lower()
        if query_lower in searchable:
            entry = f"### {code} — {title}\n{desc}"
            if criteria:
                entry += f"\n\nDiagnosekriterien:\n{criteria[:600]}"
            elif keywords:
                entry += f"\nSchlüsselsymptome: {', '.join(keywords[:5])}"
            results.append(entry)
            if len(results) >= 3:
                break

    # Fall back to full-text search in ICD-10 markdown
    if not results:
        texts = _load_icd10_texts()
        for _name, content in texts.items():
            for block in re.split(r"\n###+ ", content):
                if query_lower in block.lower():
                    snippet = block[:500].strip()
                    results.append(f"### {snippet}")
                    if len(results) >= 3:
                        break
            if len(results) >= 3:
                break

    if not results:
        return f"Keine ICD-10-Einträge gefunden für: {query}"

    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Tool: transcript (factory — returns a closure capturing the transcript text)
# ---------------------------------------------------------------------------

def make_search_transcript(transcript_text: str) -> Callable[[str], str]:
    """Create a transcript search tool bound to a specific transcript.

    Args:
        transcript_text: The full formatted transcript (one utterance per line).

    Returns:
        A tool function that searches the bound transcript.
    """
    lines = transcript_text.split("\n")

    def search_transcript(query: str) -> str:
        """Search the therapy transcript for specific terms or phrases.

        Use this to verify whether a patient or therapist actually said
        something, or to find exact quotes as evidence for a diagnosis.

        Args:
            query: The term or phrase to search for.

        Returns:
            Matching utterances with context, or a no-match message.
        """
        query_lower = query.lower()
        matches: list[str] = []

        for i, line in enumerate(lines):
            if query_lower in line.lower():
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = "\n".join(lines[start:end])
                matches.append(context)
                if len(matches) >= 5:
                    break

        if not matches:
            return f"Kein Treffer für '{query}' im Transkript gefunden."

        return (
            f"Gefundene Stellen ({len(matches)} Treffer):\n\n"
            + "\n---\n".join(matches)
        )

    return search_transcript
