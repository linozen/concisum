"""Advanced RAG tools optimized for constrained models."""

import logging
from typing import List
from dataclasses import dataclass

from pydantic_ai import RunContext

from concisum.diagnosis.vectorstore import ICD10VectorStore
from concisum.diagnosis.retrieval import HybridRetriever, RetrievalConfig
from concisum.diagnosis.models import Symptom

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisContextAdvanced:
    """Enhanced context with hybrid retriever."""

    vectorstore: ICD10VectorStore
    symptoms: List[Symptom]
    retriever: HybridRetriever


async def search_candidate_diagnoses(
    ctx: RunContext[DiagnosisContextAdvanced], symptom_summary: str
) -> str:
    """
    Stage 1: Broad search for candidate diagnoses.

    Returns: Short list of candidate codes with titles only.
    This minimizes context for the initial retrieval step.
    """
    logger.info(f"Stage 1: Searching candidates for: {symptom_summary}")

    # Use hybrid retriever for better results
    candidates = await ctx.deps.retriever.retrieve_for_symptoms(ctx.deps.symptoms)

    if not candidates:
        return "Keine passenden Kandidatendiagnosen gefunden."

    # Return MINIMAL context: just codes and titles
    result = "Kandidatendiagnosen:\n"
    for i, entry in enumerate(candidates, 1):
        result += f"{i}. {entry.code} - {entry.title}\n"

    return result


async def validate_diagnosis_criteria(
    ctx: RunContext[DiagnosisContextAdvanced], icd_code: str
) -> str:
    """
    Stage 2: Detailed criteria for a specific diagnosis.

    Returns: Full criteria for validation. Only called when agent
    wants to validate a specific diagnosis.
    """
    logger.info(f"Stage 2: Validating criteria for {icd_code}")

    entry = await ctx.deps.vectorstore.get_exact_criteria(icd_code)

    if not entry:
        return f"Code {icd_code} nicht gefunden."

    # Format for SYSTEMATIC validation
    result = f"**{entry.code} - {entry.title}**\n\n"
    result += "DIAGNOSEKRITERIEN (müssen systematisch geprüft werden):\n\n"

    for i, criterion in enumerate(entry.criteria, 1):
        result += f"[Kriterium {i}]\n{criterion}\n\n"
        result += f"→ Erfüllt durch vorliegende Symptome? [ ]\n\n"

    # Add differential hints
    if entry.differential:
        result += "\nDIFFERENTIALDIAGNOSTISCH ABGRENZEN VON:\n"
        result += ", ".join(entry.differential)

    return result


async def compare_differential_diagnoses(
    ctx: RunContext[DiagnosisContextAdvanced], codes: List[str]
) -> str:
    """
    Stage 3: Side-by-side comparison of multiple diagnoses.

    Useful for differential diagnosis reasoning.
    """
    logger.info(f"Stage 3: Comparing {len(codes)} diagnoses")

    if len(codes) > 3:
        return "Bitte max. 3 Diagnosen zum Vergleich angeben."

    entries = []
    for code in codes:
        entry = await ctx.deps.vectorstore.get_exact_criteria(code)
        if entry:
            entries.append(entry)

    if not entries:
        return "Keine der angegebenen Diagnosen gefunden."

    # Format as comparison table
    result = "DIFFERENTIALDIAGNOSTISCHER VERGLEICH\n\n"

    for entry in entries:
        result += f"## {entry.code} - {entry.title}\n"
        result += f"**Hauptkriterien:** {entry.criteria[0] if entry.criteria else 'N/A'}\n"
        result += f"**Besonderheiten:** {entry.description[:100]}...\n\n"

    return result


# Example: Create retrieval-optimized context
def create_advanced_context(
    vectorstore: ICD10VectorStore, symptoms: List[Symptom]
) -> DiagnosisContextAdvanced:
    """
    Create context with optimized retrieval configuration.

    Use this instead of basic DiagnosisContext for better results
    with constrained models.
    """
    # Configure retrieval for small models
    config = RetrievalConfig(
        semantic_top_k=5,  # Start with 5 candidates
        use_keyword_boost=True,  # Boost exact symptom matches
        use_reranking=True,  # Re-rank by criteria coverage
        rerank_top_k=3,  # Final: 3 best candidates
    )

    retriever = HybridRetriever(vectorstore, config)

    return DiagnosisContextAdvanced(
        vectorstore=vectorstore, symptoms=symptoms, retriever=retriever
    )
