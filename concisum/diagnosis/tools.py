"""Pydantic-AI tools for RAG-enhanced diagnosis generation."""

import logging
from typing import List
from dataclasses import dataclass

from pydantic_ai import RunContext

from concisum.diagnosis.vectorstore import ICD10VectorStore, ICD10Entry
from concisum.diagnosis.models import Symptom

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisContext:
    """Context for diagnosis generation with RAG."""

    vectorstore: ICD10VectorStore
    symptoms: List[Symptom]


async def search_icd10_by_symptoms(
    ctx: RunContext[DiagnosisContext], symptom_query: str
) -> str:
    """
    Search for relevant ICD-10 diagnoses based on symptom descriptions.

    This tool performs semantic search over the ICD-10 vector database to find
    diagnoses that match the provided symptoms.

    Args:
        ctx: Runtime context containing vectorstore and symptoms
        symptom_query: Description of symptoms to search for

    Returns:
        Formatted string containing top matching diagnoses with codes, titles, and criteria
    """
    logger.info(f"Searching ICD-10 database for: {symptom_query}")

    # Perform semantic search
    results = await ctx.deps.vectorstore.search_by_symptoms(
        [symptom_query], top_k=3
    )

    if not results:
        return "Keine passenden ICD-10 Diagnosen gefunden. Bitte nutze dein klinisches Fachwissen."

    # Format results for the agent
    formatted_results = []
    for entry in results:
        result_text = f"**{entry.code} - {entry.title}**\n"
        result_text += f"{entry.description}\n\n"
        result_text += "Diagnostische Kriterien:\n"
        for i, criterion in enumerate(entry.criteria, 1):
            result_text += f"{i}. {criterion}\n"

        if entry.differential:
            result_text += "\nDifferentialdiagnostisch abzugrenzen von:\n"
            result_text += ", ".join(entry.differential)

        formatted_results.append(result_text)

    return "\n\n---\n\n".join(formatted_results)


async def get_icd10_criteria(
    ctx: RunContext[DiagnosisContext], icd_code: str
) -> str:
    """
    Retrieve exact diagnostic criteria for a specific ICD-10 code.

    This tool performs exact lookup of diagnostic criteria to validate
    whether a specific diagnosis applies.

    Args:
        ctx: Runtime context containing vectorstore
        icd_code: ICD-10 code to look up (e.g., 'F32.1')

    Returns:
        Formatted string with diagnostic criteria, or error message if not found
    """
    logger.info(f"Looking up exact criteria for ICD-10 code: {icd_code}")

    entry = await ctx.deps.vectorstore.get_exact_criteria(icd_code)

    if not entry:
        return f"ICD-10 Code {icd_code} wurde nicht in der Datenbank gefunden."

    # Format criteria for validation
    result_text = f"**{entry.code} - {entry.title}**\n\n"
    result_text += f"{entry.description}\n\n"
    result_text += "Diagnostische Kriterien (alle müssen erfüllt sein):\n"

    for i, criterion in enumerate(entry.criteria, 1):
        result_text += f"{i}. {criterion}\n"

    if entry.differential:
        result_text += "\nDifferentialdiagnostisch beachten:\n"
        result_text += ", ".join(entry.differential)

    return result_text


async def list_available_diagnoses(ctx: RunContext[DiagnosisContext]) -> str:
    """
    List all available ICD-10 codes in the database.

    Useful for understanding what diagnoses are available.

    Args:
        ctx: Runtime context containing vectorstore

    Returns:
        Comma-separated list of ICD-10 codes
    """
    codes = ctx.deps.vectorstore.get_all_codes()
    logger.info(f"Vector store contains {len(codes)} ICD-10 codes")

    if not codes:
        return "Keine ICD-10 Codes in der Datenbank verfügbar."

    # Sort codes for better readability
    codes_sorted = sorted(codes)
    return f"Verfügbare ICD-10 Codes ({len(codes_sorted)}): " + ", ".join(
        codes_sorted
    )
