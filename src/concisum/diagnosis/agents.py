import logging
from typing import List, Dict, Any

from pydantic_ai import Agent

from concisum.diagnosis.models import SymptomList, Diagnosis
from concisum.summary.models import Utterance
from concisum.config import model

logger = logging.getLogger(__name__)

# Agent for extracting symptoms from transcript chunks
symptom_extractor = Agent(
    model,
    output_type=SymptomList,
    instructions=(
        "Du bist ein Experte für die Identifikation psychologischer Symptome aus Therapietranskripten. "
        "Deine Aufgabe ist es, psychische Symptome aus einem Teil eines Therapietranskripts zu identifizieren "
        "und zu extrahieren. Identifiziere alle Symptome, die auf psychische Erkrankungen hindeuten könnten. "
        "Beachte klinisch relevante Anzeichen wie Stimmungsprobleme, kognitive Veränderungen, Verhaltensmuster, "
        "physiologische Symptome und soziale Beeinträchtigungen. "
        "Für jedes Symptom gib den Namen, eine kurze Beschreibung und einen konkreten Beleg aus dem Text an. "
        "Sei präzise und halte dich an klinisch anerkannte Symptombeschreibungen."
    ),
)


def _create_diagnosis_agent(
    tools: list[str] | None = None,
    transcript_text: str = "",
) -> Agent[None, Diagnosis]:
    """Create a diagnosis agent with optional tools.

    Args:
        tools: List of tool names to enable. Supported: "icd10", "transcript".
               If None or empty, the agent operates without tools.
        transcript_text: Full formatted transcript text. Required when the
                         "transcript" tool is enabled.
    """
    enabled = set(tools or [])

    tool_hint = ""
    if "icd10" in enabled:
        tool_hint += (
            "\n\nDir steht das Werkzeug 'lookup_icd10' zur Verfügung. "
            "Nutze es, um diagnostische Kriterien nachzuschlagen, bevor du eine Diagnose stellst."
        )
    if "transcript" in enabled:
        tool_hint += (
            "\n\nDir steht das Werkzeug 'search_transcript' zur Verfügung. "
            "Nutze es, um konkrete Belege im Originaltranskript zu finden und Zitate zu verifizieren."
        )

    agent: Agent[None, Diagnosis] = Agent(
        model,
        output_type=Diagnosis,
        instructions=(
            "Du bist ein psychiatrischer Experte für die Diagnoseerstellung nach ICD-10. "
            "Deine Aufgabe ist es, basierend auf einer Liste von Symptomen eine ICD-10-Diagnose (Kapitel V, F00-F99) "
            "zu stellen. Überprüfe systematisch alle Diagnosekriterien und begründe deine Entscheidung fachlich korrekt. "
            "Gib die vollständige ICD-10-Diagnose mit Code, Bezeichnung und ggf. Schweregrad an. Bei Komorbiditäten "
            "nenne auch Nebendiagnosen. Stelle max. 3 Diagnosen."
            + tool_hint
        ),
    )

    if "icd10" in enabled:
        from concisum.tools import lookup_icd10
        agent.tool_plain(lookup_icd10)
        logger.info("Diagnosis agent: icd10 tool enabled")

    if "transcript" in enabled:
        from concisum.tools import make_search_transcript
        agent.tool_plain(make_search_transcript(transcript_text))
        logger.info("Diagnosis agent: transcript tool enabled")

    return agent


class DiagnosisOrchestrator:
    """Orchestrates the process of extracting symptoms and generating diagnoses."""

    def __init__(
        self,
        tools: list[str] | None = None,
        transcript_text: str = "",
    ):
        """Initialize the DiagnosisOrchestrator.

        Args:
            tools: List of tool names to enable for the diagnosis agent.
                   Supported: "icd10", "transcript". Default: no tools.
            transcript_text: Full formatted transcript. Required when
                             "transcript" is in *tools*.
        """
        self.tools = tools
        self.diagnosis_agent = _create_diagnosis_agent(
            tools=tools, transcript_text=transcript_text
        )

    async def extract_symptoms_from_chunk(self, chunk: List[Utterance]) -> SymptomList:
        """Extract symptoms from a chunk of transcript utterances.

        Args:
            chunk: List of utterances in the chunk

        Returns:
            List of extracted symptoms
        """
        # Format the chunk for the prompt
        formatted_chunk = "\n".join(
            [
                f"{'[Therapeut]' if utt.speaker == '1' else '[Patient]'}: {utt.text}"
                for utt in chunk
            ]
        )

        prompt = (
            "Identifiziere psychische Symptome aus folgendem Teil eines Therapiegesprächs:\n\n"
            f"{formatted_chunk}"
        )

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                result = await symptom_extractor.run(prompt)
                logger.info(
                    f"Extracted {len(result.output.symptoms)} symptoms from chunk"
                )
                return result.output
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Symptom extraction failed ({retry_count}/{max_retries}): {e}"
                )
                if retry_count >= max_retries:
                    logger.error(
                        f"Failed to extract symptoms after {max_retries} attempts: {e}"
                    )
                    return SymptomList(symptoms=[])
                # Vary the prompt to encourage different output
                prompt = (
                    "Bitte identifiziere alle psychischen Symptome aus folgendem Teil eines Therapiegesprächs. "
                    "Achte auf klare und präzise Symptombeschreibungen:\n\n"
                    f"{formatted_chunk}"
                )

        # This line ensures a return value on all code paths
        return SymptomList(symptoms=[])

    async def extract_symptoms_from_chunks(
        self, chunks: List[List[Utterance]]
    ) -> SymptomList:
        """Extract symptoms from multiple chunks and combine them.

        Args:
            chunks: List of utterance chunks

        Returns:
            Combined list of extracted symptoms (max 5 valid symptoms)
        """
        all_symptoms = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting symptoms from chunk {i+1}/{len(chunks)}")
            symptom_list = await self.extract_symptoms_from_chunk(chunk)
            all_symptoms.extend(symptom_list.symptoms)

        # Filter symptoms to only include those with all required attributes
        valid_symptoms = []
        for symptom in all_symptoms:
            try:
                if (
                    hasattr(symptom, "name")
                    and symptom.name
                    and hasattr(symptom, "description")
                    and symptom.description
                    and hasattr(symptom, "evidence")
                    and symptom.evidence
                ):
                    valid_symptoms.append(symptom)
            except Exception as e:
                logger.warning(f"Skipping invalid symptom: {str(e)}")
                continue

        # If no valid symptoms were found, create at least one default symptom to avoid downstream errors
        if not valid_symptoms and all_symptoms:
            try:
                from concisum.diagnosis.models import Symptom

                # Try to create a valid symptom from whatever data we have
                for s in all_symptoms:
                    try:
                        name = (
                            getattr(s, "name", "Unbekanntes Symptom")
                            if hasattr(s, "name")
                            else "Unbekanntes Symptom"
                        )
                        description = (
                            getattr(
                                s,
                                "description",
                                "Keine detaillierte Beschreibung verfügbar",
                            )
                            if hasattr(s, "description")
                            else "Keine detaillierte Beschreibung verfügbar"
                        )
                        evidence = (
                            getattr(s, "evidence", "Keine konkreten Belege verfügbar")
                            if hasattr(s, "evidence")
                            else "Keine konkreten Belege verfügbar"
                        )

                        valid_symptoms.append(
                            Symptom(
                                name=name, description=description, evidence=evidence
                            )
                        )
                        # Just need one valid symptom as a fallback
                        break
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Could not create default symptom: {str(e)}")

        # Remove duplicate symptoms (same name)
        unique_symptoms = {}
        for symptom in valid_symptoms:
            try:
                if symptom.name not in unique_symptoms:
                    unique_symptoms[symptom.name] = symptom
                    # Stop adding if we already have 5 symptoms
                    if len(unique_symptoms) >= 5:
                        break
            except Exception as e:
                logger.warning(f"Error while deduplicating symptom: {str(e)}")
                continue

        logger.info(f"Found {len(unique_symptoms)} unique valid symptoms (max 5)")
        logger.info("Extracted symptoms: \n")
        for symptom in unique_symptoms.values():
            try:
                logger.info(f"- {symptom.name}: {symptom.description}")
            except Exception as e:
                logger.warning(f"Could not log symptom info: {str(e)}")

        return SymptomList(symptoms=list(unique_symptoms.values()))

    async def generate_diagnosis(self, symptoms: SymptomList) -> Diagnosis:
        """Generate a diagnosis based on extracted symptoms.

        Args:
            symptoms: List of extracted symptoms

        Returns:
            Generated diagnosis
        """
        symptom_text_parts = []
        for s in symptoms.symptoms:
            if hasattr(s, "description") and s.description:
                symptom_text_parts.append(f"- {s.name}: {s.description}")
            else:
                symptom_text_parts.append(f"- {s.name}")

        symptom_text = "\n".join(symptom_text_parts)

        prompt = (
            "Erstelle eine psychiatrische Diagnose nach ICD-10 basierend auf folgenden Symptomen:\n\n"
            f"{symptom_text}"
        )

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                result = await self.diagnosis_agent.run(prompt)
                logger.info(f"Generated diagnosis: {result.output.icd_10_diagnose}")
                return result.output
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Diagnosis generation failed (attempt {retry_count}/{max_retries}): {str(e)}"
                )

                if retry_count >= max_retries:
                    logger.error(
                        f"Failed to generate diagnosis after {max_retries} attempts: {e}"
                    )
                    # Return a default diagnosis as fallback
                    from concisum.diagnosis.models import Diagnosis

                    return Diagnosis(
                        icd_10_diagnose="F99 - Psychische Störung ohne nähere Angabe",
                        icd_10_begruendung="Eine genauere Diagnose konnte aufgrund unzureichender oder nicht eindeutiger Symptome nicht gestellt werden. Die vorliegenden Symptome deuten auf eine psychische Störung hin, aber die genaue Natur der Störung konnte nicht bestimmt werden.",
                        icd_10_sicherheit=0.5,
                    )

                # Add slight variation to the prompt on retry
                if retry_count == 1:
                    prompt = (
                        "Erstelle eine präzise psychiatrische Diagnose nach ICD-10 auf Basis dieser Symptome:\n\n"
                        f"{symptom_text}"
                    )
                elif retry_count == 2:
                    # Simplify the task for the last attempt
                    prompt = (
                        "Als psychiatrischer Experte stelle eine kurze ICD-10-Diagnose (F-Kategorie) für diese Symptome:\n\n"
                        f"{symptom_text}\n\n"
                        "Gib einen ICD-10-Code mit Bezeichnung, eine kurze Begründung und eine Sicherheit zwischen 0 und 1 an."
                    )

        # This line ensures a return value on all code paths
        # Create a default diagnosis as fallback
        from concisum.diagnosis.models import Diagnosis

        return Diagnosis(
            icd_10_diagnose="F99 - Psychische Störung ohne nähere Angabe",
            icd_10_begruendung="Eine genauere Diagnose konnte aufgrund technischer Probleme nicht erstellt werden.",
            icd_10_sicherheit=0.3,
        )

    async def process_transcript(self, chunks: List[List[Utterance]]) -> Dict[str, Any]:
        """Process a transcript to extract symptoms and generate a diagnosis.

        Args:
            chunks: List of utterance chunks from the transcript

        Returns:
            Dictionary with symptoms and diagnosis
        """
        # Extract symptoms from all chunks
        symptoms = await self.extract_symptoms_from_chunks(chunks)

        # Generate diagnosis based on symptoms
        diagnosis = await self.generate_diagnosis(symptoms)

        return {"symptoms": symptoms, "diagnosis": diagnosis}
