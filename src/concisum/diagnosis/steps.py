from __future__ import annotations

import logging
from typing import Any, Callable

from concisum.diagnosis.models import Diagnosis, ICD10Entry, SymptomList
from concisum.pipeline.registry import StepRegistry
from concisum.pipeline.step import ParamType, Step, StepDef, StepParamDef
from concisum.summary.models import Utterance

LOG = logging.getLogger(__name__)


@StepRegistry.register_step
class ExtractSymptomsStep(Step):
    """Extract symptoms from transcript chunks."""

    @classmethod
    def definition(cls) -> StepDef:
        return StepDef(
            id="extract_symptoms",
            label="Extract Symptoms",
            description="Extract psychological symptoms from transcript chunks",
            category="diagnosis",
            input_type="list[list[Utterance]]",
            output_type="SymptomList",
            params=[
                StepParamDef(
                    name="max_symptoms",
                    label="Max symptoms",
                    type=ParamType.INT,
                    default=5,
                    min_value=1,
                    max_value=20,
                    description="Maximum number of unique symptoms to extract",
                ),
                StepParamDef(
                    name="therapist_speaker",
                    label="Therapist speaker ID",
                    type=ParamType.STRING,
                    default="0",
                    description="Speaker number for the therapist",
                ),
            ],
        )

    async def run(
        self,
        input_data: Any,
        params: dict[str, Any],
        on_progress: Callable[[str], None] | None = None,
    ) -> SymptomList:
        from concisum.diagnosis.agents import make_symptom_extractor

        symptom_extractor = make_symptom_extractor()

        max_symptoms = int(params.get("max_symptoms", 5))
        therapist = str(params.get("therapist_speaker", "0"))
        chunks: list[list[Utterance]] = input_data
        all_symptoms = []

        for i, chunk in enumerate(chunks):
            if on_progress:
                on_progress(f"Extracting symptoms ({i + 1}/{len(chunks)})")

            formatted = "\n".join(
                f"{'[Therapeut]' if utt.speaker == therapist else '[Patient]'}: {utt.text}"
                for utt in chunk
            )
            prompt = (
                "Identifiziere psychische Symptome aus folgendem Teil eines Therapiegesprächs:\n\n"
                f"{formatted}"
            )

            try:
                result = await symptom_extractor.run(prompt)
                all_symptoms.extend(result.output.symptoms)
            except Exception as e:
                LOG.warning("Symptom extraction failed for chunk %d: %s", i + 1, e)

        # Deduplicate by name, cap at max
        unique: dict[str, Any] = {}
        for symptom in all_symptoms:
            if (
                hasattr(symptom, "name")
                and symptom.name
                and hasattr(symptom, "description")
                and symptom.description
                and symptom.name not in unique
            ):
                unique[symptom.name] = symptom
                if len(unique) >= max_symptoms:
                    break

        LOG.info("Extracted %d unique symptoms (max %d)", len(unique), max_symptoms)
        return SymptomList(symptoms=list(unique.values()))


@StepRegistry.register_step
class DiagnoseStep(Step):
    """Generate ICD-10 diagnosis from symptoms."""

    @classmethod
    def definition(cls) -> StepDef:
        return StepDef(
            id="diagnose",
            label="Diagnose",
            description="Generate ICD-10 diagnosis from extracted symptoms",
            category="diagnosis",
            input_type="SymptomList",
            output_type="Diagnosis",
            params=[
                StepParamDef(
                    name="tools",
                    label="Diagnosis tools",
                    type=ParamType.STRING,
                    default="",
                    description="Comma-separated tool names: icd10, transcript",
                ),
            ],
        )

    async def run(
        self,
        input_data: Any,
        params: dict[str, Any],
        on_progress: Callable[[str], None] | None = None,
    ) -> Diagnosis:
        from concisum.diagnosis.agents import _create_diagnosis_agent

        tools_str = str(params.get("tools", ""))
        tools = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else None

        symptoms: SymptomList = input_data

        if on_progress:
            on_progress("Generating diagnosis")

        # Build transcript text for the transcript tool if needed
        # (the transcript is passed via pipeline context if available)
        transcript_text = str(params.get("_transcript_text", ""))

        agent = _create_diagnosis_agent(tools=tools, transcript_text=transcript_text)

        symptom_parts = []
        for s in symptoms.symptoms:
            desc = getattr(s, "description", "")
            if desc:
                symptom_parts.append(f"- {s.name}: {desc}")
            else:
                symptom_parts.append(f"- {s.name}")

        prompt = (
            "Erstelle eine psychiatrische Diagnose nach ICD-10 basierend auf folgenden Symptomen:\n\n"
            f"{chr(10).join(symptom_parts)}\n\n"
            "Antworte mit: hauptdiagnose (code, title, severity), nebendiagnosen (Liste), begruendung, sicherheit (0-1)."
        )

        fallback = Diagnosis(
            hauptdiagnose=ICD10Entry(
                code="F99",
                title="Psychische Störung ohne nähere Angabe",
                severity="",
            ),
            nebendiagnosen=[],
            begruendung="Eine genauere Diagnose konnte aufgrund technischer Probleme nicht erstellt werden.",
            sicherheit=0.3,
        )

        for attempt in range(3):
            try:
                result = await agent.run(prompt)
                LOG.info(
                    "Generated diagnosis: %s %s",
                    result.output.hauptdiagnose.code,
                    result.output.hauptdiagnose.title,
                )
                return result.output
            except Exception as e:
                LOG.warning("Diagnosis attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    LOG.error("All diagnosis attempts failed")
                    return fallback

        return fallback
