from __future__ import annotations

import logging
from typing import Any, Callable

from concisum.pipeline.registry import StepRegistry
from concisum.pipeline.step import ParamType, Step, StepDef, StepParamDef
from concisum.summary.models import ChunkSummary, FullSummary, Utterance

LOG = logging.getLogger(__name__)


@StepRegistry.register_step
class ChunkTranscriptStep(Step):
    """Split utterances into chunks and format with speaker labels."""

    @classmethod
    def definition(cls) -> StepDef:
        return StepDef(
            id="chunk_transcript",
            label="Chunk Transcript",
            description="Split utterances into chunks of configurable size",
            category="summary",
            input_type="dict",
            output_type="list[list[Utterance]]",
            params=[
                StepParamDef(
                    name="chunk_size",
                    label="Chunk size",
                    type=ParamType.INT,
                    default=50,
                    min_value=10,
                    max_value=200,
                    description="Number of utterances per chunk",
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
    ) -> list[list[Utterance]]:
        chunk_size = int(params.get("chunk_size", 50))

        # input_data is a dict with "utterances" key from pipeline input
        raw_utterances = input_data.get("utterances", [])
        utterances = [
            Utterance(text=u["text"], speaker=str(u["speaker"]))
            for u in raw_utterances
        ]

        chunks = [
            utterances[i : i + chunk_size]
            for i in range(0, len(utterances), chunk_size)
        ]

        if on_progress:
            on_progress(f"Split into {len(chunks)} chunks")

        LOG.info("Chunked %d utterances into %d chunks", len(utterances), len(chunks))
        return chunks


@StepRegistry.register_step
class SummarizeChunksStep(Step):
    """Summarize each transcript chunk using LLM."""

    @classmethod
    def definition(cls) -> StepDef:
        return StepDef(
            id="summarize_chunks",
            label="Summarize Chunks",
            description="Summarize each transcript chunk using LLM",
            category="summary",
            input_type="list[list[Utterance]]",
            output_type="list[ChunkSummary]",
            params=[
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
    ) -> list[ChunkSummary]:
        from concisum.summary.agents import chunk_summarizer

        therapist = str(params.get("therapist_speaker", "0"))
        chunks: list[list[Utterance]] = input_data
        results: list[ChunkSummary] = []

        for i, chunk in enumerate(chunks):
            if on_progress:
                on_progress(f"Summarize ({i + 1}/{len(chunks)})")

            formatted = "\n".join(
                f"{'[Therapeut*in]' if utt.speaker == therapist else '[Klient*in]'}: {utt.text}"
                for utt in chunk
            )
            prompt = (
                "Fasse den folgenden Teil eines therapeutischen Gesprächs auf Deutsch zusammen:\n\n"
                f"{formatted}"
            )
            result = await chunk_summarizer.run(prompt)
            results.append(result.output)

        return results


@StepRegistry.register_step
class CombineSummariesStep(Step):
    """Merge chunk summaries into a final summary."""

    @classmethod
    def definition(cls) -> StepDef:
        return StepDef(
            id="combine_summaries",
            label="Combine Summaries",
            description="Merge chunk summaries into a coherent final summary",
            category="summary",
            input_type="list[ChunkSummary]",
            output_type="FullSummary",
            params=[
                StepParamDef(
                    name="max_words",
                    label="Max words",
                    type=ParamType.INT,
                    default=300,
                    min_value=100,
                    max_value=1000,
                    description="Maximum word count for the final summary",
                ),
            ],
        )

    async def run(
        self,
        input_data: Any,
        params: dict[str, Any],
        on_progress: Callable[[str], None] | None = None,
    ) -> FullSummary:
        from concisum.summary.agents import full_summarizer

        max_words = int(params.get("max_words", 300))
        chunk_summaries: list[ChunkSummary] = input_data

        if on_progress:
            on_progress("Generating final summary")

        combined = "\n\n".join(
            f"Teil {i + 1}:\n{s.content}" for i, s in enumerate(chunk_summaries)
        )
        prompt = (
            "Erstelle eine zusammenhängende deutsche Gesamtzusammenfassung aus den folgenden "
            "Teilzusammenfassungen einer psychotherapeutischen Sitzung. Achte darauf, dass du "
            "konsequent genderst. Schreibe also IMMER 'die Klient*in' und 'die Therapeut*in'. "
            "Es gibt max. 1 Therapeut*in und max. 1 Klient*in. "
            f"Die Zusammenfassung MUSS weniger als {max_words} Wörter enthalten:\n\n"
            f"{combined}"
        )

        result = await full_summarizer.run(prompt)

        # Check word count and retry once if needed
        words = result.output.content.split()
        if len(words) > max_words:
            LOG.warning("Summary too long (%d words), requesting shorter version", len(words))
            prompt = (
                "Erstelle eine kürzere Zusammenfassung der psychotherapeutischen Sitzung zwischen "
                "EINER Klient*in und EINER Therapeut*in. Nutze immer "
                "gendersensible Sprache und gehe immer nur von diesen beiden "
                "Gesprächspartner:innen aus. "
                f"Die Zusammenfassung darf MAXIMAL {max_words} Wörter enthalten. "
                f"Die aktuelle Zusammenfassung hat {len(words)} Wörter "
                f"und ist zu lang:\n\n{result.output.content}"
            )
            result = await full_summarizer.run(prompt)

        return FullSummary(content=result.output.content)
