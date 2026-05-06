"""Topic extraction and aggregation pipeline steps."""

from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from concisum.config import build_model
from concisum.pipeline.registry import StepRegistry
from concisum.pipeline.step import ParamType, Step, StepDef, StepParamDef
from concisum.summary.models import Utterance
from concisum.topic.models import (
    ChunkTopicList,
    Topic,
    TopicReport,
    TopicWithProportion,
)

LOG = logging.getLogger(__name__)


def make_topic_extractor(
    model: OpenAIModel | None = None,
) -> Agent[None, ChunkTopicList]:
    """Build a per-call agent for per-chunk topic extraction."""
    return Agent(
        model or build_model(),
        output_type=ChunkTopicList,
        retries=3,
        instructions=(
            "Du bist ein Experte für die Analyse psychotherapeutischer Gesprächsinhalte. "
            "Deine Aufgabe ist es, die behandelten Themen in einem Abschnitt eines "
            "Therapietranskripts zu identifizieren. "
            "\n\n"
            "Richtlinien:\n"
            "- Identifiziere konkrete Themen (z.B. 'Beruflicher Stress', "
            "'Beziehungskonflikt', 'Traumaverarbeitung', 'Bewältigungsstrategien', "
            "'Selbstwertprobleme', 'Schlafstörungen').\n"
            "- Verwende kurze, aussagekräftige Themenbezeichnungen auf Deutsch.\n"
            "- Bewerte die Konfidenz (0.0–1.0): 1.0 = zentrales, ausführlich "
            "diskutiertes Thema; 0.5 = am Rande erwähnt.\n"
            "- Ignoriere reine Begrüßungs- oder Verabschiedungsfloskeln.\n"
            "- Gib max. 6 Themen pro Abschnitt an.\n"
            "- Jedes Thema MUSS eine kurze Beschreibung haben."
        ),
    )


def make_topic_aggregator(
    model: OpenAIModel | None = None,
) -> Agent[None, TopicReport]:
    """Build a per-call agent for session-level topic aggregation."""
    return Agent(
        model or build_model(),
        output_type=TopicReport,
        retries=3,
        instructions=(
            "Du bist ein Experte für die thematische Analyse psychotherapeutischer Sitzungen. "
            "Deine Aufgabe ist es, thematische Ergebnisse aus einzelnen Abschnitten eines "
            "Transkripts zu einer Gesamtübersicht zusammenzufassen. "
            "\n\n"
            "Richtlinien:\n"
            "- Führe ähnliche Themen zusammen (z.B. 'Arbeitsstress' und "
            "'Berufliche Überlastung' → ein Thema).\n"
            "- Sortiere die Themen nach Häufigkeit (häufigste zuerst).\n"
            "- Berechne für jedes Thema den Anteil der Abschnitte, in denen es vorkommt.\n"
            "- Schreibe eine kurze thematische Gesamtschau der Sitzung (2–3 Sätze).\n"
            "- Alle Ausgaben auf Deutsch."
        ),
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@StepRegistry.register_step
class ExtractTopicsStep(Step):
    """Extract topics from each transcript chunk."""

    @classmethod
    def definition(cls) -> StepDef:
        return StepDef(
            id="extract_topics",
            label="Extract Topics",
            description="Identify therapeutic topics in each transcript chunk",
            category="topic",
            input_type="list[list[Utterance]]",
            output_type="list[ChunkTopicList]",
            params=[
                StepParamDef(
                    name="max_topics_per_chunk",
                    label="Max topics per chunk",
                    type=ParamType.INT,
                    default=6,
                    min_value=1,
                    max_value=15,
                    description="Maximum number of topics to extract per chunk",
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
    ) -> list[ChunkTopicList]:
        topic_extractor = make_topic_extractor()

        max_topics = int(params.get("max_topics_per_chunk", 6))
        therapist = str(params.get("therapist_speaker", "0"))
        chunks: list[list[Utterance]] = input_data
        results: list[ChunkTopicList] = []

        for i, chunk in enumerate(chunks):
            if on_progress:
                on_progress(f"Extracting topics ({i + 1}/{len(chunks)})")

            formatted = "\n".join(
                f"{'[Therapeut*in]' if utt.speaker == therapist else '[Klient*in]'}: {utt.text}"
                for utt in chunk
            )

            prompt = (
                f"Identifiziere die behandelten Themen in folgendem Abschnitt "
                f"eines Therapiegesprächs (max. {max_topics} Themen):\n\n"
                f"{formatted}"
            )

            try:
                result = await topic_extractor.run(prompt)
                # Cap at max_topics
                topics = result.output.topics[:max_topics]
                results.append(ChunkTopicList(topics=topics))
                LOG.info(
                    "Chunk %d/%d: extracted %d topics",
                    i + 1,
                    len(chunks),
                    len(topics),
                )
            except Exception as e:
                LOG.warning("Topic extraction failed for chunk %d: %s", i + 1, e)
                results.append(ChunkTopicList(topics=[]))

        return results


@StepRegistry.register_step
class AggregateTopicsStep(Step):
    """Merge per-chunk topics into a session-level topic report."""

    @classmethod
    def definition(cls) -> StepDef:
        return StepDef(
            id="aggregate_topics",
            label="Aggregate Topics",
            description="Merge per-chunk topics into a session-level report",
            category="topic",
            input_type="list[ChunkTopicList]",
            output_type="TopicReport",
            params=[],
        )

    async def run(
        self,
        input_data: Any,
        params: dict[str, Any],
        on_progress: Callable[[str], None] | None = None,
    ) -> TopicReport:
        topic_aggregator = make_topic_aggregator()

        if on_progress:
            on_progress("Aggregating topics across chunks")

        chunk_topic_lists: list[ChunkTopicList] = input_data
        total_chunks = len(chunk_topic_lists)

        # --- Local merging: group similar topics by label similarity ---
        # This provides a deterministic baseline. The LLM then refines it.
        label_hits: dict[str, list[tuple[int, Topic]]] = {}
        for chunk_idx, ctl in enumerate(chunk_topic_lists):
            for topic in ctl.topics:
                key = topic.label.lower().strip()
                # Simple exact-match grouping (LLM handles fuzzy merging)
                if key not in label_hits:
                    label_hits[key] = []
                label_hits[key].append((chunk_idx, topic))

        # Build a structured summary for the LLM
        chunk_summaries: list[str] = []
        for i, ctl in enumerate(chunk_topic_lists):
            topic_labels = ", ".join(t.label for t in ctl.topics)
            chunk_summaries.append(f"Abschnitt {i + 1}: {topic_labels}")

        pre_merged: list[str] = []
        for label, hits in label_hits.items():
            chunk_indices = [h[0] for h in hits]
            proportion = len(set(chunk_indices)) / max(total_chunks, 1)
            avg_confidence = sum(h[1].confidence for h in hits) / len(hits)
            pre_merged.append(
                f"- \"{hits[0][1].label}\" (Proportion: {proportion:.0%}, "
                f"Konfidenz: {avg_confidence:.2f}, "
                f"in Abschnitten: {[i + 1 for i in chunk_indices]})"
            )

        prompt = (
            "Fasse die folgenden thematischen Ergebnisse aus einzelnen Abschnitten "
            "einer Therapiesitzung zu einer Gesamtübersicht zusammen.\n\n"
            f"Gesamtanzahl Abschnitte: {total_chunks}\n\n"
            "Themen pro Abschnitt:\n"
            + "\n".join(chunk_summaries)
            + "\n\nVorgruppierte Themen:\n"
            + "\n".join(pre_merged)
            + "\n\n"
            "Führe ähnliche Themen zusammen, berechne die Proportionen, "
            "sortiere nach Häufigkeit und schreibe eine kurze "
            "thematische Gesamtschau (2–3 Sätze)."
        )

        try:
            result = await topic_aggregator.run(prompt)
            report = result.output
            report.total_chunks = total_chunks
            LOG.info(
                "Aggregated into %d session topics from %d chunks",
                len(report.topics),
                total_chunks,
            )
            return report
        except Exception as e:
            LOG.error("Topic aggregation failed: %s", e)
            # Fallback: return the pre-merged data without LLM refinement
            fallback_topics: list[TopicWithProportion] = []
            for label, hits in sorted(
                label_hits.items(), key=lambda x: len(x[1]), reverse=True
            ):
                chunk_indices = list({h[0] for h in hits})
                proportion = len(chunk_indices) / max(total_chunks, 1)
                representative = hits[0][1]
                fallback_topics.append(
                    TopicWithProportion(
                        label=representative.label,
                        description=representative.description,
                        confidence=representative.confidence,
                        proportion=proportion,
                        chunk_indices=chunk_indices,
                    )
                )
            return TopicReport(
                topics=fallback_topics,
                session_summary="Thematische Übersicht (Fallback ohne LLM-Aggregation).",
                total_chunks=total_chunks,
            )
