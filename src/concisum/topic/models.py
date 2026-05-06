"""Pydantic models for topic extraction and aggregation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Topic(BaseModel):
    """A single topic identified in a transcript chunk."""

    label: str = Field(
        description=(
            "Kurze, prägnante Themenbezeichnung auf Deutsch "
            "(z.B. 'Beruflicher Stress', 'Beziehungskonflikt', "
            "'Traumaverarbeitung', 'Bewältigungsstrategien')"
        )
    )
    description: str = Field(
        description=(
            "1–2 Sätze, die beschreiben, worum es bei diesem Thema geht "
            "und wie es im Gespräch vorkommt"
        )
    )
    confidence: float = Field(
        description=(
            "Konfidenz der Themenerkennung (0.0–1.0). "
            "1.0 = Thema ist eindeutig und zentral, "
            "0.5 = Thema ist am Rande erwähnt"
        ),
        ge=0.0,
        le=1.0,
    )


class ChunkTopicList(BaseModel):
    """Topics extracted from a single transcript chunk."""

    topics: list[Topic] = Field(
        description="Liste der in diesem Abschnitt identifizierten Themen",
        default_factory=list,
    )


class TopicWithProportion(Topic):
    """A topic enriched with coverage proportion across the whole session."""

    proportion: float = Field(
        description=(
            "Anteil der Chunks, in denen dieses Thema vorkommt (0.0–1.0). "
            "Ein Thema das in 3 von 6 Chunks auftaucht hat proportion=0.5"
        ),
        ge=0.0,
        le=1.0,
    )
    chunk_indices: list[int] = Field(
        description="Indices der Chunks, in denen dieses Thema identifiziert wurde",
        default_factory=list,
    )


class TopicReport(BaseModel):
    """Aggregated topic report for an entire therapy session."""

    topics: list[TopicWithProportion] = Field(
        description=(
            "Sortierte Liste der Sitzungsthemen, absteigend nach Proportion "
            "(häufigste Themen zuerst)"
        ),
        default_factory=list,
    )
    session_summary: str = Field(
        description=(
            "2–3 Sätze, die die thematische Struktur der gesamten Sitzung "
            "zusammenfassen (auf Deutsch)"
        ),
        default="",
    )
    total_chunks: int = Field(
        description="Anzahl der Chunks, in die das Transkript geteilt wurde",
        default=0,
    )
