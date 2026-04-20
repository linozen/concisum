from pydantic import BaseModel, Field
from typing import List, Optional

from concisum.diagnosis.models import Diagnosis, SymptomList


class Utterance(BaseModel):
    """Represents a single utterance made by a single speaker."""

    text: str = Field(description="Der transkribierte Text der Äußerung")
    speaker: str = Field(description="Die Sprecherkennung")


class UtteranceList(BaseModel):
    """Represents a list of utterances."""

    utterances: List[Utterance] = Field(description="Eine Liste von Äußerungen")


class ChunkSummary(BaseModel):
    """Represents a summary of a chunk of transcript text."""

    content: str = Field(
        description="Der Inhalt der Zusammenfassung eines Teils des Transkripts"
    )


class FullSummary(BaseModel):
    """Represents a summary of a full transcript of a psychotherapy session."""

    content: str = Field(
        description="Der Inhalt der Zusammenfassung des kompletten Transkripts",
    )
    diagnosis: Optional[Diagnosis] = Field(
        description="Die diagnostische Einschätzung basierend auf dem Transkript",
        default=None
    )
    symptoms: Optional[SymptomList] = Field(
        description="Liste der identifizierten Symptome aus dem Transkript",
        default=None
    )
