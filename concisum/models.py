from pydantic import BaseModel, Field


class Utterance(BaseModel):
    """Represents a single utterance in a transcript."""

    text: str = Field(description="The transcribed text of the utterance")
    speaker: str = Field(description="The speaker identifier")


class TherapySummary(BaseModel):
    zusammenfassung: str = Field(
        description="""Eine ausführliche Zusammenfassung des Therapiegesprächs.
        Beschreibe detailliert den Verlauf, die Hauptthemen, psychische
        Symptome, biografischen Hintergründe, Ressourcen und therapeutische
        Interventionen."""
    )
    icd_10_diagnose: str = Field(
        description="""Die vollständige ICD-10-Diagnose (Kapitel V, F00-F99) mit
        Code, Bezeichnung und ggf. Schweregrad. Bei Komorbiditäten auch
        Nebendiagnosen angeben.""",
    )
    icd_10_begruendung: str = Field(
        description="""Eine diagnostische Begründung mit systematischer
        Überprüfung aller relevanten Diagnosekriterien. Belege jedes Kriterium
        mit konkreten Beispielen aus dem Gespräch.""",
    )
