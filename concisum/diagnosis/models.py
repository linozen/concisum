from pydantic import BaseModel, Field
from typing import List


class Symptom(BaseModel):
    """Represents a psychological symptom with evidence."""

    name: str = Field(description="Der Name des Symptoms")
    description: str = Field(
        description="Beschreibung, wie sich das Symptom manifestiert"
    )
    evidence: str = Field(
        description="Konkreter Beleg aus dem Kontext, der dieses Symptom unterstützt"
    )


class SymptomList(BaseModel):
    """A list of psychological symptoms."""

    symptoms: List[Symptom] = Field(
        description="Liste der aus dem kompletten Transkript extrahierten psychischen Symptome",
        default_factory=list,
    )


class Diagnosis(BaseModel):
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
    icd_10_sicherheit: float = Field(
        description="""Eine Bewertung der Sicherheit der Diagnose auf einer Skala von 0 bis 1,
        wobei 1 die höchste Sicherheit darstellt.""",
    )
