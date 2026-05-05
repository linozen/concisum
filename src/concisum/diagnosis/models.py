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


class ICD10Entry(BaseModel):
    """A single ICD-10 diagnosis entry."""

    code: str = Field(
        description="Der ICD-10-Code (z.B. 'F32.1', 'F41.0'). Nur der Code, keine Bezeichnung."
    )
    title: str = Field(
        description="Die offizielle Bezeichnung der Diagnose (z.B. 'Mittelgradige depressive Episode')"
    )
    severity: str = Field(
        description="Schweregrad falls zutreffend (z.B. 'leicht', 'mittelgradig', 'schwer'), sonst leer",
        default="",
    )


class Diagnosis(BaseModel):
    hauptdiagnose: ICD10Entry = Field(
        description="Die Hauptdiagnose nach ICD-10 (Kapitel V, F00-F99)"
    )
    nebendiagnosen: List[ICD10Entry] = Field(
        description="Liste der Nebendiagnosen (Komorbiditäten). Leer wenn keine vorhanden.",
        default_factory=list,
    )
    begruendung: str = Field(
        description="""Eine diagnostische Begründung mit systematischer
        Überprüfung aller relevanten Diagnosekriterien. Belege jedes Kriterium
        mit konkreten Beispielen aus dem Gespräch.""",
    )
    sicherheit: float = Field(
        description="""Eine Bewertung der Sicherheit der Diagnose auf einer Skala von 0 bis 1,
        wobei 1 die höchste Sicherheit darstellt.""",
    )
