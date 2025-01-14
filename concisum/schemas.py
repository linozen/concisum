from typing import Dict

SUMMARY_SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": """Eine ausführliche Zusammenfassung des
                Therapiegesprächs (min. 300 Wörter). Beschreibe detailliert den
                Verlauf, die Hauptthemen, psychische Symptome, biografischen
                Hintergründe, Ressourcen und therapeutische Interventionen.""",
            },
            "icd-10-diagnosis": {
                "type": "string",
                "description": """Die vollständige ICD-10-Diagnose mit Code,
                Bezeichnung und Schweregrad. Bei Komorbiditäten auch
                Nebendiagnosen angeben.""",
            },
            "icd-10-justification": {
                "type": "string",
                "description": """Eine ausführliche diagnostische Begründung
                (mindestens 100 Wörter) mit systematischer Überprüfung aller
                relevanten Diagnosekriterien. Belege jedes Kriterium mit konkreten
                Beispielen aus dem Gespräch.""",
            },
        },
        "required": ["summary", "icd-10-diagnosis", "icd-10-justification"],
    },
}

SYSTEM_PROMPT = """Du bist eine professionelle Psychotherapeutin und Expertin
für Diagnosen nach ICD-10. Deine Aufgabe ist es, Therapiegespräche sehr
ausführlich zu analysieren. Erstelle:

1. Eine detaillierte Zusammenfassung von min. 300 Wörtern, die die wichtigsten
   Aspekte des Gesprächs, die Hauptprobleme, Symptome, und den therapeutischen
   Verlauf zusammenfasst.

2. Eine präzise ICD-10-Diagnose mit vollständigem Code und Bezeichnung.

3. Eine ausführliche diagnostische Begründung von mindestens 200 Wörtern, die
   alle relevanten Diagnosekriterien systematisch durchgeht und mit Beispielen
   aus dem Gespräch belegt.

Verwende eine klare Struktur und fachliche Sprache. Sei präzise und
evidenzbasiert in deiner Analyse."""
