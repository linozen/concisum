"""Analyze concisum agentic vs non-agentic experiment results.

Parses all outputs from an experiment run into a structured DataFrame.
Extracts timing, word counts, diagnostic accuracy, tool usage, and
quality metrics.

Usage:
    uv run python scripts/analyze_experiment.py results/anon_2026-04-20/
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Ground truth ICD-10 codes for each session
GROUND_TRUTH_ICD10 = {
    "108_01": "F10.2",  # Alcohol dependence (Alkoholabhängigkeit role script)
    "108_02": "F45.0",  # Somatization disorder (Somatisierungsstörung role script)
    "129_01": "F43.1",  # PTSD (PTBS role script)
    "129_02": "F10.2",  # Alcohol dependence (Alkoholabhängigkeit role script)
}

# Gender-neutral terms expected in agentic output
GENDER_NEUTRAL_TERMS = ["Klient*in", "Therapeut*in", "Patient*in"]
GENDERED_TERMS = [
    "Klient ", "Klientin ", "Patient ", "Patientin ",
    "Therapeut ", "Therapeutin ",
]

# Character names from role scripts that should be anonymized
KNOWN_CHARACTER_NAMES = [
    "Krämer", "Pankow", "Tenner", "Seefeld", "Sabine",
    "Anerose", "Anne-Rose", "Meier", "Viktoria", "Herrcher",
    "Laura", "Kralinge", "Kallinger", "Renate", "Weiss", "Schreiner",
]


def parse_time_file(path: Path) -> dict:
    """Parse a .time file into a dict."""
    data = {}
    for line in path.read_text().splitlines():
        if ": " in line:
            key, val = line.split(": ", 1)
            data[key] = val
    return data


def parse_md_file(path: Path) -> dict:
    """Extract structured metrics from a .md output file."""
    if not path.exists():
        return {"exists": False}

    text = path.read_text(encoding="utf-8")
    words = len(text.split())

    result = {
        "exists": True,
        "word_count": words,
        "has_diagnosis": "## Diagnose" in text or "### ICD-10 Diagnose" in text,
        "has_symptoms": "## Identifizierte Symptome" in text,
    }

    # Extract ICD-10 code — search the diagnosis section for F-codes
    diag_section = re.search(
        r"## Diagnose\s*\n(.*?)(?=\n## [^#]|\Z)", text, re.DOTALL
    )
    diag_text = diag_section.group(1) if diag_section else text
    # Find all F-codes in the diagnosis section
    codes = re.findall(r"F\d+(?:\.\d+)?", diag_text)
    # Deduplicate while preserving order
    seen = set()
    unique_codes = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique_codes.append(c)
    result["icd10_codes"] = unique_codes
    result["icd10_primary"] = unique_codes[0] if unique_codes else None

    # Extract confidence score
    conf_match = re.search(
        r"(?:Diagnosesicherheit|Sicherheitsbewertung)\s*\n+([\d.]+)",
        text,
    )
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
        except ValueError:
            result["confidence"] = None
    else:
        result["confidence"] = None

    # Count symptoms (### headers under ## Identifizierte Symptome)
    symptoms_section = re.search(
        r"## Identifizierte Symptome\s*\n(.*)",
        text,
        re.DOTALL,
    )
    if symptoms_section:
        symptom_headers = re.findall(r"^### .+", symptoms_section.group(1), re.MULTILINE)
        result["num_symptoms"] = len(symptom_headers)
    else:
        result["num_symptoms"] = 0

    # Gender-neutral language compliance
    neutral_count = sum(text.count(t) for t in GENDER_NEUTRAL_TERMS)
    gendered_count = sum(text.count(t) for t in GENDERED_TERMS)
    result["gender_neutral_count"] = neutral_count
    result["gendered_count"] = gendered_count
    result["gender_neutral_ratio"] = (
        neutral_count / (neutral_count + gendered_count)
        if (neutral_count + gendered_count) > 0
        else None
    )

    # Name leakage check — both pseudonym tags and actual names
    person_tags = re.findall(r"\[PERSON_\d+\]", text)
    result["person_tag_count"] = len(person_tags)
    # Check for known role-script character names that should be anonymized
    leaked_names = []
    for name in KNOWN_CHARACTER_NAMES:
        if name in text:
            leaked_names.append(name)
    result["leaked_names"] = leaked_names
    result["name_leak_count"] = len(leaked_names)

    # --- Summary content analysis ---
    # Split summary from diagnosis section
    summary_match = re.search(
        r"# Therapiesitzung Zusammenfassung\s*\n(.*?)(?=\n## Diagnose|\Z)",
        text, re.DOTALL,
    )
    summary_text = summary_match.group(1).strip() if summary_match else ""
    result["summary_only_words"] = len(summary_text.split())
    result["summary_paragraphs"] = len(
        [p for p in summary_text.split("\n\n") if p.strip()]
    )

    # Structural analysis
    result["has_markdown_structure"] = bool(
        re.search(r"^#{1,3} ", text, re.MULTILINE)
    )
    result["num_sections"] = len(re.findall(r"^#{1,3} ", text, re.MULTILINE))

    # Clinical content coverage — check for key clinical dimensions
    summary_lower = summary_text.lower()
    result["mentions_presenting_problem"] = any(
        t in summary_lower for t in [
            "leberwerte", "alkohol", "schmerz", "unfall", "trauma",
            "beschwerd", "konsum", "trinken",
        ]
    )
    result["mentions_therapeutic_intervention"] = any(
        t in summary_lower for t in [
            "intervention", "wunderfrage", "achtsamkeit", "atemübung",
            "selbstbeobachtung", "psychoedukation", "beobachtungsaufgabe",
            "vereinbar", "übung",
        ]
    )
    result["mentions_therapist_role"] = any(
        t in summary_lower for t in [
            "therapeut", "empathisch", "validier", "explorat",
            "gesprächsführung",
        ]
    )
    result["mentions_biographical_context"] = any(
        t in summary_lower for t in [
            "trennung", "mutter", "kindheit", "biograf", "lebensereignis",
            "schwiegermutter", "nichte", "geschwister",
        ]
    )
    result["mentions_coping_mechanisms"] = any(
        t in summary_lower for t in [
            "bewältigung", "coping", "selbstmedikation", "regulation",
            "vermeidung",
        ]
    )

    # Comorbidity detection — count distinct ICD-10 diagnoses mentioned
    result["num_diagnoses"] = len(unique_codes)
    result["has_comorbidity"] = len(unique_codes) > 1

    # Diagnosis reasoning quality
    result["has_structured_reasoning"] = bool(
        re.search(r"(?:Begründung|Kriterien)", text)
    )
    result["has_evidence_quotes"] = bool(
        re.search(r'[„"«].+?["\x93»]', diag_text)
    )

    return result


def parse_log_file(path: Path) -> dict:
    """Extract metrics from a .log file."""
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")

    # Count HTTP requests (LLM calls)
    http_count = text.count("HTTP Request")

    # Count tool invocations by looking at tool_call responses
    # Pattern: assistant message contains tool_calls with function name
    lookup_calls = len(re.findall(r"'name': 'lookup_icd10'", text))
    search_calls = len(re.findall(r"'name': 'search_transcript'", text))
    # Each tool call name appears in both the tool definition AND in actual calls.
    # Tool definitions appear in every request's tools array.
    # Actual calls appear in assistant messages' tool_calls array.
    # Better pattern: count in tool_call_id context
    actual_lookup = len(re.findall(r"'function': \{'name': 'lookup_icd10'", text))
    actual_search = len(re.findall(r"'function': \{'name': 'search_transcript'", text))

    return {
        "llm_calls": http_count,
        "lookup_icd10_calls": actual_lookup,
        "search_transcript_calls": actual_search,
        "total_tool_calls": actual_lookup + actual_search,
    }


def evaluate_icd10_accuracy(primary_code: str | None, session: str) -> dict:
    """Compare extracted ICD-10 code against ground truth."""
    gt = GROUND_TRUTH_ICD10.get(session)
    if not gt or not primary_code:
        return {"icd10_correct": None, "icd10_category_correct": None}

    # Exact match (up to subcode level)
    exact = primary_code == gt

    # Category match (F10 vs F10, F45 vs F45)
    gt_cat = gt.split(".")[0]
    pred_cat = primary_code.split(".")[0]
    category_correct = pred_cat == gt_cat

    return {
        "icd10_correct": exact,
        "icd10_category_correct": category_correct,
        "icd10_ground_truth": gt,
    }


def analyze_results_dir(results_dir: Path) -> pd.DataFrame:
    """Parse all results in a directory into a DataFrame."""
    rows = []

    time_files = sorted(results_dir.glob("*.time"))
    if not time_files:
        raise FileNotFoundError(f"No .time files found in {results_dir}")

    for time_file in time_files:
        stem = time_file.stem  # e.g., "108_01_simple"
        timing = parse_time_file(time_file)

        session = timing.get("session", "")
        mode = timing.get("mode", "")
        status = timing.get("status", "")

        md_file = time_file.with_suffix(".md")
        log_file = time_file.with_suffix(".log")

        md_metrics = parse_md_file(md_file)
        log_metrics = parse_log_file(log_file)

        # ICD-10 accuracy
        accuracy = evaluate_icd10_accuracy(
            md_metrics.get("icd10_primary"), session
        )

        row = {
            "session": session,
            "mode": mode,
            "status": status,
            "duration_ms": int(timing.get("duration_ms", 0)),
            "duration_s": int(timing.get("duration_ms", 0)) / 1000,
            **md_metrics,
            **log_metrics,
            **accuracy,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print analysis summary to stdout."""
    print("=" * 70)
    print("EXPERIMENT ANALYSIS SUMMARY")
    print("=" * 70)

    # Performance comparison
    print("\n## Performance by Mode")
    print("-" * 50)
    perf = df.groupby("mode").agg(
        mean_duration_s=("duration_s", "mean"),
        mean_llm_calls=("llm_calls", "mean"),
        mean_word_count=("word_count", "mean"),
        success_rate=("status", lambda x: (x == "success").mean()),
    ).round(1)
    print(perf.to_string())

    # Diagnostic accuracy
    print("\n\n## Diagnostic Accuracy")
    print("-" * 50)
    diag = df[df["has_diagnosis"] == True].copy()
    if not diag.empty:
        acc = diag.groupby("mode").agg(
            category_accuracy=("icd10_category_correct", "mean"),
            exact_accuracy=("icd10_correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_symptoms=("num_symptoms", "mean"),
        ).round(2)
        print(acc.to_string())

    # Per-session diagnostic detail
    print("\n\n## Diagnosis Detail (per session × mode)")
    print("-" * 50)
    cols = ["session", "mode", "icd10_primary", "icd10_ground_truth",
            "icd10_category_correct", "confidence", "num_symptoms"]
    available_cols = [c for c in cols if c in df.columns]
    detail = df[df["has_diagnosis"] == True][available_cols]
    print(detail.to_string(index=False))

    # Gender-neutral language
    print("\n\n## Gender-Neutral Language Compliance")
    print("-" * 50)
    gender = df[df["exists"] == True].groupby("mode").agg(
        mean_neutral=("gender_neutral_count", "mean"),
        mean_gendered=("gendered_count", "mean"),
        mean_ratio=("gender_neutral_ratio", "mean"),
    ).round(2)
    print(gender.to_string())

    # Tool usage
    print("\n\n## Tool Usage")
    print("-" * 50)
    tools = df.groupby("mode").agg(
        mean_lookup_icd10=("lookup_icd10_calls", "mean"),
        mean_search_transcript=("search_transcript_calls", "mean"),
        mean_total_tools=("total_tool_calls", "mean"),
    ).round(1)
    print(tools.to_string())

    # Name leakage
    print("\n\n## Name/PII Leakage")
    print("-" * 50)
    leakage_df = df[df["exists"] == True][["session", "mode", "person_tag_count", "name_leak_count", "leaked_names"]]
    has_leaks = leakage_df[
        (leakage_df["person_tag_count"] > 0) | (leakage_df["name_leak_count"] > 0)
    ]
    if has_leaks.empty:
        print("No PII leakage detected in any output.")
    else:
        print(has_leaks.to_string(index=False))

    # Summary content analysis
    print("\n\n## Summary Content Analysis")
    print("-" * 50)
    valid = df[df["exists"] == True]
    content = valid.groupby("mode").agg(
        mean_summary_words=("summary_only_words", "mean"),
        mean_paragraphs=("summary_paragraphs", "mean"),
        mean_sections=("num_sections", "mean"),
        presenting_problem_pct=("mentions_presenting_problem",
                                lambda x: x.mean() * 100),
        therapeutic_intervention_pct=("mentions_therapeutic_intervention",
                                      lambda x: x.mean() * 100),
        therapist_role_pct=("mentions_therapist_role",
                            lambda x: x.mean() * 100),
        biographical_context_pct=("mentions_biographical_context",
                                   lambda x: x.mean() * 100),
        coping_mechanisms_pct=("mentions_coping_mechanisms",
                                lambda x: x.mean() * 100),
    ).round(1)
    print(content.to_string())

    # Diagnosis quality
    print("\n\n## Diagnosis Quality")
    print("-" * 50)
    diag_q = valid.groupby("mode").agg(
        mean_num_diagnoses=("num_diagnoses", "mean"),
        comorbidity_pct=("has_comorbidity", lambda x: x.mean() * 100),
        structured_reasoning_pct=("has_structured_reasoning",
                                   lambda x: x.mean() * 100),
        evidence_quotes_pct=("has_evidence_quotes",
                              lambda x: x.mean() * 100),
    ).round(1)
    print(diag_q.to_string())

    # Per-session summary word counts
    print("\n\n## Summary Word Count Detail (summary only, excl. diagnosis)")
    print("-" * 50)
    wc = valid.pivot_table(
        values="summary_only_words", index="session", columns="mode",
        aggfunc="first",
    )
    print(wc.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Analyze concisum experiment results"
    )
    parser.add_argument(
        "results_dir", type=Path,
        help="Path to experiment results directory",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Output CSV path (optional)",
    )
    parser.add_argument(
        "--json", type=Path, default=None, dest="json_out",
        help="Output JSON path (optional)",
    )
    args = parser.parse_args()

    df = analyze_results_dir(args.results_dir)
    print_summary(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nCSV written to: {args.csv}")

    if args.json_out:
        df.to_json(args.json_out, orient="records", indent=2)
        print(f"\nJSON written to: {args.json_out}")


if __name__ == "__main__":
    main()
