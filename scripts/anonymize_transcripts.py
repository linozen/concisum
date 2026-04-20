"""Anonymize therapy transcripts by replacing PII with pseudonyms.

Two-pass approach:
  1. `detect`  — Use spaCy German NER to find PERSON candidates, write
                 a review file (_pii_candidates.json) for manual curation.
  2. `apply`   — Read a curated entity map and replace all occurrences.

Usage:
    # Step 1: detect candidates
    uv run --with spacy python scripts/anonymize_transcripts.py detect \
        INPUT_DIR --out-map candidates.json

    # Step 2: edit candidates.json — remove false positives, add missed names

    # Step 3: apply
    uv run --with spacy python scripts/anonymize_transcripts.py apply \
        INPUT_DIR OUTPUT_DIR --map candidates.json
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import spacy

LOG = logging.getLogger(__name__)

# Common German words that spaCy frequently misclassifies as PER in
# conversational therapy transcripts. This is not exhaustive — the
# detect step is meant to be followed by manual review.
FALSE_POSITIVE_PERSONS = {
    # Interjections / discourse markers
    "Nee", "nee", "Ne", "Na", "Tja", "Ach", "Ah", "Aha", "Hi", "Boah",
    "Oh", "Naja", "Yeah", "Danke", "danke", "Bitte", "Schön", "Sagt",
    "Merken", "Darf", "Lass", "Bin", "Gott", "Einfach", "Freut",
    "Angenehm", "Unangenehm", "Schlecht", "Leichter", "Wütend", "Müde",
    "Spaß", "Jung", "Schönes", "Mache",
    # Common nouns / kinship terms
    "Mann", "Mama", "Papa", "Bruder", "Schwester", "Mutter", "Vater",
    "Oma", "Opa", "Kind", "Kinder", "Tochter", "Sohn", "Nichte",
    # Phrases misclassified as names
    "oh Gott", "Oh Gott", "ach Gott", "Ach Gott", "guck mal",
    "oh guck mal", "Ach komm", "ach komm", "ach nee", "danke schön",
    "Gute Mama", "Mein Bruder",
    # Clinical / medical terms
    "Herzschmerzen", "Luftprobleme", "Schweinehund", "Kindersitz",
    "Krankenwagenpolizei", "Gleitzeit", "Körpensport",
    # Misc false positives
    "Jahren", "Grote", "Krach", "Parian",
}


def load_transcripts(input_dir: Path) -> list[tuple[Path, dict]]:
    """Load all JSON transcript files from a directory."""
    files = sorted(input_dir.glob("*.json"))
    result = []
    for fp in files:
        if fp.name.startswith("_"):
            continue
        with open(fp, encoding="utf-8") as f:
            result.append((fp, json.load(f)))
    return result


def collect_texts(transcripts: list[tuple[Path, dict]]) -> list[str]:
    """Extract all utterance texts from loaded transcripts."""
    texts = []
    for _fp, data in transcripts:
        for utt in data.get("utterances", []):
            texts.append(utt.get("text", utt.get("ref_text", "")))
    return texts


def detect_persons(
    texts: list[str], nlp: spacy.language.Language
) -> dict[str, int]:
    """Run spaCy NER on all texts and return person entity counts."""
    candidates: Counter[str] = Counter()

    for doc in nlp.pipe(texts, batch_size=64):
        for ent in doc.ents:
            if ent.label_ == "PER":
                text = ent.text.strip()
                if text and text not in FALSE_POSITIVE_PERSONS:
                    candidates[text] += 1

    return dict(candidates)


def cmd_detect(args: argparse.Namespace) -> None:
    """Detect PII candidates and write a review file."""
    input_dir = Path(args.input_dir)
    out_map = Path(args.out_map)

    transcripts = load_transcripts(input_dir)
    LOG.info("Loaded %d transcript files", len(transcripts))

    texts = collect_texts(transcripts)
    LOG.info("Collected %d utterances", len(texts))

    LOG.info("Loading spaCy model '%s'...", args.spacy_model)
    nlp = spacy.load(args.spacy_model)

    LOG.info("Running NER...")
    candidates = detect_persons(texts, nlp)

    # Build review structure: candidate → {count, pseudonym, keep}
    review: dict[str, dict] = {}
    counter = 0
    for name, count in sorted(candidates.items(), key=lambda x: -x[1]):
        counter += 1
        review[name] = {
            "count": count,
            "pseudonym": f"[PERSON_{counter}]",
            "keep": True,
        }

    with open(out_map, "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    LOG.info(
        "Wrote %d candidates to %s — review and set 'keep: false' for "
        "false positives, then run 'apply'.",
        len(review),
        out_map,
    )


def cmd_apply(args: argparse.Namespace) -> None:
    """Apply a curated entity map to anonymize transcripts."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    map_path = Path(args.map)

    with open(map_path, encoding="utf-8") as f:
        review = json.load(f)

    # Build replacement map from curated entries
    entity_map: dict[str, str] = {}
    for original, info in review.items():
        if info.get("keep", True):
            entity_map[original] = info["pseudonym"]

    LOG.info("Applying %d replacements", len(entity_map))

    transcripts = load_transcripts(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fp, data in transcripts:
        for utt in data.get("utterances", []):
            if "text" in utt:
                utt["text"] = _replace(utt["text"], entity_map)
            if "ref_text" in utt:
                utt["ref_text"] = _replace(utt["ref_text"], entity_map)
            for word in utt.get("words", []):
                if "word" in word:
                    word["word"] = _replace(word["word"], entity_map)

        out_path = output_dir / fp.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        LOG.info("Saved: %s", out_path)

    LOG.info("Anonymization complete. %d files processed.", len(transcripts))


def _replace(text: str, entity_map: dict[str, str]) -> str:
    """Replace all entity occurrences, longest-first."""
    for original in sorted(entity_map, key=len, reverse=True):
        text = re.sub(re.escape(original), entity_map[original], text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize therapy transcripts (two-pass: detect → apply)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # detect
    p_detect = sub.add_parser("detect", help="Detect PII candidates via NER")
    p_detect.add_argument("input_dir", type=Path)
    p_detect.add_argument(
        "--out-map", default="pii_candidates.json",
        help="Output path for candidate review file",
    )
    p_detect.add_argument("--spacy-model", default="de_core_news_lg")

    # apply
    p_apply = sub.add_parser("apply", help="Apply curated entity map")
    p_apply.add_argument("input_dir", type=Path)
    p_apply.add_argument("output_dir", type=Path)
    p_apply.add_argument("--map", required=True, help="Curated entity map JSON")

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "apply":
        cmd_apply(args)


if __name__ == "__main__":
    main()
