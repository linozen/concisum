#!/usr/bin/env python3
"""Incrementally update the ICD-10 vector store."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from concisum.diagnosis.vectorstore import ICD10VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_vectorstore(json_path: Path, mode: str = "upsert"):
    """
    Update vector store with new/modified entries.

    Args:
        json_path: Path to JSON file with new entries
        mode: "upsert" (add/update), "delete", or "rebuild"
    """
    vectorstore = ICD10VectorStore(persist_dir="./data/icd10_db")

    if mode == "rebuild":
        logger.info("Rebuilding entire vector store...")
        vectorstore.clear()
        count = vectorstore.populate_from_json(json_path)
        logger.info(f"Rebuilt with {count} entries")

    elif mode == "upsert":
        logger.info("Upserting entries (add new, update existing)...")
        # ChromaDB handles upserts automatically via IDs
        count = vectorstore.populate_from_json(json_path)
        logger.info(f"Upserted {count} entries")

    elif mode == "delete":
        logger.info("Deleting entries...")
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)

        codes_to_delete = [entry['code'] for entry in data.get('entries', [])]
        vectorstore.collection.delete(ids=codes_to_delete)
        logger.info(f"Deleted {len(codes_to_delete)} entries")

    # Verify
    total = vectorstore.collection.count()
    logger.info(f"Total entries in vector store: {total}")
    logger.info(f"Available codes: {', '.join(sorted(vectorstore.get_all_codes()))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update ICD-10 vector store")
    parser.add_argument("json_file", type=Path, help="JSON file with ICD-10 entries")
    parser.add_argument(
        "--mode",
        choices=["upsert", "delete", "rebuild"],
        default="upsert",
        help="Update mode (default: upsert)"
    )

    args = parser.parse_args()
    update_vectorstore(args.json_file, args.mode)
