#!/usr/bin/env python3
"""Initialize the ICD-10 vector store with sample data."""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from concisum.diagnosis.vectorstore import ICD10VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize vector store with sample ICD-10 data."""
    # Initialize vector store
    vectorstore = ICD10VectorStore(persist_dir="./data/icd10_db")

    # Clear existing data (optional)
    logger.info("Clearing existing vector store data...")
    vectorstore.clear()

    # Populate from sample data
    json_path = Path("./data/icd10_sample.json")
    if not json_path.exists():
        logger.error(f"Sample data file not found: {json_path}")
        logger.error("Please ensure you're running this script from the project root.")
        sys.exit(1)

    logger.info(f"Populating vector store from {json_path}...")
    count = vectorstore.populate_from_json(json_path)

    logger.info(f"Successfully initialized vector store with {count} ICD-10 entries")

    # Verify by listing codes
    codes = vectorstore.get_all_codes()
    logger.info(f"Available codes: {', '.join(sorted(codes))}")


if __name__ == "__main__":
    main()
