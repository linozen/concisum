import os
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from concisum.summary.config import SummarizerConfig
from concisum.summary.summarizer import TranscriptSummarizer
from concisum.summary.utils import load_json_transcript, save_as_markdown

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    """Main entry point for the summarization CLI."""
    parser = argparse.ArgumentParser(
        description="Summarize therapy transcripts using LLM and generate ICD-10 diagnoses"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to input transcript text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSON file (default: input_summary.json)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the GGUF model file",
        default="models/mistral-nemo-instruct-2407-q6_k.gguf",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for generation (default: 0.3)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Validate input file
    if not args.input_file.exists():
        LOG.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    if not args.input_file.is_file():
        LOG.error(f"Input path is not a file: {args.input_file}")
        sys.exit(1)

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = args.input_file.parent / f"{args.input_file.stem}_summary.md"

    try:
        # Read input file
        LOG.info(f"Reading transcript from {args.input_file}")
        utterances = load_json_transcript(str(args.input_file))

        if not utterances:
            LOG.error("No utterances found in input file")
            sys.exit(1)

        # Initialize summarizer
        config = SummarizerConfig(
            model_path=args.model_path,
            use_gpu=args.gpu,
            temperature=args.temperature,
        )

        summarizer = TranscriptSummarizer(config)

        # Generate summary
        LOG.info("Generating summary and diagnosis...")
        summary = summarizer.summarize(utterances)

        # Save results
        save_as_markdown(summary, output_path)
        LOG.info(f"Summary saved to {output_path}")

        # Print summary to console
        print("--------")
        print(f"Zusammenfassung: {summary['summary']}")
        print(f"ICD-10 Diagnose: {summary['icd-10-diagnosis']}")
        print(f"ICD-10 Begründung: {summary['icd-10-justification']}")
        print("--------")

    except Exception as e:
        LOG.error(f"Error processing transcript: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
