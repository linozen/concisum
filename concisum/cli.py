import logging
import traceback
from pathlib import Path
from typing import Optional
import asyncio

import typer
from rich.console import Console
from rich.panel import Panel

from concisum.load_json import load_utterances_from_json
from concisum.summary.agents import SummaryOrchestrator
from concisum.summary.models import FullSummary


# Setup logging
LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)


# Setup rich
console = Console()

# Setup typer
app = typer.Typer(
    help="Summarize therapy transcripts using LLM and generate ICD-10 diagnoses"
)


def validate_input_file(input_file: Path) -> None:
    """Validate that input file exists and is a file."""
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        raise typer.Exit(code=1)

    if not input_file.is_file():
        console.print(f"[red]Error:[/red] Input path is not a file: {input_file}")
        raise typer.Exit(code=1)


async def summarize_transcript(
    input_file: Path, with_diagnosis: bool = False, use_rag: bool = True
) -> FullSummary:
    """
    Summarize a transcript file using hierarchical processing.

    Args:
        input_file: Path to the input JSON transcript file
        with_diagnosis: Whether to include diagnosis generation
        use_rag: Whether to use the vector database (RAG) for diagnosis generation

    Returns:
        FullSummary object containing the complete transcript summary
    """
    # Load utterances from JSON file
    utterance_list = load_utterances_from_json(input_file)

    # Initialize the transcript summarizer
    summarizer = SummaryOrchestrator(
        chunk_size=50,
        therapist_speaker_number=1,
        generate_diagnosis=with_diagnosis,
        use_rag=use_rag,
    )  # Configurable chunk size

    # Process the transcript through hierarchical summarization
    full_summary = await summarizer.process_transcript(utterance_list)
    return full_summary


def save_as_markdown(summary: FullSummary, output_path: Path) -> None:
    """
    Save the summary as a markdown file.

    Args:
        summary: FullSummary object containing the summary content
        output_path: Path where the markdown file should be saved
    """
    markdown_content = f"""# Therapiesitzung Zusammenfassung

{summary.content}
"""

    # Add diagnosis information if available
    if summary.diagnosis:
        markdown_content += f"""

## Diagnose

### ICD-10 Diagnose
{summary.diagnosis.icd_10_diagnose}

### Begründung
{summary.diagnosis.icd_10_begruendung}

### Diagnosesicherheit
{summary.diagnosis.icd_10_sicherheit:.2f}
"""

    # Add symptoms if available
    if summary.symptoms and summary.symptoms.symptoms:
        markdown_content += "\n\n## Identifizierte Symptome\n"
        for symptom in summary.symptoms.symptoms:
            markdown_content += f"\n### {symptom.name}\n"
            markdown_content += f"**Beschreibung:** {symptom.description}\n\n"
            markdown_content += f"**Belege:** {symptom.evidence}\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    console.print(f"[green]Summary saved to:[/green] {output_path}")


@app.command()
def summarize(
    input_file: Path = typer.Argument(
        ...,
        help="Path to input transcript text file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to output markdown file (default: input_summary.md)",
    ),
    with_diagnosis: bool = typer.Option(
        False,
        "--diagnosis",
        "-d",
        help="Generate ICD-10 diagnosis from transcript",
    ),
    no_rag: bool = typer.Option(
        False,
        "--no-rag",
        help="Disable the use of the vector database (RAG) for diagnosis generation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """
    Analyze therapy transcripts and generate summaries with ICD-10 diagnoses.
    """
    try:
        # Setup logging
        setup_logging(verbose)

        # Validate input file
        validate_input_file(input_file)

        # Set output path
        output_path = output or input_file.parent / f"{input_file.stem}_summary.md"

        # Read input file
        with console.status("[bold green]Reading transcript...[/bold green]"):
            utterances = load_utterances_from_json(input_file)
            if verbose:
                console.print(f"Loaded {len(utterances.utterances)} utterances")

        if not utterances:
            console.print("[red]Error:[/red] No utterances found in input file")
            raise typer.Exit(code=1)

        with console.status(
            "[bold green]Generating summary and diagnosis through hierarchical processing...[/bold green]"
        ):
            # Run async summarization function with diagnosis if requested
            summary = asyncio.run(
                summarize_transcript(
                    input_file, with_diagnosis=with_diagnosis, use_rag=not no_rag
                )
            )

        # Save results
        save_as_markdown(summary, output_path)

        # Display summary preview
        console.print(
            Panel(
                summary.content[:500] + "..."
                if len(summary.content) > 500
                else summary.content,
                title="Vorschau",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print("[red]Traceback:[/red]")
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
