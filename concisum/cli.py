import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from concisum.summarizer import SYSTEM_PROMPT, USER_PROMPT, TranscriptSummarizer
from concisum.utils import (
    load_json_transcript,
    save_as_markdown,
    format_conversation_for_display,
    prepare_transcript_text,
)

LOG = logging.getLogger(__name__)
console = Console()
app = typer.Typer(
    help="Summarize therapy transcripts using LLM and generate ICD-10 diagnoses"
)


def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)


def validate_input_file(input_file: Path) -> None:
    """Validate that input file exists and is a file."""
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        raise typer.Exit(code=1)

    if not input_file.is_file():
        console.print(f"[red]Error:[/red] Input path is not a file: {input_file}")
        raise typer.Exit(code=1)


def print_summary(summary: dict):
    """Print summary to console in a formatted way."""
    console.print(
        Panel.fit(
            f"""[bold blue]Zusammenfassung:[/bold blue]
{summary['summary']}

[bold blue]ICD-10 Diagnose:[/bold blue]
{summary['icd-10-diagnosis']}

[bold blue]ICD-10 Begründung:[/bold blue]
{summary['icd-10-justification']}""",
            title="Therapiegespräch Analyse",
            border_style="cyan",
        )
    )


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
    model: str = typer.Option(
        "mistral-nemo", "--model", "-m", help="Set model name (Ollama)"
    ),
    temperature: float = typer.Option(
        0.3,
        "--temperature",
        "-t",
        help="Temperature for generation",
        min=0.0,
        max=1.0,
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
            utterances = load_json_transcript(str(input_file))

        if not utterances:
            console.print("[red]Error:[/red] No utterances found in input file")
            raise typer.Exit(code=1)

        # Prepare transcript text
        transcript_text = prepare_transcript_text(utterances)

        # If verbose, show the prompts
        if verbose:
            console.print("\n[bold purple]Input Details:[/bold purple]")
            console.print(
                Panel(
                    format_conversation_for_display(
                        transcript_text=transcript_text,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=USER_PROMPT,
                    ),
                    title="Conversation Setup",
                    border_style="blue",
                )
            )
            console.print()  # Add blank line for spacing

        # Initialize summarizer and generate summary
        summarizer = TranscriptSummarizer(
            model=model,
            temperature=temperature,
        )

        with console.status(
            "[bold green]Generating summary and diagnosis...[/bold green]"
        ):
            summary = summarizer.summarize(utterances)

        # Save results
        save_as_markdown(summary, output_path)
        console.print(f"[green]Summary saved to:[/green] {output_path}")

        # Print summary to console
        print_summary(summary)

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
