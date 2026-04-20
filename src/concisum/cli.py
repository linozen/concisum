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
    input_file: Path,
    with_diagnosis: bool = False,
    simple: bool = False,
    tools: list[str] | None = None,
) -> FullSummary:
    """Summarize a transcript file."""
    utterance_list = load_utterances_from_json(input_file)

    if simple:
        from concisum.simple import simple_summarize
        return await simple_summarize(utterance_list, with_diagnosis=with_diagnosis)

    summarizer = SummaryOrchestrator(
        chunk_size=50,
        therapist_speaker_number=1,
        generate_diagnosis=with_diagnosis,
        tools=tools,
    )
    return await summarizer.process_transcript(utterance_list)


def save_as_markdown(summary: FullSummary, output_path: Path) -> None:
    """Save the summary as a markdown file."""
    markdown_content = f"""# Therapiesitzung Zusammenfassung

{summary.content}
"""

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

    if summary.symptoms and summary.symptoms.symptoms:
        markdown_content += "\n\n## Identifizierte Symptome\n"
        for symptom in summary.symptoms.symptoms:
            markdown_content += f"\n### {symptom.name}\n"
            markdown_content += f"**Beschreibung:** {symptom.description}\n\n"
            markdown_content += f"**Belege:** {symptom.evidence}\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    console.print(f"[green]Summary saved to:[/green] {output_path}")


def save_as_json(summary: FullSummary, output_path: Path) -> None:
    """Save the summary as a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))

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
        help="Path to output file (default: input_summary.md or .json)",
    ),
    fmt: str = typer.Option(
        "md",
        "--format",
        "-f",
        help="Output format: md (markdown) or json",
    ),
    with_diagnosis: bool = typer.Option(
        False,
        "--diagnosis",
        "-d",
        help="Generate ICD-10 diagnosis from transcript",
    ),
    simple: bool = typer.Option(
        False,
        "--simple",
        "-s",
        help="Use single-prompt (non-agentic) processing instead of hierarchical pipeline",
    ),
    tools: Optional[str] = typer.Option(
        None,
        "--tools",
        "-t",
        help="Comma-separated list of tools to enable for diagnosis (e.g. 'icd10,transcript')",
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
        setup_logging(verbose)
        validate_input_file(input_file)

        ext = ".json" if fmt == "json" else ".md"
        output_path = output or input_file.parent / f"{input_file.stem}_summary{ext}"

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
            tool_list = [t.strip() for t in tools.split(",")] if tools else None
            summary = asyncio.run(
                summarize_transcript(
                    input_file,
                    with_diagnosis=with_diagnosis,
                    simple=simple,
                    tools=tool_list,
                )
            )

        if fmt == "json":
            save_as_json(summary, output_path)
        else:
            save_as_markdown(summary, output_path)

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
