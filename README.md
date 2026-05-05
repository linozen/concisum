# Concisum

LLM-based summarization and ICD-10 diagnosis generation for therapy session transcripts. Processes structured utterances through configurable pipelines of chunked summarization, symptom extraction, and diagnostic classification.

Built with [Pydantic AI](https://ai.pydantic.dev/), [Litestar](https://litestar.dev/), and local LLMs via [Ollama](https://ollama.com).

Part of the [PS200 research project](https://doi.org/10.1159/000549546) (Taubitz, Sehn & Alpers, 2025).

## Quick Start

```bash
uv sync

# CLI usage
uv run concisum <transcript.json> -o output.md --diagnosis

# Start API server
uv run concisum serve --port 8090
```

Requires Ollama running with a supported model (default: `qwen3:8b`).

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/jobs/` | Submit summarization job |
| `GET` | `/jobs/{job_id}` | Poll job status and retrieve results |
| `POST` | `/pipelines/run` | Run a named pipeline template |
| `GET` | `/pipelines/templates` | List available pipeline templates |
| `GET` | `/pipelines/templates/{id}` | Get template details and step parameters |

### Pipeline Templates

Pipelines compose steps into reusable workflows:

| Template | Steps |
|----------|-------|
| `summary` | Chunk, Summarize, Combine |
| `summary_with_diagnosis` | Chunk, Summarize, Combine, Extract Symptoms, Diagnose |
| `diagnosis_only` | Chunk, Extract Symptoms, Diagnose |

## Architecture

```
src/concisum/
├── cli.py                   # Typer CLI: summarize, serve
├── config.py                # LLM configuration (Ollama endpoint + model)
├── load_json.py             # Transcript JSON parsing
├── summary/
│   ├── agents.py            # Chunk summarizer + full summarizer agents
│   ├── models.py            # Utterance, ChunkSummary, FullSummary
│   └── steps.py             # Pipeline step definitions
├── diagnosis/
│   ├── agents.py            # Symptom extractor + diagnosis agent
│   ├── models.py            # Symptom, ICD10Entry, Diagnosis
│   └── steps.py             # Pipeline step definitions
├── pipeline/
│   ├── pipeline.py          # Generic pipeline runner
│   ├── registry.py          # Template registry
│   └── step.py              # Step abstraction
├── tools.py                 # RAG tools: ICD-10 lookup, transcript search
└── api/
    ├── app.py               # Litestar API
    ├── controllers.py       # Job endpoints
    ├── pipeline_controller.py  # Pipeline endpoints
    └── job_store.py         # In-memory job state
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434/v1` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3:8b` | LLM model name |

## Development

```bash
uv run ruff check .   # lint
uv run pyright        # type check
uv run pytest         # test
```

## Related Projects

- [Thoth](https://github.com/linozen/thoth) -- web UI for transcription and summarization
- [Clio](https://github.com/linozen/clio) -- batch transcription + speaker diarization API
