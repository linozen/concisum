# Development Workflow

## Edit-Validation Loop

1. **Edit**: Make changes to the codebase
2. **Validate**:
   - Run `~/.local/bin/uv run ruff check .` - Check code style and linting
   - Run `~/.local/bin/uv run pyright` - Type checking
   - Run `~/.local/bin/uv run pytest` - Execute tests

## Architecture

LLM-based summarization CLI with hierarchical processing:

- `src/concisum/` - Main package directory
  - `cli.py` - CLI interface (Typer)
  - `config.py` - LLM model configuration (Ollama)
  - `load_json.py` - JSON transcript parsing (multiple formats)
  - `summary/` - Summarization module
    - `agents.py` - Pydantic AI agents for chunked summarization
    - `models.py` - Data models for utterances and summaries
  - `diagnosis/` - ICD-10 diagnosis module
    - `agents.py` - Pydantic AI agents for symptom extraction and diagnosis
    - `models.py` - Data models for symptoms and diagnoses
- `tests/` - Test suite for JSON parsing

The application processes therapy session transcripts through hierarchical summarization: chunks → chunk summaries → final summary (≤250 words). Optional ICD-10 diagnosis generation with symptom extraction and confidence scoring.

## Environment Setup

Sync dependencies:
```bash
~/.local/bin/uv sync
```

Run CLI:
```bash
~/.local/bin/uv run concisum <input.json> [options]
```

Configuration:
- Expects Ollama-compatible endpoint (default: `http://localhost:11434/v1`)
- Default model: `qwen3:8b`
- Set `OLLAMA_HOST` and `OLLAMA_MODEL` environment variables to override
