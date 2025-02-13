# `concisum`

`concisum` is a CLI tool that analyzes therapy session transcripts using LLMs to
generate comprehensive summaries and ICD-10 diagnoses. It's designed to assist
mental health professionals in documenting and analyzing therapy sessions.

## Features

- Analyzes therapy transcripts and generates detailed summaries
- Provides ICD-10 diagnoses (Chapter V, F00-F99) with justifications
- Supports local LLM inference through Ollama
- Outputs results in both console and markdown format
- Handles JSON-formatted transcript files

## Prerequisites

- Python 3.11 or higher
- Ollama installed and running locally (with models like mistral-nemo)
- uv (https://github.com/astral-sh/uv)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/concisum.git
cd concisum
```

2. Create and activate a virtual environment using uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
uv sync
```

## Usage

The basic command syntax is:

```bash
concisum <input_file> [options]
```

### Options

- `-o, --output`: Specify output markdown file path (default: input_summary.md)
- `-m, --model`: Set the Ollama model name (default: mistral-nemo)
- `-t, --temperature`: Set generation temperature (default: 0.3)
- `-v, --verbose`: Enable verbose logging
- `--help`: Show help message

### Example

```bash
concisum transcript.json -o summary.md -m mistral-nemo -t 0.3 -v
```

### Input Format

The input file should be a JSON file with the following structure:

```json
{
  "utterances": [
    {
      "hyp_text": "utterance text",
      "hyp_spk": "speaker_id"
    },
    ...
  ]
}
```

### Output

The tool generates:
1. A console output with the summary and diagnosis
2. A markdown file containing:
   - Detailed session summary
   - ICD-10 diagnosis
   - Diagnostic justification

## Development

To set up the development environment:

1. Install dependencies using uv:
```bash
uv pip install -e .
```

2. The project uses uv for dependency management. To update dependencies:
```bash
uv pip compile pyproject.toml -o uv.lock
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
