#!/usr/bin/env bash
# Concisum agentic vs non-agentic experiment
# Runs 4 sessions × 4 modes = 16 runs with gemma4:31b on osi-klips-80
set -euo pipefail

CONCISUM_DIR="$HOME/concisum_experiment/concisum"
TRANSCRIPT_DIR="$HOME/concisum_experiment/transcripts_anonymized"
OUTPUT_DIR="$HOME/concisum_experiment/results/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

export OLLAMA_HOST="http://localhost:11434/v1"
export OLLAMA_MODEL="gemma4:26b"

cd "$CONCISUM_DIR"

TRANSCRIPTS=(
    "2spk_de_108_01_TableDJIMic2_00h29m26s__parakeet.json"
    "2spk_de_108_02_TableDJIMic2_00h26m44s__parakeet.json"
    "2spk_de_129_01_LavalDJIMic2_00h34m44s__parakeet.json"
    "2spk_de_129_02_LavalDJIMic2_00h41m37s__parakeet.json"
)

# Mode definitions: name, flags
MODES=(
    "simple|--diagnosis --simple"
    "agentic|--diagnosis"
    "agentic_icd10|--diagnosis --tools icd10"
    "agentic_icd10_transcript|--diagnosis --tools icd10,transcript"
)

# Log environment
{
    echo "=== Experiment Environment ==="
    echo "Date: $(date -Iseconds)"
    echo "Host: $(hostname)"
    echo "Model: $OLLAMA_MODEL"
    echo "OLLAMA_HOST: $OLLAMA_HOST"
    echo "Python: $(uv run python --version)"
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "N/A"
    echo ""
} > "$OUTPUT_DIR/environment.txt"

TOTAL=$((${#TRANSCRIPTS[@]} * ${#MODES[@]}))
RUN=0

for transcript in "${TRANSCRIPTS[@]}"; do
    session=$(echo "$transcript" | sed -E 's/2spk_de_([0-9]+_[0-9]+)_.*/\1/')
    input_file="$TRANSCRIPT_DIR/$transcript"

    for mode_spec in "${MODES[@]}"; do
        IFS='|' read -r mode_name flags <<< "$mode_spec"
        RUN=$((RUN + 1))

        output_file="$OUTPUT_DIR/${session}_${mode_name}.md"
        log_file="$OUTPUT_DIR/${session}_${mode_name}.log"
        timing_file="$OUTPUT_DIR/${session}_${mode_name}.time"

        echo "[$RUN/$TOTAL] Session $session, mode: $mode_name"
        echo "  Input:  $input_file"
        echo "  Output: $output_file"

        # Run with timing
        start_ts=$(date +%s%N)
        if uv run concisum "$input_file" -o "$output_file" $flags -v \
            > "$log_file" 2>&1; then
            status="success"
        else
            status="failed (exit $?)"
        fi
        end_ts=$(date +%s%N)
        duration_ms=$(( (end_ts - start_ts) / 1000000 ))

        # Write timing info
        {
            echo "session: $session"
            echo "mode: $mode_name"
            echo "status: $status"
            echo "duration_ms: $duration_ms"
            echo "duration_s: $(echo "scale=1; $duration_ms / 1000" | bc)"
            echo "timestamp: $(date -Iseconds)"
        } > "$timing_file"

        echo "  Status: $status, Duration: ${duration_ms}ms"
        echo ""
    done
done

# Generate summary
echo "=== Generating experiment summary ==="
{
    echo "# Concisum Experiment Results"
    echo ""
    echo "Model: $OLLAMA_MODEL"
    echo "Date: $(date -Iseconds)"
    echo ""
    echo "| Session | Mode | Status | Duration (s) | Output Words |"
    echo "|---------|------|--------|-------------|-------------|"

    for timing in "$OUTPUT_DIR"/*.time; do
        session=$(grep '^session:' "$timing" | cut -d' ' -f2)
        mode=$(grep '^mode:' "$timing" | cut -d' ' -f2)
        status=$(grep '^status:' "$timing" | cut -d' ' -f2-)
        duration=$(grep '^duration_s:' "$timing" | cut -d' ' -f2)
        md_file="${timing%.time}.md"
        if [ -f "$md_file" ]; then
            words=$(wc -w < "$md_file" | tr -d ' ')
        else
            words="N/A"
        fi
        echo "| $session | $mode | $status | $duration | $words |"
    done
} > "$OUTPUT_DIR/summary.md"

echo ""
echo "=== Experiment complete ==="
echo "Results in: $OUTPUT_DIR"
cat "$OUTPUT_DIR/summary.md"
