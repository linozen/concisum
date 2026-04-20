#!/usr/bin/env bash
# Re-run only the failed agentic experiments (with fixed retry logic)
set -euo pipefail

CONCISUM_DIR="$HOME/concisum_experiment/concisum"
TRANSCRIPT_DIR="$HOME/concisum_experiment/transcripts"
OUTPUT_DIR="$HOME/concisum_experiment/results/2026-04-20_080919"

export OLLAMA_HOST="http://localhost:11434/v1"
export OLLAMA_MODEL="gemma4:26b"

cd "$CONCISUM_DIR"

TRANSCRIPTS=(
    "2spk_de_108_01_TableDJIMic2_00h29m26s__parakeet.json"
    "2spk_de_108_02_TableDJIMic2_00h26m44s__parakeet.json"
    "2spk_de_129_01_LavalDJIMic2_00h34m44s__parakeet.json"
    "2spk_de_129_02_LavalDJIMic2_00h41m37s__parakeet.json"
)

# Only agentic modes (simple already succeeded)
MODES=(
    "agentic|--diagnosis"
    "agentic_icd10|--diagnosis --tools icd10"
    "agentic_icd10_transcript|--diagnosis --tools icd10,transcript"
)

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

        start_ts=$(date +%s%N)
        if uv run concisum "$input_file" -o "$output_file" $flags -v \
            > "$log_file" 2>&1; then
            status="success"
        else
            status="failed (exit $?)"
        fi
        end_ts=$(date +%s%N)
        duration_ms=$(( (end_ts - start_ts) / 1000000 ))

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

# Regenerate summary
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

echo "=== Re-run complete ==="
cat "$OUTPUT_DIR/summary.md"
