# Analysis Plan: Agentic vs Non-Agentic Summarization Experiment

## Context

Experiment completed on `osi-klips-80` using `gemma4:26b` (MoE, Q4_K_M) on
RTX A6000. 4 therapy sessions × 4 modes = 16 runs on anonymized transcripts.
Results in `~/concisum_experiment/results/2026-04-20_105248/` on klips80.

### Modes
1. **simple** — single-prompt, no chunking, no structured output, no retry
2. **agentic** — hierarchical chunking + pydantic-ai structured output + diagnosis
3. **agentic_icd10** — agentic + ICD-10 lookup tool
4. **agentic_icd10_transcript** — agentic + ICD-10 + transcript search tool

### Sessions (anonymized, best-mic Parakeet transcripts)
- 108_01 (184 utt, ~29 min) — alcohol dependence case
- 108_02 (131 utt, ~27 min) — somatoform disorder case
- 129_01 (165 utt, ~35 min) — PTSD case
- 129_02 (326 utt, ~42 min) — alcohol dependence case

### Known Issues
- 108_02 agentic failed in both runs (empty model response timeout)
- All other 15 runs succeeded

---

## Step 1: Pull and Organize Results

```bash
scp -r osi-klips-80:~/concisum_experiment/results/2026-04-20_105248/ \
    results/anon_2026-04-20/
```

Each run produces:
- `{session}_{mode}.md` — summary + diagnosis output
- `{session}_{mode}.log` — verbose execution log
- `{session}_{mode}.time` — timing metadata

---

## Step 2: Structured Data Extraction

Create `scripts/analyze_experiment.py` that parses all outputs into a
structured DataFrame with columns:

| Column | Source |
|--------|--------|
| session | filename |
| mode | filename |
| duration_s | .time file |
| summary_word_count | .md file |
| has_diagnosis | .md contains "## Diagnose" |
| icd10_code | parsed from diagnosis section |
| icd10_confidence | parsed from "Diagnosesicherheit" |
| num_symptoms | count of "### " under "## Identifizierte Symptome" |
| gender_neutral | count of "Klient\*in" / "Therapeut\*in" vs "Klient" / "Patient" |
| name_leakage | check for person names or [PERSON_N] tags remaining |
| structured_output | bool: diagnosis fields parsed into correct sections |
| llm_calls | count of HTTP requests in .log |
| tool_calls | count of tool invocations in .log (lookup_icd10 / search_transcript) |

---

## Step 3: Comparison Dimensions

### 3a. Summary Quality (qualitative)
- **Gender-neutral language compliance**: Count gendered vs gender-neutral terms
- **Name leakage**: Does simple mode leak character names? Does agentic prevent it?
- **Word count adherence**: Agentic targets ≤300 words; does simple overshoot?
- **Structure**: Agentic produces markdown sections; simple produces free text

### 3b. Diagnostic Accuracy
- **ICD-10 code correctness**: Compare assigned codes against ground truth
  (from role scripts in `sources/role_scripts/`)
  - 108_01: F10.2 (Alkoholabhängigkeit)
  - 108_02: F45.0 (Somatisierungsstörung)
  - 129_01: F43.1 (PTBS)
  - 129_02: F10.2 (Alkoholabhängigkeit)
- **Confidence scores**: Agentic produces structured 0-1 scores; simple cannot
- **Comorbidity detection**: Does the model identify secondary diagnoses?
- **Evidence quality**: Do symptoms have concrete transcript quotes?

### 3c. Tool Use Impact
- **ICD-10 tool**: Does lookup_icd10 improve code accuracy or specificity?
- **Transcript tool**: Does search_transcript produce better evidence quotes?
- Compare agentic (no tools) vs agentic_icd10 vs agentic_icd10_transcript

### 3d. Performance
- **Duration**: simple (~40s) vs agentic (~300-650s)
- **LLM calls**: simple (1-2) vs agentic (many, depends on chunk count)
- **Reliability**: failure rate per mode (108_02 agentic consistently fails)

---

## Step 4: Visualizations

Create `scripts/plot_experiment.py` using plotnine. Output PDFs to
`../00_papers/thesis/figures/`.

1. **Duration comparison** — grouped bar chart, session × mode
2. **Diagnostic accuracy table** — markdown/LaTeX table for thesis
3. **Feature comparison heatmap** — modes × quality dimensions
4. **Tool use analysis** — bar chart of tool calls per mode

---

## Step 5: Write Thesis Sections

Fill TODO placeholders in `thesis.typ`:
- Results (~line 328): summarization comparison, diagnostic accuracy, tool impact
- Discussion (~line 380): when agentic overhead is justified, tool use value,
  limitations (empty model responses, MoE reliability)

---

## Step 6: Re-run Failed Case (Optional)

108_02 agentic fails due to gemma4:26b empty response on that transcript.
Options:
- Accept 15/16 (sufficient for thesis)
- Retry with different chunk size or prompt
- Document as MoE model limitation
