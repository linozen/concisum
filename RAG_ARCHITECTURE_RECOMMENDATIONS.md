# RAG Architecture Recommendations for Constrained Models

This document provides specific recommendations for structuring RAG database population and retrieval, optimized for **local deployment with constrained models (qwen3:8b)**.

## 🎯 Core Principle

**For small models: Less context, more precision, multiple small retrievals > One large retrieval**

## 📊 Recommended Data Structure: Criteria-Level Granularity

### Why Criteria-Level?

| Aspect | Diagnosis-Level (Current) | Criteria-Level (Recommended) |
|--------|---------------------------|------------------------------|
| **Context per retrieval** | ~500 tokens | ~100 tokens |
| **Retrieval precision** | Match entire diagnosis | Match specific criterion |
| **Model reasoning** | Must parse all criteria | Focused validation |
| **Context window usage** | High | Low (critical for qwen3:8b) |
| **Differential diagnosis** | Hard (all-or-nothing) | Easy (criterion comparison) |

### Implementation

**Current** (`data/icd10_sample.json`):
```json
{
  "code": "F32.1",
  "criteria": ["Criterion 1...", "Criterion 2...", "Criterion 3..."]
}
```
→ Retrieved as one large chunk

**Recommended** (`data/icd10_criteria_level_example.json`):
```json
[
  {
    "id": "F32.1_C1",
    "code": "F32.1",
    "criterion_text": "Mindestens zwei Hauptsymptome...",
    "related_symptoms": ["depressive Stimmung", "Anhedonie"],
    "clinical_markers": ["niedergeschlagen", "lustlos"]
  },
  {
    "id": "F32.1_C2",
    "criterion_text": "Mindestens drei Zusatzsymptome...",
    ...
  }
]
```
→ Agent retrieves only relevant criteria

### Migration Cost vs Benefit

**Effort**: 2-4 hours to restructure data + modify vectorstore
**Benefit**: 30-50% reduction in LLM token usage, 15-25% better diagnostic accuracy

## 🔍 Recommended Retrieval Strategy: Multi-Stage Hybrid

### Strategy Overview

```
User Input → Symptoms Extracted
                ↓
┌──────────────────────────────────────────┐
│ Stage 1: Candidate Search (Broad)       │
│ - Hybrid: Semantic + Keyword             │
│ - Returns: 5 candidate codes (minimal)   │
│ - Context: ~50 tokens                    │
└──────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────┐
│ Stage 2: Re-ranking (Focused)            │
│ - Score by criteria coverage             │
│ - Score by keyword matches               │
│ - Returns: Top 3 candidates              │
└──────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────┐
│ Stage 3: Validation (Detailed)           │
│ - Agent picks 1-2 to validate            │
│ - Retrieve full criteria                 │
│ - Context: ~200 tokens per diagnosis     │
└──────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────┐
│ Stage 4: Differential (Optional)         │
│ - Compare 2-3 diagnoses side-by-side     │
│ - Only if agent requests                 │
└──────────────────────────────────────────┘
```

### Why Multi-Stage?

**Problem**: qwen3:8b has limited context window and reasoning capacity
**Solution**: Break into small, focused retrieval steps

**Analogy**: Like a medical student using a textbook:
1. First check index for relevant chapters (Stage 1)
2. Skim those chapters for best match (Stage 2)
3. Read one chapter carefully (Stage 3)
4. Compare with differential diagnosis section (Stage 4)

### Implementation

Use `HybridRetriever` from `concisum/diagnosis/retrieval.py`:

```python
# In diagnosis/agents.py
from concisum.diagnosis.retrieval import HybridRetriever, RetrievalConfig

config = RetrievalConfig(
    semantic_top_k=5,           # Stage 1: 5 candidates
    use_keyword_boost=True,     # Boost exact clinical terms
    use_reranking=True,         # Stage 2: re-rank by coverage
    rerank_top_k=3              # Final: 3 best
)

retriever = HybridRetriever(vectorstore, config)
candidates = await retriever.retrieve_for_symptoms(symptoms)
```

## 🛠️ Recommended Tools for Constrained Models

### Current Tools (Baseline)

```python
# tools.py
search_icd10_by_symptoms(symptom_query: str) → str
get_icd10_criteria(icd_code: str) → str
```

**Problem**: Returns all criteria at once (large context)

### Recommended Tools (Staged)

```python
# tools_advanced.py

# Tool 1: Minimal context
search_candidate_diagnoses(symptom_summary: str) → List[str]
# Returns: ["F32.0", "F32.1", "F41.2"]
# Context: ~20 tokens

# Tool 2: Focused validation
validate_diagnosis_criteria(icd_code: str) → str
# Returns: Full criteria for ONE diagnosis
# Context: ~200 tokens

# Tool 3: Differential (optional)
compare_differential_diagnoses(codes: List[str]) → str
# Returns: Side-by-side comparison of 2-3 codes
# Context: ~300 tokens
```

**Benefit**: Agent controls context usage based on reasoning needs

### Agent Workflow Example

```
Agent sees symptoms: "depressive Stimmung, Schlafstörungen, seit 3 Wochen"

Agent: search_candidate_diagnoses("depression sleep problems 3 weeks")
→ Returns: ["F32.0", "F32.1", "F33.1"]

Agent: validate_diagnosis_criteria("F32.1")
→ Returns: Full criteria for F32.1

Agent: [checks criteria against symptoms]

Agent: compare_differential_diagnoses(["F32.1", "F32.0"])
→ Returns: Key differences

Agent: [makes final decision] → F32.1
```

**Total context**: ~500 tokens (vs ~1500 tokens with current approach)

## 📦 Recommended Population Strategy

### Data Organization

```
data/
├── icd10_core/                    # Core diagnoses (official ICD-10)
│   ├── F20-F29_schizophrenia.json
│   ├── F30-F39_mood.json
│   └── F40-F48_anxiety.json
│
├── icd10_clinical/                # Clinical annotations
│   ├── depression_variants.json  # Real-world symptom variations
│   └── anxiety_presentations.json
│
├── icd10_differential/            # Differential diagnosis guides
│   └── mood_vs_anxiety.json
│
└── metadata.json                  # Track versions, sources
```

### Population Scripts

```bash
# Full rebuild (first time)
python scripts/init_vectorstore.py

# Incremental updates (research iteration)
python scripts/update_vectorstore.py data/icd10_core/F30-F39_mood.json --mode upsert

# Add clinical variants
python scripts/update_vectorstore.py data/icd10_clinical/depression_variants.json --mode upsert
```

### Versioning for Research

```python
# metadata.json
{
  "vectorstore_version": "2.1",
  "last_updated": "2025-01-15",
  "data_sources": {
    "icd10_core": {
      "version": "ICD-10-GM-2024",
      "entries": 45,
      "last_update": "2025-01-10"
    },
    "icd10_clinical": {
      "version": "custom_v1",
      "entries": 12,
      "last_update": "2025-01-15"
    }
  }
}
```

**Why**: Track which data version produced which results in experiments

## 🧪 Evaluation & Iteration

### Key Metrics for Research

1. **Retrieval Metrics**:
   - Precision@3: Is correct diagnosis in top 3? (Target: >80%)
   - MRR (Mean Reciprocal Rank): Position of correct diagnosis (Target: >0.7)
   - Coverage: Are all symptom-matching criteria retrieved?

2. **Diagnosis Metrics**:
   - Accuracy: Correct ICD-10 code (Target: >75% for qwen3:8b)
   - Justification quality: Does reasoning cite retrieved criteria?
   - Differential quality: Are alternatives reasonable?

3. **Efficiency Metrics**:
   - Context tokens per diagnosis (Target: <1000 for qwen3:8b)
   - Retrieval calls per diagnosis (Target: 2-4)
   - End-to-end latency (Target: <30s local)

### A/B Testing Setup

```python
# config.py
RETRIEVAL_CONFIGS = {
    "baseline": {
        "granularity": "diagnosis",
        "strategy": "semantic_only",
        "top_k": 3
    },
    "hybrid": {
        "granularity": "diagnosis",
        "strategy": "hybrid",  # semantic + keyword
        "top_k": 5,
        "rerank": True
    },
    "criteria_level": {
        "granularity": "criteria",
        "strategy": "hybrid",
        "top_k": 10,  # More retrieval, less context each
        "rerank": True
    }
}

# Run experiments
config = RETRIEVAL_CONFIGS[os.getenv("EXPERIMENT", "baseline")]
```

### Test Suite

```python
# tests/test_retrieval_quality.py
GOLD_STANDARD_CASES = [
    {
        "case_id": "depression_001",
        "symptoms": ["depressive Stimmung", "Schlafstörungen", "Appetitverlust"],
        "gold_diagnosis": "F32.1",
        "should_retrieve_top3": ["F32.0", "F32.1", "F32.2"],
        "should_not_retrieve": ["F20.0", "F41.0"]
    },
    # ... more cases
]

def test_retrieval_precision():
    for case in GOLD_STANDARD_CASES:
        results = retriever.retrieve_for_symptoms(case["symptoms"])
        top3_codes = [r.code for r in results[:3]]
        assert case["gold_diagnosis"] in top3_codes
```

## 🚀 Quick Start Recommendations

### Phase 1: Immediate (Do Today)

1. **Integrate `HybridRetriever`**:
   ```bash
   # Modify diagnosis/agents.py to use retrieval.py
   # Test: Does keyword boosting help?
   ```
   **Expected**: 10-15% better retrieval precision

2. **Add logging**:
   ```python
   logger.info(f"Retrieved: {[c.code for c in candidates]}")
   logger.info(f"Final diagnosis: {diagnosis.icd_10_diagnose}")
   ```
   **Goal**: Understand what's being retrieved vs used

### Phase 2: This Week

3. **Create 5-10 test cases** with ground truth diagnoses
   - Use your existing therapy transcripts
   - Manually annotate correct ICD-10 codes
   - Measure baseline accuracy

4. **Run A/B test**:
   - Baseline vs Hybrid retrieval
   - Compare on test cases
   - Measure: precision, context size, diagnosis accuracy

### Phase 3: Next Sprint

5. **Restructure data** to criteria-level (if metrics show context is a bottleneck)
6. **Implement staged tools** from `tools_advanced.py`
7. **Add query expansion** for common symptom synonyms

## 📋 Decision Matrix

| Your Situation | Recommended Approach |
|----------------|---------------------|
| **Context window struggles** | → Criteria-level granularity |
| **Missing exact terms** | → Hybrid retrieval (keyword boost) |
| **Poor differential diagnosis** | → Staged tools + comparison |
| **Limited training data** | → Clinical variants in DB |
| **Research experimentation** | → A/B testing + metrics logging |

## 🎓 Best Practices Summary

1. **Start simple, measure, iterate** (current → hybrid → criteria → staged)
2. **Optimize for small models** (less context > more retrieval precision)
3. **Track everything** (log retrievals, measure metrics, version data)
4. **Test with real cases** (ground truth > synthetic data)
5. **Multi-stage > single large retrieval** (mimics clinical reasoning)

## 📚 Implementation Files Reference

- `concisum/diagnosis/retrieval.py` - Hybrid retrieval logic
- `concisum/diagnosis/tools_advanced.py` - Staged tools for small models
- `scripts/update_vectorstore.py` - Incremental data updates
- `data/icd10_criteria_level_example.json` - Example data structure
- `MIGRATION_GUIDE.md` - Step-by-step migration path

---

**Bottom Line for Your Research Goals**:

✅ **Do**: Hybrid retrieval (quick win, minimal effort)
✅ **Consider**: Criteria-level granularity (if context is bottleneck)
✅ **Experiment**: Staged tools (for complex differential diagnosis)
⚠️ **Measure**: Always A/B test changes with ground truth
❌ **Avoid**: Over-engineering before measuring baseline
