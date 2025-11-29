"""Integration tests for RAG components."""

import tempfile
from pathlib import Path


class TestVectorStoreBasic:
    """Basic tests for ICD10VectorStore that don't require model downloads."""

    def test_imports(self):
        """Test that all RAG modules can be imported."""
        from concisum.diagnosis.vectorstore import ICD10VectorStore, ICD10Entry
        from concisum.diagnosis.tools import DiagnosisContext, search_icd10_by_symptoms
        from concisum.diagnosis.agents import DiagnosisOrchestrator

        assert ICD10VectorStore is not None
        assert ICD10Entry is not None
        assert DiagnosisContext is not None
        assert DiagnosisOrchestrator is not None

    def test_icd10_entry_model(self):
        """Test ICD10Entry Pydantic model."""
        from concisum.diagnosis.vectorstore import ICD10Entry

        entry = ICD10Entry(
            code="F32.0",
            title="Test Diagnosis",
            description="Test description",
            criteria=["Criterion 1", "Criterion 2"],
            differential=["F32.1"],
        )

        assert entry.code == "F32.0"
        assert entry.title == "Test Diagnosis"
        assert len(entry.criteria) == 2
        assert "F32.1" in entry.differential

    def test_diagnosis_orchestrator_init_no_rag(self):
        """Test DiagnosisOrchestrator can be initialized without RAG."""
        from concisum.diagnosis.agents import DiagnosisOrchestrator

        # Initialize without RAG (should not fail)
        orchestrator = DiagnosisOrchestrator(use_rag=False)

        assert orchestrator.use_rag is False
        assert orchestrator.vectorstore is None

    def test_diagnosis_orchestrator_init_with_rag_graceful_fail(self):
        """Test DiagnosisOrchestrator gracefully handles missing vectorstore."""
        from concisum.diagnosis.agents import DiagnosisOrchestrator

        # Initialize with RAG but non-existent path (should fall back to non-RAG)
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = DiagnosisOrchestrator(
                use_rag=True, vectorstore_path=f"{tmpdir}/nonexistent"
            )

            # Should initialize successfully with empty vectorstore
            assert orchestrator.vectorstore is not None


class TestRetrievalComponents:
    """Tests for advanced retrieval components (optional features)."""

    def test_retrieval_config(self):
        """Test RetrievalConfig dataclass."""
        from concisum.diagnosis.retrieval import RetrievalConfig

        config = RetrievalConfig(
            semantic_top_k=5,
            use_keyword_boost=True,
            use_reranking=True,
            rerank_top_k=3,
        )

        assert config.semantic_top_k == 5
        assert config.use_keyword_boost is True
        assert config.rerank_top_k == 3

    def test_hybrid_retriever_init(self):
        """Test HybridRetriever can be initialized."""
        from concisum.diagnosis.retrieval import HybridRetriever, RetrievalConfig
        from concisum.diagnosis.vectorstore import ICD10VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            vectorstore = ICD10VectorStore(persist_dir=tmpdir)
            config = RetrievalConfig()

            retriever = HybridRetriever(vectorstore, config)

            assert retriever.vectorstore is not None
            assert retriever.config is not None


class TestDataStructures:
    """Tests for sample data structures."""

    def test_sample_data_exists(self):
        """Test that sample ICD-10 data file exists."""
        sample_file = Path("data/icd10_sample.json")
        assert sample_file.exists(), "Sample ICD-10 data file should exist"

    def test_criteria_level_example_exists(self):
        """Test that criteria-level example file exists."""
        example_file = Path("data/icd10_criteria_level_example.json")
        assert example_file.exists(), "Criteria-level example should exist"

    def test_sample_data_valid_json(self):
        """Test that sample data is valid JSON."""
        import json

        sample_file = Path("data/icd10_sample.json")
        with open(sample_file, "r") as f:
            data = json.load(f)

        assert "entries" in data
        assert len(data["entries"]) > 0

        # Check first entry has required fields
        entry = data["entries"][0]
        assert "code" in entry
        assert "title" in entry
        assert "description" in entry
        assert "criteria" in entry
