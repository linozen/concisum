"""Vector store for ICD-10 diagnostic criteria using ChromaDB."""

import logging
from pathlib import Path
from typing import List, Optional
import json

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ICD10Entry(BaseModel):
    """Represents an ICD-10 diagnostic entry with criteria."""

    code: str = Field(description="ICD-10 code (e.g., 'F32.1')")
    title: str = Field(description="Diagnostic title in German")
    description: str = Field(description="Short description of the disorder")
    criteria: List[str] = Field(
        description="List of diagnostic criteria", default_factory=list
    )
    differential: List[str] = Field(
        description="Differential diagnostic considerations", default_factory=list
    )


class ICD10VectorStore:
    """
    Vector store for ICD-10 F-codes using ChromaDB.

    Provides semantic search over diagnostic criteria and exact code lookup.
    Uses sentence-transformers for local, privacy-preserving embeddings.
    """

    def __init__(self, persist_dir: str = "./data/icd10_db"):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to persist the ChromaDB database
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection with default embedding function
        # ChromaDB uses "all-MiniLM-L6-v2" via ONNX Runtime by default (lightweight, no PyTorch)
        self.collection = self.client.get_or_create_collection(
            name="icd10_f_codes",
            metadata={
                "description": "ICD-10 Chapter V (F00-F99) diagnostic criteria",
                "embedding_function": "default",  # ONNX-based, ~50MB vs 3GB PyTorch
            },
        )

        logger.info(
            f"ICD-10 vector store initialized with {self.collection.count()} entries"
        )

    def populate_from_json(self, json_path: str | Path) -> int:
        """
        Populate the vector store from a JSON file.

        Args:
            json_path: Path to JSON file containing ICD-10 entries

        Returns:
            Number of entries added
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        if not entries:
            logger.warning(f"No entries found in {json_path}")
            return 0

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for entry in entries:
            icd_entry = ICD10Entry(**entry)

            # Create searchable document combining all relevant text
            doc_text = f"{icd_entry.title}\n{icd_entry.description}\n"
            doc_text += "Kriterien:\n" + "\n".join(icd_entry.criteria)

            documents.append(doc_text)
            metadatas.append(icd_entry.model_dump())
            ids.append(icd_entry.code)

        # Add to ChromaDB (handles embedding automatically)
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        logger.info(f"Added {len(entries)} ICD-10 entries to vector store")
        return len(entries)

    async def search_by_symptoms(
        self, symptom_descriptions: List[str], top_k: int = 3
    ) -> List[ICD10Entry]:
        """
        Semantic search for ICD-10 codes based on symptom descriptions.

        Args:
            symptom_descriptions: List of symptom descriptions to search for
            top_k: Number of top results to return

        Returns:
            List of matching ICD-10 entries, ranked by relevance
        """
        if not symptom_descriptions:
            logger.warning("No symptoms provided for search")
            return []

        # Combine symptom descriptions into search query
        query = " ".join(symptom_descriptions)

        # Perform semantic search
        results = self.collection.query(query_texts=[query], n_results=top_k)

        # Convert results to ICD10Entry objects
        entries = []
        if results["metadatas"] and results["metadatas"][0]:
            for metadata in results["metadatas"][0]:
                try:
                    entries.append(ICD10Entry(**metadata))
                except Exception as e:
                    logger.warning(f"Error parsing metadata: {e}")
                    continue

        logger.info(
            f"Found {len(entries)} diagnoses for symptoms: {symptom_descriptions[:3]}..."
        )
        return entries

    async def get_exact_criteria(self, icd_code: str) -> Optional[ICD10Entry]:
        """
        Get exact diagnostic criteria for a specific ICD-10 code.

        Args:
            icd_code: ICD-10 code (e.g., 'F32.1')

        Returns:
            ICD10Entry if found, None otherwise
        """
        try:
            results = self.collection.get(ids=[icd_code])

            if results["metadatas"] and len(results["metadatas"]) > 0:
                return ICD10Entry(**results["metadatas"][0])

            logger.warning(f"ICD-10 code {icd_code} not found in vector store")
            return None

        except Exception as e:
            logger.error(f"Error retrieving ICD-10 code {icd_code}: {e}")
            return None

    def get_all_codes(self) -> List[str]:
        """
        Get list of all ICD-10 codes in the vector store.

        Returns:
            List of ICD-10 codes
        """
        results = self.collection.get()
        return results["ids"] if results["ids"] else []

    def clear(self):
        """Clear all entries from the vector store."""
        self.client.delete_collection("icd10_f_codes")
        self.collection = self.client.get_or_create_collection(
            name="icd10_f_codes",
            metadata={"description": "ICD-10 Chapter V (F00-F99) diagnostic criteria"},
        )
        logger.info("Vector store cleared")
