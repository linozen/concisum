"""Advanced retrieval strategies for RAG-enhanced diagnosis."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from concisum.diagnosis.vectorstore import ICD10VectorStore, ICD10Entry
from concisum.diagnosis.models import Symptom

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategies."""

    # Semantic search parameters
    semantic_top_k: int = 5
    semantic_threshold: float = 0.0  # Minimum similarity score

    # Keyword filtering
    use_keyword_boost: bool = True
    keyword_boost_factor: float = 1.5

    # Re-ranking
    use_reranking: bool = True
    rerank_top_k: int = 3

    # Query expansion
    expand_symptoms: bool = False


class HybridRetriever:
    """
    Hybrid retrieval combining semantic search with keyword filtering.

    Optimized for constrained local models that benefit from precise context.
    """

    def __init__(
        self, vectorstore: ICD10VectorStore, config: Optional[RetrievalConfig] = None
    ):
        self.vectorstore = vectorstore
        self.config = config or RetrievalConfig()

    async def retrieve_for_symptoms(
        self, symptoms: List[Symptom]
    ) -> List[ICD10Entry]:
        """
        Multi-stage retrieval optimized for small models.

        Stage 1: Semantic search over diagnoses
        Stage 2: Keyword filtering (boost exact symptom matches)
        Stage 3: Re-ranking by criteria coverage
        """
        if not symptoms:
            return []

        # Extract symptom names and descriptions
        symptom_names = [s.name for s in symptoms]
        symptom_descriptions = [
            f"{s.name}: {s.description}" for s in symptoms if s.description
        ]

        # Stage 1: Semantic search
        query = " ".join(symptom_descriptions or symptom_names)
        logger.info(f"Semantic search query: {query[:100]}...")

        candidates = await self.vectorstore.search_by_symptoms(
            symptom_descriptions or symptom_names, top_k=self.config.semantic_top_k
        )

        if not candidates:
            logger.warning("No semantic matches found")
            return []

        logger.info(f"Found {len(candidates)} semantic candidates")

        # Stage 2: Keyword boosting (if enabled)
        if self.config.use_keyword_boost:
            candidates = self._keyword_boost(candidates, symptom_names)

        # Stage 3: Re-ranking by criteria coverage (if enabled)
        if self.config.use_reranking:
            candidates = self._rerank_by_coverage(
                candidates, symptoms, top_k=self.config.rerank_top_k
            )

        logger.info(
            f"Final retrieval: {[c.code for c in candidates[:self.config.rerank_top_k]]}"
        )
        return candidates[: self.config.rerank_top_k]

    def _keyword_boost(
        self, candidates: List[ICD10Entry], symptom_names: List[str]
    ) -> List[ICD10Entry]:
        """
        Boost candidates that contain exact symptom keywords.

        Useful because semantic search might miss exact clinical terms.
        """
        keyword_scores: Dict[str, int] = {}

        for candidate in candidates:
            score = 0
            searchable_text = (
                f"{candidate.title} {candidate.description} "
                + " ".join(candidate.criteria)
            ).lower()

            for symptom in symptom_names:
                # Exact match
                if symptom.lower() in searchable_text:
                    score += 2
                # Partial match (for compound terms)
                elif any(word.lower() in searchable_text for word in symptom.split()):
                    score += 1

            keyword_scores[candidate.code] = score

        # Sort by keyword score (descending), preserve semantic order for ties
        candidates_sorted = sorted(
            candidates, key=lambda c: keyword_scores.get(c.code, 0), reverse=True
        )

        logger.info(f"Keyword scores: {keyword_scores}")
        return candidates_sorted

    def _rerank_by_coverage(
        self, candidates: List[ICD10Entry], symptoms: List[Symptom], top_k: int = 3
    ) -> List[ICD10Entry]:
        """
        Re-rank by how many criteria are supported by the symptoms.

        This helps small models focus on diagnoses with strong evidence.
        """
        coverage_scores: Dict[str, float] = {}

        for candidate in candidates:
            # Simple heuristic: count how many criteria mention the symptoms
            matched_criteria = 0
            total_criteria = len(candidate.criteria)

            for criterion in candidate.criteria:
                criterion_lower = criterion.lower()
                for symptom in symptoms:
                    if (
                        symptom.name.lower() in criterion_lower
                        or (
                            symptom.description
                            and any(
                                word.lower() in criterion_lower
                                for word in symptom.description.split()[:5]
                            )
                        )
                    ):
                        matched_criteria += 1
                        break  # Count each criterion only once

            # Coverage ratio
            coverage = matched_criteria / total_criteria if total_criteria > 0 else 0
            coverage_scores[candidate.code] = coverage

        # Sort by coverage (descending)
        candidates_sorted = sorted(
            candidates, key=lambda c: coverage_scores.get(c.code, 0), reverse=True
        )

        logger.info(f"Coverage scores: {coverage_scores}")
        return candidates_sorted


class QueryExpander:
    """
    Expand symptom queries with clinical synonyms/related terms.

    Useful for improving recall with constrained models.
    """

    # Simple synonym mapping (could be expanded with a proper medical thesaurus)
    SYMPTOM_SYNONYMS = {
        "depressive Stimmung": [
            "Niedergeschlagenheit",
            "Traurigkeit",
            "gedrückte Stimmung",
        ],
        "Angst": ["Ängstlichkeit", "Furcht", "Besorgnis", "Panik"],
        "Schlafstörungen": [
            "Insomnie",
            "Einschlafstörungen",
            "Durchschlafstörungen",
        ],
        "Konzentrationsstörungen": [
            "Aufmerksamkeitsstörungen",
            "verminderte Konzentration",
        ],
    }

    @classmethod
    def expand_symptom_query(cls, symptom: Symptom) -> List[str]:
        """
        Expand a symptom with synonyms for better retrieval.

        Returns:
            List of expanded terms including original
        """
        expanded = [symptom.name]

        # Add synonyms if available
        if symptom.name in cls.SYMPTOM_SYNONYMS:
            expanded.extend(cls.SYMPTOM_SYNONYMS[symptom.name])

        # Add key terms from description
        if symptom.description:
            # Extract potential clinical terms (simple heuristic)
            terms = [
                word
                for word in symptom.description.split()
                if len(word) > 6 and word[0].isupper()
            ]
            expanded.extend(terms[:3])  # Add max 3 terms

        return list(set(expanded))  # Remove duplicates
