"""
Advanced Retrieval System for Multimodal RAG

This module provides sophisticated retrieval capabilities combining:
- Dense (vector-based) retrieval for semantic understanding
- Sparse (keyword-based) retrieval for exact matching
- Hybrid fusion strategies for optimal results
- Advanced query processing and enhancement
"""
from typing import Dict, List

from .base_retriever import BaseRetriever, Query, RetrievalResult
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from .hybrid_retriever import HybridRetriever
from .query_processor import QueryProcessor


# Convenience function for creating retrievers
def create_retriever(retriever_type: str = "hybrid",
                     vector_store=None,
                     embedder=None,
                     config: Dict = None) -> BaseRetriever:
    """
    Factory function to create retrievers

    Args:
        retriever_type: 'dense', 'sparse', or 'hybrid'
        vector_store: Vector store instance (required for dense/hybrid)
        embedder: Embedder instance (required for dense/hybrid)
        config: Configuration dictionary
    """

    if retriever_type == "dense":
        if not vector_store or not embedder:
            raise ValueError("Dense retriever requires vector_store and embedder")
        return DenseRetriever(vector_store, embedder, config)

    elif retriever_type == "sparse":
        return SparseRetriever(config)

    elif retriever_type == "hybrid":
        if not vector_store or not embedder:
            raise ValueError("Hybrid retriever requires vector_store and embedder")
        return HybridRetriever(vector_store, embedder, config)

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


# Integration helper for complete pipeline
class RetrievalPipeline:
    """
    Complete retrieval pipeline that orchestrates query processing,
    retrieval, and result post-processing
    """

    def __init__(self, retriever: BaseRetriever, config: Dict = None):
        self.retriever = retriever
        self.config = config or {}

        # Initialize query processor
        self.query_processor = QueryProcessor(self.config.get('query_processing', {}))

        # Post-processing configuration
        self.enable_reranking = self.config.get('enable_reranking', True)
        self.enable_diversity = self.config.get('enable_diversity', True)
        self.max_results = self.config.get('max_results', 50)

    def search(self, query_text: str,
               top_k: int = 10,
               query_metadata: Dict = None) -> List[RetrievalResult]:
        """
        Complete search pipeline with query processing and result enhancement
        """

        # Process query
        processed_query = self.query_processor.process_query(query_text, query_metadata)

        # Retrieve results
        results = self.retriever.retrieve(processed_query, min(top_k * 2, self.max_results))

        # Post-process results
        if self.enable_reranking:
            results = self._rerank_results(results, processed_query)

        if self.enable_diversity:
            results = self._apply_diversity_filtering(results)

        # Take final top-k results
        final_results = results[:top_k]

        # Add pipeline metadata
        for result in final_results:
            result.metadata['pipeline_processed'] = True
            result.metadata['original_query'] = query_text
            result.metadata['processed_query'] = processed_query.text
            result.metadata['query_intent'] = processed_query.metadata.get('intent', 'unknown')

        return final_results

    def _rerank_results(self, results: List[RetrievalResult], query: Query) -> List[RetrievalResult]:
        """Apply additional re-ranking based on query intent and content analysis"""

        intent = query.metadata.get('intent', 'general')

        for result in results:
            # Boost based on intent matching
            content_lower = result.content.lower()

            if intent == 'definition' and any(
                    word in content_lower for word in ['define', 'definition', 'is a', 'refers to']):
                result.combined_score *= 1.2
            elif intent == 'procedure' and any(
                    word in content_lower for word in ['step', 'process', 'method', 'procedure']):
                result.combined_score *= 1.2
            elif intent == 'comparison' and any(
                    word in content_lower for word in ['compare', 'versus', 'difference', 'better']):
                result.combined_score *= 1.2

        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

    def _apply_diversity_filtering(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply diversity filtering to reduce redundant results"""

        if len(results) <= 1:
            return results

        diverse_results = [results[0]]  # Always include top result

        for result in results[1:]:
            # Check similarity to already selected results
            is_diverse = True

            for selected in diverse_results:
                # Check parent document similarity
                if (result.parent_document_id and
                        result.parent_document_id == selected.parent_document_id):
                    # Same document, check content overlap
                    content_similarity = self._calculate_content_overlap(
                        result.content, selected.content
                    )
                    if content_similarity > 0.7:  # 70% overlap threshold
                        is_diverse = False
                        break

            if is_diverse:
                diverse_results.append(result)

        return diverse_results

    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """Calculate content overlap between two text snippets"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        smaller_set = min(len(words1), len(words2))

        return len(intersection) / smaller_set


__all__ = [
    'BaseRetriever',
    'Query',
    'RetrievalResult',
    'DenseRetriever',
    'SparseRetriever',
    'HybridRetriever',
    'QueryProcessor',
    'create_retriever',
    'RetrievalPipeline'
]