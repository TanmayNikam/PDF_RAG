"""
Dense (vector-based) retrieval using embeddings and vector stores
"""

import time
from typing import List, Dict, Optional
import numpy as np
from src.embeddings import MultimodalContent
from .base_retriever import BaseRetriever, Query, RetrievalResult
import traceback


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using vector embeddings and similarity search

    This is our primary semantic search engine that uses the vector store
    and embedding system we built earlier.
    """

    def __init__(self, vector_store, embedder, config: Dict = None):
        super().__init__("dense_retriever", config)

        self.vector_store = vector_store
        self.embedder = embedder

        # Dense retrieval configuration
        self.similarity_threshold = self.config.get('similarity_threshold', 0.0)
        self.rerank_results = self.config.get('rerank_results', True)
        self.boost_recent = self.config.get('boost_recent', False)
        self.diversity_penalty = self.config.get('diversity_penalty', 0.1)

        self.logger.info(f"✅ Initialized Dense Retriever with {vector_store.document_count} documents")

    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using dense vector similarity"""
        start_time = time.time()

        try:
            # Generate query embedding if not provided
            if query.embedding is None:
                if query.query_type == "multimodal":
                    # Handle multimodal queries
                    multimodal_content = MultimodalContent(text=query.text)
                    embedding_result = self.embedder.embed(multimodal_content)
                else:
                    # Standard text query
                    embedding_result = self.embedder.embed_text_only(query.text)

                query.embedding = embedding_result.embedding

            # Search vector store
            search_results = self.vector_store.search(
                query_embedding=query.embedding,
                top_k=min(top_k * 2, 100),  # Get extra for post-processing
                filters=query.filters
            )

            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, result in enumerate(search_results):
                retrieval_result = RetrievalResult(
                    document_id=result.document.id,
                    content=result.document.content,
                    score=result.score,
                    rank=i,
                    retrieval_method="dense_vector",
                    metadata=result.document.metadata.copy(),
                    chunk_metadata=result.search_metadata,
                    parent_document_id=result.document.parent_document_id,
                    relevance_score=result.score
                )

                # Apply similarity threshold
                if retrieval_result.score >= self.similarity_threshold:
                    retrieval_results.append(retrieval_result)

            # Post-process results
            if self.rerank_results:
                retrieval_results = self._rerank_results(retrieval_results, query)

            # Apply diversity penalty if enabled
            if self.diversity_penalty > 0:
                retrieval_results = self._apply_diversity_penalty(retrieval_results)

            # Take top K results and update ranks
            final_results = retrieval_results[:top_k]
            for i, result in enumerate(final_results):
                result.rank = i
                result.combined_score = result.relevance_score

            # Update statistics
            query_time = time.time() - start_time
            self.update_stats(query_time)

            self.logger.debug(f"Dense retrieval: {len(final_results)} results in {query_time:.3f}s")
            return final_results

        except Exception as e:
            self.logger.error(f"Dense retrieval failed: {e}")
            traceback.print_exc()
            raise

    def _rerank_results(self, results: List[RetrievalResult], query: Query) -> List[RetrievalResult]:
        """Apply re-ranking to improve result quality"""

        for result in results:
            # Start with relevance score
            combined_score = result.relevance_score

            # Boost based on content type
            content_type = result.metadata.get('content_type', 'text')
            type_boost = self.config.get('content_type_boost', {}).get(content_type, 1.0)
            combined_score *= type_boost

            # Boost recent documents if enabled
            if self.boost_recent and 'timestamp' in result.metadata:
                recency_score = self._calculate_recency_score(result.metadata['timestamp'])
                combined_score *= (1.0 + recency_score * 0.1)
                result.recency_score = recency_score

            # Boost based on metadata matching
            metadata_boost = self._calculate_metadata_boost(result, query)
            combined_score *= metadata_boost

            result.combined_score = combined_score

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

    def _apply_diversity_penalty(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply diversity penalty to reduce redundant results"""

        if len(results) <= 1:
            return results

        diverse_results = [results[0]]  # Always include top result

        for result in results[1:]:
            # Calculate similarity to already selected results
            max_similarity = 0.0

            for selected in diverse_results:
                # Simple content-based similarity (can be enhanced with embeddings)
                content_similarity = self._calculate_content_similarity(
                    result.content, selected.content
                )
                max_similarity = max(max_similarity, content_similarity)

            # Apply diversity penalty
            diversity_score = 1.0 - (max_similarity * self.diversity_penalty)
            result.diversity_score = diversity_score
            result.combined_score *= diversity_score

            diverse_results.append(result)

        # Re-sort by adjusted scores
        diverse_results.sort(key=lambda x: x.combined_score, reverse=True)
        return diverse_results

    def _calculate_recency_score(self, timestamp: float) -> float:
        """Calculate recency score based on timestamp"""
        current_time = time.time()
        age_hours = (current_time - timestamp) / 3600

        # Exponential decay over 7 days
        max_age_hours = 24 * 7
        recency_score = max(0.0, 1.0 - (age_hours / max_age_hours))

        return recency_score

    def _calculate_metadata_boost(self, result: RetrievalResult, query: Query) -> float:
        """Calculate boost based on metadata matching"""
        boost = 1.0

        # Check for exact metadata matches
        for key, value in query.metadata.items():
            if key in result.metadata and result.metadata[key] == value:
                boost *= 1.1  # 10% boost for each exact match

        return boost

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Simple content similarity calculation"""
        # Convert to word sets for Jaccard similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)