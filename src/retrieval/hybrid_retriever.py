"""
Hybrid retrieval combining dense and sparse methods
"""

import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import re

from .base_retriever import BaseRetriever, Query, RetrievalResult
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval that intelligently combines dense and sparse retrieval

    This is the main retrieval engine that provides the best of both worlds:
    - Semantic understanding from dense retrieval
    - Exact keyword matching from sparse retrieval
    """

    def __init__(self, vector_store, embedder, config: Dict = None):
        super().__init__("hybrid_retriever", config)

        # Initialize component retrievers
        dense_config = self.config.get('dense', {})
        sparse_config = self.config.get('sparse', {})

        self.dense_retriever = DenseRetriever(vector_store, embedder, dense_config)
        self.sparse_retriever = SparseRetriever(sparse_config)

        # Hybrid configuration
        self.dense_weight = self.config.get('dense_weight', 0.7)
        self.sparse_weight = self.config.get('sparse_weight', 0.3)
        self.fusion_method = self.config.get('fusion_method', 'weighted_sum')  # weighted_sum, rrf, linear_combination
        self.adaptive_weights = self.config.get('adaptive_weights', True)

        # Query analysis for adaptive weighting
        self.keyword_indicators = ['name', 'who', 'what', 'when', 'where', 'list', 'define']
        self.semantic_indicators = ['similar', 'like', 'about', 'explain', 'understand', 'meaning']

        # Performance tracking
        self.fusion_stats = {
            'dense_preferred': 0,
            'sparse_preferred': 0,
            'balanced': 0
        }

        self.logger.info("Initialized Hybrid Retriever")

    def add_documents(self, documents: List[Dict]):
        """Add documents to both dense and sparse indexes"""
        # Documents are already in vector store for dense retrieval
        # Add to sparse retriever
        self.sparse_retriever.add_documents(documents)

        self.logger.info(f"Added {len(documents)} documents to hybrid retriever")


    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve using hybrid approach"""
        start_time = time.time()

        try:
            # Analyze query to determine optimal weighting
            if self.adaptive_weights:
                dense_weight, sparse_weight = self._analyze_query_weights(query)
            else:
                dense_weight, sparse_weight = self.dense_weight, self.sparse_weight

            # Get results from both retrievers
            # Retrieve more results for better fusion
            retrieval_k = min(top_k * 3, 50)

            dense_results = self.dense_retriever.retrieve(query, retrieval_k)
            sparse_results = self.sparse_retriever.retrieve(query, retrieval_k)

            # Fuse results
            fused_results = self._fuse_results(
                dense_results, sparse_results,
                dense_weight, sparse_weight,
                query
            )

            # Take top K and update metadata
            final_results = fused_results[:top_k]
            for i, result in enumerate(final_results):
                result.rank = i
                result.metadata['fusion_method'] = self.fusion_method
                result.metadata['dense_weight'] = dense_weight
                result.metadata['sparse_weight'] = sparse_weight

            # Update statistics
            query_time = time.time() - start_time
            self.update_stats(query_time)

            # Update fusion stats
            if dense_weight > sparse_weight + 0.2:
                self.fusion_stats['dense_preferred'] += 1
            elif sparse_weight > dense_weight + 0.2:
                self.fusion_stats['sparse_preferred'] += 1
            else:
                self.fusion_stats['balanced'] += 1

            self.logger.debug(f"Hybrid retrieval: {len(final_results)} results in {query_time:.3f}s")
            self.logger.debug(f"Weights used: dense={dense_weight:.2f}, sparse={sparse_weight:.2f}")

            return final_results

        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed: {e}")
            raise

    def _analyze_query_weights(self, query: Query) -> Tuple[float, float]:
        """Analyze query to determine optimal dense/sparse weighting"""
        query_text = query.text.lower()

        # Count indicators
        keyword_score = sum(1 for indicator in self.keyword_indicators if indicator in query_text)
        semantic_score = sum(1 for indicator in self.semantic_indicators if indicator in query_text)

        # Check for quoted phrases (favor sparse)
        quoted_phrases = len(re.findall(r'"[^"]*"', query.text))
        if quoted_phrases > 0:
            keyword_score += quoted_phrases * 2

        # Check for named entities (simple heuristic)
        capitalized_words = len(re.findall(r'\b[A-Z][a-z]+\b', query.text))
        if capitalized_words > 1:
            keyword_score += 1

        # Check query length (longer queries often benefit from semantic search)
        word_count = len(query.text.split())
        if word_count > 10:
            semantic_score += 1
        elif word_count < 5:
            keyword_score += 1

        # Calculate adaptive weights
        total_indicators = keyword_score + semantic_score

        if total_indicators == 0:
            # Default weights
            return self.dense_weight, self.sparse_weight

        # Adjust weights based on indicators
        keyword_ratio = keyword_score / total_indicators
        semantic_ratio = semantic_score / total_indicators

        # Interpolate between default weights and indicator-based weights
        alpha = 0.3  # How much to adjust from defaults

        dense_weight = self.dense_weight * (1 - alpha) + (0.3 + 0.7 * semantic_ratio) * alpha
        sparse_weight = self.sparse_weight * (1 - alpha) + (0.7 * keyword_ratio + 0.3) * alpha

        # Normalize weights
        total_weight = dense_weight + sparse_weight
        dense_weight /= total_weight
        sparse_weight /= total_weight

        return dense_weight, sparse_weight

    def _fuse_results(self, dense_results: List[RetrievalResult],
                      sparse_results: List[RetrievalResult],
                      dense_weight: float, sparse_weight: float,
                      query: Query) -> List[RetrievalResult]:
        """Fuse results from dense and sparse retrievers"""

        if self.fusion_method == 'weighted_sum':
            return self._weighted_sum_fusion(dense_results, sparse_results, dense_weight, sparse_weight)
        elif self.fusion_method == 'rrf':
            return self._reciprocal_rank_fusion(dense_results, sparse_results)
        elif self.fusion_method == 'linear_combination':
            return self._linear_combination_fusion(dense_results, sparse_results, dense_weight, sparse_weight)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _weighted_sum_fusion(self, dense_results: List[RetrievalResult],
                             sparse_results: List[RetrievalResult],
                             dense_weight: float, sparse_weight: float) -> List[RetrievalResult]:
        """Fuse results using weighted sum of normalized scores"""

        # Normalize scores within each result set
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)

        # Create document score map
        doc_scores = {}
        doc_results = {}

        # Add dense results
        for result in dense_results:
            doc_id = result.document_id
            weighted_score = result.score * dense_weight
            doc_scores[doc_id] = weighted_score
            doc_results[doc_id] = result
            doc_results[doc_id].metadata['dense_score'] = result.score
            doc_results[doc_id].metadata['dense_rank'] = result.rank

        # Add sparse results (combining scores if document already exists)
        for result in sparse_results:
            doc_id = result.document_id
            weighted_score = result.score * sparse_weight

            if doc_id in doc_scores:
                doc_scores[doc_id] += weighted_score
                doc_results[doc_id].metadata['sparse_score'] = result.score
                doc_results[doc_id].metadata['sparse_rank'] = result.rank
                doc_results[doc_id].metadata['found_in_both'] = True
            else:
                doc_scores[doc_id] = weighted_score
                doc_results[doc_id] = result
                doc_results[doc_id].metadata['sparse_score'] = result.score
                doc_results[doc_id].metadata['sparse_rank'] = result.rank
                doc_results[doc_id].metadata['found_in_both'] = False

        # Create fused results
        fused_results = []
        for doc_id, combined_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_results[doc_id]
            result.score = combined_score
            result.combined_score = combined_score
            result.retrieval_method = "hybrid_weighted_sum"
            fused_results.append(result)

        return fused_results

    def _reciprocal_rank_fusion(self, dense_results: List[RetrievalResult],
                                sparse_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion (RRF)"""

        k = 60  # RRF parameter
        doc_scores = {}
        doc_results = {}

        # Add dense results
        for rank, result in enumerate(dense_results):
            doc_id = result.document_id
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = rrf_score
            doc_results[doc_id] = result
            doc_results[doc_id].metadata['dense_rank'] = rank

        # Add sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result.document_id
            rrf_score = 1.0 / (k + rank + 1)

            if doc_id in doc_scores:
                doc_scores[doc_id] += rrf_score
                doc_results[doc_id].metadata['sparse_rank'] = rank
                doc_results[doc_id].metadata['found_in_both'] = True
            else:
                doc_scores[doc_id] = rrf_score
                doc_results[doc_id] = result
                doc_results[doc_id].metadata['sparse_rank'] = rank
                doc_results[doc_id].metadata['found_in_both'] = False

        # Create fused results
        fused_results = []
        for doc_id, rrf_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_results[doc_id]
            result.score = rrf_score
            result.combined_score = rrf_score
            result.retrieval_method = "hybrid_rrf"
            fused_results.append(result)

        return fused_results

    def _linear_combination_fusion(self, dense_results: List[RetrievalResult],
                                   sparse_results: List[RetrievalResult],
                                   dense_weight: float, sparse_weight: float) -> List[RetrievalResult]:
        """Fuse results using linear combination of raw scores"""

        doc_scores = {}
        doc_results = {}

        # Add dense results
        for result in dense_results:
            doc_id = result.document_id
            doc_scores[doc_id] = result.score * dense_weight
            doc_results[doc_id] = result

        # Add sparse results
        for result in sparse_results:
            doc_id = result.document_id
            if doc_id in doc_scores:
                doc_scores[doc_id] += result.score * sparse_weight
            else:
                doc_scores[doc_id] = result.score * sparse_weight
                doc_results[doc_id] = result

        # Create fused results
        fused_results = []
        for doc_id, combined_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_results[doc_id]
            result.score = combined_score
            result.combined_score = combined_score
            result.retrieval_method = "hybrid_linear_combination"
            fused_results.append(result)

        return fused_results

    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to [0, 1] range"""
        if not results:
            return results

        scores = [result.score for result in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same
            for result in results:
                result.score = 1.0
        else:
            # Min-max normalization
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)

        return results

    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics from all components"""
        return {
            'hybrid': self.get_stats(),
            'dense': self.dense_retriever.get_stats(),
            'sparse': self.sparse_retriever.get_stats(),
            'fusion_stats': self.fusion_stats.copy(),
            # 'sparse_vocabulary': self.sparse_retriever.get_vocabulary_stats()
        }

