"""
Hybrid vector store combining multiple backends and custom optimizations
"""
import logging
import time
from typing import List, Dict, Optional, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .base_vector_store import BaseVectorStore, VectorDocument, SearchResult
from .faiss_vector_store import FAISSVectorStore

class HybridVectorStore(BaseVectorStore):
    """
    Advanced vector store that combines multiple approaches:
    1. FAISS for high-performance similarity search
    2. Custom filters and post-processing
    3. Intelligent caching and optimization
    4. Multi-modal document handling
    """

    def __init__(self, dimension: int, config: Dict = None):
        super().__init__(dimension, config)

        self.logger = logging.getLogger(__name__)

        # Initialize primary store (FAISS)
        faiss_config = self.config.get('faiss', {})
        self.primary_store = FAISSVectorStore(dimension, faiss_config)

        # Performance optimizations
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = self.config.get('cache_size', 1000)
        self.enable_parallel_search = self.config.get('enable_parallel_search', True)

        # Search result caching
        if self.enable_caching:
            from functools import lru_cache
            self._cache = {}
            self._cache_lock = threading.RLock()

        # Advanced search features
        self.reranking_enabled = self.config.get('enable_reranking', True)
        self.diversity_boost = self.config.get('diversity_boost', 0.1)

        # Document type handling
        self.content_type_weights = self.config.get('content_type_weights', {
            'text': 1.0,
            'image': 1.0,
            'multimodal': 1.1  # Slight boost for multimodal content
        })

        self.is_initialized = True
        self.logger.info("Initialized Hybrid Vector Store")

    def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents with preprocessing and optimization"""
        if not documents:
            return []

        try:
            # Preprocess documents
            processed_docs = self._preprocess_documents(documents)

            # Add to primary store
            doc_ids = self.primary_store.add_documents(processed_docs)

            # Update our statistics
            self.document_count = self.primary_store.document_count

            # Clear cache if enabled
            if self.enable_caching:
                self._clear_cache()

            self.logger.info(f"Added {len(documents)} documents to hybrid store")
            return doc_ids

        except Exception as e:
            self.logger.error(f"Failed to add documents to hybrid store: {e}")
            raise

    def search(self, query_embedding: np.ndarray,
               top_k: int = 10,
               filters: Dict = None,
               search_params: Dict = None) -> List[SearchResult]:
        """Advanced search with multiple optimizations"""

        search_params = search_params or {}


        # Check cache first
        if self.enable_caching:
            cache_key = self._get_cache_key(query_embedding, top_k, filters)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

        start_time = time.time()

        try:
            # Get more results than requested for post-processing
            search_k = min(top_k * 2, self.document_count)


            # Primary search using FAISS
            primary_results = self.primary_store.search(
                query_embedding, search_k, filters
            )

            # Apply advanced post-processing
            processed_results = self._post_process_results(
                primary_results, query_embedding, search_params
            )

            # Apply content type weighting
            weighted_results = self._apply_content_type_weighting(processed_results)

            # Re-rank if enabled
            if self.reranking_enabled:
                reranked_results = self._rerank_results(
                    weighted_results, query_embedding, search_params
                )
            else:
                reranked_results = weighted_results

            # Apply diversity if requested
            if search_params.get('enable_diversity', False):
                diverse_results = self._apply_diversity_boost(reranked_results)
            else:
                diverse_results = reranked_results

            # Take top K results
            final_results = diverse_results[:top_k]

            # Update ranks
            for i, result in enumerate(final_results):
                result.rank = i
                result.search_metadata.update({
                    'total_search_time': time.time() - start_time,
                    'hybrid_processing': True
                })

            # Cache the result
            if self.enable_caching:
                self._add_to_cache(cache_key, final_results)

            return final_results

        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            raise

    def multi_vector_search(self, query_embeddings: List[np.ndarray],
                            weights: List[float] = None,
                            top_k: int = 10) -> List[SearchResult]:
        """
        Search using multiple query vectors (e.g., text + image) particularly for multimodal queries
        """

        if not query_embeddings:
            return []

        weights = weights or [1.0] * len(query_embeddings)

        if len(weights) != len(query_embeddings):
            raise ValueError("Number of weights must match number of query embeddings")

        try:
            # Perform searches for each query vector
            all_results = []

            if self.enable_parallel_search and len(query_embeddings) > 1:
                # Parallel search
                with ThreadPoolExecutor(max_workers=len(query_embeddings)) as executor:
                    future_to_weight = {
                        executor.submit(self.primary_store.search, qe, top_k * 2): w
                        for qe, w in zip(query_embeddings, weights)
                    }

                    for future in as_completed(future_to_weight):
                        weight = future_to_weight[future]
                        results = future.result()

                        # Apply weight to scores
                        for result in results:
                            result.score *= weight
                            result.search_metadata['query_weight'] = weight

                        all_results.extend(results)
            else:
                # Sequential search
                for query_embedding, weight in zip(query_embeddings, weights):
                    results = self.primary_store.search(query_embedding, top_k * 2)

                    # Apply weight to scores
                    for result in results:
                        result.score *= weight
                        result.search_metadata['query_weight'] = weight

                    all_results.extend(results)

            # Combine and deduplicate results
            combined_results = self._combine_multi_vector_results(all_results)

            # Sort by combined score
            combined_results.sort(key=lambda x: x.score, reverse=True)

            # Take top K and update ranks
            final_results = combined_results[:top_k]
            for i, result in enumerate(final_results):
                result.rank = i
                result.search_metadata['multi_vector_search'] = True

            return final_results

        except Exception as e:
            self.logger.error(f"Multi-vector search failed: {e}")
            raise

    def semantic_search(self, query: str, embedder,
                        top_k: int = 10,
                        search_params: Dict = None) -> List[SearchResult]:
        """
        High-level semantic search that handles embedding generation
        """
        try:
            # Generate query embedding
            query_result = embedder.embed(query)
            query_embedding = query_result.embedding

            # Perform search
            return self.search(query_embedding, top_k, search_params=search_params)

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            raise

    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Retrieve document with caching"""
        return self.primary_store.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and clear cache"""
        success = self.primary_store.delete_document(doc_id)

        if success and self.enable_caching:
            self._clear_cache()
            self.document_count = self.primary_store.document_count

        return success

    def save(self, path: str) -> bool:
        """Save the hybrid vector store"""
        return self.primary_store.save(path)

    def load(self, path: str) -> bool:
        """Load the hybrid vector store"""
        success = self.primary_store.load(path)

        if success:
            self.document_count = self.primary_store.document_count
            if self.enable_caching:
                self._clear_cache()

        return success

    def _preprocess_documents(self, documents: List[VectorDocument]) -> List[VectorDocument]:
        """Preprocess documents before adding to store"""
        processed_docs = []

        for doc in documents:
            # Ensure embedding is normalized for cosine similarity
            if self.config.get('normalize_embeddings', True):
                norm = np.linalg.norm(doc.embedding)
                if norm > 0:
                    doc.embedding = doc.embedding / norm

            # Add preprocessing metadata
            doc.metadata['preprocessing'] = {
                'normalized': True,
                'processed_at': time.time(),
                'hybrid_store': True
            }

            processed_docs.append(doc)

        return processed_docs

    def _post_process_results(self, results: List[SearchResult],
                              query_embedding: np.ndarray,
                              search_params: Dict) -> List[SearchResult]:
        """Apply post-processing to search results"""

        # Apply custom scoring adjustments
        for result in results:
            # Adjust score based on content length (optional)
            if search_params.get('length_penalty', False):
                content_length = len(result.document.content)
                length_factor = min(1.0, content_length / 1000)  # Normalize to 1000 chars
                result.score *= (0.8 + 0.2 * length_factor)

            # Boost recent documents (optional)
            if search_params.get('recency_boost', False) and result.document.timestamp:
                age_hours = (time.time() - result.document.timestamp) / 3600
                recency_factor = max(0.5, 1.0 - (age_hours / (24 * 7)))  # Week decay
                result.score *= recency_factor

            # Add custom metadata
            result.search_metadata['post_processed'] = True

        return results

    def _apply_content_type_weighting(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply content type specific weighting"""

        for result in results:
            content_type = result.document.content_type
            weight = self.content_type_weights.get(content_type, 1.0)
            result.score *= weight
            result.search_metadata['content_type_weight'] = weight

        return results

    def _rerank_results(self, results: List[SearchResult],
                        query_embedding: np.ndarray,
                        search_params: Dict) -> List[SearchResult]:
        """Advanced re-ranking of search results"""

        if not results:
            return results

        # Custom re-ranking algorithms can be added here
        # For now, we'll implement a simple semantic clustering boost

        # Group similar documents and boost diversity
        reranked = []
        used_embeddings = []

        for result in results:
            doc_embedding = result.document.embedding

            # Calculate similarity to already selected documents
            if used_embeddings:
                similarities = [
                    np.dot(doc_embedding, used_emb)
                    for used_emb in used_embeddings
                ]
                max_similarity = max(similarities)

                # Apply diversity penalty
                diversity_penalty = 1.0 - (max_similarity * self.diversity_boost)
                result.score *= diversity_penalty
                result.search_metadata['diversity_penalty'] = diversity_penalty

            reranked.append(result)
            used_embeddings.append(doc_embedding)

        # Sort by adjusted scores
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    def _apply_diversity_boost(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply diversity boost to reduce redundant results"""

        if len(results) <= 1:
            return results

        diverse_results = [results[0]]  # Always include top result

        for result in results[1:]:
            # Check similarity to already selected results
            should_include = True

            for selected in diverse_results:
                similarity = np.dot(result.document.embedding, selected.document.embedding)

                # Skip if too similar to already selected result
                if similarity > 0.95:  # Very high similarity threshold
                    should_include = False
                    break

            if should_include:
                diverse_results.append(result)

        return diverse_results

    def _combine_multi_vector_results(self, all_results: List[SearchResult]) -> List[SearchResult]:
        """Combine results from multiple query vectors"""

        # Group results by document ID
        doc_results = {}

        for result in all_results:
            doc_id = result.document.id

            if doc_id not in doc_results:
                doc_results[doc_id] = {
                    'document': result.document,
                    'scores': [],
                    'metadata': []
                }

            doc_results[doc_id]['scores'].append(result.score)
            doc_results[doc_id]['metadata'].append(result.search_metadata)

        # Combine scores and create final results
        combined_results = []

        for doc_id, data in doc_results.items():
            scores = data['scores']

            # Combine scores (can use different strategies)
            combined_score = max(scores)  # Max score strategy
            # Alternative: combined_score = sum(scores) / len(scores)  # Average
            # Alternative: combined_score = sum(scores)  # Sum

            combined_metadata = {
                'individual_scores': scores,
                'score_combination': 'max',
                'query_count': len(scores),
                'all_metadata': data['metadata']
            }

            result = SearchResult(
                document=data['document'],
                score=combined_score,
                rank=0,  # Will be set later
                search_metadata=combined_metadata
            )

            combined_results.append(result)

        return combined_results

    def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Retrieve results from cache"""
        try:
            with self._cache_lock:
                return self._cache.get(cache_key)
        except:
            return None

    def _get_cache_key(self, query_embedding: np.ndarray, top_k: int, filters: Dict) -> str:
        """Generate cache key for search results"""
        import hashlib

        # Create a hash of the query parameters
        key_data = {
            'embedding_hash': hashlib.md5(query_embedding.tobytes()).hexdigest()[:16],
            'top_k': top_k,
            'filters': str(sorted(filters.items())) if filters else 'none'
        }

        key_string = f"{key_data['embedding_hash']}_{key_data['top_k']}_{key_data['filters']}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _add_to_cache(self, cache_key: str, results: List[SearchResult]):
        """Add results to cache"""
        try:
            with self._cache_lock:
                # Implement LRU-style cache
                if len(self._cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

                self._cache[cache_key] = results
        except:
            pass  # Cache is optional, don't fail on cache errors

    def _clear_cache(self):
        """Clear the search cache"""
        try:
            with self._cache_lock:
                self._cache.clear()
        except:
            pass

    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics about the vector store"""
        base_stats = self.get_stats()
        faiss_stats = self.primary_store.get_stats()
        faiss_info = self.primary_store.get_index_info()

        return {
            'hybrid_store': base_stats,
            'faiss_store': faiss_stats,
            'faiss_index': faiss_info,
            'caching': {
                'enabled': self.enable_caching,
                'cache_size': len(self._cache) if self.enable_caching else 0,
                'max_cache_size': self.cache_size
            },
            'search_optimizations': {
                'parallel_search': self.enable_parallel_search,
                'reranking': self.reranking_enabled,
                'diversity_boost': self.diversity_boost
            },
            'content_type_weights': self.content_type_weights
        }