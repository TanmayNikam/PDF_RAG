"""
ColPali-specific retriever for document image retrieval
"""

import time
import logging
from typing import List, Dict, Optional
import numpy as np

from .base_retriever import BaseRetriever, Query, RetrievalResult


class ColPaliRetriever(BaseRetriever):
    """
    Specialized retriever for ColPali embeddings

    Handles document page images and provides visual document retrieval
    without requiring text extraction.
    """

    def __init__(self, vector_store, colpali_embedder, config: Dict = None):
        super().__init__("colpali_retriever", config)

        self.vector_store = vector_store
        self.colpali_embedder = colpali_embedder

        # ColPali-specific configuration
        self.visual_similarity_threshold = self.config.get('visual_similarity_threshold', 0.0)
        self.boost_visual_content = self.config.get('boost_visual_content', True)
        self.page_context_window = self.config.get('page_context_window', 2)  # Include nearby pages

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized ColPali Retriever")

    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve document pages using ColPali embeddings"""
        start_time = time.time()

        try:
            # Generate query embedding using ColPali
            if query.embedding is None:
                query_result = self.colpali_embedder.embed_query(query.text)
                query.embedding = query_result.embedding

            # Search vector store
            search_results = self.vector_store.search(
                query_embedding=query.embedding,
                top_k=min(top_k * 2, 100),  # Get extra for post-processing
                filters=self._build_colpali_filters(query.filters)
            )



            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, result in enumerate(search_results):
                retrieval_result = RetrievalResult(
                    document_id=result.document.id,
                    content=result.document.content,
                    score=result.score,
                    rank=i,
                    retrieval_method="colpali_visual",
                    metadata=result.document.metadata.copy(),
                    chunk_metadata=result.search_metadata,
                    parent_document_id=result.document.parent_document_id,
                    relevance_score=result.score,
                )

                retrieval_result.metadata['page_image'] = self._get_original_page_image(result.document.id)

                # Add ColPali-specific metadata
                retrieval_result.metadata.update({
                    'visual_retrieval': True,
                    'colpali_processed': True,
                    'page_based': True,
                    'page_image': result.document.metadata.get('page_image')
                })

                # Apply similarity threshold
                if retrieval_result.score >= self.visual_similarity_threshold:
                    retrieval_results.append(retrieval_result)

            # Post-process for ColPali-specific optimizations
            if self.boost_visual_content:
                retrieval_results = self._boost_visual_content(retrieval_results)

            # Add page context if requested
            if self.page_context_window > 0:
                retrieval_results = self._add_page_context(retrieval_results)

            # Take top K results
            final_results = retrieval_results[:top_k]
            for i, result in enumerate(final_results):
                result.rank = i
                result.combined_score = result.relevance_score

            # Update statistics
            query_time = time.time() - start_time
            self.update_stats(query_time)

            self.logger.debug(f"ColPali retrieval: {len(final_results)} results in {query_time:.3f}s")
            return final_results

        except Exception as e:
            self.logger.error(f"ColPali retrieval failed: {e}")
            raise

    def _build_colpali_filters(self, base_filters: Dict) -> Dict:
        """Build filters specific to ColPali content"""
        filters = base_filters.copy() if base_filters else {}

        # Only retrieve ColPali-processed content
        filters['content_type'] = 'document_image'

        return filters

    def _boost_visual_content(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Boost results that likely contain visual elements"""
        for result in results:
            # Boost based on content type and metadata
            metadata = result.metadata

            # Boost pages that likely have figures, charts, etc.
            visual_indicators = [
                'figure', 'chart', 'diagram', 'table', 'graph',
                'illustration', 'image', 'plot', 'visualization'
            ]

            content_lower = result.content.lower()
            visual_score = sum(1 for indicator in visual_indicators
                               if indicator in content_lower)

            if visual_score > 0:
                boost_factor = 1.0 + (visual_score * 0.1)  # 10% boost per visual indicator
                result.score *= boost_factor
                result.metadata['visual_boost'] = boost_factor

        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _add_page_context(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Add context from nearby pages"""
        # Group results by parent document
        doc_pages = {}
        for result in results:
            doc_id = result.parent_document_id or result.document_id
            if doc_id not in doc_pages:
                doc_pages[doc_id] = []
            doc_pages[doc_id].append(result)

        # For each high-scoring page, consider adding nearby pages
        enhanced_results = []
        added_pages = set()

        for result in results:
            if result.document_id not in added_pages:
                enhanced_results.append(result)
                added_pages.add(result.document_id)

                # Add context pages if this is a high-scoring result
                if result.rank < 5:  # Top 5 results get context
                    context_pages = self._get_context_pages(result, doc_pages)
                    for context_page in context_pages:
                        if context_page.document_id not in added_pages:
                            context_page.metadata['context_page'] = True
                            context_page.metadata['context_for'] = result.document_id
                            enhanced_results.append(context_page)
                            added_pages.add(context_page.document_id)

        return enhanced_results

    def _get_context_pages(self, main_result: RetrievalResult,
                           doc_pages: Dict) -> List[RetrievalResult]:
        """Get nearby pages for context"""
        context_pages = []
        doc_id = main_result.parent_document_id or main_result.document_id

        if doc_id not in doc_pages:
            return context_pages

        main_page_num = main_result.metadata.get('page_number', 0)

        # Get pages within context window
        for result in doc_pages[doc_id]:
            page_num = result.metadata.get('page_number', 0)
            page_distance = abs(page_num - main_page_num)

            if 0 < page_distance <= self.page_context_window:
                # Reduce score for context pages
                context_result = result
                context_result.score *= 0.8  # 20% reduction for context pages
                context_pages.append(context_result)

        return context_pages

    def _get_original_page_image(self, document_id: str) -> bytes:
        """Retrieve original page image from separate storage"""
        # Load from file system, database, or cache
        # Based on the document_id/chunk_id
        image_path = f"/content/MultModalRAGS/MultiModalRAGS/processed_images/{document_id}.png"
        with open(image_path, 'rb') as f:
            return f.read()
