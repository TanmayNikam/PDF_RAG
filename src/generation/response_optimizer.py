"""
Response optimization and post-processing for multimodal RAG
"""

import re
from typing import Dict, List, Optional, Any
import logging
import numpy as np


class ResponseOptimizer:
    """
    Advanced response optimization and post-processing
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Optimization settings
        self.enable_citation_extraction = self.config.get('enable_citation_extraction', True)
        self.enable_formatting = self.config.get('enable_formatting', True)
        self.enable_fact_checking = self.config.get('enable_fact_checking', False)
        self.max_response_length = self.config.get('max_response_length', 2000)
        self.enable_source_tracking = self.config.get('enable_source_tracking', True)

        # Citation patterns for different formats
        self.citation_patterns = [
            r'\[Source:\s*([^\]]+)\]',  # [Source: doc_id]
            r'\(([^)]+\.pdf[^)]*)\)',  # (document.pdf)
            r'according to ([^,\.]+)',  # according to document
            r'as shown in ([^,\.]+)',  # as shown in figure
            r'Figure\s+(\d+)',  # Figure 1
            r'Table\s+(\d+)',  # Table 1
        ]

    def optimize_response(self, response_text: str, context_docs: List[Dict], query: str) -> Dict:
        """
        Comprehensive response optimization

        Returns:
            Dict with optimized response and metadata
        """
        optimized = {
            'original_text': response_text,
            'optimized_text': response_text,
            'citations': [],
            'confidence_metrics': {},
            'formatting_applied': [],
            'warnings': [],
            'source_attribution': {},
            'content_analysis': {}
        }

        try:
            text = response_text

            # Step 1: Basic formatting
            if self.enable_formatting:
                text = self._apply_formatting(text)
                optimized['formatting_applied'].append('basic_formatting')

            # Step 2: Citation extraction and enhancement
            if self.enable_citation_extraction:
                text, citations = self._extract_and_enhance_citations(text, context_docs)
                optimized['citations'] = citations
                optimized['formatting_applied'].append('citations')

            # Step 3: Source attribution analysis
            if self.enable_source_tracking:
                source_attribution = self._analyze_source_attribution(text, context_docs)
                optimized['source_attribution'] = source_attribution

            # Step 4: Content analysis
            content_analysis = self._analyze_content_quality(text, context_docs, query)
            optimized['content_analysis'] = content_analysis

            # Step 5: Length optimization
            if len(text) > self.max_response_length:
                text = self._intelligent_truncation(text)
                optimized['warnings'].append('response_truncated')

            # Step 6: Calculate comprehensive confidence metrics
            confidence_metrics = self._calculate_comprehensive_confidence(
                text, context_docs, query, optimized
            )
            optimized['confidence_metrics'] = confidence_metrics

            optimized['optimized_text'] = text

        except Exception as e:
            self.logger.error(f"Response optimization failed: {e}")
            optimized['warnings'].append(f'optimization_failed: {str(e)}')

        return optimized

    def _apply_formatting(self, text: str) -> str:
        """Apply comprehensive text formatting"""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines

        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

        # Capitalize sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip().capitalize() if s else s for s in sentences]
        text = ' '.join(sentences)

        # Format lists and bullet points
        text = re.sub(r'\n(\d+)\.\s*', r'\n\1. ', text)  # Numbered lists
        text = re.sub(r'\n[•\-*]\s*', r'\n• ', text)  # Bullet points

        # Clean up citations
        text = re.sub(r'\[\s*Source\s*:\s*([^]]+)\s*]', r'[Source: \1]', text)

        return text.strip()

    def _extract_and_enhance_citations(self, text: str, context_docs: List[Dict]) -> tuple:
        """Extract and enhance citations in the response"""

        citations = []
        enhanced_text = text

        # Extract existing citations
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation_text = match.group(1)
                citations.append({
                    'text': citation_text,
                    'pattern': pattern,
                    'position': match.span(),
                    'type': self._classify_citation_type(pattern)
                })

        # Add implicit citations based on content analysis
        implicit_citations = self._find_implicit_citations(text, context_docs)
        citations.extend(implicit_citations)

        # Enhance citations with document metadata
        enhanced_citations = []
        for citation in citations:
            enhanced_citation = self._enhance_citation_with_metadata(citation, context_docs)
            enhanced_citations.append(enhanced_citation)

        return enhanced_text, enhanced_citations

    def _find_implicit_citations(self, text: str, context_docs: List[Dict]) -> List[Dict]:
        """Find content that should be cited but isn't explicitly cited"""

        implicit_citations = []
        text_words = set(text.lower().split())

        for doc in context_docs:
            doc_content = doc.get('content', '').lower()
            doc_words = set(doc_content.split())

            # Calculate word overlap
            overlap = text_words.intersection(doc_words)
            if len(overlap) >= 5:  # Significant overlap
                similarity_score = len(overlap) / min(len(text_words), len(doc_words))

                if similarity_score > 0.1:  # 10% similarity threshold
                    implicit_citations.append({
                        'document_id': doc.get('document_id', 'unknown'),
                        'similarity_score': similarity_score,
                        'overlapping_words': list(overlap)[:10],  # First 10 overlapping words
                        'type': 'implicit',
                        'source_type': doc.get('metadata', {}).get('content_type', 'text')
                    })

        return implicit_citations

    def _enhance_citation_with_metadata(self, citation: Dict, context_docs: List[Dict]) -> Dict:
        """Enhance citation with document metadata"""

        enhanced = citation.copy()

        # Try to match citation to documents
        for doc in context_docs:
            doc_id = doc.get('document_id', '')
            doc_content = doc.get('content', '')

            # Check if citation references this document
            if (doc_id in citation.get('text', '') or
                    any(word in doc_content.lower() for word in citation.get('text', '').lower().split()[:3])):
                enhanced.update({
                    'document_id': doc_id,
                    'document_metadata': doc.get('metadata', {}),
                    'content_type': doc.get('metadata', {}).get('content_type', 'text'),
                    'page_number': doc.get('metadata', {}).get('page_number'),
                    'section': doc.get('metadata', {}).get('section')
                })
                break

        return enhanced

    def _classify_citation_type(self, pattern: str) -> str:
        """Classify the type of citation based on pattern"""

        if 'Source:' in pattern:
            return 'explicit_source'
        elif 'Figure' in pattern:
            return 'figure_reference'
        elif 'Table' in pattern:
            return 'table_reference'
        elif 'according to' in pattern:
            return 'attribution'
        elif '.pdf' in pattern:
            return 'document_reference'
        else:
            return 'general_reference'

    def _analyze_source_attribution(self, text: str, context_docs: List[Dict]) -> Dict:
        """Analyze how well the response attributes information to sources"""

        attribution_analysis = {
            'total_sources_available': len(context_docs),
            'sources_referenced': 0,
            'attribution_quality': 0.0,
            'uncited_content_ratio': 0.0,
            'source_distribution': {}
        }

        # Count different content types in sources
        content_types = {}
        for doc in context_docs:
            content_type = doc.get('metadata', {}).get('content_type', 'text')
            content_types[content_type] = content_types.get(content_type, 0) + 1

        attribution_analysis['source_distribution'] = content_types

        # Analyze if response references available sources appropriately
        text_lower = text.lower()
        referenced_sources = 0

        for doc in context_docs:
            doc_id = doc.get('document_id', '')
            content = doc.get('content', '')

            # Check for explicit references
            if doc_id.lower() in text_lower:
                referenced_sources += 1
                continue

            # Check for content similarity (implicit reference)
            doc_words = set(content.lower().split()[:20])  # First 20 words
            text_words = set(text_lower.split())

            overlap = len(doc_words.intersection(text_words))
            if overlap >= 3:  # At least 3 words overlap
                referenced_sources += 1

        attribution_analysis['sources_referenced'] = referenced_sources

        if len(context_docs) > 0:
            attribution_analysis['attribution_quality'] = referenced_sources / len(context_docs)

        return attribution_analysis

    def _analyze_content_quality(self, text: str, context_docs: List[Dict], query: str) -> Dict:
        """Analyze the quality of the generated content"""

        analysis = {
            'response_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_sentence_length': 0.0,
            'readability_score': 0.0,
            'query_relevance': 0.0,
            'context_utilization': 0.0,
            'multimodal_references': 0
        }

        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            analysis['avg_sentence_length'] = analysis['word_count'] / len(sentences)

        # Simple readability score (based on sentence and word length)
        if analysis['word_count'] > 0:
            avg_word_length = sum(len(word) for word in text.split()) / analysis['word_count']
            analysis['readability_score'] = max(0, min(1, 1 - (avg_word_length - 5) / 10))

        # Query relevance (word overlap)
        query_words = set(query.lower().split())
        response_words = set(text.lower().split())
        if query_words:
            analysis['query_relevance'] = len(query_words.intersection(response_words)) / len(query_words)

        # Context utilization
        if context_docs:
            total_context_words = set()
            for doc in context_docs:
                total_context_words.update(doc.get('content', '').lower().split()[:50])

            if total_context_words:
                context_used = len(total_context_words.intersection(response_words))
                analysis['context_utilization'] = context_used / len(total_context_words)

        # Count multimodal references
        multimodal_terms = ['figure', 'image', 'chart', 'table', 'diagram', 'graph', 'visual']
        analysis['multimodal_references'] = sum(
            1 for term in multimodal_terms if term in text.lower()
        )

        return analysis

    def _intelligent_truncation(self, text: str) -> str:
        """Intelligently truncate response while preserving meaning"""

        if len(text) <= self.max_response_length:
            return text

        # Try to truncate at paragraph boundaries first
        paragraphs = text.split('\n\n')
        truncated = ""

        for paragraph in paragraphs:
            if len(truncated + paragraph + '\n\n') <= self.max_response_length - 10:
                truncated += paragraph + '\n\n'
            else:
                break

        if truncated and len(truncated.strip()) > self.max_response_length // 2:
            return truncated.strip() + "\n\n[Response truncated for length]"

        # Fallback to sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        truncated = ""

        for sentence in sentences:
            if len(truncated + sentence + ' ') <= self.max_response_length - 20:
                truncated += sentence + ' '
            else:
                break

        if truncated:
            return truncated.strip() + "... [Response truncated]"
        else:
            # Hard truncation as last resort
            return text[:self.max_response_length - 20] + "... [Truncated]"

    def _calculate_comprehensive_confidence(self, response: str, context_docs: List[Dict],
                                            query: str, optimization_data: Dict) -> Dict:
        """Calculate comprehensive confidence metrics"""

        metrics = {}

        # Base metrics
        content_analysis = optimization_data.get('content_analysis', {})

        metrics['content_quality'] = {
            'length_score': min(1.0, content_analysis.get('word_count', 0) / 150),  # Prefer 150+ words
            'readability_score': content_analysis.get('readability_score', 0.5),
            'sentence_structure_score': min(1.0, content_analysis.get('avg_sentence_length', 0) / 15)
        }

        # Context utilization
        metrics['context_usage'] = {
            'utilization_score': content_analysis.get('context_utilization', 0.0),
            'source_attribution_score': optimization_data.get('source_attribution', {}).get('attribution_quality', 0.0),
            'multimodal_integration_score': min(1.0, content_analysis.get('multimodal_references', 0) / 3)
        }

        # Query alignment
        metrics['query_alignment'] = {
            'relevance_score': content_analysis.get('query_relevance', 0.0),
            'completeness_score': min(1.0, len(response.split()) / 50)  # Prefer 50+ word responses
        }

        # Citation quality
        citations = optimization_data.get('citations', [])
        metrics['citation_quality'] = {
            'citation_count': len(citations),
            'explicit_citations': len([c for c in citations if c.get('type') != 'implicit']),
            'citation_diversity': len(set(c.get('document_id') for c in citations if c.get('document_id')))
        }

        # Overall confidence (weighted combination)
        weights = {
            'content_quality': 0.25,
            'context_usage': 0.35,
            'query_alignment': 0.25,
            'citation_quality': 0.15
        }

        # Calculate weighted average of sub-scores
        overall_score = 0.0
        total_weight = 0.0

        for category, weight in weights.items():
            if category in metrics:
                category_scores = [v for v in metrics[category].values() if isinstance(v, (int, float))]
                if category_scores:
                    category_avg = sum(category_scores) / len(category_scores)
                    overall_score += category_avg * weight
                    total_weight += weight

        if total_weight > 0:
            metrics['overall_confidence'] = overall_score / total_weight
        else:
            metrics['overall_confidence'] = 0.5

        # Confidence level categorization
        confidence_level = metrics['overall_confidence']
        if confidence_level >= 0.8:
            metrics['confidence_level'] = 'high'
        elif confidence_level >= 0.6:
            metrics['confidence_level'] = 'medium'
        elif confidence_level >= 0.4:
            metrics['confidence_level'] = 'low'
        else:
            metrics['confidence_level'] = 'very_low'

        return metrics


