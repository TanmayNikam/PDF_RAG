"""
Enhanced sparse retrieval using BM25 and NLTK for professional text processing
"""

import time
from typing import List, Dict, Set, Tuple
import re
from collections import defaultdict

# NLTK imports

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from .base_retriever import BaseRetriever, Query, RetrievalResult


def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        # nltk.data.find('corpora/wordnet')

        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("NLTK data downloaded")

        download_nltk_data()


class SparseRetriever(BaseRetriever):
    """
    Sparse retrieval using BM25 and NLTK

    This provides state-of-the-art keyword-based search using:
    - BM25 scoring (better than TF-IDF)
    - NLTK for robust text processing
    - Professional stemming and lemmatization
    - Smart stopword handling
    """

    def __init__(self, config: Dict = None):
        super().__init__("sparse_retriever", config)

        # Document storage
        self.documents = {}  # doc_id -> document content
        self.document_metadata = {}  # doc_id -> metadata
        self.processed_documents = []  # List of processed document tokens
        self.doc_id_list = []  # Maintains order for BM25

        # BM25 configuration
        self.bm25_variant = self.config.get('bm25_variant', 'okapi')  # okapi, l, plus
        self.bm25_index = None

        # Text processing configuration
        self.use_stemming = self.config.get('use_stemming', True)
        self.use_lemmatization = self.config.get('use_lemmatization', False)  # Either stem or lemmatize
        self.remove_stopwords = self.config.get('remove_stopwords', True)
        self.min_term_length = self.config.get('min_term_length', 2)
        self.max_term_length = self.config.get('max_term_length', 50)
        self.lowercase = self.config.get('lowercase', True)

        self.stop_words = set(stopwords.words('english'))
        if self.use_stemming and not self.use_lemmatization:
            self.stemmer = PorterStemmer()
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()


        # Performance tracking
        self.index_stats = {
            'total_documents': 0,
            'vocabulary_size': 0,
            'avg_doc_length': 0.0,
            'index_build_time': 0.0
        }

        self.logger.info(f"   Initialized Sparse Retriever")
        self.logger.info(f"   BM25 variant: {self.bm25_variant}")
        self.logger.info(f"   Stemming: {self.use_stemming}")
        self.logger.info(f"   Lemmatization: {self.use_lemmatization}")

    def add_documents(self, documents: List[Dict]):
        """Add documents and build BM25 index"""
        start_time = time.time()

        # Process each document
        for doc_dict in documents:
            doc_id = doc_dict['id']
            content = doc_dict['content']
            metadata = doc_dict.get('metadata', {})

            self.documents[doc_id] = content
            self.document_metadata[doc_id] = metadata
            self.doc_id_list.append(doc_id)

            # Process document text
            processed_tokens = self._preprocess_text(content)
            self.processed_documents.append(processed_tokens)

        # Build BM25 index
        if self.processed_documents:
            self._build_bm25_index()

        self.index_stats.update({
            'total_documents': len(self.documents),
            'vocabulary_size': len(self._get_vocabulary()),
            'avg_doc_length': 0 if len(self.processed_documents) == 0 else sum(len(doc) for doc in self.processed_documents) / len(self.processed_documents)
        })


    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using BM25 scoring"""
        start_time = time.time()

        try:
            if not self.bm25_index:
                self.logger.warning("BM25 index not built")
                return []

            # Process query
            query_tokens = self._preprocess_text(query.text)

            if not query_tokens:
                return []

            # Get BM25 scores
            doc_scores = self.bm25_index.get_scores(query_tokens)

            # Get top documents
            top_indices = doc_scores.argsort()[-top_k:][::-1]

            # Convert to RetrievalResult objects
            results = []
            for rank, doc_idx in enumerate(top_indices):
                score = doc_scores[doc_idx]

                # Skip documents with zero score
                if score <= 0:
                    continue

                doc_id = self.doc_id_list[doc_idx]

                # Calculate additional metrics
                matched_terms = self._get_matched_terms(query_tokens, doc_idx)
                query_coverage = len(matched_terms) / len(query_tokens) if query_tokens else 0

                result = RetrievalResult(
                    document_id=doc_id,
                    content=self.documents[doc_id],
                    score=float(score),
                    rank=rank,
                    retrieval_method=f"bm25_{self.bm25_variant}",
                    metadata=self.document_metadata[doc_id].copy(),
                    relevance_score=float(score)
                )

                # Add BM25-specific metadata
                result.metadata.update({
                    'matched_terms': matched_terms,
                    'query_coverage': query_coverage,
                    'bm25_variant': self.bm25_variant,
                    'processed_query_length': len(query_tokens)
                })

                results.append(result)

            # Update statistics
            query_time = time.time() - start_time
            self.update_stats(query_time)

            self.logger.debug(f"BM25 retrieval: {len(results)} results in {query_time:.3f}s")
            return results

        except Exception as e:
            self.logger.error(f"BM25 retrieval failed: {e}")
            raise


    def _preprocess_text(self, text: str) -> List[str]:
        """Enhanced text preprocessing using NLTK"""

        if not text or not text.strip():
            return []

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Tokenization
            # Use NLTK tokenizer (handles punctuation better)
        tokens = word_tokenize(text)

        # Filter and process tokens
        processed_tokens = []

        for token in tokens:
            # Skip if not alphabetic (removes numbers, punctuation)
            if not token.isalpha():
                continue

            # Check length constraints
            if len(token) < self.min_term_length or len(token) > self.max_term_length:
                continue

            # Remove stopwords
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue

            # Apply stemming or lemmatization
            if self.use_lemmatization:
                # Lemmatization (requires POS tagging for best results)
                pos = self._get_wordnet_pos(token)
                token = self.lemmatizer.lemmatize(token, pos)
            elif self.use_stemming:
                # Stemming
                token = self.stemmer.stem(token)

            processed_tokens.append(token)

        return processed_tokens

    def _get_wordnet_pos(self, word: str) -> str:
        """Get WordNet POS tag for better lemmatization"""

        try:
            tag = pos_tag([word])[0][1][0].upper()
            tag_dict = {
                'J': 'a',  # Adjective
                'N': 'n',  # Noun
                'V': 'v',  # Verb
                'R': 'r'  # Adverb
            }
            return tag_dict.get(tag, 'n')  # Default to noun
        except:
            return 'n'

    def _build_bm25_index(self):
        """Build BM25 index from processed documents"""

        if self.bm25_variant == 'okapi':
            self.bm25_index = BM25Okapi(self.processed_documents)
        elif self.bm25_variant == 'l':
            self.bm25_index = BM25L(self.processed_documents)
        elif self.bm25_variant == 'plus':
            self.bm25_index = BM25Plus(self.processed_documents)
        else:
            self.logger.warning(f"Unknown BM25 variant: {self.bm25_variant}, using Okapi")
            self.bm25_index = BM25Okapi(self.processed_documents)

        self.logger.debug(f"Built BM25 index with {len(self.processed_documents)} documents")

    def _get_matched_terms(self, query_tokens: List[str], doc_idx: int) -> List[str]:
        """Get terms that matched between query and document"""
        doc_tokens = set(self.processed_documents[doc_idx])
        matched_terms = [term for term in query_tokens if term in doc_tokens]
        return matched_terms

    def _get_vocabulary(self) -> Set[str]:
        """Get the complete vocabulary from all documents"""
        vocabulary = set()
        for doc_tokens in self.processed_documents:
            vocabulary.update(doc_tokens)
        return vocabulary

    def get_enhanced_stats(self) -> Dict:
        """Get comprehensive statistics including BM25-specific metrics"""
        base_stats = self.get_stats()

        vocabulary = self._get_vocabulary()
        top_terms = self._get_top_terms()

        enhanced_stats = {
            'base_stats': base_stats,
            'index_stats': self.index_stats.copy(),
            'bm25_config': {
                'variant': self.bm25_variant,
                'stemming': self.use_stemming,
                'lemmatization': self.use_lemmatization,
                'remove_stopwords': self.remove_stopwords
            },
            'vocabulary_stats': {
                'size': len(vocabulary),
                'top_terms': top_terms,
                'avg_term_length': sum(len(term) for term in vocabulary) / len(vocabulary) if vocabulary else 0
            },
            'text_processing': {
                'total_tokens': sum(len(doc) for doc in self.processed_documents),
                'unique_tokens': len(vocabulary),
                'compression_ratio': len(vocabulary) / sum(
                    len(doc) for doc in self.processed_documents) if self.processed_documents else 0
            }
        }

        return enhanced_stats

    def _get_top_terms(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequent terms across all documents"""
        term_counts = defaultdict(int)
        # can't we use counter?
        for doc_tokens in self.processed_documents:
            for token in doc_tokens:
                term_counts[token] += 1

        return sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def search_similar_terms(self, term: str, top_k: int = 5) -> List[str]:
        """Find terms similar to the given term (simple implementation)"""
        try:
            vocabulary = self._get_vocabulary()
            similar_terms = []

            # Simple similarity based on edit distance or common prefixes
            for vocab_term in vocabulary:
                if term != vocab_term:
                    # Check if terms share common prefix (3+ characters)
                    if len(term) >= 3 and len(vocab_term) >= 3:
                        if term[:3] == vocab_term[:3]:
                            similar_terms.append(vocab_term)

                    # Check if one term contains the other
                    if term in vocab_term or vocab_term in term:
                        similar_terms.append(vocab_term)

            return similar_terms[:top_k]

        except Exception as e:
            self.logger.warning(f"Similar terms search failed: {e}")
            return []

    def explain_score(self, query: str, doc_id: str) -> Dict:
        """Explain BM25 score for debugging and transparency"""
        if doc_id not in self.documents:
            return {}

        try:
            query_tokens = self._preprocess_text(query)
            doc_idx = self.doc_id_list.index(doc_id)

            # Get overall score
            scores = self.bm25_index.get_scores(query_tokens)
            total_score = scores[doc_idx]

            # Get individual term contributions (approximation)
            term_scores = {}
            for token in query_tokens:
                single_token_scores = self.bm25_index.get_scores([token])
                term_scores[token] = single_token_scores[doc_idx]

            doc_tokens = self.processed_documents[doc_idx]
            matched_terms = [term for term in query_tokens if term in doc_tokens]

            explanation = {
                'total_score': float(total_score),
                'query_tokens': query_tokens,
                'matched_terms': matched_terms,
                'term_scores': {k: float(v) for k, v in term_scores.items()},
                'document_length': len(doc_tokens),
                'query_coverage': len(matched_terms) / len(query_tokens) if query_tokens else 0,
                'bm25_variant': self.bm25_variant
            }

            return explanation

        except Exception as e:
            self.logger.error(f"Score explanation failed: {e}")
            return {'error': str(e)}


