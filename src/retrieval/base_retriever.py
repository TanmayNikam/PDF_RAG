"""
Base retriever interfaces and core classes for the retrieval system
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time
import logging


@dataclass
class Query:
    """Represents a user query with metadata"""
    text: str
    query_type: str = "semantic"  # semantic, keyword, hybrid, multimodal
    filters: Optional[Dict] = None
    metadata: Optional[Dict] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.filters is None:
            self.filters = {}


@dataclass
class RetrievalResult:
    """Enhanced result with retrieval-specific metadata"""
    document_id: str
    content: str
    score: float
    rank: int
    retrieval_method: str
    metadata: Dict
    chunk_metadata: Optional[Dict] = None
    parent_document_id: Optional[str] = None

    # Additional retrieval metrics
    relevance_score: Optional[float] = None
    diversity_score: Optional[float] = None
    recency_score: Optional[float] = None
    combined_score: Optional[float] = None


class BaseRetriever(ABC):
    """Abstract base class for all retrievers"""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'cache_hits': 0
        }

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        pass

    def update_stats(self, query_time: float, cache_hit: bool = False):
        """Update retrieval statistics"""
        self.retrieval_stats['total_queries'] += 1
        self.retrieval_stats['total_time'] += query_time
        self.retrieval_stats['avg_time'] = (
                self.retrieval_stats['total_time'] / self.retrieval_stats['total_queries']
        )
        if cache_hit:
            self.retrieval_stats['cache_hits'] += 1

    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        return {
            'name': self.name,
            'config': self.config,
            'stats': self.retrieval_stats.copy()
        }
