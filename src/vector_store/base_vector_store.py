"""
Base vector store interface and abstract classes
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
import uuid
import time
import json
from pathlib import Path


@dataclass
class VectorDocument:
    """Represents a document in the vector store"""
    id: str
    embedding: np.ndarray
    content: str
    metadata: Dict
    content_type: str  # 'text', 'image', 'multimodal'
    chunk_id: Optional[str] = None
    parent_document_id: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class SearchResult:
    """Represents a search result from vector store"""
    document: VectorDocument
    score: float
    rank: int
    search_metadata: Dict


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""

    def __init__(self, dimension: int, config: Dict = None):
        self.dimension = dimension
        self.config = config or {}
        self.document_count = 0
        self.is_initialized = False

    @abstractmethod
    def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents to the vector store"""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray,
               top_k: int = 10,
               filters: Dict = None) -> List[SearchResult]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Retrieve a document by ID"""
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        """Save the vector store to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        """Load the vector store from disk"""
        pass

    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            'document_count': self.document_count,
            'dimension': self.dimension,
            'config': self.config,
            'initialized': self.is_initialized
        }




