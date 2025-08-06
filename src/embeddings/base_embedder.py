"""
Base embedder class and interfaces for multimodal embedding generation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging


@dataclass
class EmbeddingResult:
    """Represents an embedding result with metadata"""
    embedding: np.ndarray
    dimension: int
    model_name: str
    content_type: str  # 'text', 'image', 'multimodal'
    metadata: Dict
    processing_time: Optional[float] = None


class BaseEmbedder(ABC):
    """Abstract base class for all embedders"""

    def __init__(self, model_name: str, config: Dict = None):
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.is_loaded = False

    @abstractmethod
    def load_model(self):
        """Load the embedding model"""
        pass

    @abstractmethod
    def embed(self, content: Union[str, bytes, np.ndarray], metadata: Dict = None) -> EmbeddingResult:
        """Generate embedding for content"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass

    def ensure_loaded(self):
        """Ensure model is loaded"""
        if not self.is_loaded:
            self.load_model()