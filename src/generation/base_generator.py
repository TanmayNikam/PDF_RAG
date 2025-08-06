"""
Base classes and interfaces for the generation system
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import time
import logging


@dataclass
class GenerationRequest:
    """Request for text generation"""
    query: str
    context_documents: List[Dict]
    generation_config: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GenerationResponse:
    """Response from text generation"""
    generated_text: str
    confidence_score: float
    generation_time: float
    token_count: int
    model_name: str
    metadata: Dict

    # Source tracking
    sources_used: List[str]
    context_relevance: float

    # generation details not required

    # Generation details
    # prompt_template: Optional[str] = None
    # total_tokens: Optional[int] = None
    # cost_estimate: Optional[float] = None


class BaseGenerator(ABC):
    """Abstract base class for text generators"""

    def __init__(self, model_name: str, config: Dict = None):
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.generation_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'avg_time_per_request': 0.0,
            'avg_tokens_per_request': 0.0
        }

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text based on query and context"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the generator is available and ready"""
        pass

    def update_stats(self, response: GenerationResponse):
        """Update generation statistics"""
        self.generation_stats['total_requests'] += 1
        self.generation_stats['total_tokens'] += response.token_count
        self.generation_stats['total_time'] += response.generation_time

        # Calculate averages
        total_requests = self.generation_stats['total_requests']
        self.generation_stats['avg_time_per_request'] = (
                self.generation_stats['total_time'] / total_requests
        )
        self.generation_stats['avg_tokens_per_request'] = (
                self.generation_stats['total_tokens'] / total_requests
        )

    def get_stats(self) -> Dict:
        """Get generation statistics"""
        return {
            'model_name': self.model_name,
            'config': self.config,
            'stats': self.generation_stats.copy()
        }


