"""
Complete Multimodal RAG Generation Module

This module provides a comprehensive generation system for multimodal RAG,
including LLM integration via LangChain, advanced prompt management,
response optimization, and specialized multimodal content handling.
"""
from typing import Dict

from .base_generator import BaseGenerator, GenerationRequest, GenerationResponse
from .langchain_generator import LangChainGenerator
from .prompt_manager import AdvancedPromptManager, PromptExample
from .response_optimizer import ResponseOptimizer
from .generation_pipeline import MultimodalGenerationPipeline

__all__ = [
    'BaseGenerator',
    'GenerationRequest',
    'GenerationResponse',
    'LangChainGenerator',
    'AdvancedPromptManager',
    'PromptExample',
    'ResponseOptimizer',
    'MultimodalGenerationPipeline'
]

# Configuration helpers
def create_generation_pipeline(config: Dict = None) -> MultimodalGenerationPipeline:
    """
    Convenience function to create a generation pipeline with common configurations

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured MultimodalGenerationPipeline instance
    """

    default_config = {
        "primary_generator": {
            "provider": "ollama",
            "model": "qwen2.5vl:3b",
            "config": {
                "temperature": 0.1,
                "max_tokens": 4096,
                "use_chat_model": True,
                'vision_enabled': True
            }
        },
        "enable_fallback": True,
        "enable_optimization": True,
        "default_template": "multimodal",
        "optimizer_config": {
            "enable_citation_extraction": True,
            "enable_formatting": True,
            "max_response_length": 2000
        }
    }

    if config:
        default_config.update(config)

    return MultimodalGenerationPipeline(default_config)


# Template configurations for common use cases
COMMON_TEMPLATES = {
    "pdf_qa": {
        "primary_generator": {
            "provider": "ollama",
            "model": "qwen2.5vl:3b",
            'config': {
                'temperature': 0.1,
                'max_tokens': 4096,
                'use_chat_model': True,
                'vision_enabled': True
            }
        },
        "default_template": "pdf_analysis",
        "enable_optimization": True
    },

    "research_assistant": {
        "primary_generator": {
            "provider": "ollama",
            "model": "llama3.2:3b",
            'config': {
                'temperature': 0.1,
                'max_tokens': 4096,
                'use_chat_model': True
            }
        },
        "default_template": "citation_qa",
        "optimizer_config": {
            "enable_citation_extraction": True,
            "enable_source_tracking": True
        }
    },

    "document_summarizer": {
        "primary_generator": {
            "provider": "ollama",
            "model": "qwen2.5vl:3b",
            'config': {
                'temperature': 0.1,
                'max_tokens': 4096,
                'use_chat_model': True,
                'vision_enabled': True
            }
        },
        "default_template": "summary",
        "optimizer_config": {
            "max_response_length": 1500
        }
    }
}


def create_specialized_pipeline(pipeline_type: str, **kwargs) -> MultimodalGenerationPipeline:
    """
    Create a specialized generation pipeline for common use cases

    Args:
        pipeline_type: Type of pipeline ('pdf_qa', 'research_assistant', 'document_summarizer')
        **kwargs: Additional configuration overrides

    Returns:
        Specialized MultimodalGenerationPipeline instance
    """

    if pipeline_type not in COMMON_TEMPLATES:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {list(COMMON_TEMPLATES.keys())}")

    config = COMMON_TEMPLATES[pipeline_type].copy()
    config.update(kwargs)

    return MultimodalGenerationPipeline(config)