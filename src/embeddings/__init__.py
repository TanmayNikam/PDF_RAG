"""
Multimodal embedding generation system
"""

from .base_embedder import BaseEmbedder, EmbeddingResult
from .text_embedder import SentenceTransformerEmbedder, HuggingFaceEmbedder
from .image_embedder import CLIPImageEmbedder
from .multimodal_embedder import MultimodalEmbedder, MultimodalContent


# Convenience function for easy embedding generation
def create_embedder(embedder_type: str = "multimodal", **kwargs) -> BaseEmbedder:
    """
    Factory function to create embedders

    Args:
        embedder_type: 'text', 'image', 'multimodal'
        **kwargs: Configuration for the embedder
    """

    if embedder_type == "text":
        return SentenceTransformerEmbedder(**kwargs)
    elif embedder_type == "image":
        return CLIPImageEmbedder(**kwargs)
    elif embedder_type == "multimodal":
        return MultimodalEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


__all__ = [
    'BaseEmbedder',
    'EmbeddingResult',
    'SentenceTransformerEmbedder',
    'HuggingFaceEmbedder',
    'CLIPImageEmbedder',
    'MultimodalEmbedder',
    'MultimodalContent',
    'create_embedder'
]