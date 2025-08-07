"""
Multimodal embedding fusion and cross-modal understanding
"""

import time
from typing import Union, List, Dict, Tuple, Optional

import PIL
import numpy as np
from dataclasses import dataclass

from .base_embedder import BaseEmbedder, EmbeddingResult
from .text_embedder import SentenceTransformerEmbedder
from .image_embedder import CLIPImageEmbedder


@dataclass
class MultimodalContent:
    """Represents multimodal content for embedding"""
    text: Optional[str] = None
    image: Optional[Union[bytes, 'PIL.Image.Image']] = None
    metadata: Optional[Dict] = None


class MultimodalEmbedder(BaseEmbedder):
    """
    Advanced multimodal embedder that can handle text, images, and their combinations

    This is the core innovation - we create unified embeddings for multimodal content
    """

    def __init__(self, config: Dict = None):
        default_config = {
            'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'image_model': 'openai/clip-vit-base-patch32',
            'fusion_method': 'concatenation',  # 'concatenation', 'weighted_average', 'learned_fusion'
            'text_weight': 0.6,
            'image_weight': 0.4,
            'normalize_before_fusion': True,
            'normalize_after_fusion': True
        }

        if config:
            default_config.update(config)

        super().__init__("multimodal_embedder", default_config)

        self.text_embedder = None
        self.image_embedder = None
        self.dimension = None

    def load_model(self):
        """Load all component models"""
        try:
            self.logger.info("Loading multimodal embedder components...")

            # Load text embedder
            self.text_embedder = SentenceTransformerEmbedder(
                model_name=self.config['text_model']
            )
            self.text_embedder.load_model()

            # Load image embedder (CLIP can do both text and images)
            self.image_embedder = CLIPImageEmbedder(
                model_name=self.config['image_model']
            )
            self.image_embedder.load_model()

            # Calculate fusion dimension
            self.dimension = self._calculate_fusion_dimension()

        except Exception as e:
            self.logger.error(f"Failed to load multimodal embedder: {e}")
            raise

    def embed(self, content: Union[MultimodalContent, List[MultimodalContent]],
              metadata: Dict = None) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Generate multimodal embeddings"""
        self.ensure_loaded()

        start_time = time.time()

        # Handle single content vs list
        is_single = isinstance(content, MultimodalContent)
        contents = [content] if is_single else content

        results = []

        for i, multimodal_content in enumerate(contents):

            # print("type of multimodal content", type(multimodal_content))
            # Generate individual embeddings
            text_embedding = None
            image_embedding = None

            if multimodal_content.text:
                text_result = self.text_embedder.embed(multimodal_content.text)
                text_embedding = text_result.embedding

            if multimodal_content.image:
                image_result = self.image_embedder.embed(multimodal_content.image)
                image_embedding = image_result.embedding

            # Fuse embeddings
            fused_embedding = self._fuse_embeddings(text_embedding, image_embedding)

            # Create result
            result_metadata = {
                'fusion_method': self.config['fusion_method'],
                'has_text': text_embedding is not None,
                'has_image': image_embedding is not None,
                'text_model': self.config['text_model'],
                'image_model': self.config['image_model'],
                'text_weight': self.config['text_weight'],
                'image_weight': self.config['image_weight']
            }

            if metadata:
                result_metadata.update(metadata)
            if multimodal_content.metadata:
                result_metadata.update(multimodal_content.metadata)

            result = EmbeddingResult(
                embedding=fused_embedding,
                dimension=self.dimension,
                model_name=self.model_name,
                content_type='multimodal',
                metadata=result_metadata,
                processing_time=(time.time() - start_time) / len(contents)
            )
            results.append(result)

        return results[0] if is_single else results

    def embed_text_only(self, text: Union[str, List[str]], metadata: Dict = None):
        """Embed text using the multimodal space"""

        # Use CLIP text embedder for consistency with multimodal space
        return self.image_embedder.embed_text(text, metadata)

    def embed_image_only(self, image, metadata: Dict = None):
        """Embed image using the multimodal space"""

        return self.image_embedder.embed(image, metadata)

    def _fuse_embeddings(self, text_embedding: Optional[np.ndarray],
                         image_embedding: Optional[np.ndarray]) -> np.ndarray:
        """Fuse text and image embeddings using configured method"""

        if text_embedding is None and image_embedding is None:
            raise ValueError("At least one embedding must be provided")

        # Handle single modality cases
        if text_embedding is None:
            return self._pad_embedding(image_embedding, 'image')
        if image_embedding is None:
            return self._pad_embedding(text_embedding, 'text')

        # Normalize before fusion if configured
        if self.config['normalize_before_fusion']:
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            image_embedding = image_embedding / np.linalg.norm(image_embedding)

        # Apply fusion method
        fusion_method = self.config['fusion_method']

        if fusion_method == 'concatenation':
            fused = np.concatenate([text_embedding, image_embedding])

        elif fusion_method == 'weighted_average':
            # Ensure same dimension (use smaller dimension)
            min_dim = min(len(text_embedding), len(image_embedding))
            text_trunc = text_embedding[:min_dim]
            image_trunc = image_embedding[:min_dim]

            fused = (self.config['text_weight'] * text_trunc +
                     self.config['image_weight'] * image_trunc)

        elif fusion_method == 'element_wise_max':
            min_dim = min(len(text_embedding), len(image_embedding))
            text_trunc = text_embedding[:min_dim]
            image_trunc = image_embedding[:min_dim]

            fused = np.maximum(text_trunc, image_trunc)

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Normalize after fusion if configured
        if self.config['normalize_after_fusion']:
            fused = fused / np.linalg.norm(fused)

        return fused

    def _pad_embedding(self, embedding: np.ndarray, modality: str) -> np.ndarray:
        """Pad single modality embedding to match multimodal dimension"""
        if self.config['fusion_method'] == 'concatenation':
            # Pad with zeros for missing modality
            if modality == 'text':
                # Pad for missing image part
                image_dim = self.image_embedder.get_dimension()
                padding = np.zeros(image_dim)
                return np.concatenate([embedding, padding])
            else:  # image
                # Pad for missing text part
                text_dim = self.text_embedder.get_dimension()
                padding = np.zeros(text_dim)
                return np.concatenate([padding, embedding])
        else:
            # For other methods, just return the embedding
            return embedding

    def _calculate_fusion_dimension(self) -> int:
        """Calculate the dimension of fused embeddings"""
        text_dim = self.text_embedder.get_dimension()
        image_dim = self.image_embedder.get_dimension()

        if self.config['fusion_method'] == 'concatenation':
            return text_dim + image_dim
        elif self.config['fusion_method'] in ['weighted_average', 'element_wise_max']:
            return min(text_dim, image_dim)
        else:
            return max(text_dim, image_dim)

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.dimension is None:
            self.load_model()
        return self.dimension

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        # Ensure embeddings are normalized
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)

        return np.dot(norm1, norm2)