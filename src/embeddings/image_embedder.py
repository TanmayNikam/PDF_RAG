"""
Image embedding using CLIP and other vision models
"""

import time
from typing import Union, List, Dict
import numpy as np
from PIL import Image
import io


try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from .base_embedder import BaseEmbedder, EmbeddingResult

class CLIPImageEmbedder(BaseEmbedder):


    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", config: Dict = None):
        super().__init__(model_name, config)
        self.processor = None
        self.model = None
        self.dimension = None

        # CLIP model configurations
        self.model_configs = {
            "openai/clip-vit-base-patch32": {
                "dimension": 512,
                "image_size": 224,
                "description": "Standard CLIP model, good balance of speed and quality"
            },
            "openai/clip-vit-large-patch14": {
                "dimension": 768,
                "image_size": 224,
                "description": "Larger CLIP model, higher quality but slower"
            }
        }

    def load_model(self):
        """Load CLIP model and processor"""
        if not CLIP_AVAILABLE:
            raise ImportError("transformers with CLIP support not available")
        try:
            self.logger.info(f"Loading CLIP model: {self.model_name}")

            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)

            # Set to evaluation mode
            self.model.eval()

            # Get dimension
            if self.model_name in self.model_configs:
                self.dimension = self.model_configs[self.model_name]["dimension"]
            else:
                self.dimension = self.model.config.projection_dim

            # self.is_loaded = True
            # self.logger.info(f" Loaded {self.model_name}, dimension: {self.dimension}")

        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise

    def embed(self, content: Union[bytes, Image.Image, List[Union[bytes, Image.Image]]],
              metadata: Dict = None) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Generate image embeddings using CLIP"""
        self.ensure_loaded()

        start_time = time.time()

        # Handle different input types
        images = self._prepare_images(content)
        is_single = not isinstance(content, list)

        try:
            embeddings = []

            # Process images in batches
            batch_size = self.config.get('batch_size', 8)

            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]

                # Process images
                inputs = self.processor(images=batch_images, return_tensors="pt")

                # Generate embeddings
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                    # Normalize embeddings
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

                    embeddings.extend(image_features.numpy())

            processing_time = time.time() - start_time

            # Create results
            results = []
            for i, embedding in enumerate(embeddings):
                result_metadata = {
                    'image_mode': images[i].mode if i < len(images) else 'unknown',
                    'image_size': images[i].size if i < len(images) else None,
                    'model_info': self.model_configs.get(self.model_name, {}),
                    'normalized': True
                }
                if metadata:
                    result_metadata.update(metadata)

                result = EmbeddingResult(
                    embedding=embedding,
                    dimension=self.dimension,
                    model_name=self.model_name,
                    content_type='image',
                    metadata=result_metadata,
                    processing_time=processing_time / len(embeddings)
                )
                results.append(result)

            return results[0] if is_single else results

        except Exception as e:
            self.logger.error(f"CLIP image embedding failed: {e}")
            raise

    def embed_text(self, text: Union[str, List[str]], metadata: Dict = None) -> Union[
        EmbeddingResult, List[EmbeddingResult]]:
        """Generate text embeddings using CLIP (for cross-modal similarity)"""
        # self.ensure_loaded()

        start_time = time.time()

        # Handle single string vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        try:
            # Process text
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)

            # Generate embeddings
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

                # Normalize embeddings
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

                embeddings = text_features.numpy()

            processing_time = time.time() - start_time

            # Create results
            results = []
            for i, (text_content, embedding) in enumerate(zip(texts, embeddings)):
                result_metadata = {
                    'text_length': len(text_content),
                    'model_info': self.model_configs.get(self.model_name, {}),
                    'content_type': 'text_via_clip',
                    'normalized': True
                }
                if metadata:
                    result_metadata.update(metadata)

                result = EmbeddingResult(
                    embedding=embedding,
                    dimension=self.dimension,
                    model_name=self.model_name,
                    content_type='text',
                    metadata=result_metadata,
                    processing_time=processing_time / len(texts)
                )
                results.append(result)

            return results[0] if is_single else results

        except Exception as e:
            self.logger.error(f"CLIP text embedding failed: {e}")
            raise

    def _prepare_images(self, content) -> List[Image.Image]:
        """Convert various image inputs to PIL Images"""
        if isinstance(content, list):
            return [self._to_pil_image(item) for item in content]
        else:
            return [self._to_pil_image(content)]

    def _to_pil_image(self, content: Union[bytes, Image.Image]) -> Image.Image:
        """Convert bytes or maintain PIL Image"""
        if isinstance(content, Image.Image):
            return content.convert('RGB')
        elif isinstance(content, bytes):
            return Image.open(io.BytesIO(content)).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(content)}")

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.dimension is None:
            self.load_model()
        return self.dimension
