"""
ColPali embedder for direct document image processing
"""

import time
from typing import List, Union, Dict, Optional
import numpy as np
from PIL import Image
import io
import torch

try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    from colpali_engine.utils.torch_utils import get_torch_device
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False

from .base_embedder import BaseEmbedder, EmbeddingResult


class ColPaliEmbedder(BaseEmbedder):
    """
    ColPali embedder for processing document pages as images

    ColPali works directly on document page images and generates patch-level embeddings
    that can answer queries about visual and textual content without OCR.
    """

    def __init__(self, model_name: str = "vidore/colpali", config: Dict = None):
        super().__init__(model_name, config)

        if not COLPALI_AVAILABLE:
            raise ImportError("ColPali not available")

        self.model = None
        self.processor = None
        self.device = None
        self.max_patches = self.config.get('max_patches', 1024)
        self.batch_size = self.config.get('batch_size', 1)

    def load_model(self):
        """Load ColPali model and processor"""
        try:
            self.logger.info(f"Loading ColPali model: {self.model_name}")

            # Get device
            self.device = get_torch_device("auto")

            print("device: ", self.device)

            # Load model and processor
            self.model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                device_map=self.device
            )

            self.processor = ColPaliProcessor.from_pretrained(self.model_name)

            # Set to evaluation mode
            self.model.eval()

            self.is_loaded = True
            self.logger.info(f"✅ Loaded ColPali model on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load ColPali model: {e}")
            raise

    def embed(self, content: Union[bytes, Image.Image, List[Union[bytes, Image.Image]]],
              metadata: Dict = None) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Generate ColPali embeddings for document images"""
        self.ensure_loaded()

        start_time = time.time()

        # Handle different input types
        images = self._prepare_images(content)
        is_single = not isinstance(content, list)

        try:
            embeddings = []

            # Process images in batches
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i:i + self.batch_size]

                # Process images
                inputs = self.processor.process_images(batch_images)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    image_embeddings = self.model(**inputs)

                    # ColPali returns patch-level embeddings
                    # We'll use the mean pooled representation for compatibility
                    if isinstance(image_embeddings, tuple):
                        # Handle tuple output (embeddings, attention_mask)
                        embeddings_tensor = image_embeddings[0]
                    else:
                        embeddings_tensor = image_embeddings

                    # Pool patch embeddings to get document-level embedding
                    doc_embeddings = torch.mean(embeddings_tensor, dim=1)  # Average over patches

                    # Move to CPU and convert to numpy
                    doc_embeddings = doc_embeddings.cpu().numpy()
                    embeddings.extend(doc_embeddings)

            processing_time = time.time() - start_time

            # Create results
            results = []
            for i, embedding in enumerate(embeddings):
                result_metadata = {
                    'model_info': {
                        'name': self.model_name,
                        'type': 'colpali',
                        'patches_processed': True
                    },
                    'processing_time': processing_time / len(embeddings),
                    'device': str(self.device),
                    'image_index': i if not is_single else 0
                }

                if metadata:
                    result_metadata.update(metadata)

                result = EmbeddingResult(
                    embedding=embedding,
                    dimension=len(embedding),
                    model_name=self.model_name,
                    content_type='document_image',
                    metadata=result_metadata,
                    processing_time=processing_time / len(embeddings)
                )
                results.append(result)

            return results[0] if is_single else results

        except Exception as e:
            self.logger.error(f"ColPali embedding failed: {e}")
            raise

    def embed_query(self, query: str, metadata: Dict = None) -> EmbeddingResult:
        """Generate query embedding for retrieval"""
        self.ensure_loaded()

        start_time = time.time()

        try:
            # Process query text
            inputs = self.processor.process_queries([query])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate query embedding
            with torch.no_grad():
                query_embedding = self.model(**inputs)

                if isinstance(query_embedding, tuple):
                    query_embedding = query_embedding[0]

                # Pool to get single vector
                query_vector = torch.mean(query_embedding, dim=1).squeeze()
                query_vector = query_vector.cpu().numpy()

            processing_time = time.time() - start_time

            result_metadata = {
                'model_info': {
                    'name': self.model_name,
                    'type': 'colpali_query'
                },
                'query_text': query,
                'processing_time': processing_time,
                'device': str(self.device)
            }

            if metadata:
                result_metadata.update(metadata)

            return EmbeddingResult(
                embedding=query_vector,
                dimension=len(query_vector),
                model_name=self.model_name,
                content_type='query',
                metadata=result_metadata,
                processing_time=processing_time
            )

        except Exception as e:
            self.logger.error(f"ColPali query embedding failed: {e}")
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
        if not self.is_loaded:
            self.load_model()

        # Test with dummy query to get dimension
        try:
            test_result = self.embed_query("test")
            return test_result.dimension
        except:
            # Fallback to model config
            return getattr(self.model.config, 'hidden_size', 768)