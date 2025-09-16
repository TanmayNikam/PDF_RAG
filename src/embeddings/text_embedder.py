"""
Text embedding using Sentence Transformers and HuggingFace models
"""

import time
from typing import List, Union, Dict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base_embedder import BaseEmbedder, EmbeddingResult


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Text embedder using Sentence Transformers

    This is our primary text embedder - fast, efficient, and high-quality
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", config: Dict = None):
        super().__init__(model_name, config)
        self.model = None
        self.dimension = None

        # Model configurations
        self.model_configs = {
            "sentence-transformers/all-MiniLM-L6-v2": {
                "dimension": 384,
                "max_seq_length": 256,
                "description": "Fast, lightweight model good for general use"
            },
            "sentence-transformers/all-mpnet-base-v2": {
                "dimension": 768,
                "max_seq_length": 384,
                "description": "Higher quality, slower model"
            },
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
                "dimension": 384,
                "max_seq_length": 512,
                "description": "Optimized for question-answering"
            }
        }

    def load_model(self):
        """Load Sentence Transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")

        try:
            self.logger.info(f"Loading Sentence Transformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Get dimension
            if self.model_name in self.model_configs:
                self.dimension = self.model_configs[self.model_name]["dimension"]
            else:
                # Test with dummy text to get dimension
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                self.dimension = test_embedding.shape[1]

            self.is_loaded = True
            # self.logger.info(f" Loaded {self.model_name}, dimension: {self.dimension}")

        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def embed(self, content: Union[str, List[str]], metadata: Dict = None) -> Union[
        EmbeddingResult, List[EmbeddingResult]]:
        """Generate text embeddings"""
        # can check later if ensure_loaded is required
        self.ensure_loaded()


        start_time = time.time()

        # Handle single string vs list
        is_single = isinstance(content, str)
        texts = [content] if is_single else content

        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for better similarity
                batch_size=self.config.get('batch_size', 32)
            )

            processing_time = time.time() - start_time

            # Create results
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                result_metadata = {
                    'text_length': len(text),
                    'model_info': self.model_configs.get(self.model_name, {}),
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
            self.logger.error(f"Embedding generation failed: {e}")
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.dimension is None:
            self.load_model()
        return self.dimension


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Text embedder using raw HuggingFace transformers

    This gives us more control and understanding of the embedding process
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", config: Dict = None):
        super().__init__(model_name, config)
        self.tokenizer = None
        self.model = None
        self.dimension = None

    def load_model(self):
        """Load HuggingFace model and tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")

        try:
            self.logger.info(f"Loading HuggingFace model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            # Set to evaluation mode
            self.model.eval()

            # Get dimension from model config
            self.dimension = self.model.config.hidden_size

            # self.is_loaded = True
            # self.logger.info(f" Loaded {self.model_name}, dimension: {self.dimension}")

        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}")
            raise

    def embed(self, content: Union[str, List[str]], metadata: Dict = None) -> Union[
        EmbeddingResult, List[EmbeddingResult]]:
        """Generate embeddings using HuggingFace model"""
        # will implement if required
        self.ensure_loaded()

        start_time = time.time()

        # Handle single string vs list
        is_single = isinstance(content, str)
        texts = [content] if is_single else content

        try:
            embeddings = []

            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)

                    # Use mean pooling of last hidden states
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']

                    # Mean pooling
                    masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                    summed = torch.sum(masked_hidden_states, dim=1)
                    counts = torch.sum(attention_mask, dim=1, keepdim=True)
                    mean_pooled = summed / counts

                    # L2 normalize
                    embedding = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

                    embeddings.append(embedding.squeeze().numpy())

            processing_time = time.time() - start_time

            # Create results
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                result_metadata = {
                    'text_length': len(text),
                    'tokenizer_type': self.tokenizer.__class__.__name__,
                    'pooling_method': 'mean',
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
            self.logger.error(f"HuggingFace embedding generation failed: {e}")
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.dimension is None:
            self.load_model()
        return self.dimension