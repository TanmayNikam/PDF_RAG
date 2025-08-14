"""
FAISS-based vector store implementation
"""

import logging
import time
import uuid
from typing import List, Dict, Optional, Tuple
import numpy as np
import pickle
import json
from pathlib import Path
import traceback

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .base_vector_store import BaseVectorStore, VectorDocument, SearchResult


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store

    This combines FAISS's performance with the custom document management and advanced search capabilities.
    """

    def __init__(self, dimension: int, config: Dict = None):
        super().__init__(dimension, config)

        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available.")

        self.logger = logging.getLogger(__name__)

        # FAISS configuration
        self.index_type = self.config.get('index_type', 'IndexFlatIP')
        self.metric_type = self.config.get('metric_type', 'INNER_PRODUCT') # Inner Product for cosine similarity
        self.nlist = self.config.get('nlist', 100)  # For IVF indices
        self.nprobe = self.config.get('nprobe', 10)  # For IVF search

        # Initialize FAISS index
        self.index = None
        self.documents = {}  # Document storage: id -> VectorDocument
        self.id_to_index = {}  # Map document ID to FAISS index position
        self.index_to_id = {}  # Map FAISS index position to document ID

        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'avg_search_time': 0.0
        }

        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index based on configuration"""
        try:
            if self.index_type == 'IndexFlatIP':
                # Exact search using inner product (best for cosine similarity with normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimension)

            elif self.index_type == 'IndexFlatL2':
                # Exact search using L2 distance
                self.index = faiss.IndexFlatL2(self.dimension)

            elif self.index_type == 'IndexIVFFlat':
                # Inverted file index for faster approximate search
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)

            elif self.index_type == 'IndexHNSWFlat':
                # Hierarchical Navigable Small World graphs - great for high recall
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efSearch = 64

            elif self.index_type == 'IndexLSH':
                # Locality Sensitive Hashing - memory efficient
                self.index = faiss.IndexLSH(self.dimension, 64)

            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

            self.is_initialized = True
            self.logger.info(f"✅ Initialized FAISS index: {self.index_type}, dimension: {self.dimension}")

        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            raise

    def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents to the FAISS vector store"""
        if not documents:
            return []

        try:
            # Prepare embeddings matrix
            embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)

            # Normalize embeddings for cosine similarity (if using inner product)
            if self.index_type in ['IndexFlatIP', 'IndexIVFFlat']:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)  # Avoid division by zero

            # Get starting index position
            start_idx = self.index.ntotal

            # print(embeddings, type(embeddings))

            # Add to FAISS index

            print(f"Faiss Embeddings dimension: {self.index.d}")

            print(f"Documents dimension: {embeddings.shape}")

            self.index.add(embeddings)

            # Store documents and maintain mappings
            doc_ids = []
            for i, doc in enumerate(documents):
                # Generate ID if not provided
                if not doc.id:
                    doc.id = str(uuid.uuid4())

                doc_ids.append(doc.id)
                faiss_idx = start_idx + i
                # do we have a better approach to generated faiss indices?

                # Store document
                self.documents[doc.id] = doc

                # Maintain bidirectional mapping
                self.id_to_index[doc.id] = faiss_idx
                self.index_to_id[faiss_idx] = doc.id

            self.document_count = len(self.documents)

            # Train index if needed (for IVF indices)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if self.index.ntotal >= self.nlist:
                    self.logger.info("Training FAISS index...")
                    self.index.train(embeddings)
                    self.logger.info("FAISS index trained")

            self.logger.info(f"Added {len(documents)} documents to FAISS store")
            return doc_ids

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Failed to add documents to FAISS store: {e}")
            raise

    def search(self, query_embedding: np.ndarray, top_k: int = 10, filters: Dict = None) -> List[SearchResult]:
        """Search for similar documents using FAISS"""

        if self.document_count == 0:
            return []

        start_time = time.time()

        try:
            # Prepare query embedding
            query = query_embedding.reshape(1, -1).astype(np.float32)

            print("query embeddding: ", query)

            # Normalize for cosine similarity
            if self.index_type in ['IndexFlatIP', 'IndexIVFFlat']:
                norm = np.linalg.norm(query)
                if norm > 0:
                    query = query / norm

            # Perform FAISS search
            search_k = min(top_k * 2, self.document_count)  # Get extra results for filtering


            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe

            scores, indices = self.index.search(query, search_k)

            print(f"score: {scores}, indices: {indices}")

            # Convert results to SearchResult objects
            results = []
            rank = 0

            for i in range(len(indices[0])):
                faiss_idx = indices[0][i]
                score = float(scores[0][i])

                # Skip invalid indices
                if faiss_idx == -1:
                    continue

                # Get document ID
                doc_id = self.index_to_id.get(faiss_idx)
                if not doc_id:
                    continue

                # Get document
                document = self.documents.get(doc_id)
                if not document:
                    continue

                # Apply filters if provided
                if filters and not self._apply_filters(document, filters):
                    continue

                # Create search result
                search_result = SearchResult(
                    document=document,
                    score=score,
                    rank=rank,
                    search_metadata={
                        'faiss_index': int(faiss_idx),
                        'search_time': time.time() - start_time,
                        'index_type': self.index_type
                    }
                )

                results.append(search_result)
                rank += 1

                # Stop when we have enough results
                if len(results) >= top_k:
                    break

            # Update search statistics
            search_time = time.time() - start_time
            self._update_search_stats(search_time)

            self.logger.debug(f"FAISS search completed: {len(results)} results in {search_time:.3f}s")
            return results

        except Exception as e:
            self.logger.error(f"FAISS search failed: {e}")
            raise

    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Retrieve a document by ID"""
        return self.documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            if doc_id not in self.documents:
                return False

            # Note: FAISS doesn't support individual deletions efficiently
            # For production, you'd need to rebuild the index periodically
            # For now, we just remove from our document store

            faiss_idx = self.id_to_index.get(doc_id)

            # Remove from our mappings
            del self.documents[doc_id]
            if faiss_idx is not None:
                del self.id_to_index[doc_id]
                del self.index_to_id[faiss_idx]

            self.document_count = len(self.documents)

            self.logger.info(f" Marked document {doc_id} as deleted")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def batch_search(self, query_embeddings: np.ndarray,
                     top_k: int = 10) -> List[List[SearchResult]]:
        """Perform batch search for multiple queries"""

        if self.document_count == 0:
            return [[] for _ in range(len(query_embeddings))]

        try:
            # Prepare queries
            queries = query_embeddings.astype(np.float32)

            # Normalize for cosine similarity
            if self.index_type in ['IndexFlatIP', 'IndexIVFFlat']:
                norms = np.linalg.norm(queries, axis=1, keepdims=True)
                queries = queries / (norms + 1e-8)

            # Perform batch search
            scores, indices = self.index.search(queries, top_k)

            # Convert to SearchResult objects
            batch_results = []

            for q_idx in range(len(queries)):
                query_results = []

                for r_idx in range(top_k):
                    faiss_idx = indices[q_idx][r_idx]
                    score = float(scores[q_idx][r_idx])

                    if faiss_idx == -1:
                        continue

                    doc_id = self.index_to_id.get(faiss_idx)
                    if not doc_id:
                        continue

                    document = self.documents.get(doc_id)
                    if not document:
                        continue

                    search_result = SearchResult(
                        document=document,
                        score=score,
                        rank=r_idx,
                        search_metadata={'faiss_index': int(faiss_idx), 'batch_search': True}
                    )

                    query_results.append(search_result)

                batch_results.append(query_results)

            return batch_results

        except Exception as e:
            self.logger.error(f"Batch search failed: {e}")
            raise

    def get_similar_documents(self, doc_id: str, top_k: int = 10) -> List[SearchResult]:
        """Find documents similar to a given document"""
        document = self.get_document(doc_id)
        if not document:
            return []

        # Search using the document's embedding
        results = self.search(document.embedding, top_k + 1)  # +1 to exclude self

        # Remove the document itself from results
        return [r for r in results if r.document.id != doc_id][:top_k]

    def save(self, path: str) -> bool:
        """Save the FAISS vector store to disk"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_path = save_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))

            # Save documents and metadata
            metadata = {
                'documents': {doc_id: {
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'content_type': doc.content_type,
                    'chunk_id': doc.chunk_id,
                    'parent_document_id': doc.parent_document_id,
                    'timestamp': doc.timestamp,
                    'embedding': doc.embedding.tolist()  # Convert to list for JSON
                } for doc_id, doc in self.documents.items()},
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'config': self.config,
                'dimension': self.dimension,
                'document_count': self.document_count,
                'search_stats': self.search_stats
            }

            metadata_path = save_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved FAISS vector store to {save_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load the FAISS vector store from disk"""
        try:
            load_path = Path(path)

            # Load FAISS index
            index_path = load_path / "faiss_index.bin"
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

            self.index = faiss.read_index(str(index_path))

            # Load metadata
            metadata_path = load_path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Restore documents
            self.documents = {}
            for doc_id, doc_data in metadata['documents'].items():
                embedding = np.array(doc_data['embedding'], dtype=np.float32)
                doc = VectorDocument(
                    id=doc_data['id'],
                    embedding=embedding,
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    content_type=doc_data['content_type'],
                    chunk_id=doc_data.get('chunk_id'),
                    parent_document_id=doc_data.get('parent_document_id'),
                    timestamp=doc_data.get('timestamp')
                )
                self.documents[doc_id] = doc

            # Restore mappings
            self.id_to_index = {k: int(v) for k, v in metadata['id_to_index'].items()}
            self.index_to_id = {int(k): v for k, v in metadata['index_to_id'].items()}

            # Restore other attributes
            self.config.update(metadata.get('config', {}))
            self.dimension = metadata['dimension']
            self.document_count = metadata['document_count']
            self.search_stats = metadata.get('search_stats', self.search_stats)

            self.is_initialized = True

            self.logger.info(f"Loaded FAISS vector store from {load_path}")
            self.logger.info(f"Loaded {self.document_count} documents")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            return False

    def _apply_filters(self, document: VectorDocument, filters: Dict) -> bool:
        """Apply filters to determine if document matches criteria"""
        for key, value in filters.items():
            if key == 'content_type':
                if document.content_type != value:
                    return False
            elif key == 'parent_document_id':
                if document.parent_document_id != value:
                    return False
            elif key in document.metadata:
                if document.metadata[key] != value:
                    return False
            # Add more filter types as needed
        return True

    def _update_search_stats(self, search_time: float):
        """Update search performance statistics"""
        self.search_stats['total_searches'] += 1
        self.search_stats['total_search_time'] += search_time
        self.search_stats['avg_search_time'] = (
                self.search_stats['total_search_time'] / self.search_stats['total_searches']
        )

    def optimize_index(self):
        """Optimizing the FAISS index for better performance"""
        try:
            if self.index_type == 'IndexIVFFlat' and hasattr(self.index, 'make_direct_map'):
                # Adding direct map for faster ID lookups
                self.index.make_direct_map()
                self.logger.info("✅ Added direct map to IVF index")
        except Exception as e:
            self.logger.warning(f"Index optimization failed: {e}")

    def get_index_info(self) -> Dict:
        """Get information about the FAISS index"""
        info = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal if self.index else 0,
            'is_trained': getattr(self.index, 'is_trained', True),
            'metric_type': self.metric_type
        }

        if hasattr(self.index, 'nlist'):
            info['nlist'] = self.index.nlist
        if hasattr(self.index, 'nprobe'):
            info['nprobe'] = self.index.nprobe

        return info


