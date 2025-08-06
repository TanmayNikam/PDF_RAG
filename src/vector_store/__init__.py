"""
Vector Store System for Multimodal RAG
"""
from typing import Dict, List

from .base_vector_store import BaseVectorStore, VectorDocument, SearchResult
from .faiss_vector_store import FAISSVectorStore
from .hybrid_vector_store import HybridVectorStore


# Convenience function for creating vector stores
def create_vector_store(store_type: str = "hybrid", dimension: int = 384, config: Dict = None) -> BaseVectorStore:
    """
    Factory function to create vector stores

    Args:
        store_type: 'faiss' or 'hybrid'
        dimension: Embedding dimension
        config: Configuration dictionary
    """

    if store_type == "faiss":
        return FAISSVectorStore(dimension, config)
    elif store_type == "hybrid":
        return HybridVectorStore(dimension, config)
    else:
        raise ValueError(f"Unknown store type: {store_type}")


# Integration helper for document processor
def documents_to_vector_documents(processed_documents: List, embedder) -> List[VectorDocument]:
    """
    Convert processed documents from document processor to VectorDocuments

    This bridges our document processor with the vector store
    """
    vector_docs = []

    for doc in processed_documents:
        for chunk_type in ['text_chunks', 'image_chunks', 'table_chunks', 'mixed_chunks']:
            chunks = doc.get(chunk_type, [])

            for chunk in chunks:
                # Generate embedding based on chunk type
                if chunk_type == 'text_chunks':
                    embedding_result = embedder.embed(chunk['content'])
                    embedding = embedding_result.embedding
                    content_type = 'text'
                    content = chunk['content']

                elif chunk_type == 'image_chunks':
                    if 'image_data' in chunk:
                        embedding_result = embedder.embed(chunk['image_data'])
                        embedding = embedding_result.embedding
                        content_type = 'image'
                        content = chunk.get('extracted_text', f"[IMAGE: {chunk['chunk_id']}]")
                    else:
                        continue

                elif chunk_type in ['table_chunks', 'mixed_chunks']:
                    # Handle as text for now
                    embedding_result = embedder.embed(chunk['content'])
                    embedding = embedding_result.embedding
                    content_type = 'text'
                    content = chunk['content']

                # Create VectorDocument
                vector_doc = VectorDocument(
                    id=chunk['chunk_id'],
                    embedding=embedding,
                    content=content,
                    metadata={
                        'chunk_type': chunk_type[:-7],  # Remove '_chunks' suffix
                        'document_id': doc['document_id'],
                        'page_number': chunk.get('page_number'),
                        'chunk_metadata': chunk.get('metadata', {}),
                        'file_path': doc.get('file_path'),
                        'processing_timestamp': doc['metadata'].get('processing_timestamp')
                    },
                    content_type=content_type,
                    chunk_id=chunk['chunk_id'],
                    parent_document_id=doc['document_id']
                )

                vector_docs.append(vector_doc)

    return vector_docs


__all__ = [
    'BaseVectorStore',
    'VectorDocument',
    'SearchResult',
    'FAISSVectorStore',
    'HybridVectorStore',
    'create_vector_store',
    'documents_to_vector_documents'
]