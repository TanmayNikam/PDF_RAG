"""
Vector Store System for Multimodal RAG
"""
from typing import Dict, List

from .base_vector_store import BaseVectorStore, VectorDocument, SearchResult
from .faiss_vector_store import FAISSVectorStore
from .hybrid_vector_store import HybridVectorStore

import sys
from pathlib import Path


# current_dir = Path(__file__).parent
# src_dir = current_dir.parent  # Go up from embeddings to src
# sys.path.insert(0, str(src_dir))
#
# print(sys.path)

# project_root = Path("/kaggle/working/MultiModalRAGS/MultiModalRAGS")
# sys.path.insert(0, str(project_root))

try:
    from src.embeddings import MultimodalContent
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


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
            # print("chunk type is: ", chunk_type)
            for chunk in chunks:
                # Generate embedding based on chunk type
                if chunk_type == 'text_chunks':
                    # embedding_result = embedder.embed(chunk['content'])
                    # multiModalContent = MultimodalContent(text=chunk['content'])
                    #

                    multimodal_content = MultimodalContent(text=chunk['content'])
                    embedding_result = embedder.embed(multimodal_content)

                    embedding = embedding_result.embedding
                    content_type = 'text'
                    content = chunk['content']

                elif chunk_type == 'image_chunks':
                    if 'image_data' in chunk:
                        # print("image chunk type as per python: ", type(chunk['content']))

                        # embedding_result = embedder.embed_image_only(chunk['image_data'])
                        multimodal_content = MultimodalContent(image=chunk['image_data'])
                        embedding_result = embedder.embed(multimodal_content)
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


def colpali_documents_to_vector_documents(processed_documents: List, colpali_embedder) -> List[VectorDocument]:
    """
    Convert ColPali processed documents to VectorDocuments

    This bridges ColPali document processor with the vector store
    """
    vector_docs = []

    for doc in processed_documents:
        # print("each document: ", doc)
        colpali_chunks = doc.get('colpali_chunks', [])

        for chunk in colpali_chunks:
            # Generate embedding for the page image

            page_image = chunk['page_image']
            embedding_result = colpali_embedder.embed(page_image)

            import base64
            img_base64 = base64.b64encode(page_image).decode('utf-8')

            # Create VectorDocument for the page
            vector_doc = VectorDocument(
                id=chunk['chunk_id'],
                embedding=embedding_result.embedding,
                content=chunk['content'],
                metadata={
                    'chunk_type': 'colpali_page',
                    'document_id': doc['document_id'],
                    'page_number': chunk['page_number'],
                    'page_size': chunk['page_size'],
                    'file_path': doc.get('file_path'),
                    'processing_timestamp': doc['metadata'].get('processing_timestamp'),
                    'dpi': chunk['metadata'].get('dpi'),
                    'original_size': chunk['metadata'].get('original_size'),
                    'rendering_method': 'colpali',
                    'page_image': img_base64,
                    'has_page_image': True,
                    'image_format': chunk['metadata'].get('image_format', 'PNG'),
                    'has_visual_data': True
                },
                content_type='document_image',
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
    'documents_to_vector_documents',
    'colpali_documents_to_vector_documents'
]