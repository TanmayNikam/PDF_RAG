"""
Multimodal Document Processing Pipeline

This module provides comprehensive document processing capabilities for
multimodal RAG systems, including PDF parsing, image extraction, OCR,
and intelligent text chunking.
"""



import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

from .pdf_processor import PDFProcessor, DocumentChunk
from .image_processor import ImageProcessor
from .multi_modal_document_processor import MultimodalDocumentProcessor
# from .text_chunker import AdvancedTextChunker, TextChunk


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)



if __name__ == "__main__":
    # Example configuration
    config = {
        'pdf': {
            'extract_images': True,
            'min_image_size': (100, 100),
            'ocr_enabled': True,
            'chunk_size': 400,
            'chunk_overlap': 40,
        },
        'image': {
            'ocr_enabled': True,
            'enhance_images': True,
            'target_size': (512, 512),
        },
        'chunking': {
            'chunk_size': 400,
            'chunk_overlap': 40,
            'respect_sentence_boundaries': True,
        }
    }

    # Initialize processor
    processor = MultimodalDocumentProcessor(config)

    # Example: Process a single document
    # result = processor.process_document("path/to/document.pdf")
    # print(f"Processed {result['total_chunks']} chunks")

    # Example: Batch processing
    # file_paths = ["doc1.pdf", "doc2.pdf", "image1.png"]
    # batch_result = processor.process_batch(file_paths, output_dir="./processed_docs")

    print("Document processor ready!")
    print("Processing capabilities:", processor.get_processing_stats())