import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

from .pdf_processor import PDFProcessor, DocumentChunk
from .image_processor import ImageProcessor
# from .text_chunker import AdvancedTextChunker, TextChunk
from .hybrid_text_chunker import HybridTextChunker, TextChunk


class MultimodalDocumentProcessor:
    """
    Main orchestrator for multimodal document processing

    This class coordinates PDF processing, image extraction, OCR,
    and text chunking to create a comprehensive document processing pipeline.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the multimodal document processor

        Args:
            config: Configuration dictionary for all processors
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

        # Initialize individual processors
        self.pdf_processor = PDFProcessor(self.config.get('pdf', {}))
        self.image_processor = ImageProcessor(self.config.get('image', {}))
        self.text_chunker = HybridTextChunker(self.config.get('chunking', {}))

        self.logger.info("Multimodal Document Processor initialized")

    def _default_config(self) -> Dict:
        """Default configuration for all processors"""
        return {
            'pdf': {
                'extract_images': True,
                'min_image_size': (100, 100),
                'ocr_enabled': True,
                'chunk_size': 512,
                'chunk_overlap': 50,
            },
            'image': {
                'ocr_enabled': True,
                'enhance_images': True,
                'target_size': (512, 512),
                'quality_threshold': 0.7,
            },
            'chunking': {
                'chunk_size': 512,
                'chunk_overlap': 50,
                'respect_sentence_boundaries': True,
                'respect_paragraph_boundaries': True,
            },
            'output': {
                'save_processed_images': True,
                'image_format': 'PNG',
                'include_metadata': True,
            }
        }

    def process_document(self, file_path: Union[str, Path],
                         document_id: Optional[str] = None) -> Dict:
        """
        Process a document and extract all multimodal content

        Args:
            file_path: Path to the document file
            document_id: Optional custom document identifier

        Returns:
            Dictionary containing all processed content and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if not document_id:
            document_id = file_path.stem

        self.logger.info(f"Processing document: {file_path}")

        try:
            # Determine file type and process accordingly
            if file_path.suffix.lower() == '.pdf':
                return self._process_pdf_document(file_path, document_id)
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                return self._process_image_document(file_path, document_id)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def _process_pdf_document(self, file_path: Path, document_id: str) -> Dict:
        """Process a PDF document"""

        # Extract content from PDF
        pdf_chunks = self.pdf_processor.process_pdf(str(file_path))

        processed_result = {
            'document_id': document_id,
            'file_path': str(file_path),
            'document_type': 'pdf',
            'total_chunks': len(pdf_chunks),
            'text_chunks': [],
            'image_chunks': [],
            'table_chunks': [],
            'mixed_chunks': [],
            'metadata': {
                'processing_timestamp': self._get_timestamp(),
                'file_size_bytes': file_path.stat().st_size,
                'total_pages': max((chunk.page_number for chunk in pdf_chunks), default=0) + 1,
            }
        }

        # Process each chunk
        for chunk in pdf_chunks:
            processed_chunk = self._process_document_chunk(chunk)

            # Categorize chunks
            if chunk.chunk_type == 'text':
                processed_result['text_chunks'].append(processed_chunk)
            elif chunk.chunk_type == 'image':
                processed_result['image_chunks'].append(processed_chunk)
            elif chunk.chunk_type == 'table':
                processed_result['table_chunks'].append(processed_chunk)
            else:
                processed_result['mixed_chunks'].append(processed_chunk)

        # Update metadata
        processed_result['metadata'].update({
            'text_chunks_count': len(processed_result['text_chunks']),
            'image_chunks_count': len(processed_result['image_chunks']),
            'table_chunks_count': len(processed_result['table_chunks']),
            'mixed_chunks_count': len(processed_result['mixed_chunks']),
        })

        self.logger.info(f"PDF processing complete: {processed_result['total_chunks']} chunks extracted")

        return processed_result

    def _process_image_document(self, file_path: Path, document_id: str) -> Dict:
        """Process a standalone image document"""

        # Read image data
        with open(file_path, 'rb') as f:
            image_data = f.read()

        # Process image
        image_result = self.image_processor.process_image(image_data, document_id)

        processed_result = {
            'document_id': document_id,
            'file_path': str(file_path),
            'document_type': 'image',
            'total_chunks': 1,
            'text_chunks': [],
            'image_chunks': [],
            'metadata': {
                'processing_timestamp': self._get_timestamp(),
                'file_size_bytes': file_path.stat().st_size,
                'original_size': image_result['original_size'],
                'image_type': image_result['image_type'],
                'quality_score': image_result['quality_score'],
            }
        }

        # Create image chunk
        image_chunk = {
            'chunk_id': document_id,
            'content': image_result['extracted_text'],
            'chunk_type': 'image',
            'image_data': image_result['processed_image_data'],
            'text_regions': image_result['text_regions'],
            'metadata': image_result
        }

        processed_result['image_chunks'].append(image_chunk)

        # If text was extracted, create text chunks too
        if image_result['extracted_text'].strip():
            text_chunks = self.text_chunker.chunk_text(
                image_result['extracted_text'],
                {'source_type': 'image_ocr', 'document_id': document_id}
            )

            for i, text_chunk in enumerate(text_chunks):
                chunk_data = {
                    'chunk_id': f"{document_id}_text_{i}",
                    'content': text_chunk.content,
                    'chunk_type': 'text_from_image',
                    'metadata': text_chunk.metadata
                }
                processed_result['text_chunks'].append(chunk_data)

        processed_result['total_chunks'] = len(processed_result['text_chunks']) + len(processed_result['image_chunks'])

        return processed_result

    def _process_document_chunk(self, chunk: DocumentChunk) -> Dict:
        """Process an individual document chunk"""

        processed_chunk = {
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'chunk_type': chunk.chunk_type,
            'page_number': chunk.page_number,
            'metadata': chunk.metadata.copy()
        }

        # Handle image chunks
        if chunk.image_data:
            try:
                # Process the image
                image_result = self.image_processor.process_image(
                    chunk.image_data,
                    chunk.chunk_id
                )

                processed_chunk.update({
                    'image_data': image_result['processed_image_data'],
                    'extracted_text': image_result['extracted_text'],
                    'text_regions': image_result['text_regions'],
                    'image_metadata': {
                        'image_type': image_result['image_type'],
                        'quality_score': image_result['quality_score'],
                        'original_size': image_result['original_size']
                    }
                })

                # If image contains text, update content
                if image_result['extracted_text'].strip():
                    if chunk.content == f"[IMAGE: {chunk.chunk_id}]":
                        processed_chunk['content'] = image_result['extracted_text']
                    else:
                        processed_chunk['content'] += f"\n\nExtracted text: {image_result['extracted_text']}"

            except Exception as e:
                self.logger.warning(f"Failed to process image in chunk {chunk.chunk_id}: {e}")

        # Enhanced text chunking for large text chunks
        if chunk.chunk_type == 'text' and len(chunk.content) > self.config['chunking']['chunk_size']:
            # which text chunker to use?
            text_chunks = self.text_chunker.chunk_text(
                chunk.content,
                chunk.metadata
            )

            # If multiple sub-chunks were created, mark this for later handling
            if len(text_chunks) > 1:
                processed_chunk['sub_chunks'] = []
                for i, text_chunk in enumerate(text_chunks):
                    sub_chunk = {
                        'chunk_id': f"{chunk.chunk_id}_sub_{i}",
                        'content': text_chunk.content,
                        'chunk_type': 'text_sub_chunk',
                        'metadata': text_chunk.metadata
                    }
                    processed_chunk['sub_chunks'].append(sub_chunk)

        return processed_chunk

    #  process multiple documents in batch
    def process_batch(self, file_paths: List[Union[str, Path]],
                      output_dir: Optional[Path] = None) -> Dict:
        """
        Process multiple documents in batch

        Args:
            file_paths: List of file paths to process
            output_dir: Optional directory to save processed results

        Returns:
            Dictionary containing batch processing results
        """
        self.logger.info(f"Starting batch processing of {len(file_paths)} documents")

        batch_results = {
            'total_documents': len(file_paths),
            'successful_documents': 0,
            'failed_documents': 0,
            'documents': {},
            'batch_metadata': {
                'processing_timestamp': self._get_timestamp(),
                'total_chunks': 0,
                'total_text_chunks': 0,
                'total_image_chunks': 0,
            }
        }

        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(f"Processing document {i + 1}/{len(file_paths)}: {file_path}")

                result = self.process_document(file_path)
                document_id = result['document_id']

                batch_results['documents'][document_id] = result
                batch_results['successful_documents'] += 1

                # Update batch metadata
                batch_results['batch_metadata']['total_chunks'] += result['total_chunks']
                batch_results['batch_metadata']['total_text_chunks'] += len(result.get('text_chunks', []))
                batch_results['batch_metadata']['total_image_chunks'] += len(result.get('image_chunks', []))

                # Save individual result if output directory specified
                if output_dir:
                    self._save_processed_document(result, output_dir)

            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                batch_results['failed_documents'] += 1
                batch_results['documents'][str(file_path)] = {
                    'error': str(e),
                    'status': 'failed'
                }

        self.logger.info(
            f"Batch processing complete: {batch_results['successful_documents']} successful, {batch_results['failed_documents']} failed")

        return batch_results

    def _save_processed_document(self, result: Dict, output_dir: Path):
        """Save processed document results to disk"""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        document_id = result['document_id']

        # Save main result as JSON
        result_file = output_dir / f"{document_id}_processed.json"

        # Create a JSON-serializable copy
        json_result = result.copy()

        # Handle binary data
        for chunk_list_key in ['text_chunks', 'image_chunks', 'table_chunks', 'mixed_chunks']:
            if chunk_list_key in json_result:
                for chunk in json_result[chunk_list_key]:
                    if 'image_data' in chunk and chunk['image_data']:
                        # Save image separately and store path
                        image_file = output_dir / f"{chunk['chunk_id']}.png"
                        with open(image_file, 'wb') as f:
                            f.write(chunk['image_data'])
                        chunk['image_file_path'] = str(image_file)
                        del chunk['image_data']  # Remove binary data from JSON

        # Save JSON
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)

        self.logger.debug(f"Saved processed document: {result_file}")

    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_processing_stats(self) -> Dict:
        """Get statistics about the processing capabilities"""
        return {
            'processors': {
                'pdf_processor': {
                    'extract_images': self.pdf_processor.config['extract_images'],
                    'ocr_enabled': self.pdf_processor.config['ocr_enabled'],
                    'chunk_size': self.pdf_processor.config['chunk_size'],
                },
                'image_processor': {
                    'ocr_enabled': self.image_processor.config['ocr_enabled'],
                    'enhance_images': self.image_processor.config['enhance_images'],
                    'target_size': self.image_processor.config['target_size'],
                },
                'text_chunker': {
                    'chunk_size': self.text_chunker.config['chunk_size'],
                    'chunk_overlap': self.text_chunker.config['chunk_overlap'],
                    # 'respect_boundaries': self.text_chunker.config['respect_sentence_boundaries'],
                }
            },
            'supported_formats': {
                'documents': ['.pdf'],
                'images': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
            }
        }
