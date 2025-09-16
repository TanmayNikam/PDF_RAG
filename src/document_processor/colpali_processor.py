"""
ColPali-specific document processor that works with page images
"""

import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
from dataclasses import dataclass

from .pdf_processor import DocumentChunk


@dataclass
class ColPaliPageChunk:
    """Represents a document page for ColPali processing"""
    content: str  # Page description
    chunk_type: str  # Always 'colpali_page'
    page_number: int
    chunk_id: str
    metadata: Dict
    page_image: bytes  # Raw page image
    page_size: tuple  # (width, height)


class ColPaliDocumentProcessor:
    """
    Document processor specifically for ColPali

    Converts PDF pages to high-quality images without text extraction,
    as ColPali handles text understanding directly from images.
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> Dict:
        return {
            'dpi': 150,  # Higher DPI for better ColPali performance
            'image_format': 'PNG',
            'max_pages': None,  # Process all pages
            'min_page_size': (100, 100),
            'target_size': None,  # Keep original size
        }

    def process_document(self, file_path: Union[str, Path],
                         document_id: Optional[str] = None) -> Dict:
        """Process a single document for ColPali (maintains interface compatibility)"""
        file_path = Path(file_path)

        print("Process Document Started")

        if not file_path.exists():
            print("Invalid File Path, Document not found.")
            raise FileNotFoundError(f"Document not found: {file_path}")

        if not document_id:
            document_id = file_path.stem

        self.logger.info(f"Processing document for ColPali: {file_path}")

        try:
            if file_path.suffix.lower() == '.pdf':
                chunks = self.process_pdf_for_colpali(str(file_path))

                # Convert to the expected format
                processed_result = {
                    'document_id': document_id,
                    'file_path': str(file_path),
                    'document_type': 'pdf_colpali',
                    'total_chunks': len(chunks),
                    'colpali_chunks': [],
                    'metadata': {
                        'processing_timestamp': self._get_timestamp(),
                        'file_size_bytes': file_path.stat().st_size,
                        'total_pages': len(chunks),
                        'processing_method': 'colpali',
                        'dpi': self.config['dpi']
                    }
                }

                # Convert chunks to expected format
                for chunk in chunks:
                    processed_chunk = {
                        'chunk_id': chunk.chunk_id,
                        'content': chunk.content,
                        'chunk_type': chunk.chunk_type,
                        'page_number': chunk.page_number,
                        'page_image': chunk.page_image,
                        'page_size': chunk.page_size,
                        'metadata': chunk.metadata
                    }
                    processed_result['colpali_chunks'].append(processed_chunk)

                return processed_result
            else:
                raise ValueError(f"ColPali processor only supports PDF files, got: {file_path.suffix}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def process_batch(self, file_paths: List[Union[str, Path]],
                      output_dir: Optional[Path] = None) -> Dict:
        """Process multiple documents in batch for ColPali"""
        self.logger.info(f"Starting ColPali batch processing of {len(file_paths)} documents")

        batch_results = {
            'total_documents': len(file_paths),
            'successful_documents': 0,
            'failed_documents': 0,
            'documents': {},
            'batch_metadata': {
                'processing_timestamp': self._get_timestamp(),
                'total_pages': 0,
                'total_chunks': 0,
                'processing_method': 'colpali_batch'
            }
        }

        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(f"Processing document {i + 1}/{len(file_paths)}: {file_path}")
                import copy

                result = self.process_document(file_path)
                document_id = result['document_id']
                batch_results['documents'][document_id] = copy.deepcopy(result)
                batch_results['successful_documents'] += 1

                # Update batch metadata
                batch_results['batch_metadata']['total_pages'] += result['metadata']['total_pages']
                batch_results['batch_metadata']['total_chunks'] += result['total_chunks']

                # # Save individual result if output directory specified
                if output_dir:
                    self._save_colpali_document(result, output_dir)

            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                batch_results['failed_documents'] += 1
                batch_results['documents'][str(file_path)] = {
                    'error': str(e),
                    'status': 'failed'
                }

        self.logger.info(
            f"ColPali batch processing complete: {batch_results['successful_documents']} successful, "
            f"{batch_results['failed_documents']} failed"
        )

        self.logger.info(f"batch Process documents: {batch_results['documents']}")

        return batch_results


    def process_pdf_for_colpali(self, pdf_path: str) -> List[ColPaliPageChunk]:

        """Process PDF for ColPali by converting pages to images"""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            doc = fitz.open(str(pdf_path))
            chunks = []

            # print("self.config ", self.config)

            max_pages = self.config.get('max_pages', None) or len(doc)

            # print(f" max pages: {self.config['max_pages']} len doc: {len(doc)}, max_pages: {max_pages}")

            for page_num in range(min(len(doc), max_pages)):
                page = doc.load_page(page_num)
                page_chunk = self._process_page_for_colpali(page, page_num, pdf_path.stem)
                if page_chunk:
                    chunks.append(page_chunk)

            doc.close()

            self.logger.info(f"Processed {pdf_path} for ColPali: {len(chunks)} pages")
            return chunks

        except Exception as e:
            self.logger.error(f"Error processing PDF for ColPali {pdf_path}: {str(e)}")
            raise

    def _process_page_for_colpali(self, page, page_num: int, doc_name: str) -> Optional[ColPaliPageChunk]:
        """Convert a single page to high-quality image for ColPali"""
        try:
            # Get page dimensions
            rect = page.rect
            page_width, page_height = rect.width, rect.height

            # Check minimum size requirements
            if (page_width < self.config['min_page_size'][0] or
                    page_height < self.config['min_page_size'][1]):
                self.logger.warning(f"Page {page_num} too small, skipping")
                return None

            # Render page to image with high DPI
            mat = fitz.Matrix(self.config['dpi'] / 72, self.config['dpi'] / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Resize if target size specified
            if self.config.get('target_size',False):
                img = img.resize(self.config['target_size'], Image.Resampling.LANCZOS)

            # Convert back to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format=self.config['image_format'], quality=95)
            final_img_data = img_buffer.getvalue()

            # Create chunk
            chunk_id = f"{doc_name}_colpali_page_{page_num:04d}"

            chunk = ColPaliPageChunk(
                content=f"Page {page_num + 1} of {doc_name}",
                chunk_type='colpali_page',
                page_number=page_num,
                chunk_id=chunk_id,
                page_image=final_img_data,
                page_size=(img.width, img.height),
                metadata={
                    'document_name': doc_name,
                    'page_number': page_num,
                    'original_size': (page_width, page_height),
                    'rendered_size': (img.width, img.height),
                    'dpi': self.config['dpi'],
                    'image_format': self.config['image_format'],
                    'file_size_bytes': len(final_img_data),
                    'processing_method': 'colpali'
                }
            )

            pix = None  # Clean up
            return chunk

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.warning(f"Failed to process page {page_num} for ColPali: {e}")
            return None

    def _save_colpali_document(self, result: Dict, output_dir: Path):
        """Save processed ColPali document results to disk"""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        document_id = result['document_id']

        # Save main result as JSON (without binary image data)
        result_file = output_dir / f"{document_id}_colpali_processed.json"

        # Create a JSON-serializable copy
        json_result = result.copy()

        colpali_chunks = json_result.get('colpali_chunks', [])

        # Handle binary image data
        for chunk in colpali_chunks:
            if 'page_image' in chunk and chunk['page_image']:
                # Save image separately
                image_file = output_dir / f"{chunk['chunk_id']}.png"
                with open(image_file, 'wb') as f:
                    f.write(chunk['page_image'])
                chunk['page_image_path'] = str(image_file)
                del chunk['page_image']  # Remove binary data from JSON

        # Save JSON
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)

        self.logger.debug(f"Saved ColPali processed document: {result_file}")

    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()