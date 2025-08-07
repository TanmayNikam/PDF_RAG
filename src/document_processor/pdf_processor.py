import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import base64
from pathlib import Path


@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata"""
    content: str
    chunk_type: str  # 'text', 'image', 'table', 'mixed'
    page_number: int
    chunk_id: str
    metadata: Dict
    image_data: Optional[bytes] = None
    image_description: Optional[str] = None


class PDFProcessor:
    """Advanced PDF processor that extracts text, images, and maintains layout context"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> Dict:
        return {
            'extract_images': True,
            'min_image_size': (100, 100),  # Minimum image dimensions
            'image_quality': 85,
            'ocr_enabled': True,
            'preserve_layout': True,
            'chunk_size': 512,
            'chunk_overlap': 50,
        }

    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Main method to process a PDF and extract multimodal content

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of DocumentChunk objects containing processed content
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Use PyMuPDF for comprehensive extraction
            doc = fitz.open(str(pdf_path))
            chunks = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_chunks = self._process_page(page, page_num, pdf_path.stem)
                chunks.extend(page_chunks)

            doc.close()

            self.logger.info(f"Processed {pdf_path}: {len(chunks)} chunks extracted")
            return chunks

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise


    def _process_page(self, page, page_num: int, doc_name: str) -> List[DocumentChunk]:
        """Process a single page and extract all content types"""
        chunks = []

        # Extract text with layout preservation
        text_chunks = self._extract_text_chunks(page, page_num, doc_name)
        chunks.extend(text_chunks)

        # Extract images if enabled
        if self.config['extract_images']:
            image_chunks = self._extract_images(page, page_num, doc_name)
            chunks.extend(image_chunks)

        # Extract tables (basic implementation) (skipping as of now, focusing on text and images)
        # table_chunks = self._extract_tables(page, page_num, doc_name)
        # chunks.extend(table_chunks)

        return chunks


    # extract text
    def _extract_text_chunks(self, page, page_num: int, doc_name: str) -> List[DocumentChunk]:
        """Extract and chunk text content with context preservation"""
        chunks = []

        # Get text with layout information
        text_dict = page.get_text("dict")
        full_text = page.get_text()

        if not full_text.strip():
            return chunks

        # Basic text chunking with overlap
        text_chunks = self._chunk_text(full_text)

        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{doc_name}_page_{page_num}_text_{i}"

            chunk = DocumentChunk(
                content=chunk_text,
                chunk_type='text',
                page_number=page_num,
                chunk_id=chunk_id,
                metadata={
                    'document_name': doc_name,
                    'page_number': page_num,
                    'chunk_index': i,
                    'text_length': len(chunk_text),
                    'contains_numbers': any(char.isdigit() for char in chunk_text),
                    # 'font_info': self._extract_font_info(text_dict)
                }
            )
            chunks.append(chunk)

        return chunks

    # extract images
    def _extract_images(self, page, page_num: int, doc_name: str) -> List[DocumentChunk]:
        """Extract images from the page with metadata"""
        chunks = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                # Extract image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)

                # Skip small images
                if pix.width < self.config['min_image_size'][0] or \
                        pix.height < self.config['min_image_size'][1]:
                    pix = None
                    continue

                # Convert to PIL Image
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    img_pil = Image.open(io.BytesIO(img_data))
                else:  # CMYK: convert to RGB first
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix1.tobytes("png")
                    img_pil = Image.open(io.BytesIO(img_data))
                    pix1 = None

                # Create image chunk
                chunk_id = f"{doc_name}_page_{page_num}_img_{img_index}"

                chunk = DocumentChunk(
                    content=f"[IMAGE: {chunk_id}]",
                    chunk_type='image',
                    page_number=page_num,
                    chunk_id=chunk_id,
                    image_data=img_data,
                    metadata={
                        'document_name': doc_name,
                        'page_number': page_num,
                        'image_index': img_index,
                        'width': pix.width,
                        'height': pix.height,
                        'format': 'PNG',
                        'size_bytes': len(img_data)
                    }
                )
                chunks.append(chunk)

                pix = None

            except Exception as e:
                self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                continue

        return chunks

    # extract tables (if present in the page)

    # chunking text (can't we use recursive text splitters
    # over here rather than defining the chunking algorithm on own)?
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.config['chunk_size']:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config['chunk_size']

            # Try to break at sentence or paragraph boundaries
            if end < len(text):
                # Look for sentence endings
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)

                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.config['chunk_overlap']

        return chunks

    # extracting font info might not be required will implement latter if seems to be imp


    # following two functions are for detecting tables will check implement only if they are required.
    def _detect_table_regions(self, text_dict: Dict) -> List[Dict]:
        """Simple table detection based on text alignment"""
        # This is a basic implementation
        # For production, consider using specialized table detection libraries

        potential_tables = []

        try:
            # Look for blocks with multiple aligned columns
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue

                lines = block["lines"]
                if len(lines) < 3:  # Need at least 3 lines for a table
                    continue

                # Analyze text alignment and spacing
                x_positions = []
                for line in lines:
                    line_x_positions = []
                    for span in line.get("spans", []):
                        line_x_positions.append(span.get("bbox", [0, 0, 0, 0])[0])
                    x_positions.append(line_x_positions)

                # Simple heuristic: if multiple lines have similar x-positions, it might be a table
                if self._is_likely_table(x_positions):
                    table_text = "\n".join([
                        "".join([span.get("text", "") for span in line.get("spans", [])])
                        for line in lines
                    ])

                    potential_tables.append({
                        'text': table_text,
                        'rows': len(lines),
                        'cols': max(len(pos) for pos in x_positions) if x_positions else 0,
                        'bbox': block.get("bbox", [])
                    })

        except Exception as e:
            self.logger.warning(f"Error in table detection: {e}")

        return potential_tables

    def _is_likely_table(self, x_positions: List[List[float]]) -> bool:
        """Heuristic to determine if text layout suggests a table"""
        if len(x_positions) < 3:
            return False

        # Check if multiple lines have similar starting positions
        first_positions = [pos[0] if pos else 0 for pos in x_positions]

        # Simple variance check
        if len(set(first_positions)) > len(first_positions) * 2:  # Too much variation
            return False

        # Check for multiple columns (multiple x-positions per line)
        avg_cols = sum(len(pos) for pos in x_positions) / len(x_positions)

        return avg_cols >= 2  # At least 2 columns on average


