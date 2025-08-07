from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import base64
import io
import logging

class ImageProcessor:
    """Advanced image processor for multimodal RAG"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> Dict:
        return {
            'ocr_enabled': True,
            'ocr_language': 'eng',
            'enhance_images': True, # setting false as of now, will switch to True if the compute permits
            'target_size': (512, 512),
            'quality_threshold': 0.7,
            'extract_text_regions': True,
        }

    def process_image(self, image_data: bytes, image_id: str) -> Dict:
        """
        Process an image and extract text, descriptions, and metadata

        Args:
            image_data: Raw image bytes
            image_id: Unique identifier for the image

        Returns:
            Dictionary containing processed image information
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))

            result = {
                'image_id': image_id,
                'original_size': image.size,
                'format': image.format,
                'mode': image.mode,
                'extracted_text': '',
                'text_regions': [],
                'image_type': self._classify_image_type(image),
                'quality_score': self._assess_image_quality(image),
                'processed_image_data': None
            }

            # Enhance image if needed
            if self.config['enhance_images']:
                enhanced_image = self._enhance_image(image)
            else:
                enhanced_image = image

            # Extract text using OCR
            if self.config['ocr_enabled']:
                ocr_result = self._extract_text_ocr(enhanced_image)
                result['extracted_text'] = ocr_result['text']
                result['text_regions'] = ocr_result['regions']


            # Resize and prepare for embedding
            processed_image = self._prepare_for_embedding(enhanced_image)

            # Convert back to bytes
            img_buffer = io.BytesIO()
            processed_image.save(img_buffer, format='PNG')
            result['processed_image_data'] = img_buffer.getvalue()

            return result

        except Exception as e:
            self.logger.error(f"Error processing image {image_id}: {str(e)}")
            raise

    # enhancing image if needed (only if compute permits)
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR and analysis"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)

            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)

            # Apply slight denoising
            image = image.filter(ImageFilter.SMOOTH_MORE)

            return image

        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image

    # extracting text from the images
    def _extract_text_ocr(self, image: Image.Image) -> Dict:
        """Extract text using OCR with region information"""
        try:
            # Convert PIL to OpenCV for advanced processing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                cv_image,
                lang=self.config['ocr_language'],
                output_type=pytesseract.Output.DICT
            )

            # Extract text and regions
            extracted_text = pytesseract.image_to_string(
                cv_image,
                lang=self.config['ocr_language']
            ).strip()

            # Process regions
            text_regions = []
            if self.config['extract_text_regions']:
                text_regions = self._process_ocr_regions(ocr_data)

            return {
                'text': extracted_text,
                'regions': text_regions
            }

        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}")
            return {'text': '', 'regions': []}

    def _process_ocr_regions(self, ocr_data: Dict) -> List[Dict]:
        """Process OCR data to extract text regions with confidence scores"""
        regions = []

        try:
            n_boxes = len(ocr_data['text'])

            for i in range(n_boxes):
                confidence = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()

                if confidence > 30 and text:  # Filter low-confidence detections
                    x, y, w, h = (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['width'][i],
                        ocr_data['height'][i]
                    )

                    regions.append({
                        'text': text,
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidence,
                        'word_num': ocr_data['word_num'][i],
                        'block_num': ocr_data['block_num'][i]
                    })

        except Exception as e:
            self.logger.warning(f"Error processing OCR regions: {e}")

        return regions

    def _classify_image_type(self, image: Image.Image) -> str:
        """Basic image type classification"""
        try:
            # Simple heuristics based on image properties
            width, height = image.size
            aspect_ratio = width / height

            # Check if image is likely a chart/diagram
            if self._is_likely_chart(image):
                return 'chart'
            elif self._is_likely_diagram(image):
                return 'diagram'
            elif aspect_ratio > 2 or aspect_ratio < 0.5:
                return 'banner_or_header'
            else:
                return 'general'

        except Exception as e:
            self.logger.warning(f"Image classification failed: {e}")
            return 'unknown'

    def _is_likely_chart(self, image: Image.Image) -> bool:
        """Heuristic to detect if image is a chart/graph"""
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')

            # Look for regular patterns that might indicate charts
            # This is a simplified heuristic
            img_array = np.array(gray)

            # Check for horizontal/vertical lines (common in charts)
            horizontal_lines = cv2.morphologyEx(
                img_array,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            )
            vertical_lines = cv2.morphologyEx(
                img_array,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            )

            line_pixels = np.sum(horizontal_lines > 0) + np.sum(vertical_lines > 0)
            total_pixels = img_array.size

            return (line_pixels / total_pixels) > 0.01  # 1% threshold

        except Exception:
            return False

    def _is_likely_diagram(self, image: Image.Image) -> bool:
        """Heuristic to detect if image is a diagram"""
        try:
            # Look for geometric shapes and connections
            # This is a simplified implementation
            gray = np.array(image.convert('L'))

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Count edge pixels
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size

            # Diagrams typically have more edges
            return (edge_pixels / total_pixels) > 0.05  # 5% threshold

        except Exception:
            return False

    def _assess_image_quality(self, image: Image.Image) -> float:
        """Assess image quality for processing suitability"""
        try:
            # Convert to grayscale
            gray = np.array(image.convert('L'))

            # Calculate image variance (sharpness indicator)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Simple quality score based on variance
            # Higher variance typically indicates sharper images
            quality_score = min(variance / 1000.0, 1.0)  # Normalize to 0-1

            return quality_score

        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default medium quality


    # prepare image for embedding
    def _prepare_for_embedding(self, image: Image.Image) -> Image.Image:
        """Prepare image for embedding generation"""
        try:
            # Resize to target size while maintaining aspect ratio
            target_size = self.config['target_size']

            # Calculate resize dimensions
            width, height = image.size
            target_width, target_height = target_size

            # Calculate scaling factor
            scale_w = target_width / width
            scale_h = target_height / height
            scale = min(scale_w, scale_h)

            # Resize
            new_width = int(width * scale)
            new_height = int(height * scale)

            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create a new image with target size and paste the resized image
            final_image = Image.new('RGB', target_size, (255, 255, 255))

            # Center the image
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2

            final_image.paste(resized_image, (x_offset, y_offset))

            return final_image

        except Exception as e:
            self.logger.error(f"Image preparation failed: {e}")
            return image.resize(self.config['target_size'])