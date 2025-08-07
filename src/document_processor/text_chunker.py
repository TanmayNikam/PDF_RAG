import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    content: str
    start_idx: int
    end_idx: int
    chunk_type: str
    metadata: Dict


class AdvancedTextChunker:
    """Advanced text chunking with semantic awareness"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict:
        return {
            'chunk_size': 512,
            'chunk_overlap': 50,
            'min_chunk_size': 50,
            'respect_sentence_boundaries': True,
            'respect_paragraph_boundaries': True,
            'semantic_chunking': False,  # For future enhancement
        }

    def chunk_text(self, text: str, source_metadata: Dict = None) -> List[TextChunk]:
        """
        Advanced text chunking with multiple strategies

        Args:
            text: Input text to chunk
            source_metadata: Metadata about the source document

        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) < self.config['min_chunk_size']:
            return []

        # Choose chunking strategy based on text characteristics
        if self._is_structured_text(text):
            return self._chunk_structured_text(text, source_metadata)
        else:
            return self._chunk_plain_text(text, source_metadata)

    def _chunk_plain_text(self, text: str, source_metadata: Dict = None) -> List[TextChunk]:
        """Chunk plain text with boundary respect"""
        chunks = []
        text = text.strip()

        if len(text) <= self.config['chunk_size']:
            chunk = TextChunk(
                content=text,
                start_idx=0,
                end_idx=len(text),
                chunk_type='single',
                metadata=self._create_chunk_metadata(text, source_metadata)
            )
            return [chunk]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.config['chunk_size']

            # Find optimal break point
            if end < len(text):
                end = self._find_break_point(text, start, end)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = TextChunk(
                    content=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    chunk_type='plain',
                    metadata=self._create_chunk_metadata(
                        chunk_text,
                        source_metadata,
                        chunk_index
                    )
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            start = end - self.config['chunk_overlap']
            if start <= 0:
                break

        return chunks

    def _chunk_structured_text(self, text: str, source_metadata: Dict = None) -> List[TextChunk]:
        """Chunk structured text (with headers, lists, etc.)"""
        chunks = []

        # Split by major structural elements first
        sections = self._split_by_structure(text)

        chunk_index = 0
        for section in sections:
            if len(section['content']) <= self.config['chunk_size']:
                # Small section, keep as one chunk
                chunk = TextChunk(
                    content=section['content'],
                    start_idx=section['start'],
                    end_idx=section['end'],
                    chunk_type=section['type'],
                    metadata=self._create_chunk_metadata(
                        section['content'],
                        source_metadata,
                        chunk_index
                    )
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Large section, need to chunk further
                section_chunks = self._chunk_plain_text(
                    section['content'],
                    source_metadata
                )

                for chunk in section_chunks:
                    chunk.chunk_type = f"structured_{section['type']}"
                    chunk.metadata['section_type'] = section['type']
                    chunks.append(chunk)
                    chunk_index += 1

        return chunks

    def _find_break_point(self, text: str, start: int, preferred_end: int) -> int:
        """Find optimal break point respecting boundaries"""

        # Look for paragraph breaks first
        if self.config['respect_paragraph_boundaries']:
            paragraph_break = text.rfind('\n\n', start, preferred_end)
            if paragraph_break > start:
                return paragraph_break + 2

        # Look for sentence breaks
        if self.config['respect_sentence_boundaries']:
            sentence_patterns = [r'\.[\s\n]', r'![\s\n]', r'\?[\s\n]']

            for pattern in sentence_patterns:
                matches = list(re.finditer(pattern, text[start:preferred_end]))
                if matches:
                    last_match = matches[-1]
                    return start + last_match.end()

        # Look for other natural breaks
        natural_breaks = ['\n', '. ', '! ', '? ', '; ', ', ']

        for break_char in natural_breaks:
            break_pos = text.rfind(break_char, start, preferred_end)
            if break_pos > start:
                return break_pos + len(break_char)

        # No good break found, use preferred end
        return preferred_end

    def _is_structured_text(self, text: str) -> bool:
        """Detect if text has clear structure (headers, lists, etc.)"""
        # Simple heuristics for structure detection
        structure_indicators = [
            r'^\s*#+\s',  # Markdown headers
            r'^\s*\d+\.\s',  # Numbered lists
            r'^\s*[-•*]\s',  # Bullet lists
            r'^\s*[A-Z][^a-z]*:',  # Section headers (all caps followed by colon)
        ]

        line_count = len(text.split('\n'))
        structured_lines = 0

        for line in text.split('\n')[:50]:  # Check first 50 lines
            for pattern in structure_indicators:
                if re.match(pattern, line):
                    structured_lines += 1
                    break

        # If more than 10% of lines show structure, consider it structured
        return structured_lines / max(line_count, 1) > 0.1

    def _split_by_structure(self, text: str) -> List[Dict]:
        """Split text by structural elements"""
        sections = []
        lines = text.split('\n')

        current_section = {
            'content': '',
            'type': 'content',
            'start': 0,
            'end': 0
        }

        char_position = 0

        for line in lines:
            line_with_newline = line + '\n'

            # Detect section type
            section_type = self._detect_line_type(line)

            if section_type != current_section['type'] and current_section['content']:
                # Save current section
                current_section['end'] = char_position
                sections.append(current_section.copy())

                # Start new section
                current_section = {
                    'content': line_with_newline,
                    'type': section_type,
                    'start': char_position,
                    'end': 0
                }
            else:
                current_section['content'] += line_with_newline

            char_position += len(line_with_newline)

        # Add last section
        if current_section['content']:
            current_section['end'] = char_position
            sections.append(current_section)

        return sections

    def _detect_line_type(self, line: str) -> str:
        """Detect the type of a text line"""
        line = line.strip()

        if not line:
            return 'empty'

        # Header patterns
        if re.match(r'^\s*#+\s', line):
            return 'markdown_header'
        elif re.match(r'^\s*[A-Z][^a-z]*:\s*', line):
            return 'section_header'
        # List patterns
        elif re.match(r'^\s*\d+\.\s', line):
            return 'numbered_list'
        elif re.match(r'^\s*[-•*]\s', line):
            return 'bullet_list'
        # Special content
        elif re.match(r'^\s*```', line):
            return 'code_block'
        elif re.match(r'^\s*\|.*\|', line):
            return 'table_row'

        return 'content'


    def _create_chunk_metadata(self, chunk_text: str, source_metadata: Dict = None,
                               chunk_index: int = None) -> Dict:
        """Create comprehensive metadata for a chunk"""
        metadata = {
            'length': len(chunk_text),
            'word_count': len(chunk_text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', chunk_text)),
            'has_numbers': bool(re.search(r'\d', chunk_text)),
            'has_special_chars': bool(re.search(r'[^\w\s]', chunk_text)),
            'language_hints': self._detect_language_hints(chunk_text),
        }

        if chunk_index is not None:
            metadata['chunk_index'] = chunk_index

        if source_metadata:
            metadata.update(source_metadata)

        return metadata


    def _detect_language_hints(self, text: str) -> Dict:
        """Simple language detection hints"""
        hints = {
            'has_code': bool(re.search(r'[{}();]|def |class |import |function', text)),
            'has_math': bool(re.search(r'[∑∫∂α-ωΑ-Ω]|\\[a-zA-Z]+', text)),
            'has_citations': bool(re.search(r'\[[0-9]+\]|\([0-9]{4}\)', text)),
            'has_urls': bool(re.search(r'https?://|www\.', text)),
        }
        return hints