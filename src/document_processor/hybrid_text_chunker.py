"""
Hybrid Text Chunker: Custom implementation with selective LangChain integration
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import re
from abc import ABC, abstractmethod



try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class TextChunk:
    """Custom chunk representation with rich metadata"""
    content: str
    start_idx: int
    end_idx: int
    chunk_type: str
    metadata: Dict
    semantic_density: Optional[float] = None  # Custom metric
    chunk_id: Optional[str] = None


class BaseChunker(ABC):
    """Abstract base for different chunking strategies"""

    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        pass


class CustomRecursiveChunker(BaseChunker):
    """
    Custom implementation of recursive text splitting

    This demonstrates understanding of the recursive splitting algorithm
    while providing more control over the process than LangChain's implementation.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Define separators in order of preference (custom hierarchy)
        self.separators = [
            "\n\n\n",  # Triple newline (section breaks)
            "\n\n",  # Double newline (paragraph breaks)
            "\n",  # Single newline (line breaks)
            ". ",  # Sentence ends
            "! ",  # Exclamation sentences
            "? ",  # Question sentences
            "; ",  # Semicolon breaks
            ", ",  # Comma breaks
            " ",  # Word breaks
            ""  # Character breaks (last resort)
        ]

    def chunk_text(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        Custom recursive chunking implementation

        This shows understanding of how recursive splitting works:
        1. Try each separator in order
        2. Split text if it's too long
        3. Recursively process each part
        4. Combine with overlap
        """
        if len(text) <= self.chunk_size:
            return [self._create_chunk(text, 0, len(text), metadata, "single")]

        return self._recursive_split(text, self.separators.copy(), metadata)

    def _recursive_split(self, text: str, separators: List[str],
                         metadata: Dict = None) -> List[TextChunk]:
        """Core recursive splitting algorithm"""

        if not separators:
            # No more separators, force split by characters
            return self._force_split(text, metadata)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split
            return self._force_split(text, metadata)

        # Split by current separator
        splits = text.split(separator)

        # If we get good splits, process them
        if len(splits) > 1:
            return self._process_splits(splits, separator, remaining_separators, metadata)
        else:
            # Current separator didn't work, try next one
            return self._recursive_split(text, remaining_separators, metadata)

    def _process_splits(self, splits: List[str], separator: str,
                        remaining_separators: List[str], metadata: Dict = None) -> List[TextChunk]:
        """Process the splits from a separator"""
        chunks = []
        current_chunk = ""
        start_idx = 0

        for i, split in enumerate(splits):
            # Add separator back (except for last split)
            if i < len(splits) - 1:
                split_with_sep = split + separator
            else:
                split_with_sep = split

            # Check if adding this split would exceed

            potential_chunk = current_chunk + split_with_sep

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is ready, save it
                if current_chunk:
                    chunk_end = start_idx + len(current_chunk)
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), start_idx, chunk_end, metadata, "recursive"
                    ))

                    # Handle overlap
                    if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        start_idx = chunk_end - len(overlap_text)
                        current_chunk = overlap_text + split_with_sep
                    else:
                        start_idx = chunk_end
                        current_chunk = split_with_sep
                else:
                    # Split is too large, needs further splitting
                    if len(split_with_sep) > self.chunk_size:
                        sub_chunks = self._recursive_split(split_with_sep, remaining_separators, metadata)
                        chunks.extend(sub_chunks)
                        start_idx += len(split_with_sep)
                        current_chunk = ""
                    else:
                        current_chunk = split_with_sep

        # Add final chunk
        if current_chunk.strip():
            chunk_end = start_idx + len(current_chunk)
            chunks.append(self._create_chunk(
                current_chunk.strip(), start_idx, chunk_end, metadata, "recursive"
            ))

        return chunks

    def _force_split(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """Force split by characters when no good separators work"""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(self._create_chunk(
                chunk_text, start, end, metadata, "forced"
            ))

            start = end - self.chunk_overlap

        return chunks

    def _create_chunk(self, content: str, start: int, end: int,
                      metadata: Dict = None, chunk_type: str = "custom") -> TextChunk:
        """Create a TextChunk with comprehensive metadata"""

        chunk_metadata = {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': self._count_sentences(content),
            'has_code': self._contains_code(content),
            'has_math': self._contains_math(content),
            'readability_score': self._calculate_readability(content),
            'chunker_type': 'custom_recursive'
        }

        if metadata:
            chunk_metadata.update(metadata)

        return TextChunk(
            content=content,
            start_idx=start,
            end_idx=end,
            chunk_type=chunk_type,
            metadata=chunk_metadata,
            semantic_density=self._calculate_semantic_density(content)
        )

    def _count_sentences(self, text: str) -> int:
        """Custom sentence counting"""
        sentence_endings = re.findall(r'[.!?]+', text)
        return len(sentence_endings)

    def _contains_code(self, text: str) -> bool:
        """Detect code snippets"""
        code_indicators = [
            r'```',  # Code blocks
            r'def\s+\w+\(',  # Function definitions
            r'class\s+\w+',  # Class definitions
            r'import\s+\w+',  # Import statements
            r'\w+\.\w+\(',  # Method calls
        ]
        return any(re.search(pattern, text) for pattern in code_indicators)

    def _contains_math(self, text: str) -> bool:
        """Detect mathematical content"""
        math_indicators = [
            r'\$.*?\$',  # LaTeX math
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'[α-ωΑ-Ω]',  # Greek letters
            r'\b\d+\.\d+\b',  # Decimal numbers
            r'[∑∫∂∇]',  # Math symbols
        ]
        return any(re.search(pattern, text) for pattern in math_indicators)

    def _calculate_readability(self, text: str) -> float:
        """Simple readability score (Flesch-like)"""
        words = text.split()
        sentences = self._count_sentences(text)

        if not words or not sentences:
            return 0.0

        avg_sentence_length = len(words) / sentences
        # Simplified readability: lower score = more complex
        return max(0, 100 - (avg_sentence_length * 2))

    def _calculate_semantic_density(self, text: str) -> float:
        """Custom metric for semantic information density"""
        words = text.split()
        if not words:
            return 0.0

        # Simple heuristic: more unique words = higher density
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        return len(unique_words) / len(words)


class LangChainChunker(BaseChunker):
    """
    LangChain wrapper for comparison and fallback

    This shows how to integrate LangChain components while maintaining
    our custom interface and metadata structure.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_text(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """Use LangChain but convert to our format"""

        # Create LangChain document
        doc = Document(page_content=text, metadata=metadata or {})

        # Split using LangChain
        splits = self.splitter.split_documents([doc])

        # Convert to our format
        chunks = []
        for i, split in enumerate(splits):
            # Estimate start/end positions
            start_idx = self._estimate_start_position(text, split.page_content, i)
            end_idx = start_idx + len(split.page_content)

            chunk_metadata = {
                'length': len(split.page_content),
                'word_count': len(split.page_content.split()),
                'chunker_type': 'langchain_recursive',
                'chunk_index': i
            }

            if metadata:
                chunk_metadata.update(metadata)

            chunk = TextChunk(
                content=split.page_content,
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_type="langchain",
                metadata=chunk_metadata
            )
            chunks.append(chunk)

        return chunks

    def _estimate_start_position(self, full_text: str, chunk_content: str, chunk_index: int) -> int:
        """Estimate start position (LangChain doesn't provide exact indices)"""
        # Simple estimation based on chunk index
        estimated_position = chunk_index * (len(chunk_content) * 0.8)  # Account for overlap

        # Try to find actual position
        actual_position = full_text.find(chunk_content, int(estimated_position))

        return max(0, actual_position if actual_position >= 0 else int(estimated_position))


class SemanticChunker(BaseChunker):
    """
    Custom semantic chunker that groups related content

    This is a more advanced approach that goes beyond simple text splitting
    to create semantically coherent chunks.
    """

    def __init__(self, chunk_size: int = 512, similarity_threshold: float = 0.7):
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk_text(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        Semantic chunking based on content similarity

        This is a simplified version - in production, you'd use embeddings
        """
        # First, split into sentences
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Group related sentences
        semantic_groups = self._group_similar_sentences(sentences)

        # Convert groups to chunks
        chunks = []
        char_position = 0

        for i, group in enumerate(semantic_groups):
            group_text = " ".join(group)

            # If group is too large, split it further
            if len(group_text) > self.chunk_size:
                sub_chunks = self._split_large_group(group_text, char_position, metadata)
                chunks.extend(sub_chunks)
                char_position += len(group_text)
            else:
                chunk = TextChunk(
                    content=group_text,
                    start_idx=char_position,
                    end_idx=char_position + len(group_text),
                    chunk_type="semantic",
                    metadata=self._create_semantic_metadata(group_text, metadata, i)
                )
                chunks.append(chunk)
                char_position += len(group_text)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _group_similar_sentences(self, sentences: List[str]) -> List[List[str]]:
        """
        Group similar sentences together

        This is a simplified approach - in production, use semantic embeddings
        """
        if not sentences:
            return []

        groups = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            # Simple similarity based on shared words
            similarity = self._calculate_sentence_similarity(
                current_group[-1], sentences[i]
            )

            if similarity >= self.similarity_threshold:
                current_group.append(sentences[i])
            else:
                # Start new group
                groups.append(current_group)
                current_group = [sentences[i]]

        # Add last group
        if current_group:
            groups.append(current_group)

        return groups

    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Simple word-based similarity calculation

        In production, use sentence embeddings for better accuracy
        """
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _split_large_group(self, text: str, start_pos: int, metadata: Dict = None) -> List[TextChunk]:
        """Split large semantic groups using custom chunker"""
        custom_chunker = CustomRecursiveChunker(self.chunk_size, 50)
        sub_chunks = custom_chunker.chunk_text(text, metadata)

        # Adjust positions
        for chunk in sub_chunks:
            chunk.start_idx += start_pos
            chunk.end_idx += start_pos
            chunk.chunk_type = "semantic_split"

        return sub_chunks

    def _create_semantic_metadata(self, text: str, metadata: Dict = None, group_index: int = None) -> Dict:
        """Create metadata for semantic chunks"""
        chunk_metadata = {
            'length': len(text),
            'word_count': len(text.split()),
            'chunker_type': 'semantic',
            'semantic_group_index': group_index,
            'coherence_score': self._calculate_coherence(text)
        }

        if metadata:
            chunk_metadata.update(metadata)

        return chunk_metadata

    def _calculate_coherence(self, text: str) -> float:
        """Calculate coherence score for the text"""
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return 1.0

        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._calculate_sentence_similarity(sentences[i], sentences[i + 1])
            similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0


class HybridTextChunker:
    """
    Main chunker that combines multiple strategies

    This demonstrates how to use different chunking approaches based on
    content type and requirements, while maintaining full control.
    """

    def __init__(self, config: Dict = None):
        # self.config = config or self._default_config()

        self.config = self._default_config()

        # Initialize different chunkers
        self.custom_chunker = CustomRecursiveChunker(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap']
        )

        if LANGCHAIN_AVAILABLE and self.config.get('use_langchain_fallback', True):
            self.langchain_chunker = LangChainChunker(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
        else:
            self.langchain_chunker = None

        if self.config.get('enable_semantic_chunking', False):
            self.semantic_chunker = SemanticChunker(
                chunk_size=self.config['chunk_size'],
                similarity_threshold=self.config.get('semantic_threshold', 0.7)
            )
        else:
            self.semantic_chunker = None

    def _default_config(self) -> Dict:
        return {
            'chunk_size': 512,
            'chunk_overlap': 50,
            'strategy': 'langchain',  # 'custom', 'langchain', 'semantic', 'adaptive'
            'use_langchain_fallback': True,
            'enable_semantic_chunking': False,
            'semantic_threshold': 0.7,
            'min_chunk_size': 50,
        }

    def chunk_text(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        Main chunking method that selects appropriate strategy
        """
        if not text or len(text.strip()) < self.config['min_chunk_size']:
            return []

        strategy = self._select_strategy(text)

        if strategy == 'semantic' and self.semantic_chunker:
            chunks = self.semantic_chunker.chunk_text(text, metadata)
        elif strategy == 'langchain' and self.langchain_chunker:
            chunks = self.langchain_chunker.chunk_text(text, metadata)
        else:
            chunks = self.custom_chunker.chunk_text(text, metadata)

        # Post-process chunks
        return self._post_process_chunks(chunks, text)

    def _select_strategy(self, text: str) -> str:
        """
        Intelligent strategy selection based on content analysis
        """
        if self.config['strategy'] != 'adaptive':
            return self.config['strategy']

        # Analyze text characteristics
        analysis = self._analyze_text(text)

        if analysis['has_clear_structure'] and self.semantic_chunker:
            return 'semantic'
        elif analysis['is_complex'] and self.langchain_chunker:
            return 'langchain'
        else:
            return 'custom'

    def _analyze_text(self, text: str) -> Dict:
        """Analyze text to determine best chunking strategy"""
        lines = text.split('\n')

        # Check for structure indicators
        structured_lines = sum(1 for line in lines if re.match(r'^\s*[#\-*\d]+[.)]\s', line))
        structure_ratio = structured_lines / len(lines) if lines else 0

        # Check complexity
        avg_sentence_length = len(text.split()) / max(len(re.findall(r'[.!?]', text)), 1)

        return {
            'has_clear_structure': structure_ratio > 0.1,
            'is_complex': avg_sentence_length > 20,
            'structure_ratio': structure_ratio,
            'avg_sentence_length': avg_sentence_length,
            'total_length': len(text)
        }

    def _post_process_chunks(self, chunks: List[TextChunk], original_text: str) -> List[TextChunk]:
        """Post-process chunks for quality and consistency"""
        if not chunks:
            return chunks

        processed_chunks = []

        for i, chunk in enumerate(chunks):
            # Add chunk ID
            chunk.chunk_id = f"chunk_{i:04d}"

            # Update metadata with post-processing info
            chunk.metadata.update({
                'chunk_id': chunk.chunk_id,
                'total_chunks': len(chunks),
                'chunk_index': i,
                'relative_position': i / len(chunks) if len(chunks) > 1 else 0.0,
                'post_processed': True
            })

            processed_chunks.append(chunk)

        return processed_chunks

    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict:
        """Generate statistics about the chunking results"""
        if not chunks:
            return {}

        chunk_lengths = [len(chunk.content) for chunk in chunks]
        chunk_types = [chunk.chunk_type for chunk in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'chunk_types': dict(zip(*zip(*[(t, chunk_types.count(t)) for t in set(chunk_types)]))),
            'avg_semantic_density': sum(c.semantic_density for c in chunks if c.semantic_density) / len(chunks),
            'total_characters': sum(chunk_lengths),
        }
