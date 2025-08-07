# test_document_processor.py
"""
Test script for the multimodal document processor

"""

import sys
import os
from pathlib import Path
import json
import requests
from typing import List


# Add src to path for imports - this ensures PyCharm can find modules
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    # Import our custom modules
    from src.document_processor import MultimodalDocumentProcessor
    from src.document_processor.hybrid_text_chunker import HybridTextChunker

    # from document_processor.multi_modal_document_processor import MultimodalDocumentProcessor
    # from srcdocument_processor.hybrid_text_chunker import HybridTextChunker

    print("✅ Successfully imported custom modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've created the src/document_processor directory and files")
    sys.exit(1)


def download_test_pdf(url: str, filename: str) -> Path:
    """Download a test PDF if it doesn't exist"""
    test_data_dir = Path("data/test_samples")
    test_data_dir.mkdir(parents=True, exist_ok=True)

    filepath = test_data_dir / filename

    if not filepath.exists():
        print(f"Downloading test PDF: {filename}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✓ Downloaded: {filepath}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            return None
    else:
        print(f"✓ Test file exists: {filepath}")

    return filepath


def setup_test_data() -> List[Path]:
    """Setup test data for processing"""
    test_files = []

    # Download some test PDFs (small academic papers)
    test_pdfs = [
        {
            'url': 'https://arxiv.org/pdf/1706.03762.pdf',  # Attention Is All You Need
            'filename': 'attention_paper.pdf'
        },
        {
            'url': 'https://arxiv.org/pdf/1810.04805.pdf',  # BERT paper  
            'filename': 'bert_paper.pdf'
        }
    ]

    for pdf_info in test_pdfs:
        filepath = download_test_pdf(pdf_info['url'], pdf_info['filename'])
        if filepath and filepath.exists():
            test_files.append(filepath)

    return test_files


def test_basic_functionality():
    """Test basic processor functionality"""
    print("\n" + "=" * 50)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 50)

    # Initialize processor with test config
    config = {
        'pdf': {
            'extract_images': True,
            'min_image_size': (50, 50),  # Lower threshold for test
            'ocr_enabled': True,
            'chunk_size': 300,  # Smaller chunks for testing
            'chunk_overlap': 30,
        },
        'image': {
            'ocr_enabled': True,
            'enhance_images': True,
            'target_size': (256, 256),  # Smaller for faster processing
        },
        'chunking': {
            'chunk_size': 300,
            'chunk_overlap': 30,
            'strategy': 'adaptive',  # Use our hybrid approach
        }
    }

    try:
        processor = MultimodalDocumentProcessor(config)

        # Test processor stats
        stats = processor.get_processing_stats()
        print("Processor capabilities:")
        print(json.dumps(stats, indent=2))

        return processor
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return None


def test_text_chunking():
    """Test text chunking functionality"""
    print("\n" + "=" * 50)
    print("TESTING TEXT CHUNKING")
    print("=" * 50)

    try:
        chunker = HybridTextChunker({
            'chunk_size': 200,
            'chunk_overlap': 20,
            'strategy': 'adaptive'
        })

        test_text = """
        This is a test document. It contains multiple sentences and paragraphs.

        The second paragraph discusses advanced text processing techniques. These techniques 
        are essential for building robust RAG systems. They help maintain context and 
        improve retrieval accuracy.

        Finally, the third paragraph concludes our test. It demonstrates how the chunking
        algorithm handles different text structures and maintains coherent boundaries.
        """

        chunks = chunker.chunk_text(test_text.strip())

        print(f"Original text length: {len(test_text)}")
        print(f"Number of chunks created: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i + 1}:")
            print(f"  Type: {chunk.chunk_type}")
            print(f"  Length: {chunk.metadata['length']}")
            print(f"  Chunker: {chunk.metadata.get('chunker_type', 'unknown')}")
            print(f"  Content: {repr(chunk.content[:50])}...")

        # Test chunking stats
        stats = chunker.get_chunking_stats(chunks)
        print(f"\nChunking Statistics:")
        print(f"  Average length: {stats['avg_chunk_length']:.1f}")
        print(f"  Chunk types: {stats['chunk_types']}")

        return len(chunks) > 0

    except Exception as e:
        print(f"❌ Text chunking test failed: {e}")
        return False


def test_single_document(processor, file_path: Path):
    """Test processing a single document"""
    print(f"\n" + "=" * 50)
    print(f"TESTING SINGLE DOCUMENT: {file_path.name}")
    print("=" * 50)

    try:
        result = processor.process_document(file_path)

        print(f"✓ Successfully processed: {file_path.name}")
        print(f"  Document ID: {result['document_id']}")
        print(f"  Document Type: {result['document_type']}")
        print(f"  Total Chunks: {result['total_chunks']}")
        print(f"  Text Chunks: {len(result.get('text_chunks', []))}")
        print(f"  Image Chunks: {len(result.get('image_chunks', []))}")
        print(f"  Table Chunks: {len(result.get('table_chunks', []))}")

        if result.get('metadata', {}).get('total_pages'):
            print(f"  Total Pages: {result['metadata']['total_pages']}")

        # Show sample content from first few chunks
        print("\nSample Content:")
        for chunk_type in ['text_chunks', 'image_chunks', 'table_chunks']:
            chunks = result.get(chunk_type, [])
            if chunks:
                chunk = chunks[0]
                content_preview = chunk['content'][:100].replace('\n', ' ')
                print(f"  {chunk_type[:-7].title()}: {content_preview}...")

        return True

    except Exception as e:
        print(f"✗ Failed to process {file_path.name}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing(processor, file_paths: List[Path]):
    """Test batch processing"""
    print(f"\n" + "=" * 50)
    print(f"TESTING BATCH PROCESSING")
    print("=" * 50)

    if not file_paths:
        print("No files available for batch processing")
        return False

    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        batch_result = processor.process_batch(file_paths, output_dir)

        print(f"✓ Batch processing completed")
        print(f"  Total Documents: {batch_result['total_documents']}")
        print(f"  Successful: {batch_result['successful_documents']}")
        print(f"  Failed: {batch_result['failed_documents']}")
        print(f"  Total Chunks: {batch_result['batch_metadata']['total_chunks']}")
        print(f"  Text Chunks: {batch_result['batch_metadata']['total_text_chunks']}")
        print(f"  Image Chunks: {batch_result['batch_metadata']['total_image_chunks']}")

        # Show processed files
        print(f"\nProcessed files saved to: {output_dir}")
        for file in output_dir.glob("*_processed.json"):
            print(f"  - {file.name}")

        return batch_result['successful_documents'] > 0

    except Exception as e:
        print(f"✗ Batch processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_text_file():
    """Create a sample text file for testing"""
    test_dir = Path("data/test_samples")
    test_dir.mkdir(parents=True, exist_ok=True)

    sample_file = test_dir / "sample_text.txt"

    sample_content = """
# Multimodal RAG Systems: A Comprehensive Overview

## Introduction

Retrieval-Augmented Generation (RAG) systems have revolutionized how we approach 
question-answering tasks by combining the power of large language models with 
external knowledge retrieval.

## Key Components

### 1. Document Processing
- Text extraction and chunking
- Image analysis and OCR
- Metadata preservation

### 2. Embedding Generation
- Text embeddings using transformer models
- Image embeddings using vision models
- Multimodal fusion techniques

### 3. Vector Storage
- Efficient similarity search
- Scalable indexing
- Real-time updates

## Challenges and Solutions

The main challenges in multimodal RAG include:

1. **Context Preservation**: Maintaining relationships between text and images
2. **Scalability**: Handling large document collections efficiently
3. **Quality Control**: Ensuring high-quality retrieval results

## Conclusion

Multimodal RAG systems represent the future of intelligent document processing
and question-answering systems.
"""

    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content.strip())

    print(f"✓ Created sample text file: {sample_file}")
    return sample_file


def quick_test():
    """Quick test for development"""
    print("🔥 QUICK TEST MODE")
    print("=" * 30)

    try:
        # Test basic functionality
        processor = test_basic_functionality()
        if not processor:
            print("\n❌ Processor initialization failed")
            return False

        # Test text chunking
        chunking_works = test_text_chunking()
        if not chunking_works:
            print("\n❌ Text chunking failed")
            return False

        print("\n✅ Quick test PASSED - Basic functionality works!")
        print("\nNext steps:")
        print("1. Run full test: python test_document_processor.py --full")
        print("2. Check that src/document_processor/ directory exists")
        print("3. Verify all Python files are in the correct locations")

        return True

    except Exception as e:
        print(f"\n❌ Quick test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run all tests"""
    print("🚀 STARTING MULTIMODAL DOCUMENT PROCESSOR TESTS")
    print("=" * 60)

    # Test results tracking
    results = {
        'basic_functionality': False,
        'text_chunking': False,
        'single_document': False,
        'batch_processing': False,
    }

    try:
        # Test 0: Dependencies
        # Test 1: Basic functionality
        processor = test_basic_functionality()
        results['basic_functionality'] = processor is not None

        if not processor:
            print("❌ Stopping tests - processor initialization failed")
            return False

        # Test 2: Text chunking
        results['text_chunking'] = test_text_chunking()

        # Test 3: Setup test data
        print(f"\n" + "=" * 50)
        print("SETTING UP TEST DATA")
        print("=" * 50)

        test_files = setup_test_data()
        sample_text = create_sample_text_file()

        # Test 4: Single document processing
        if test_files:
            results['single_document'] = test_single_document(processor, test_files[0])
        else:
            print("⚠️  No test files available, skipping document tests")

        # Test 5: Batch processing
        if test_files and len(test_files) > 1:
            results['batch_processing'] = test_batch_processing(processor, test_files[:2])
        else:
            print("⚠️  Not enough test files for batch processing")

    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Print final results
    print(f"\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title(): <20} {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your document processor is ready!")
    elif passed_tests > total_tests // 2:
        print(f"\n⚠️  Most tests passed. Check the failed ones above.")
    else:
        print(f"\n❌ Many tests failed. Check your setup and dependencies.")

    # Provide next steps
    print(f"\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)

    if results['basic_functionality']:
        print("✓ Basic setup works - you can start building embeddings next")
    else:
        print("❌ Fix basic setup first - check dependencies and file structure")

    if results['single_document']:
        print("✓ Document processing works - ready for vector store integration")
    else:
        print("❌ Document processing issues - check file permissions and dependencies")

    print("\n📁 Generated files:")
    print("- data/test_samples/ - Downloaded test PDFs")
    print("- data/processed/ - Processed document outputs")

    return passed_tests >= total_tests // 2  # Pass if at least half the tests work


def main():
    """Main function with argument parsing"""
    import argparse

    parser = argparse.ArgumentParser(description="Test the multimodal document processor")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--full", action="store_true", help="Run comprehensive test")
    parser.add_argument("--deps", action="store_true", help="Check dependencies only")

    args = parser.parse_args()

    if args.quick:
        success = quick_test()
    elif args.full:
        success = run_comprehensive_test()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
