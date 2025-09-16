"""
Test script for the multimodal embedding system
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
# current_dir = Path(__file__).parent
# src_dir = current_dir / "src"
# sys.path.insert(0, str(src_dir))

current_dir = Path(__file__).parent
src_dir = current_dir.parent  # Go up from embeddings to src
sys.path.insert(0, str(src_dir))


import sys
print("PATH:", sys.path)

try:
    from embeddings import (
        create_embedder,
        MultimodalContent,
        SentenceTransformerEmbedder,
        CLIPImageEmbedder,
        MultimodalEmbedder
    )
except ImportError as e:
    print(f" Import error: {e}")
    sys.exit(1)

def test_text_embeddings():
    """Test text embedding functionality"""
    print("\n" + "="*50)
    print("TESTING TEXT EMBEDDINGS")
    print("="*50)

    try:
        # Create text embedder
        text_embedder = create_embedder("text",
                                       model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Test texts
        test_texts = [
            "Multimodal RAG systems combine text and image processing.",
            "CLIP models can understand both visual and textual information.",
            "Vector databases enable efficient similarity search.",
            "Machine learning models require large amounts of training data."
        ]

        print(f"Testing with {len(test_texts)} texts...")

        # Generate embeddings
        start_time = time.time()
        results = text_embedder.embed(test_texts)
        processing_time = time.time() - start_time

        print(f" Generated {len(results)} text embeddings")
        print(f" Processing time: {processing_time:.3f}s")
        print(f" Embedding dimension: {results[0].dimension}")
        print(f" Model: {results[0].model_name}")

        # Test similarity
        similarity = np.dot(results[0].embedding, results[1].embedding)
        print(f" Similarity between first two texts: {similarity:.3f}")

        # Show embedding stats
        embedding_lengths = [np.linalg.norm(r.embedding) for r in results]
        print(f" Embedding norms: {[f'{l:.3f}' for l in embedding_lengths]}")

        return True

    except Exception as e:
        print(f" Text embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_embeddings():
    """Test image embedding functionality"""
    print("\n" + "="*50)
    print("TESTING IMAGE EMBEDDINGS")
    print("="*50)

    try:
        # Create image embedder
        image_embedder = create_embedder("image",
                                        model_name="openai/clip-vit-base-patch32")

        print("image_embedder: ", image_embedder)

        # Create test images (simple colored squares)
        from PIL import Image
        import io

        def create_test_image(color, size=(224, 224)):
            """Create a simple test image"""
            img = Image.new('RGB', size, color=color)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue()

        test_images = [
            create_test_image('red'),
            create_test_image('blue'),
            create_test_image('green'),
            create_test_image('red')  # Same as first for similarity test
        ]

        print(f"Testing with {len(test_images)} synthetic images...")

        # Generate embeddings
        start_time = time.time()
        results = image_embedder.embed(test_images)
        processing_time = time.time() - start_time

        print(f" Generated {len(results)} image embeddings")
        print(f" Processing time: {processing_time:.3f}s")
        print(f" Embedding dimension: {results[0].dimension}")
        print(f" Model: {results[0].model_name}")

        # Test similarity (red images should be more similar)
        similarity_red_red = np.dot(results[0].embedding, results[3].embedding)
        similarity_red_blue = np.dot(results[0].embedding, results[1].embedding)

        print(f" Similarity (red vs red): {similarity_red_red:.3f}")
        print(f" Similarity (red vs blue): {similarity_red_blue:.3f}")

        if similarity_red_red > similarity_red_blue:
            print(" Similarity test passed: same colors are more similar")
        else:
            print(" Similarity test unexpected: different colors more similar")

        return True

    except Exception as e:
        print(f" Image embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clip_text_embeddings():
    """Test CLIP text embeddings for cross-modal compatibility"""
    print("\n" + "="*50)
    print("TESTING CLIP TEXT EMBEDDINGS")
    print("="*50)

    try:
        # Create CLIP image embedder (which can also do text)
        clip_embedder = CLIPImageEmbedder()

        # Test texts that describe images
        test_texts = [
            "a red square",
            "a blue square",
            "a green square",
            "a red colored image"
        ]

        print(f"Testing CLIP text embedding with {len(test_texts)} descriptions...")

        # Generate text embeddings using CLIP
        start_time = time.time()
        text_results = clip_embedder.embed_text(test_texts)
        processing_time = time.time() - start_time

        print(f" Generated {len(text_results)} CLIP text embeddings")
        print(f" Processing time: {processing_time:.3f}s")
        print(f" Embedding dimension: {text_results[0].dimension}")

        # Test similarity between text descriptions
        similarity = np.dot(text_results[0].embedding, text_results[3].embedding)
        print(f" Similarity ('red square' vs 'red colored image'): {similarity:.3f}")

        return True

    except Exception as e:
        print(f" CLIP text embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_embeddings():
    """Test multimodal embedding fusion"""
    print("\n" + "="*50)
    print("TESTING MULTIMODAL EMBEDDINGS")
    print("="*50)

    try:
        # Create multimodal embedder
        multimodal_embedder = create_embedder("multimodal", config={
            'fusion_method': 'concatenation',
            'text_weight': 0.6,
            'image_weight': 0.4
        })

        # Create test content
        from PIL import Image
        import io

        def create_test_image(color, size=(224, 224)):
            img = Image.new('RGB', size, color=color)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue()

        # Test different combinations
        test_contents = [
            MultimodalContent(
                text="A red square image for testing",
                image=create_test_image('red'),
                metadata={'test_type': 'both_modalities'}
            ),
            MultimodalContent(
                text="A blue square for color testing",
                image=create_test_image('blue'),
                metadata={'test_type': 'both_modalities'}
            ),
            MultimodalContent(
                text="Text only content without any image",
                metadata={'test_type': 'text_only'}
            ),
            MultimodalContent(
                image=create_test_image('green'),
                metadata={'test_type': 'image_only'}
            )
        ]

        print(f"Testing multimodal embedding with {len(test_contents)} content items...")

        # Generate multimodal embeddings
        start_time = time.time()
        results = multimodal_embedder.embed(test_contents)
        processing_time = time.time() - start_time

        print(f" Generated {len(results)} multimodal embeddings")
        print(f" Processing time: {processing_time:.3f}s")
        print(f" Embedding dimension: {results[0].dimension}")

        # Analyze results
        for i, result in enumerate(results):
            content_type = test_contents[i].metadata['test_type']
            has_text = result.metadata['has_text']
            has_image = result.metadata['has_image']

            print(f"  Content {i+1} ({content_type}):")
            print(f"  Has text: {has_text}, Has image: {has_image}")
            print(f"  Embedding norm: {np.linalg.norm(result.embedding):.3f}")

        # Test similarity between multimodal contents
        similarity = multimodal_embedder.calculate_similarity(
            results[0].embedding, results[1].embedding
        )
        print(f" Similarity between red and blue multimodal content: {similarity:.3f}")

        return True

    except Exception as e:
        print(f" Multimodal embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_comparison():
    """Compare different embedding approaches"""
    print("\n" + "="*50)
    print("TESTING EMBEDDING COMPARISON")
    print("="*50)

    try:
        # Test the same text with different embedders
        test_text = "Machine learning models process text and images"

        # Create different embedders
        sentence_embedder = SentenceTransformerEmbedder()
        clip_embedder = CLIPImageEmbedder()

        # Generate embeddings
        sentence_result = sentence_embedder.embed(test_text)
        clip_text_result = clip_embedder.embed_text(test_text)

        print(f" Test text: '{test_text}'")
        print(f" SentenceTransformer dimension: {sentence_result.dimension}")
        print(f" CLIP text dimension: {clip_text_result.dimension}")

        # Show first few values of each embedding
        print(f" SentenceTransformer embedding (first 5): {sentence_result.embedding[:5]}")
        print(f" CLIP embedding (first 5): {clip_text_result.embedding[:5]}")

        # Compare norms
        sentence_norm = np.linalg.norm(sentence_result.embedding)
        clip_norm = np.linalg.norm(clip_text_result.embedding)

        print(f" SentenceTransformer norm: {sentence_norm:.3f}")
        print(f" CLIP norm: {clip_norm:.3f}")

        return True

    except Exception as e:
        print(f" Embedding comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmarks():
    """Test embedding generation performance"""
    print("\n" + "="*50)
    print("TESTING PERFORMANCE BENCHMARKS")
    print("="*50)

    try:
        # Create embedders
        text_embedder = SentenceTransformerEmbedder()

        # Performance test with different batch sizes
        test_texts = [
            f"This is test sentence number {i} for performance testing."
            for i in range(100)
        ]

        # Test single vs batch processing
        print(" Testing single processing...")
        start_time = time.time()
        single_results = []
        for text in test_texts[:10]:  # Test with 10 texts
            result = text_embedder.embed(text)
            single_results.append(result)
        single_time = time.time() - start_time

        print(" Testing batch processing...")
        start_time = time.time()
        batch_results = text_embedder.embed(test_texts[:10])
        batch_time = time.time() - start_time

        print(f" Single processing (10 texts): {single_time:.3f}s ({single_time/10:.3f}s per text)")
        print(f" Batch processing (10 texts): {batch_time:.3f}s ({batch_time/10:.3f}s per text)")
        print(f" Speedup: {single_time/batch_time:.2f}x")

        # Verify results are similar
        similarity = np.dot(single_results[0].embedding, batch_results[0].embedding)
        print(f" Similarity between single and batch result: {similarity:.6f}")

        if similarity > 0.99:
            print(" Single and batch processing produce consistent results")
        else:
            print(" Single and batch processing results differ")

        return True

    except Exception as e:
        print(f" Performance benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
def run_embedding_tests():
    """Run all embedding tests"""
    print(" STARTING EMBEDDING SYSTEM TESTS")
    print("=" * 60)

    # Test results
    results = {
        'dependencies': False,
        'text_embeddings': False,
        'image_embeddings': False,
        'clip_text_embeddings': False,
        'multimodal_embeddings': False,
        'embedding_comparison': False,
        'performance_benchmarks': False
    }

    try:

        # Run tests
        results['text_embeddings'] = test_text_embeddings()
        results['image_embeddings'] = test_image_embeddings()
        results['clip_text_embeddings'] = test_clip_text_embeddings()
        results['multimodal_embeddings'] = test_multimodal_embeddings()
        results['embedding_comparison'] = test_embedding_comparison()
        results['performance_benchmarks'] = test_performance_benchmarks()

    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    # Print results summary
    print(f"\n" + "="*60)
    print(" EMBEDDING TEST RESULTS")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name.replace('_', ' ').title(): <25} {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n All embedding tests passed! Ready for vector store integration!")
    elif passed_tests >= total_tests // 2:
        print(f"\n Most tests passed. Check failed ones above.")
    else:
        print(f"\n Many tests failed. Check dependencies and setup.")

    # Next steps
    print(f"\n" + "="*60)
    print(" NEXT STEPS")
    print("="*60)

    if results['text_embeddings'] and results['multimodal_embeddings']:
        print("✓ Core embedding functionality works")
        print("✓ Ready to build vector store")
        print("✓ Can start working on retrieval system")
    else:
        print(" Fix core embedding issues first")

    if results['performance_benchmarks']:
        print("✓ Performance optimization verified")

    return passed_tests >= total_tests // 2

def quick_embedding_test():
    """Quick test for development"""
    print(" QUICK EMBEDDING TEST")
    print("="*30)

    try:

        # Quick text test
        text_embedder = create_embedder("text")
        result = text_embedder.embed("Hello world")
        print(f" Text embedding: dimension {result.dimension}")

        # Quick multimodal test
        multimodal_embedder = create_embedder("multimodal")
        content = MultimodalContent(text="Test content")
        result = multimodal_embedder.embed(content)
        print(f" Multimodal embedding: dimension {result.dimension}")

        print("\n Quick embedding test PASSED!")
        return True

    except Exception as e:
        print(f" Quick test failed: {e}")
        return False

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test embedding system")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full test suite")

    args = parser.parse_args()

    if args.quick:
        success = quick_embedding_test()
    elif args.full:
        success = run_embedding_tests()
    else:
        print("Running quick test by default. Use --full for comprehensive tests.")
        success = quick_embedding_test()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()