#!/usr/bin/env python3
"""
Complete Multimodal RAG System Integration Test

This script tests the complete end-to-end integration of all 5 components:
1. Document Processing
2. Multimodal Embeddings
3. Vector Store
4. Retrieval System
5. Response Generation

Usage:
    python test_complete_rag.py --quick      # Quick integration test
    python test_complete_rag.py --full       # Comprehensive test with demo
    python test_complete_rag.py --demo       # Interactive demo only
"""

import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import logging
import traceback

# Add src to path
# current_dir = Path(__file__).parent
# if current_dir.name != 'src':
#     src_dir = current_dir / 'src'
# else:
#     src_dir = current_dir
# sys.path.insert(0, str(src_dir))

try:
    from main_rag_system import MultimodalRAGSystem, create_rag_system, quick_rag_setup
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have the complete src/ directory structure with all modules")
    sys.exit(1)



COLPALI_CONFIG_EXAMPLE = {
    'colpali': {
        'enabled': True,
        'model_name': 'vidore/colpali',
        'dpi': 150,
        'batch_size': 1,
        'visual_similarity_threshold': 0.0,
        'boost_visual_content': True,
        'page_context_window': 2
    },
    'embeddings': {
        'type': 'colpali',  # Use ColPali instead of multimodal
        'normalize_embeddings': True
    },
    'retrieval': {
        'type': 'hybrid',  # Can still use hybrid with ColPali component
        'enable_colpali': True
    }
}

def create_test_pdf_content() -> bytes:
    """Create a simple test PDF with text content"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)

        # Page 1 - Abstract and Introduction
        p.drawString(100, 750, "Multimodal AI Research Paper")
        p.drawString(100, 720, "Abstract:")
        p.drawString(120, 700, "This paper presents a novel approach to multimodal learning that combines")
        p.drawString(120, 680, "computer vision and natural language processing techniques. Our method")
        p.drawString(120, 660, "achieves state-of-the-art performance on several benchmark datasets.")

        p.drawString(100, 620, "1. Introduction")
        p.drawString(120, 600, "Multimodal AI systems have gained significant attention in recent years.")
        p.drawString(120, 580, "These systems can process and understand multiple types of data including")
        p.drawString(120, 560, "text, images, and audio simultaneously.")

        p.showPage()

        # Page 2 - Methodology
        p.drawString(100, 750, "2. Methodology")
        p.drawString(120, 720, "Our approach uses transformer-based architectures for both text and image")
        p.drawString(120, 700, "processing. We implement a cross-attention mechanism to align features")
        p.drawString(120, 680, "from different modalities.")

        p.drawString(100, 640, "2.1 Architecture")
        p.drawString(120, 620, "The model consists of separate encoders for text and images, followed")
        p.drawString(120, 600, "by a fusion layer that combines multimodal representations.")

        p.showPage()

        # Page 3 - Results
        p.drawString(100, 750, "3. Experimental Results")
        p.drawString(120, 720, "We evaluate our model on three datasets: VQA, COCO Captions, and")
        p.drawString(120, 700, "Visual Reasoning. Our method achieves 94.2% accuracy on VQA,")
        p.drawString(120, 680, "representing a 15% improvement over previous methods.")

        p.drawString(100, 640, "3.1 Performance Analysis")
        p.drawString(120, 620, "The multimodal fusion layer contributes significantly to performance")
        p.drawString(120, 600, "gains, especially on tasks requiring cross-modal understanding.")

        p.showPage()

        p.save()
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        # Fallback: create a simple text file instead
        print("⚠️  ReportLab not available, creating text file instead")
        content = """Multimodal AI Research Paper

Abstract:
This paper presents a novel approach to multimodal learning that combines
computer vision and natural language processing techniques. Our method
achieves state-of-the-art performance on several benchmark datasets.

1. Introduction
Multimodal AI systems have gained significant attention in recent years.
These systems can process and understand multiple types of data including
text, images, and audio simultaneously.

2. Methodology
Our approach uses transformer-based architectures for both text and image
processing. We implement a cross-attention mechanism to align features
from different modalities.

2.1 Architecture
The model consists of separate encoders for text and images, followed
by a fusion layer that combines multimodal representations.

3. Experimental Results
We evaluate our model on three datasets: VQA, COCO Captions, and
Visual Reasoning. Our method achieves 94.2% accuracy on VQA,
representing a 15% improvement over previous methods.

3.1 Performance Analysis
The multimodal fusion layer contributes significantly to performance
gains, especially on tasks requiring cross-modal understanding.
"""
        return content.encode('utf-8')


def create_test_documents() -> List[Path]:
    """Create test documents for the RAG system"""
    test_dir = Path(tempfile.mkdtemp(prefix="rag_test_"))
    documents = []

    # Create test PDF
    print("📄 Creating test PDF document...")
    pdf_content = create_test_pdf_content()

    if isinstance(pdf_content, bytes) and pdf_content.startswith(b'%PDF'):
        # Real PDF
        pdf_path = test_dir / "test_research_paper.pdf"
        with open(pdf_path, 'wb') as f:
            f.write(pdf_content)
        documents.append(pdf_path)
    # else:
    #     # Text fallback
    #     txt_path = test_dir / "test_research_paper.txt"
    #     with open(txt_path, 'wb') as f:
    #         f.write(pdf_content)
    #     documents.append(txt_path)

    # Create additional text documents
    additional_docs = [
        {
            'name': 'ai_overview.txt',
            'content': """Artificial Intelligence Overview

Artificial Intelligence (AI) refers to the simulation of human intelligence
in machines that are programmed to think and learn like humans. The field
encompasses several subdomains including machine learning, deep learning,
natural language processing, and computer vision.

Machine Learning Fundamentals:
- Supervised learning uses labeled data to train models
- Unsupervised learning discovers patterns in unlabeled data  
- Reinforcement learning learns through trial and error

Deep Learning Applications:
- Image recognition and computer vision
- Natural language processing and translation
- Speech recognition and synthesis
- Autonomous vehicles and robotics

Current challenges in AI include explainability, bias mitigation, and
ensuring AI systems are safe and beneficial for humanity.
"""
        },
        {
            'name': 'neural_networks.txt',
            'content': """Neural Networks and Deep Learning

Neural networks are computing systems inspired by biological neural networks.
They consist of interconnected nodes (neurons) organized in layers that
process information through weighted connections.

Architecture Components:
- Input layer: Receives raw data
- Hidden layers: Process and transform data
- Output layer: Produces final predictions
- Activation functions: Introduce non-linearity
- Weights and biases: Learnable parameters

Training Process:
1. Forward propagation: Data flows through network
2. Loss calculation: Compare predictions to targets  
3. Backpropagation: Compute gradients
4. Parameter updates: Adjust weights using optimization

Common architectures include feedforward networks, convolutional neural
networks (CNNs) for images, and recurrent neural networks (RNNs) for
sequential data like text and time series.
"""
        }
    ]

    # for doc_info in additional_docs:
    #     doc_path = test_dir / doc_info['name']
    #     with open(doc_path, 'w', encoding='utf-8') as f:
    #         f.write(doc_info['content'])
    #     documents.append(doc_path)

    print(f"✅ Created {len(documents)} test documents in {test_dir}")
    return documents, test_dir


def test_system_initialization():
    """Test system initialization"""
    print("\n" + "=" * 60)
    print("TESTING SYSTEM INITIALIZATION")
    print("=" * 60)

    try:
        print("🔧 Creating RAG system with default configuration...")
        rag_system = create_rag_system(COLPALI_CONFIG_EXAMPLE)

        print("✅ System initialized successfully")

        # Test health check
        health = rag_system.health_check()
        print(f"📊 System health: {health['status']}")

        if health['status'] == 'error':
            print("❌ System health errors:")
            for issue in health['issues']:
                print(f"  - {issue}")
            return False

        # Test system stats
        stats = rag_system.get_system_stats()
        print(f"📈 System stats:")
        print(f"  Initialized: {stats['system_status']['initialized']}")
        print(f"  Components: {sum(stats['system_status']['components_status'].values())}/5")

        return True

    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_processing(rag_system, test_documents):
    """Test document processing and indexing"""
    print("\n" + "=" * 60)
    print("TESTING DOCUMENT PROCESSING")
    print("=" * 60)

    try:
        print(f"📄 Processing {len(test_documents)} test documents...")

        # Add documents to the system
        result = rag_system.add_documents(test_documents)

        if result['success']:
            print(f"✅ Document processing successful!")
            print(f"  Documents processed: {result['documents_processed']}")
            print(f"  Vector documents created: {result['vector_documents_created']}")
            print(f"  Documents indexed: {result['documents_indexed']}")
            print(f"  Processing time: {result['processing_time']:.2f}s")

            # Check system state after adding documents
            stats = rag_system.get_system_stats()
            print(f"📊 Updated system stats:")
            print(f"  Total documents: {stats['system_status']['document_count']}")
            print(f"  Documents processed: {stats['performance_stats']['documents_processed']}")

            return True
        else:
            print(f"❌ Document processing failed: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_answering(rag_system):
    """Test end-to-end query answering"""
    print("\n" + "=" * 60)
    print("TESTING QUERY ANSWERING")
    print("=" * 60)

    test_queries = [
        {
            'question': 'What is this research about?',
            'expected_topics': ['multimodal', 'ai', 'research', 'learning']
        },
        {
            'question': 'What are the main components of neural networks?',
            'expected_topics': ['layers', 'neurons', 'weights', 'activation']
        },
        {
            'question': 'What performance was achieved in the experiments?',
            'expected_topics': ['accuracy', '94.2%', 'performance', 'results']
        },
        {
            'question': 'How do neural networks learn?',
            'expected_topics': ['training', 'backpropagation', 'optimization', 'gradient']
        },
        {
            'question': 'What are the applications of deep learning?',
            'expected_topics': ['vision', 'nlp', 'recognition', 'applications']
        }
    ]

    successful_queries = 0

    for i, test_case in enumerate(test_queries, 1):
        question = test_case['question']
        expected_topics = test_case['expected_topics']

        print(f"\n🔍 Query {i}: '{question}'")

        try:
            # Query the system
            result = rag_system.query(question, top_k=3)

            if result['success']:
                answer = result['answer']
                sources = result['sources']
                query_time = result['query_time']

                print(f"✅ Query successful (Time: {query_time:.2f}s)")
                print(f"📝 Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                print(f"📚 Sources found: {len(sources)}")

                # Check if answer contains expected topics
                answer_lower = answer.lower()
                found_topics = [topic for topic in expected_topics if topic in answer_lower]

                if found_topics:
                    print(f"✅ Found expected topics: {found_topics}")
                    successful_queries += 1
                else:
                    print(f"⚠️  Expected topics not found: {expected_topics}")

                # Show top sources
                for j, source in enumerate(sources[:2]):
                    print(f"  Source {j + 1}: Score {source['relevance_score']:.3f}, "
                          f"Type: {source['metadata']['content_type']}")

            else:
                print(f"❌ Query failed: {result['error']}")

        except Exception as e:
            print(f"❌ Query error: {e}")

    success_rate = successful_queries / len(test_queries)
    print(f"\n📊 Query Test Results:")
    print(f"  Successful queries: {successful_queries}/{len(test_queries)}")
    print(f"  Success rate: {success_rate:.1%}")

    return success_rate >= 0.6  # 60% success rate threshold


def test_system_persistence(rag_system):
    """Test saving and loading the system"""
    print("\n" + "=" * 60)
    print("TESTING SYSTEM PERSISTENCE")
    print("=" * 60)

    temp_save_dir = Path(tempfile.mkdtemp(prefix="rag_save_test_"))

    try:
        # Save the system
        print("💾 Saving RAG system...")
        save_success = rag_system.save_system(temp_save_dir)

        if not save_success:
            print("❌ Failed to save system")
            return False

        print("✅ System saved successfully")

        # Get current stats for comparison
        original_stats = rag_system.get_system_stats()
        original_doc_count = original_stats['system_status']['document_count']

        # Create new system and load
        print("📁 Creating new system and loading...")
        new_rag_system = create_rag_system(COLPALI_CONFIG_EXAMPLE)
        load_success = new_rag_system.load_system(temp_save_dir)

        if not load_success:
            print("❌ Failed to load system")
            return False

        print("✅ System loaded successfully")

        # Verify loaded system
        loaded_stats = new_rag_system.get_system_stats()
        loaded_doc_count = loaded_stats['system_status']['document_count']

        print(f"📊 Verification:")
        print(f"  Original documents: {original_doc_count}")
        print(f"  Loaded documents: {loaded_doc_count}")

        if loaded_doc_count == original_doc_count:
            print("✅ Document count matches")

            # Test a query on loaded system
            test_result = new_rag_system.query("What is multimodal AI?", top_k=2)
            if test_result['success']:
                print("✅ Query on loaded system successful")
                return True
            else:
                print("❌ Query on loaded system failed")
                return False
        else:
            print("❌ Document count mismatch")
            return False

    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        shutil.rmtree(temp_save_dir, ignore_errors=True)


def test_error_handling(rag_system):
    """Test system error handling"""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)

    try:
        # Test query with no documents (on fresh system)
        fresh_system = create_rag_system(COLPALI_CONFIG_EXAMPLE)
        result = fresh_system.query("Test query")

        if not result['success'] and 'no documents' in result['error'].lower():
            print("✅ Proper error handling for empty system")
        else:
            print("⚠️  Unexpected behavior for empty system")

        # Test invalid file path
        try:
            result = rag_system.add_documents(["/nonexistent/file.pdf"])
            if not result['success']:
                print("✅ Proper error handling for invalid files")
            else:
                print("⚠️  Expected error for invalid files")
        except Exception:
            print("✅ Exception handling for invalid files")

        # Test empty query
        result = rag_system.query("")
        print(f"Empty query handling: {'✅' if not result['success'] or len(result['answer']) > 0 else '⚠️'}")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test of the complete RAG system"""
    print("🚀 STARTING COMPREHENSIVE RAG SYSTEM TEST")
    print("=" * 70)

    # Test results tracking
    results = {
        'initialization': False,
        'document_processing': False,
        'query_answering': False,
        'persistence': False,
        'error_handling': False
    }

    test_documents = None
    test_dir = None
    rag_system = None

    try:
        # Setup test documents
        print("📄 Setting up test documents...")
        test_documents, test_dir = create_test_documents()

        # Test 1: System Initialization
        results['initialization'] = test_system_initialization()

        if results['initialization']:
            rag_system = create_rag_system(COLPALI_CONFIG_EXAMPLE)

            # Test 2: Document Processing
            results['document_processing'] = test_document_processing(rag_system, test_documents)


            # Test 3: Query Answering (only if documents were processed)
            if results['document_processing']:
                results['query_answering'] = test_query_answering(rag_system)

                # Test 4: Persistence
                results['persistence'] = test_system_persistence(rag_system)

            # Test 5: Error Handling
            results['error_handling'] = test_error_handling(rag_system)

    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if test_dir and test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

    # Print results summary
    print(f"\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title(): <25} {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    # Determine success
    core_tests = ['initialization', 'document_processing', 'query_answering']
    core_passed = sum(results[test] for test in core_tests)
    core_total = len(core_tests)

    if core_passed == core_total:
        print("\n🎉 All core tests passed! RAG system is fully functional!")
        success = True
    elif core_passed >= core_total - 1:
        print(f"\n⚠️  Most core tests passed. System is largely functional.")
        success = True
    else:
        print(f"\n❌ Core tests failed. System needs fixes.")
        success = False

    # Next steps
    print(f"\n" + "=" * 70)
    print("📋 NEXT STEPS")
    print("=" * 70)

    if success:
        print("✓ RAG system is ready for production use!")
        print("✓ You can now:")
        print("  - Add your own PDF documents")
        print("  - Query the system interactively")
        print("  - Integrate into applications")
        print("  - Save/load system state")
    else:
        print("❌ Fix the failing tests before production use")
        print("💡 Common issues:")
        print("  - Missing dependencies (check requirements)")
        print("  - LLM connectivity (Ollama, API keys)")
        print("  - File permissions")

    return success


def run_quick_test():
    """Run a quick integration test"""
    print("🔥 QUICK RAG SYSTEM INTEGRATION TEST")
    print("=" * 50)

    try:
        # Quick system test
        print("🔧 Creating RAG system...")
        rag_system = create_rag_system(COLPALI_CONFIG_EXAMPLE)

        # Health check
        health = rag_system.health_check()
        print(f"📊 System health: {health['status']}")

        if health['status'] == 'error':
            print("❌ System has critical errors:")
            for issue in health['issues']:
                print(f"  - {issue}")
            return False

        # Create simple test document
        print("📄 Creating test document...")
        test_dir = Path(tempfile.mkdtemp(prefix="quick_test_"))
        test_file = test_dir / "quick_test.txt"

        with open(test_file, 'w') as f:
            f.write("""Machine Learning Basics

Machine learning is a method of data analysis that automates analytical 
model building. It uses algorithms that iteratively learn from data, 
allowing computers to find hidden insights without being explicitly 
programmed where to look.

Types of machine learning include:
- Supervised learning: learns from labeled examples
- Unsupervised learning: finds patterns in data without labels  
- Reinforcement learning: learns through trial and error

Applications include image recognition, natural language processing, 
recommendation systems, and autonomous vehicles.
""")

        # Add document
        print("➕ Adding document to system...")
        add_result = rag_system.add_documents([test_file])

        if not add_result['success']:
            print(f" Failed to add document: {add_result['error']}")
            return False

        print(f" Added {add_result['documents_processed']} document")

        # Test query
        print("🔍 Testing query...")
        query_result = rag_system.query("What is machine learning?")

        if query_result['success']:
            print(f" Query successful!")
            print(f" Answer: {query_result['answer'][:150]}...")
            print(f" Sources: {len(query_result['sources'])}")
            print(f"️  Time: {query_result['query_time']:.2f}s")
        else:
            print(f" Query failed: {query_result['error']}")
            return False

        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

        print("\n Quick test PASSED! RAG system is working!")
        return True

    except Exception as e:
        print(f" Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_interactive_demo():
    """Run interactive demo"""
    print("🎮 INTERACTIVE RAG SYSTEM DEMO")
    print("=" * 50)

    try:
        # Setup system with sample documents
        print("🔧 Setting up RAG system with sample documents...")
        test_documents, test_dir = create_test_documents()

        rag_system = create_rag_system(COLPALI_CONFIG_EXAMPLE)
        print("rag system: ", rag_system)
        add_result = rag_system.add_documents(test_documents)

        if add_result['success']:
            print(f"✅ System ready with {add_result['documents_processed']} documents!")

            # Start interactive demo
            rag_system.interactive_demo()
        else:
            print(f"❌ Setup failed: {add_result['error']}")

        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

    except Exception as e:
        print(f"❌ Demo setup failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Complete RAG System Integration")
    parser.add_argument("--quick", action="store_true", help="Run quick integration test")
    parser.add_argument("--full", action="store_true", help="Run comprehensive test suite")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.demo:
        run_interactive_demo()
        success = True
    elif args.full:
        success = run_comprehensive_test()
    elif args.quick:
        success = run_quick_test()
    else:
        print("Running quick test by default. Use --full for comprehensive testing.")
        success = run_quick_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()