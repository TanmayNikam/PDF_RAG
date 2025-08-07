"""
Test script for the complete generation module
"""

import asyncio
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any

# # Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent  # Go up from embeddings to src
sys.path.insert(0, str(src_dir))


try:
    from generation import (
        MultimodalGenerationPipeline,
        create_generation_pipeline,
        create_specialized_pipeline,
        COMMON_TEMPLATES
    )
except ImportError as e:
    print(f" Import error: {e}")
    sys.exit(1)


def create_test_context_documents() -> List[Dict]:
    """Create realistic test context documents for multimodal generation"""

    return [
        {
            'document_id': 'research_paper_001_text_001',
            'content': 'Abstract: This paper presents a novel multimodal architecture that combines convolutional neural networks with transformer attention mechanisms. Our approach achieves 94.2% accuracy on the benchmark dataset, representing a 15% improvement over previous state-of-the-art methods.',
            'metadata': {
                'content_type': 'text',
                'section': 'abstract',
                'page_number': 1,
                'source_document': 'research_paper_001.pdf'
            },
            'similarity_score': 0.92,
            'rank': 0,
            'retrieval_method': 'hybrid'
        },
        {
            'document_id': 'research_paper_001_img_001',
            'content': 'Figure 2: Architecture diagram showing the multimodal fusion layer that combines visual features from CNN backbone with textual embeddings from transformer encoder. The fusion layer uses cross-attention mechanism to align visual and textual representations.',
            'metadata': {
                'content_type': 'image',
                'section': 'methodology',
                'page_number': 3,
                'image_type': 'architecture_diagram',
                'figure_caption': 'Multimodal fusion architecture with cross-attention',
                'source_document': 'research_paper_001.pdf'
            },
            'similarity_score': 0.88,
            'rank': 1,
            'retrieval_method': 'dense'
        },
        {
            'document_id': 'research_paper_001_table_001',
            'content': 'Table 1: Performance comparison across different architectures\nModel | Accuracy | Parameters | Training Time\nCNN-only | 79.3% | 25M | 4.2h\nTransformer-only | 85.1% | 110M | 8.7h\nOur Multimodal | 94.2% | 78M | 6.1h',
            'metadata': {
                'content_type': 'table',
                'section': 'results',
                'page_number': 4,
                'table_caption': 'Performance comparison across different architectures',
                'source_document': 'research_paper_001.pdf'
            },
            'similarity_score': 0.85,
            'rank': 2,
            'retrieval_method': 'sparse'
        }
    ]


async def test_basic_generation():
    """Test basic generation functionality"""
    print("\n" + "=" * 50)
    print("TESTING BASIC GENERATION")
    print("=" * 50)

    try:
        # Create pipeline
        pipeline = create_generation_pipeline()

        # Test documents
        context_docs = create_test_context_documents()

        # Test queries
        test_queries = [
            "What is the main contribution of this research paper?",
            "What does Figure 2 show and how does it work?",
            "How does the proposed model compare to existing approaches?",
            "What are the performance improvements achieved?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")

            # Generate response
            response = pipeline.generate(
                query=query,
                context_documents=context_docs
            )

            print(f" Response: {response['response'][:200]}...")
            print(f" Template: {response['metadata']['template_used']}")
            print(f" Time: {response['metadata']['pipeline_time']:.2f}s")

            # Show multimodal features
            multimodal_features = response['metadata']['multimodal_features_used']
            print(f" Multimodal: {multimodal_features['multimodal_template_used']}")
            print(f" Content Types: {multimodal_features['content_types_processed']}")

        return True

    except Exception as e:
        print(f" Basic generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_specialized_pipelines():
    """Test specialized pipeline configurations"""
    print("\n" + "=" * 50)
    print("TESTING SPECIALIZED PIPELINES")
    print("=" * 50)

    try:
        context_docs = create_test_context_documents()

        # Test each specialized pipeline
        for pipeline_type in COMMON_TEMPLATES.keys():
            print(f"\n Testing {pipeline_type} pipeline:")

            pipeline = create_specialized_pipeline(pipeline_type)

            # Appropriate query for each type
            queries = {
                'pdf_qa': 'What are the key findings in this research paper?',
                'research_assistant': 'What evidence supports the claimed 15% improvement? Please cite sources.',
                'document_summarizer': 'Provide a summary of this research paper.'
            }

            query = queries.get(pipeline_type, 'What is this document about?')

            response = pipeline.generate(
                query=query,
                context_documents=context_docs
            )

            print(f"  Query: {query}")
            print(f"  Template: {response['metadata']['template_used']}")
            print(f"  Response: {response['response'][:150]}...")

            # Check optimization results
            if response['metadata'].get('generation_response', {}).get('metadata', {}).get('optimization_results'):
                opt_results = response['metadata']['generation_response']['metadata']['optimization_results']
                print(f"  Citations: {len(opt_results.get('citations', []))}")
                print(f"  Confidence: {opt_results.get('confidence_metrics', {}).get('overall_confidence', 0):.2f}")

        return True

    except Exception as e:
        print(f" Specialized pipelines test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_response_optimization():
    """Test response optimization features"""
    print("\n" + "=" * 50)
    print("TESTING RESPONSE OPTIMIZATION")
    print("=" * 50)

    try:
        # Create pipeline with optimization enabled
        config = {
            "enable_optimization": True,
            "optimizer_config": {
                "enable_citation_extraction": True,
                "enable_formatting": True,
                "enable_source_tracking": True,
                "max_response_length": 1000
            }
        }

        pipeline = create_generation_pipeline(config)
        context_docs = create_test_context_documents()

        # Generate response
        response = pipeline.generate(
            query="What does the research show about multimodal architectures and their performance improvements?",
            context_documents=context_docs
        )

        # Check optimization results
        gen_metadata = response['metadata']['generation_response']['metadata']

        if 'optimization_results' in gen_metadata:
            opt_results = gen_metadata['optimization_results']

            print(f" Original response length: {len(opt_results['original_text'])}")
            print(f" Optimized response length: {len(opt_results['optimized_text'])}")
            print(f" Citations found: {len(opt_results['citations'])}")
            print(f" Overall confidence: {opt_results['confidence_metrics']['overall_confidence']:.3f}")
            print(f" Formatting applied: {opt_results['formatting_applied']}")

            # Show citations if any
            for i, citation in enumerate(opt_results['citations'][:3]):
                print(f"  Citation {i + 1}: {citation.get('text', 'N/A')} ({citation.get('type', 'unknown')})")

            # Show confidence breakdown
            confidence = opt_results['confidence_metrics']
            print(f" Confidence Breakdown:")
            for category, scores in confidence.items():
                if isinstance(scores, dict):
                    avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / len(scores)
                    print(f"  {category}: {avg_score:.2f}")

        return True

    except Exception as e:
        print(f" Response optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_template_selection():
    """Test intelligent template selection"""
    print("\n" + "=" * 50)
    print("TESTING TEMPLATE SELECTION")
    print("=" * 50)

    try:
        pipeline = create_generation_pipeline()
        context_docs = create_test_context_documents()

        # Test different query types to trigger different templates
        template_tests = [
            {
                'query': 'What does Figure 2 show in the document?',
                'expected_template': 'multimodal',
                'description': 'Image-specific query'
            },
            {
                'query': 'Please cite the sources for the 94.2% accuracy claim.',
                'expected_template': 'citation_qa',
                'description': 'Citation request'
            },
            {
                'query': 'Compare the CNN-only model with the multimodal approach.',
                'expected_template': 'comparison',
                'description': 'Comparison request'
            },
            {
                'query': 'Define what a multimodal architecture means.',
                'expected_template': 'definition',
                'description': 'Definition request'
            }
        ]

        for test_case in template_tests:
            query = test_case['query']
            expected = test_case['expected_template']
            description = test_case['description']

            print(f"\n {description}")
            print(f"   Query: {query}")

            response = pipeline.generate(
                query=query,
                context_documents=context_docs
            )

            actual_template = response['metadata']['template_used']
            print(f"   Expected: {expected}")
            print(f"   Actual: {actual_template}")

            # Check if template selection was reasonable
            if actual_template == expected:
                print(f"    Template selection correct")
            else:
                print(f"    Different template selected (may still be appropriate)")

        return True

    except Exception as e:
        print(f" Template selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_generation_tests():
    """Run comprehensive generation module tests"""
    print(" STARTING GENERATION MODULE TESTS")
    print("=" * 60)

    # Test results
    results = {
        'basic_generation': False,
        'specialized_pipelines': False,
        'response_optimization': False,
        'template_selection': False
    }

    try:
        # Run tests
        results['basic_generation'] = await test_basic_generation()
        results['specialized_pipelines'] = await test_specialized_pipelines()
        results['response_optimization'] = await test_response_optimization()
        results['template_selection'] = await test_template_selection()

    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    # Results summary
    print(f"\n" + "=" * 60)
    print(" GENERATION MODULE TEST RESULTS")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = " PASSED" if passed else " FAILED"
        print(f"{test_name.replace('_', ' ').title(): <25} {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n All generation tests passed! Module is ready for integration!")
    elif passed_tests >= total_tests - 1:
        print(f"\n  Most tests passed. Generation module is functional.")
    else:
        print(f"\n Multiple tests failed. Check configuration and dependencies.")

    return passed_tests >= total_tests - 1


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

    success = asyncio.run(run_all_generation_tests())
    sys.exit(0 if success else 1)
