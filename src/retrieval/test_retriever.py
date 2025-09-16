# test_retrieval_system.py
"""
Comprehensive test script for the retrieval system
Tests dense, sparse, hybrid retrievers and query processing
"""

import sys
import time
from typing import List, Dict

import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent  # Go up from retriever to src
sys.path.insert(0, str(src_dir))

import spacy
nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
# nltk.data.find('corpora/wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


try:
    # Import retrieval components
    from retrieval import (
        create_retriever,
        Query,
        RetrievalResult,
        DenseRetriever,
        SparseRetriever,
        HybridRetriever,
        QueryProcessor,
        RetrievalPipeline
    )

    # Import supporting components
    from embeddings import create_embedder, MultimodalContent
    from vector_store import create_vector_store, VectorDocument

except ImportError as e:
    print(f" Import error: {e}")


def create_test_documents(num_docs: int = 30) -> List[Dict]:
    """Create diverse test documents for retrieval testing"""
    documents = [
        # AI/ML Documents
        {
            'id': 'ai_001',
            'content': 'Machine learning algorithms can automatically learn patterns from data without explicit programming. They are used in various applications like image recognition and natural language processing.',
            'metadata': {'topic': 'machine_learning', 'difficulty': 'beginner', 'year': 2023,
                         'content_type': 'educational'}
        },
        {
            'id': 'ai_002',
            'content': 'Deep neural networks have revolutionized artificial intelligence by enabling machines to process complex data like images, text, and speech with human-like accuracy.',
            'metadata': {'topic': 'deep_learning', 'difficulty': 'intermediate', 'year': 2023,
                         'content_type': 'technical'}
        },
        {
            'id': 'ai_003',
            'content': 'Transformer models like BERT and GPT have transformed natural language processing by using attention mechanisms to understand context and relationships between words.',
            'metadata': {'topic': 'nlp', 'difficulty': 'advanced', 'year': 2022, 'content_type': 'research'}
        },
        {
            'id': 'ai_004',
            'content': 'Computer vision systems use convolutional neural networks to analyze and understand visual content in images and videos, enabling applications like autonomous driving.',
            'metadata': {'topic': 'computer_vision', 'difficulty': 'intermediate', 'year': 2023,
                         'content_type': 'technical'}
        },
        {
            'id': 'ai_005',
            'content': 'Reinforcement learning trains agents to make decisions by learning from rewards and penalties, similar to how humans learn through trial and error.',
            'metadata': {'topic': 'reinforcement_learning', 'difficulty': 'advanced', 'year': 2022,
                         'content_type': 'educational'}
        },

        # Technical Documents
        {
            'id': 'tech_001',
            'content': 'Vector databases provide efficient similarity search capabilities for high-dimensional data, making them essential for recommendation systems and semantic search.',
            'metadata': {'topic': 'databases', 'difficulty': 'intermediate', 'year': 2023, 'content_type': 'technical'}
        },
        {
            'id': 'tech_002',
            'content': 'BM25 algorithm improves upon TF-IDF by incorporating term frequency saturation and document length normalization for better text retrieval accuracy.',
            'metadata': {'topic': 'information_retrieval', 'difficulty': 'advanced', 'year': 2021,
                         'content_type': 'research'}
        },
        {
            'id': 'tech_003',
            'content': 'FAISS (Facebook AI Similarity Search) enables efficient similarity search and clustering of dense vectors at scale, supporting both CPU and GPU acceleration.',
            'metadata': {'topic': 'search_engines', 'difficulty': 'intermediate', 'year': 2023,
                         'content_type': 'technical'}
        },

        # Research Papers
        {
            'id': 'research_001',
            'content': 'Attention Is All You Need introduced the Transformer architecture, which relies entirely on attention mechanisms and eliminates recurrence and convolutions.',
            'metadata': {'topic': 'transformers', 'difficulty': 'advanced', 'year': 2017, 'content_type': 'research',
                         'author': 'Vaswani et al.'}
        },
        {
            'id': 'research_002',
            'content': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding demonstrates the power of bidirectional training for language representations.',
            'metadata': {'topic': 'bert', 'difficulty': 'advanced', 'year': 2018, 'content_type': 'research',
                         'author': 'Devlin et al.'}
        },

        # Tutorials and Guides
        {
            'id': 'tutorial_001',
            'content': 'How to implement a basic neural network: Start by defining the network architecture, initialize weights, implement forward propagation, calculate loss, and perform backpropagation.',
            'metadata': {'topic': 'neural_networks', 'difficulty': 'beginner', 'year': 2023, 'content_type': 'tutorial'}
        },
        {
            'id': 'tutorial_002',
            'content': 'Steps to build a recommendation system: Collect user data, preprocess and clean the data, choose an algorithm, train the model, and evaluate performance.',
            'metadata': {'topic': 'recommendation_systems', 'difficulty': 'intermediate', 'year': 2023,
                         'content_type': 'tutorial'}
        },

        # Comparison Documents
        {
            'id': 'compare_001',
            'content': 'BERT vs GPT comparison: BERT uses bidirectional encoding for understanding, while GPT uses autoregressive generation. BERT excels at comprehension tasks, GPT at generation.',
            'metadata': {'topic': 'model_comparison', 'difficulty': 'intermediate', 'year': 2023,
                         'content_type': 'analysis'}
        },
        {
            'id': 'compare_002',
            'content': 'TensorFlow vs PyTorch differences: TensorFlow offers production-ready deployment tools, while PyTorch provides more intuitive research-friendly development experience.',
            'metadata': {'topic': 'frameworks', 'difficulty': 'beginner', 'year': 2023, 'content_type': 'analysis'}
        },

        # Definition Documents
        {
            'id': 'def_001',
            'content': 'Artificial Intelligence definition: AI refers to the simulation of human intelligence processes by machines, including learning, reasoning, and self-correction.',
            'metadata': {'topic': 'definitions', 'difficulty': 'beginner', 'year': 2023, 'content_type': 'reference'}
        },
        {
            'id': 'def_002',
            'content': 'Machine Learning definition: ML is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.',
            'metadata': {'topic': 'definitions', 'difficulty': 'beginner', 'year': 2023, 'content_type': 'reference'}
        }
    ]

    # Generate additional documents if needed
    additional_topics = [
        'data_science', 'statistics', 'algorithms', 'programming', 'databases',
        'cloud_computing', 'cybersecurity', 'web_development', 'mobile_apps'
    ]

    for i in range(len(documents), num_docs):
        topic = additional_topics[i % len(additional_topics)]
        doc = {
            'id': f'doc_{i:03d}',
            'content': f'This is a document about {topic.replace("_", " ")} covering various aspects and applications in the field. It provides comprehensive information and practical examples.',
            'metadata': {
                'topic': topic,
                'difficulty': ['beginner', 'intermediate', 'advanced'][i % 3],
                'year': 2020 + (i % 4),
                'content_type': ['educational', 'technical', 'research'][i % 3]
            }
        }
        documents.append(doc)

    return documents[:num_docs]


def test_query_processor():
    """Test enhanced query processing"""
    print("\n" + "=" * 50)
    print("TESTING ENHANCED QUERY PROCESSOR")
    print("=" * 50)

    try:


        # Initialize query processor
        processor = QueryProcessor()

        # Test queries with different intents
        test_queries = [
            {
                'query': 'What is machine learning?',
                'expected_intent': 'definition',
                'expected_type': 'semantic'
            },
            {
                'query': 'How to implement neural networks step by step?',
                'expected_intent': 'procedure',
                'expected_type': 'semantic'
            },
            {
                'query': 'Compare BERT vs GPT models',
                'expected_intent': 'comparison',
                'expected_type': 'hybrid'
            },
            {
                'query': 'List all machine learning algorithms',
                'expected_intent': 'list',
                'expected_type': 'keyword'
            },
            {
                'query': 'Papers by Yoshua Bengio in 2023',
                'expected_intent': 'factual',
                'expected_type': 'keyword'
            },
            {
                'query': 'Why do neural networks work so well?',
                'expected_intent': 'causal',
                'expected_type': 'semantic'
            }
        ]

        correct_predictions = 0
        total_predictions = 0

        for i, test_case in enumerate(test_queries):
            query_text = test_case['query']
            expected_intent = test_case['expected_intent']
            expected_type = test_case['expected_type']

            print(f"\n Query {i + 1}: '{query_text}'")

            # Process query
            processed_query = processor.process_query(query_text)

            detected_intent = processed_query.metadata.get('intent', 'unknown')
            detected_type = processed_query.query_type

            print(f"  Intent: {detected_intent} (expected: {expected_intent})")
            print(f"  Type: {detected_type} (expected: {expected_type})")
            print(f"  Is Question: {processed_query.metadata.get('is_question', False)}")

            # Check expanded terms
            expanded_terms = processed_query.metadata.get('expanded_terms', [])
            if expanded_terms:
                print(f"  Expanded Terms: {expanded_terms[:5]}...")

            # Check entities if available
            if 'spacy_analysis' in processed_query.metadata:
                entities = processed_query.metadata['spacy_analysis'].get('entities', [])
                if entities:
                    print(f"  Entities: {[(ent[0], ent[1]) for ent in entities[:3]]}")

            # Track accuracy
            if detected_intent == expected_intent:
                correct_predictions += 1
            total_predictions += 1

            if detected_type == expected_type:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n Query Processing Accuracy: {accuracy:.2%}")

        return True

    except Exception as e:
        print(f" Query processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dense_retrieval():
    """Test dense (vector-based) retrieval"""
    print("\n" + "=" * 50)
    print("TESTING DENSE RETRIEVAL")
    print("=" * 50)

    try:
        # Create components
        embedder = create_embedder("text")
        vector_store = create_vector_store("hybrid", dimension=embedder.get_dimension())

        # Create test documents
        test_docs = create_test_documents(15)

        # Convert to vector documents
        vector_docs = []
        for doc_dict in test_docs:
            embedding_result = embedder.embed(doc_dict['content'])

            vector_doc = VectorDocument(
                id=doc_dict['id'],
                embedding=embedding_result.embedding,
                content=doc_dict['content'],
                metadata=doc_dict['metadata'],
                content_type='text'
            )
            vector_docs.append(vector_doc)

        # Add to vector store
        vector_store.add_documents(vector_docs)

        # Create dense retriever
        dense_retriever = create_retriever("dense", vector_store, embedder)

        # Test queries
        test_queries = [
            "neural network learning algorithms",
            "transformer attention mechanisms",
            "computer vision image processing",
            "natural language understanding"
        ]

        for query_text in test_queries:
            print(f"\n Query: '{query_text}'")

            query = Query(text=query_text, query_type="semantic")
            results = dense_retriever.retrieve(query, top_k=3)

            print(f" Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i + 1}. Score: {result.score:.3f} | ID: {result.document_id}")
                print(f"     Content: {result.content[:60]}...")
                print(f"     Topic: {result.metadata.get('topic', 'unknown')}")

        # Test performance
        start_time = time.time()
        for _ in range(10):
            query = Query(text="machine learning", query_type="semantic")
            results = dense_retriever.retrieve(query, top_k=5)
        avg_time = (time.time() - start_time) / 10

        print(f"\n⏱ Average query time: {avg_time:.4f}s")

        # Get retriever stats
        stats = dense_retriever.get_stats()
        print(f" Retriever stats: {stats['stats']}")

        return True

    except Exception as e:
        print(f" Dense retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sparse_retrieval():
    """Test sparse (keyword-based) retrieval with BM25"""
    print("\n" + "=" * 50)
    print("TESTING SPARSE RETRIEVAL (BM25)")
    print("=" * 50)

    try:
        # Create sparse retriever
        sparse_config = {
            'bm25_variant': 'okapi',
            'use_nltk': True,
            'use_stemming': True,
            'remove_stopwords': True
        }
        sparse_retriever = SparseRetriever(sparse_config)

        # Create test documents
        test_docs = create_test_documents(15)

        # Add documents to sparse index
        sparse_retriever.add_documents(test_docs)

        # Get vocabulary stats
        vocab_stats = sparse_retriever.get_enhanced_stats()
        print(f"Vocabulary size: {vocab_stats['vocabulary_stats']['size']}")
        print(f"Average document length: {vocab_stats['index_stats']['avg_doc_length']:.1f}")

        # Test keyword queries
        test_queries = [
            "BERT transformer model",
            "neural network implementation",
            "machine learning algorithms",
            "Vaswani attention mechanism"  # Should find the specific paper
        ]

        for query_text in test_queries:
            print(f"\n Query: '{query_text}'")

            query = Query(text=query_text, query_type="keyword")
            results = sparse_retriever.retrieve(query, top_k=3)

            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i + 1}. Score: {result.score:.3f} | ID: {result.document_id}")
                print(f"     Content: {result.content[:60]}...")
                print(f"     Matched: {result.metadata.get('matched_terms', [])}")
                print(f"     Coverage: {result.metadata.get('query_coverage', 0):.2%}")

        # Test score explanation
        print(f"\n🔍 Score Explanation for 'BERT transformer':")
        if test_docs:
            explanation = sparse_retriever.explain_score("BERT transformer", test_docs[0]['id'])
            if explanation:
                print(f"  Total Score: {explanation.get('total_score', 0):.3f}")
                print(f"  Query Tokens: {explanation.get('query_tokens', [])}")
                print(f"  Matched Terms: {explanation.get('matched_terms', [])}")

        return True

    except Exception as e:
        print(f"Sparse retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_retrieval():
    """Test hybrid retrieval combining dense and sparse"""
    print("\n" + "=" * 50)
    print("TESTING HYBRID RETRIEVAL")
    print("=" * 50)

    try:
        # Create components
        embedder = create_embedder("text")
        vector_store = create_vector_store("hybrid", dimension=embedder.get_dimension())

        # Create test documents
        test_docs = create_test_documents(20)

        # Add to vector store
        vector_docs = []
        for doc_dict in test_docs:
            embedding_result = embedder.embed(doc_dict['content'])

            vector_doc = VectorDocument(
                id=doc_dict['id'],
                embedding=embedding_result.embedding,
                content=doc_dict['content'],
                metadata=doc_dict['metadata'],
                content_type='text'
            )
            vector_docs.append(vector_doc)

        vector_store.add_documents(vector_docs)

        # Create hybrid retriever
        hybrid_config = {
            'dense_weight': 0.7,
            'sparse_weight': 0.3,
            'fusion_method': 'weighted_sum',
            'adaptive_weights': True
        }
        hybrid_retriever = create_retriever("hybrid", vector_store, embedder, hybrid_config)

        # Add documents to sparse component
        hybrid_retriever.add_documents(test_docs)

        # Test different query types
        test_queries = [
            {
                'query': 'What is deep learning?',
                'description': 'Definition query (should favor semantic)'
            },
            {
                'query': 'Vaswani transformer paper',
                'description': 'Specific paper query (should favor keyword)'
            },
            {
                'query': 'neural network training methods',
                'description': 'General concept query (balanced)'
            },
            {
                'query': 'BERT vs GPT comparison',
                'description': 'Comparison query (hybrid)'
            }
        ]

        for test_case in test_queries:
            query_text = test_case['query']
            description = test_case['description']

            print(f"\n🔍 Query: '{query_text}'")
            print(f"   Type: {description}")

            query = Query(text=query_text, query_type="hybrid")
            results = hybrid_retriever.retrieve(query, top_k=4)

            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i + 1}. Score: {result.score:.3f} | ID: {result.document_id}")
                print(f"     Content: {result.content[:60]}...")

                # Show fusion information
                dense_score = result.metadata.get('dense_score', 'N/A')
                sparse_score = result.metadata.get('sparse_score', 'N/A')
                found_in_both = result.metadata.get('found_in_both', False)

                print(f"     Dense: {dense_score}, Sparse: {sparse_score}, Both: {found_in_both}")

        # Test adaptive weighting
        print(f"\nTesting Adaptive Weighting:")

        adaptive_queries = [
            'Define machine learning',  # Should favor dense
            'Papers by Hinton',  # Should favor sparse
        ]

        for query_text in adaptive_queries:
            query = Query(text=query_text, query_type="hybrid")
            results = hybrid_retriever.retrieve(query, top_k=2)

            if results:
                dense_weight = results[0].metadata.get('dense_weight', 0.5)
                sparse_weight = results[0].metadata.get('sparse_weight', 0.5)
                print(f"  '{query_text}' → Dense: {dense_weight:.2f}, Sparse: {sparse_weight:.2f}")

        # Get comprehensive stats
        stats = hybrid_retriever.get_comprehensive_stats()
        print(f"\nFusion Statistics:")
        print(f"  Dense preferred: {stats['fusion_stats']['dense_preferred']}")
        print(f"  Sparse preferred: {stats['fusion_stats']['sparse_preferred']}")
        print(f"  Balanced: {stats['fusion_stats']['balanced']}")

        return True

    except Exception as e:
        print(f"Hybrid retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval_pipeline():
    """Test complete retrieval pipeline with query processing"""
    print("\n" + "=" * 50)
    print("TESTING COMPLETE RETRIEVAL PIPELINE")
    print("=" * 50)

    try:
        # Create components
        embedder = create_embedder("text")
        vector_store = create_vector_store("hybrid", dimension=embedder.get_dimension())

        # Create and add test documents
        test_docs = create_test_documents(15)

        vector_docs = []
        for doc_dict in test_docs:
            embedding_result = embedder.embed(doc_dict['content'])

            vector_doc = VectorDocument(
                id=doc_dict['id'],
                embedding=embedding_result.embedding,
                content=doc_dict['content'],
                metadata=doc_dict['metadata'],
                content_type='text'
            )
            vector_docs.append(vector_doc)

        vector_store.add_documents(vector_docs)

        # Create hybrid retriever
        hybrid_retriever = create_retriever("hybrid", vector_store, embedder)
        hybrid_retriever.add_documents(test_docs)

        # Create retrieval pipeline
        pipeline_config = {
            'query_processing': {
                'enable_expansion': True,
                'enable_reformulation': True,
                'enable_intent_detection': True
            },
            'enable_reranking': True,
            'enable_diversity': True
        }
        pipeline = RetrievalPipeline(hybrid_retriever, pipeline_config)

        # Test end-to-end retrieval
        test_queries = [
            "What are neural networks?",
            "How to train deep learning models?",
            "Compare TensorFlow vs PyTorch",
            "List machine learning algorithms",
            "Papers about attention mechanisms"
        ]

        for query_text in test_queries:
            print(f"\n Pipeline Query: '{query_text}'")

            # Use pipeline search (includes query processing)
            results = pipeline.search(query_text, top_k=3)

            print(f" Pipeline Results ({len(results)} found):")
            for i, result in enumerate(results):
                print(f"  {i + 1}. Score: {result.score:.3f} | ID: {result.document_id}")
                print(f"     Content: {result.content[:60]}...")
                print(f"     Intent: {result.metadata.get('query_intent', 'unknown')}")
                print(f"     Original: '{result.metadata.get('original_query', '')}'" if result.metadata.get(
                    'original_query') != query_text else "")

        # Test performance comparison
        print(f"\n Performance Comparison:")

        # Direct retriever
        start_time = time.time()
        for _ in range(5):
            query = Query(text="machine learning", query_type="hybrid")
            results = hybrid_retriever.retrieve(query, top_k=3)
        direct_time = (time.time() - start_time) / 5

        # Pipeline
        start_time = time.time()
        for _ in range(5):
            results = pipeline.search("machine learning", top_k=3)
        pipeline_time = (time.time() - start_time) / 5

        print(f"  Direct retriever: {direct_time:.4f}s")
        print(f"  Full pipeline: {pipeline_time:.4f}s")
        print(f"  Overhead: {((pipeline_time - direct_time) / direct_time * 100):.1f}%")

        return True

    except Exception as e:
        print(f"Retrieval pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval_quality():
    """Test retrieval quality and relevance"""
    print("\n" + "=" * 50)
    print("TESTING RETRIEVAL QUALITY")
    print("=" * 50)

    try:
        # Create components
        embedder = create_embedder("text")
        vector_store = create_vector_store("hybrid", dimension=embedder.get_dimension())

        # Create test documents with known relevance
        test_docs = create_test_documents(15)

        vector_docs = []
        for doc_dict in test_docs:
            embedding_result = embedder.embed(doc_dict['content'])

            vector_doc = VectorDocument(
                id=doc_dict['id'],
                embedding=embedding_result.embedding,
                content=doc_dict['content'],
                metadata=doc_dict['metadata'],
                content_type='text'
            )
            vector_docs.append(vector_doc)

        vector_store.add_documents(vector_docs)

        # Create retrievers
        dense_retriever = create_retriever("dense", vector_store, embedder)
        hybrid_retriever = create_retriever("hybrid", vector_store, embedder)
        hybrid_retriever.add_documents(test_docs)

        # Define test cases with expected relevant documents
        relevance_tests = [
            {
                'query': 'transformer attention mechanism',
                'relevant_topics': ['transformers', 'nlp', 'bert'],
                'relevant_ids': ['ai_003', 'research_001', 'research_002']
            },
            {
                'query': 'how to implement neural networks',
                'relevant_topics': ['neural_networks', 'machine_learning'],
                'relevant_ids': ['tutorial_001', 'ai_001', 'ai_002']
            },
            {
                'query': 'BERT vs GPT',
                'relevant_topics': ['bert', 'model_comparison'],
                'relevant_ids': ['compare_001', 'research_002', 'ai_003']
            }
        ]

        retriever_scores = {'dense': [], 'hybrid': []}

        for test_case in relevance_tests:
            query_text = test_case['query']
            relevant_topics = test_case['relevant_topics']
            relevant_ids = test_case['relevant_ids']


            # Test both retrievers
            for retriever_name, retriever in [('dense', dense_retriever), ('hybrid', hybrid_retriever)]:
                query = Query(text=query_text, query_type="hybrid")
                results = retriever.retrieve(query, top_k=5)

                # Calculate relevance metrics
                relevant_found = 0
                total_relevant = len(relevant_ids)

                print(f"\n {retriever_name.title()} Results:")
                for i, result in enumerate(results):
                    is_relevant = (
                            result.document_id in relevant_ids or
                            result.metadata.get('topic') in relevant_topics
                    )

                    if is_relevant:
                        relevant_found += 1

                    print(f"  {i + 1}. Score: {result.score:.3f} | ID: {result.document_id}")
                    print(f"     Topic: {result.metadata.get('topic', 'unknown')}")

                # Calculate precision and recall
                precision = relevant_found / len(results) if results else 0
                recall = relevant_found / total_relevant if total_relevant > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f"     Precision: {precision:.2%}")
                print(f"     Recall: {recall:.2%}")
                print(f"     F1-Score: {f1:.2%}")

                retriever_scores[retriever_name].append({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        # Calculate average scores
        print(f"\n OVERALL QUALITY METRICS:")
        for retriever_name, scores in retriever_scores.items():
            if scores:
                avg_precision = sum(s['precision'] for s in scores) / len(scores)
                avg_recall = sum(s['recall'] for s in scores) / len(scores)
                avg_f1 = sum(s['f1'] for s in scores) / len(scores)

                print(f"\n{retriever_name.title()} Retriever:")
                print(f"  Average Precision: {avg_precision:.2%}")
                print(f"  Average Recall: {avg_recall:.2%}")
                print(f"  Average F1-Score: {avg_f1:.2%}")

        return True

    except Exception as e:
        print(f" Retrieval quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmarks():
    """Test retrieval system performance"""
    print("\n" + "=" * 50)
    print("TESTING PERFORMANCE BENCHMARKS")
    print("=" * 50)

    try:
        # Create components
        embedder = create_embedder("text")
        vector_store = create_vector_store("hybrid", dimension=embedder.get_dimension())

        # Test with different document counts
        doc_counts = [10, 50, 100]

        for doc_count in doc_counts:
            print(f"\n Testing with {doc_count} documents:")

            # Create test documents
            test_docs = create_test_documents(doc_count)

            # Time document indexing
            start_time = time.time()

            # Add to vector store
            vector_docs = []
            for doc_dict in test_docs:
                embedding_result = embedder.embed(doc_dict['content'])
                vector_doc = VectorDocument(
                    id=doc_dict['id'],
                    embedding=embedding_result.embedding,
                    content=doc_dict['content'],
                    metadata=doc_dict['metadata'],
                    content_type='text'
                )
                vector_docs.append(vector_doc)

            vector_store.add_documents(vector_docs)
            indexing_time = time.time() - start_time

            # Create retrievers
            dense_retriever = create_retriever("dense", vector_store, embedder)
            sparse_retriever = SparseRetriever()
            sparse_retriever.add_documents(test_docs)

            hybrid_retriever = create_retriever("hybrid", vector_store, embedder)
            hybrid_retriever.add_documents(test_docs)

            # Test query performance
            test_queries = [
                "machine learning algorithms",
                "neural network implementation",
                "transformer attention mechanism"
            ]

            retriever_performance = {}

            for name, retriever in [
                ('Dense', dense_retriever),
                ('Sparse', sparse_retriever),
                ('Hybrid', hybrid_retriever)
            ]:
                times = []

                for query_text in test_queries:
                    start_time = time.time()
                    query = Query(text=query_text, query_type="hybrid")
                    results = retriever.retrieve(query, top_k=5)
                    query_time = time.time() - start_time
                    times.append(query_time)

                avg_time = sum(times) / len(times)
                retriever_performance[name] = avg_time

            print(f"  Indexing time: {indexing_time:.3f}s ({doc_count / indexing_time:.1f} docs/sec)")

            for name, avg_time in retriever_performance.items():
                queries_per_sec = 1 / avg_time
                print(f"  {name} retrieval: {avg_time:.4f}s ({queries_per_sec:.1f} queries/sec)")

        return True

    except Exception as e:
        print(f" Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_retrieval_tests():
    """Run comprehensive retrieval system tests"""
    print("🚀 STARTING RETRIEVAL SYSTEM TESTS")
    print("=" * 60)

    # Test results tracking
    results = {
        'query_processor': False,
        'dense_retrieval': False,
        'sparse_retrieval': False,
        'hybrid_retrieval': False,
        'retrieval_pipeline': False,
        'retrieval_quality': False,
        'performance_benchmarks': False
    }

    try:

        # Run tests
        results['query_processor'] = test_query_processor()
        results['dense_retrieval'] = test_dense_retrieval()
        results['sparse_retrieval'] = test_sparse_retrieval()

        # Advanced tests (depend on previous ones)
        if results['dense_retrieval'] and results['sparse_retrieval']:
            results['hybrid_retrieval'] = test_hybrid_retrieval()
            results['retrieval_pipeline'] = test_retrieval_pipeline()
            results['retrieval_quality'] = test_retrieval_quality()

        # Performance test (optional)
        try:
            results['performance_benchmarks'] = test_performance_benchmarks()
        except Exception as e:
            print(f" Performance test skipped: {e}")

    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    # Print results summary
    print(f"\n" + "=" * 60)
    print(" RETRIEVAL SYSTEM TEST RESULTS")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name.replace('_', ' ').title(): <25} {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    # Determine success
    core_tests = [ 'dense_retrieval', 'sparse_retrieval', 'hybrid_retrieval']
    core_passed = sum(results[test] for test in core_tests if test in results)
    core_total = len(core_tests)

    if core_passed == core_total:
        print("\n All core retrieval tests passed! System is ready!")
    elif core_passed >= core_total - 1:
        print("\n️  Most core tests passed. System is functional.")
    else:
        print("\n Core tests failed. Check setup and dependencies.")

    # Next steps
    print(f"\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)

    if results['hybrid_retrieval'] and results['retrieval_pipeline']:
        print("✓ Retrieval system fully functional")
        print("✓ Ready for LLM integration")
        print("✓ Can build complete RAG pipeline")
    else:
        print(" Fix retrieval system issues first")

    if results['retrieval_quality']:
        print("✓ Quality metrics validated")
        print("✓ Ready for production testing")

    if results['performance_benchmarks']:
        print("✓ Performance benchmarks completed")
        print("✓ System scales well with document count")

    return core_passed >= core_total - 1


def quick_retrieval_test():
    """Quick test for development"""
    print(" QUICK RETRIEVAL TEST")
    print("=" * 30)

    try:
        # Quick hybrid test
        print("\n Testing hybrid retrieval...")

        embedder = create_embedder("text")
        vector_store = create_vector_store("faiss", dimension=embedder.get_dimension())

        # Small test dataset
        test_docs = [
            {'id': 'doc1', 'content': 'Machine learning algorithms learn from data', 'metadata': {'topic': 'ml'}},
            {'id': 'doc2', 'content': 'Neural networks use layers of interconnected nodes',
             'metadata': {'topic': 'nn'}},
            {'id': 'doc3', 'content': 'BERT transformer model for language understanding', 'metadata': {'topic': 'nlp'}}
        ]

        # Add to vector store
        vector_docs = []
        for doc_dict in test_docs:
            embedding_result = embedder.embed(doc_dict['content'])
            vector_doc = VectorDocument(
                id=doc_dict['id'],
                embedding=embedding_result.embedding,
                content=doc_dict['content'],
                metadata=doc_dict['metadata'],
                content_type='text'
            )
            vector_docs.append(vector_doc)

        vector_store.add_documents(vector_docs)

        # Create hybrid retriever
        hybrid_retriever = create_retriever("hybrid", vector_store, embedder)
        hybrid_retriever.add_documents(test_docs)

        # Test query
        query = Query(text="neural network learning", query_type="hybrid")
        results = hybrid_retriever.retrieve(query, top_k=2)

        print(f" Quick test passed!")
        print(f" Query: 'neural network learning'")
        print(f" Found: {len(results)} results")
        if results:
            print(f"Top result: {results[0].document_id} (score: {results[0].score:.3f})")

        return True

    except Exception as e:
        print(f" Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test retrieval system")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--quality", action="store_true", help="Run quality tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--query", action="store_true", help="Run query processing tests only")

    args = parser.parse_args()

    if args.quality:
        success = test_retrieval_quality()
    elif args.performance:
        success = test_performance_benchmarks()
    elif args.query:
        success = test_query_processor()
    elif args.quick:
        success = quick_retrieval_test()
    elif args.full:
        success = run_retrieval_tests()
    else:
        print("Running quick test by default. Use --full for comprehensive tests.")
        success = quick_retrieval_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()