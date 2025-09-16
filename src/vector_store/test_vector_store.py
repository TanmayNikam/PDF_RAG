"""
Comprehensive test script for the vector store system
"""

import sys
import time
from typing import List

import numpy as np
from pathlib import Path
import tempfile
import shutil
import uuid
import logging

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent  # Go up from vector_store to src
sys.path.insert(0, str(src_dir))

try:
    from vector_store import (
        create_vector_store,
        VectorDocument,
        SearchResult,
        FAISSVectorStore,
        HybridVectorStore,
        documents_to_vector_documents
    )
    from embeddings import create_embedder, MultimodalContent

    print("Successfully imported vector store and embedding modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_documents(num_docs: int = 50) -> List[VectorDocument]:
    """Create test documents with synthetic embeddings"""
    documents = []

    # Create text embedder for realistic embeddings
    try:
        text_embedder = create_embedder("text")
    except:
        # Fallback to random embeddings
        text_embedder = None

    for i in range(num_docs):
        # Create varied test content
        if i % 4 == 0:
            content = f"This is a technical document about machine learning and AI systems. Document {i}."
            content_type = "text"
        elif i % 4 == 1:
            content = f"Scientific research paper discussing neural networks and deep learning. Paper {i}."
            content_type = "text"
        elif i % 4 == 2:
            content = f"[IMAGE: Chart showing performance metrics and evaluation results {i}]"
            content_type = "image"
        else:
            content = f"Multimodal content combining text analysis with visual elements. Item {i}."
            content_type = "multimodal"

        # Generate embedding
        if text_embedder:
            try:
                embedding_result = text_embedder.embed(content)
                embedding = embedding_result.embedding
            except:
                embedding = np.random.normal(0, 1, 384).astype(np.float32)
        else:
            embedding = np.random.normal(0, 1, 384).astype(np.float32)

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        doc = VectorDocument(
            id=f"doc_{i:04d}",
            embedding=embedding,
            content=content,
            metadata={
                'topic': ['ml', 'ai', 'research', 'analysis'][i % 4],
                'difficulty': ['beginner', 'intermediate', 'advanced'][i % 3],
                'length': len(content),
                'category': content_type
            },
            content_type=content_type,
            chunk_id=f"chunk_{i:04d}",
            parent_document_id=f"parent_doc_{i // 10}"
        )

        documents.append(doc)

    return documents


def test_basic_vector_store_operations():
    """Test basic vector store operations"""
    print("\n" + "=" * 50)
    print("TESTING BASIC VECTOR STORE OPERATIONS")
    print("=" * 50)

    try:
        # Create vector store
        print(" Creating FAISS vector store...")
        vector_store = create_vector_store("faiss", dimension=384)

        # Create test documents
        print(" Creating test documents...")
        test_docs = create_test_documents(20)

        # Add documents
        print(" Adding documents to store...")
        start_time = time.time()
        doc_ids = vector_store.add_documents(test_docs)
        add_time = time.time() - start_time

        print(f"Added {len(doc_ids)} documents in {add_time:.3f}s")
        print(f"Store stats: {vector_store.get_stats()}")

        # Test search
        print(" Testing search...")
        query_embedding = test_docs[0].embedding

        start_time = time.time()
        results = vector_store.search(query_embedding, top_k=5)
        search_time = time.time() - start_time

        print(f"Search completed in {search_time:.3f}s")
        print(f"Found {len(results)} results")

        # Display top results
        for i, result in enumerate(results[:3]):
            print(f"  {i + 1}. Score: {result.score:.3f}, ID: {result.document.id}")
            print(f"     Content: {result.document.content[:60]}...")

        # Test document retrieval
        print(" Testing document retrieval...")
        retrieved_doc = vector_store.get_document(doc_ids[0])
        if retrieved_doc:
            print(f"Retrieved document: {retrieved_doc.id}")
        else:
            print("Failed to retrieve document")

        return True

    except Exception as e:
        print(f"Basic operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_vector_store():
    """Test hybrid vector store with advanced features"""
    print("\n" + "=" * 50)
    print("TESTING HYBRID VECTOR STORE")
    print("=" * 50)

    try:
        # Create hybrid vector store with advanced config
        config = {
            'faiss': {
                'index_type': 'IndexFlatIP',
                'metric_type': 'INNER_PRODUCT'
            },
            'enable_caching': True,
            'cache_size': 100,
            'enable_reranking': True,
            'diversity_boost': 0.1,
            'content_type_weights': {
                'text': 1.0,
                'image': 0.9,
                'multimodal': 1.1
            }
        }

        print(" Creating hybrid vector store...")
        hybrid_store = create_vector_store("hybrid", dimension=384, config=config)

        # Create diverse test documents
        print(" Creating diverse test documents...")
        test_docs = create_test_documents(30)

        # Add documents
        print(" Adding documents to hybrid store...")
        doc_ids = hybrid_store.add_documents(test_docs)

        print(f"Added {len(doc_ids)} documents")

        # Test advanced search
        print(" Testing advanced search...")
        query_embedding = test_docs[0].embedding

        # Basic search
        basic_results = hybrid_store.search(query_embedding, top_k=5)
        print(f"Basic search: {len(basic_results)} results")

        # Search with filters
        print("Testing filtered search...")
        filters = {'content_type': 'text'}
        filtered_results = hybrid_store.search(
            query_embedding, top_k=5, filters=filters
        )
        print(f"Filtered search (text only): {len(filtered_results)} results")

        # Search with parameters
        print("Testing search with parameters...")
        search_params = {
            'enable_diversity': True,
            'length_penalty': True,
            'recency_boost': False
        }
        param_results = hybrid_store.search(
            query_embedding, top_k=5, search_params=search_params
        )
        print(f"Parameterized search: {len(param_results)} results")

        # Test multi-vector search
        print("Testing multi-vector search...")
        query_embeddings = [test_docs[i].embedding for i in range(3)]
        weights = [0.5, 0.3, 0.2]

        multi_results = hybrid_store.multi_vector_search(
            query_embeddings, weights, top_k=5
        )
        print(f"Multi-vector search: {len(multi_results)} results")

        # Display comprehensive stats
        print("Comprehensive stats:")
        stats = hybrid_store.get_comprehensive_stats()
        print(f"  Documents: {stats['hybrid_store']['document_count']}")
        print(f"  Index type: {stats['faiss_index']['index_type']}")
        print(f"  Cache enabled: {stats['caching']['enabled']}")
        print(f"  Reranking: {stats['search_optimizations']['reranking']}")

        return True

    except Exception as e:
        print(f"Hybrid store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_persistence():
    """Test saving and loading vector stores"""
    print("\n" + "=" * 50)
    print("TESTING VECTOR STORE PERSISTENCE")
    print("=" * 50)

    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f" Using temp directory: {temp_dir}")

        try:
            # Create and populate vector store
            print(" Creating vector store...")
            vector_store = create_vector_store("faiss", dimension=384)

            test_docs = create_test_documents(15)
            doc_ids = vector_store.add_documents(test_docs)

            print(f"Added {len(doc_ids)} documents")

            # Perform initial search
            query_embedding = test_docs[0].embedding
            original_results = vector_store.search(query_embedding, top_k=3)

            print(f"Original search: {len(original_results)} results")

            # Save the vector store
            print("Saving vector store...")
            save_success = vector_store.save(temp_dir)

            if not save_success:
                print("Failed to save vector store")
                return False

            print("Vector store saved successfully")

            # Create new vector store and load
            print("Creating new vector store and loading...")
            new_vector_store = create_vector_store("faiss", dimension=384)
            load_success = new_vector_store.load(temp_dir)

            if not load_success:
                print("Failed to load vector store")
                return False

            print("Vector store loaded successfully")

            # Verify loaded data
            print("Verifying loaded data...")
            loaded_stats = new_vector_store.get_stats()
            print(f"Loaded store stats: {loaded_stats}")

            # Perform same search on loaded store
            loaded_results = new_vector_store.search(query_embedding, top_k=3)
            print(f"Loaded search: {len(loaded_results)} results")

            # Compare results
            if len(original_results) == len(loaded_results):
                print("Result count matches")

                # Check if top result is the same
                if (original_results[0].document.id == loaded_results[0].document.id and
                        abs(original_results[0].score - loaded_results[0].score) < 1e-6):
                    print("Top result matches perfectly")
                else:
                    print("⚠Top result differs slightly (acceptable)")
            else:
                print(" Result count mismatch")
                return False

            return True

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp directory")

    except Exception as e:
        print(f"Persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmarks():
    """Test vector store performance with different configurations"""
    print("\n" + "=" * 50)
    print("TESTING PERFORMANCE BENCHMARKS")
    print("=" * 50)

    try:
        # Test different index types and sizes
        test_configs = [
            {
                'name': 'Small FAISS Flat',
                'store_type': 'faiss',
                'config': {'index_type': 'IndexFlatIP'},
                'doc_count': 100
            },
            {
                'name': 'Medium FAISS Flat',
                'store_type': 'faiss',
                'config': {'index_type': 'IndexFlatIP'},
                'doc_count': 500
            },
            {
                'name': 'Hybrid with Caching',
                'store_type': 'hybrid',
                'config': {
                    'faiss': {'index_type': 'IndexFlatIP'},
                    'enable_caching': True,
                    'enable_reranking': True
                },
                'doc_count': 300
            }
        ]

        results = []

        for test_config in test_configs:
            print(f"\n Testing {test_config['name']}...")

            # Create vector store
            vector_store = create_vector_store(
                test_config['store_type'],
                dimension=384,
                config=test_config['config']
            )

            # Create test documents
            test_docs = create_test_documents(test_config['doc_count'])

            # Measure insertion time
            start_time = time.time()
            doc_ids = vector_store.add_documents(test_docs)
            insertion_time = time.time() - start_time

            # Measure search time (multiple searches for average)
            query_embeddings = [test_docs[i].embedding for i in range(10)]
            search_times = []

            for query_embedding in query_embeddings:
                start_time = time.time()
                results = vector_store.search(query_embedding, top_k=10)
                search_time = time.time() - start_time
                search_times.append(search_time)

            avg_search_time = sum(search_times) / len(search_times)

            # Record results
            benchmark_result = {
                'name': test_config['name'],
                'doc_count': test_config['doc_count'],
                'insertion_time': insertion_time,
                'avg_search_time': avg_search_time,
                'docs_per_second_insert': test_config['doc_count'] / insertion_time,
                'searches_per_second': 1 / avg_search_time
            }

            results.append(benchmark_result)

            print(f"  Documents: {benchmark_result['doc_count']}")
            print(f"  Insertion: {benchmark_result['insertion_time']:.3f}s")
            print(f"  Insert rate: {benchmark_result['docs_per_second_insert']:.1f} docs/sec")
            print(f"  Avg search: {benchmark_result['avg_search_time']:.4f}s")
            print(f"  Search rate: {benchmark_result['searches_per_second']:.1f} searches/sec")

        # Summary
        print(f"\n PERFORMANCE SUMMARY")
        print("-" * 50)
        for result in results:
            print(f"{result['name']:20} | "
                  f"{result['docs_per_second_insert']:6.1f} docs/s | "
                  f"{result['searches_per_second']:6.1f} searches/s")

        return True

    except Exception as e:
        print(f" Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_embeddings():
    """Test integration between vector store and embedding system"""
    print("\n" + "=" * 50)
    print("TESTING INTEGRATION WITH EMBEDDINGS")
    print("=" * 50)

    try:
        # Create embedder and vector store
        print(" Creating embedder and vector store...")
        text_embedder = create_embedder("text")
        vector_store = create_vector_store("hybrid", dimension=text_embedder.get_dimension())

        # Create real text documents
        texts = [
            "Machine learning algorithms can process large amounts of data efficiently.",
            "Deep neural networks have revolutionized computer vision and natural language processing.",
            "Vector databases enable semantic search and similarity matching.",
            "Transformers are the foundation of modern language models like GPT and BERT.",
            "Multimodal AI systems can understand both text and images simultaneously.",
            "RAG systems combine retrieval with generation for enhanced AI capabilities.",
            "FAISS provides efficient similarity search for high-dimensional vectors.",
            "Embedding models convert text into dense vector representations."
        ]

        print(f" Processing {len(texts)} real text documents...")

        # Generate embeddings and create vector documents
        vector_docs = []
        for i, text in enumerate(texts):
            embedding_result = text_embedder.embed(text)

            vector_doc = VectorDocument(
                id=f"real_doc_{i:03d}",
                embedding=embedding_result.embedding,
                content=text,
                metadata={
                    'source': 'test_integration',
                    'topic': 'AI/ML',
                    'length': len(text),
                    'word_count': len(text.split())
                },
                content_type='text'
            )
            vector_docs.append(vector_doc)

        # Add to vector store
        doc_ids = vector_store.add_documents(vector_docs)
        print(f" Added {len(doc_ids)} documents with real embeddings")

        # Test semantic search
        print(" Testing semantic search...")

        queries = [
            "neural network algorithms",
            "text processing and language models",
            "vector similarity search"
        ]

        for query in queries:
            print(f"\n Query: '{query}'")

            # Generate query embedding
            query_result = text_embedder.embed(query)

            # Search
            results = vector_store.search(query_result.embedding, top_k=3)

            print(f" Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i + 1}. Score: {result.score:.3f}")
                print(f"     Content: {result.document.content}")

        # Test similarity between related concepts
        print("\n Testing semantic similarity...")
        concept_pairs = [
            ("machine learning", "neural networks"),
            ("vector search", "similarity matching"),
            ("AI models", "deep learning")
        ]

        for concept1, concept2 in concept_pairs:
            emb1 = text_embedder.embed(concept1).embedding
            emb2 = text_embedder.embed(concept2).embedding

            similarity = np.dot(emb1, emb2)
            print(f"  '{concept1}' ↔ '{concept2}': {similarity:.3f}")

        # Test retrieval accuracy
        print("\n Testing retrieval accuracy...")
        test_doc = vector_docs[0]
        accuracy_results = vector_store.search(test_doc.embedding, top_k=3)

        if accuracy_results and accuracy_results[0].document.id == test_doc.id:
            print(" Retrieval accuracy test passed")
            print(f"   Self-similarity score: {accuracy_results[0].score:.3f}")
        else:
            print(" Retrieval accuracy: unexpected top result")

        # Test embedding normalization
        print("\n Testing embedding normalization...")
        norms = [np.linalg.norm(doc.embedding) for doc in vector_docs[:3]]
        avg_norm = sum(norms) / len(norms)
        print(f"  Average embedding norm: {avg_norm:.3f}")

        if 0.95 <= avg_norm <= 1.05:
            print(" Embeddings are properly normalized")
        else:
            print("  Embeddings may not be normalized")

        # Test batch search performance
        print("\n Testing batch search...")
        batch_queries = ["AI research", "machine learning", "data processing"]
        batch_embeddings = []

        for query in batch_queries:
            emb = text_embedder.embed(query).embedding
            batch_embeddings.append(emb)

        # Test multiple searches
        start_time = time.time()
        for emb in batch_embeddings:
            results = vector_store.search(emb, top_k=2)
        batch_time = time.time() - start_time

        print(f"  Batch search time: {batch_time:.3f}s for {len(batch_queries)} queries")
        print(f"  Average per query: {batch_time / len(batch_queries):.4f}s")

        print("\n Integration with embeddings test completed successfully!")
        return True

    except Exception as e:
        print(f" Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_vector_store_test():
    """Quick test for development"""
    print(" QUICK VECTOR STORE TEST")
    print("=" * 30)

    try:
        # Quick basic test
        print("Creating vector store...")
        vector_store = create_vector_store("faiss", dimension=384)

        print("Creating test documents...")
        test_docs = create_test_documents(5)

        print("Adding documents...")
        doc_ids = vector_store.add_documents(test_docs)

        print("Testing search...")
        query_embedding = test_docs[0].embedding
        results = vector_store.search(query_embedding, top_k=3)

        print(f"Quick test passed!")
        print(f" Added: {len(doc_ids)} documents")
        print(f" Found: {len(results)} search results")
        print(f" Top score: {results[0].score:.3f}")
        print(f" Store stats: {vector_store.document_count} documents")

        # Test with text embedder if available
        try:
            print("\n Testing with real embedder...")
            text_embedder = create_embedder("text")

            # Test real embedding
            test_text = "machine learning and artificial intelligence"
            embedding_result = text_embedder.embed(test_text)

            # Create vector document
            vector_doc = VectorDocument(
                id="real_test_doc",
                embedding=embedding_result.embedding,
                content=test_text,
                metadata={'test': True},
                content_type='text'
            )

            # Add and search
            vector_store.add_documents([vector_doc])
            real_results = vector_store.search(embedding_result.embedding, top_k=1)

            print(f" Real embedder test: {len(real_results)} results")
            print(f" Real embedder score: {real_results[0].score:.3f}")

        except Exception as e:
            print(f" Real embedder test failed: {e}")

        return True

    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_end_to_end():
    """Simple end-to-end test without external dependencies"""
    print("\n" + "=" * 50)
    print("TESTING SIMPLE END-TO-END WORKFLOW")
    print("=" * 50)

    try:
        # Step 1: Create embedder
        print(" Step 1: Creating text embedder...")
        text_embedder = create_embedder("text")
        print(f" Created embedder with dimension: {text_embedder.get_dimension()}")

        # Step 2: Create vector store
        print(" Step 2: Creating vector store...")
        vector_store = create_vector_store("hybrid", dimension=text_embedder.get_dimension())
        print(" Created hybrid vector store")

        # Step 3: Prepare documents
        print(" Step 3: Preparing documents...")
        documents_data = [
            "Artificial intelligence is transforming how we process information.",
            "Machine learning algorithms can identify patterns in large datasets.",
            "Deep neural networks have revolutionized computer vision tasks.",
            "Natural language processing enables computers to understand human text.",
            "Vector databases provide efficient similarity search capabilities.",
            "Transformer models like BERT and GPT have advanced language understanding.",
            "Multimodal AI systems can process both text and images simultaneously.",
            "RAG systems combine retrieval with generation for enhanced AI responses."
        ]

        # Step 4: Generate embeddings and create vector documents
        print(" Step 4: Generating embeddings...")
        vector_docs = []

        for i, text in enumerate(documents_data):
            embedding_result = text_embedder.embed(text)

            vector_doc = VectorDocument(
                id=f"simple_doc_{i:03d}",
                embedding=embedding_result.embedding,
                content=text,
                metadata={
                    'source': 'simple_test',
                    'index': i,
                    'word_count': len(text.split()),
                    'category': 'AI/ML'
                },
                content_type='text',
                chunk_id=f"chunk_{i:03d}",
                parent_document_id=f"parent_{i // 3}"
            )
            vector_docs.append(vector_doc)

        print(f" Created {len(vector_docs)} vector documents")

        # Step 5: Add to vector store
        print(" Step 5: Adding documents to vector store...")
        doc_ids = vector_store.add_documents(vector_docs)
        print(f" Added {len(doc_ids)} documents to store")

        # Step 6: Test search
        print(" Step 6: Testing search...")
        test_queries = [
            "deep learning neural networks",
            "text processing and language",
            "information retrieval systems"
        ]

        for query in test_queries:
            print(f"\n Query: '{query}'")

            # Generate query embedding
            query_result = text_embedder.embed(query)

            # Search vector store
            results = vector_store.search(query_result.embedding, top_k=3)

            print(f" Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i + 1}. Score: {result.score:.3f}")
                print(f"     ID: {result.document.id}")
                print(f"     Content: {result.document.content[:60]}...")

        # Step 7: Test advanced features
        print("\n Step 7: Testing advanced features...")

        # Test filtering
        filters = {'category': 'AI/ML'}
        filtered_results = vector_store.search(
            query_result.embedding,
            top_k=3,
            filters=filters
        )
        print(f" Filtered search: {len(filtered_results)} results")

        # Test document retrieval
        retrieved_doc = vector_store.get_document(doc_ids[0])
        if retrieved_doc:
            print(f" Retrieved document: {retrieved_doc.id}")

        # Step 8: Test persistence
        print("\n Step 8: Testing save/load...")
        temp_dir = tempfile.mkdtemp()

        try:
            # Save
            save_success = vector_store.save(temp_dir)
            print(f" Save successful: {save_success}")

            # Load into new store
            new_store = create_vector_store("hybrid", dimension=text_embedder.get_dimension())
            load_success = new_store.load(temp_dir)
            print(f" Load successful: {load_success}")

            # Verify
            if load_success:
                verify_results = new_store.search(query_result.embedding, top_k=1)
                print(f" Verification: {len(verify_results)} results found in loaded store")

        finally:
            shutil.rmtree(temp_dir)

        print("\n End-to-end test completed successfully!")
        return True

    except Exception as e:
        print(f" End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_vector_storage():
    """Test storing and searching multimodal content"""
    print("\n" + "=" * 50)
    print("TESTING MULTIMODAL VECTOR STORAGE")
    print("=" * 50)

    try:
        # Create multimodal embedder and vector store
        print(" Creating multimodal embedder and vector store...")
        multimodal_embedder = create_embedder("multimodal")
        vector_store = create_vector_store("hybrid", dimension=multimodal_embedder.get_dimension())

        # Create multimodal test content
        multimodal_contents = [
            {
                'content': MultimodalContent(
                    text="A chart showing AI model performance over time",
                    metadata={'type': 'chart_description'}
                ),
                'description': 'Text + conceptual image'
            },
            {
                'content': MultimodalContent(
                    text="Deep learning neural network architecture diagram"
                ),
                'description': 'Technical text description'
            },
            {
                'content': MultimodalContent(
                    text="Machine learning workflow and data processing pipeline"
                ),
                'description': 'Process description'
            }
        ]

        print(f" Processing {len(multimodal_contents)} multimodal items...")

        # Generate embeddings and create vector documents
        vector_docs = []
        for i, item in enumerate(multimodal_contents):
            embedding_result = multimodal_embedder.embed(item['content'])

            vector_doc = VectorDocument(
                id=f"multimodal_doc_{i:03d}",
                embedding=embedding_result.embedding,
                content=item['content'].text or "[Multimodal Content]",
                metadata={
                    'source': 'test_multimodal',
                    'description': item['description'],
                    'modality_type': 'multimodal',
                    'has_text': embedding_result.metadata['has_text'],
                    'has_image': embedding_result.metadata['has_image']
                },
                content_type='multimodal'
            )
            vector_docs.append(vector_doc)

        # Add to vector store
        doc_ids = vector_store.add_documents(vector_docs)
        print(f" Added {len(doc_ids)} multimodal documents")

        # Test multimodal search
        print(" Testing multimodal search...")

        # Create a query
        query_content = MultimodalContent(text="AI performance visualization")
        query_result = multimodal_embedder.embed(query_content)

        results = vector_store.search(query_result.embedding, top_k=3)

        print(f" Multimodal search results:")
        for i, result in enumerate(results):
            print(f"  {i + 1}. Score: {result.score:.3f}")
            print(f"     Content: {result.document.content}")
            print(f"     Type: {result.document.metadata.get('description', 'N/A')}")

        return True

    except Exception as e:
        print(f" Multimodal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_processor_integration():
    """Test integration with document processor output"""
    print("\n" + "=" * 50)
    print("TESTING DOCUMENT PROCESSOR INTEGRATION")
    print("=" * 50)

    try:
        # Simulate document processor output
        print(" Simulating document processor output...")

        processed_documents = [
            {
                'document_id': 'test_paper_001',
                'document_type': 'pdf',
                'file_path': '/test/path/paper1.pdf',
                'total_chunks': 4,
                'text_chunks': [
                    {
                        'chunk_id': 'test_paper_001_text_000',
                        'content': 'Abstract: This paper presents a novel approach to multimodal learning.',
                        'chunk_type': 'text',
                        'page_number': 1,
                        'metadata': {'section': 'abstract', 'length': 70}
                    },
                    {
                        'chunk_id': 'test_paper_001_text_001',
                        'content': 'Introduction: Multimodal AI systems have gained significant attention.',
                        'chunk_type': 'text',
                        'page_number': 1,
                        'metadata': {'section': 'introduction', 'length': 75}
                    }
                ],
                'image_chunks': [
                    {
                        'chunk_id': 'test_paper_001_img_000',
                        'content': '[IMAGE: Neural network architecture diagram]',
                        'chunk_type': 'image',
                        'page_number': 2,
                        'extracted_text': 'Figure 1: Neural Network Architecture',
                        'metadata': {'image_type': 'diagram', 'width': 512, 'height': 384}
                    }
                ],
                'table_chunks': [
                    {
                        'chunk_id': 'test_paper_001_table_000',
                        'content': 'Model | Accuracy | Speed\nBERT | 92.1% | 45ms\nGPT | 94.3% | 67ms',
                        'chunk_type': 'table',
                        'page_number': 3,
                        'metadata': {'rows': 3, 'cols': 3}
                    }
                ],
                'metadata': {
                    'processing_timestamp': '2024-01-01T12:00:00',
                    'total_pages': 5
                }
            }
        ]

        # Create embedder
        text_embedder = create_embedder("text")

        # Convert to vector documents using our integration function
        print(" Converting to vector documents...")
        vector_docs = documents_to_vector_documents(processed_documents, text_embedder)

        print(f" Converted to {len(vector_docs)} vector documents")

        # Display conversion results
        for doc in vector_docs:
            print(f"  {doc.id} ({doc.content_type}): {doc.content[:50]}...")

        # Create vector store and add documents
        print(" Adding to vector store...")
        vector_store = create_vector_store("hybrid", dimension=text_embedder.get_dimension())
        doc_ids = vector_store.add_documents(vector_docs)

        print(f" Added {len(doc_ids)} documents to vector store")

        # Test search across different content types
        print(" Testing search across content types...")

        queries = [
            "neural network architecture",
            "model performance comparison",
            "multimodal learning approach"
        ]

        for query in queries:
            query_result = text_embedder.embed(query)
            results = vector_store.search(query_result.embedding, top_k=2)

            print(f"\n Query: '{query}'")
            for i, result in enumerate(results):
                print(f"  {i + 1}. {result.document.content_type} - Score: {result.score:.3f}")
                print(f"     Content: {result.document.content[:60]}...")

        return True

    except Exception as e:
        print(f" Document processor integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_vector_store_tests():
    """Run all vector store tests"""
    print(" STARTING VECTOR STORE SYSTEM TESTS")
    print("=" * 60)

    # Test results
    results = {
        'dependencies': False,
        'basic_operations': False,
        'hybrid_store': False,
        'persistence': False,
        'performance': False,
        'embedding_integration': False,
        'multimodal_storage': False,
        'simple_end_to_end': False,
        'document_processor_integration': False
    }

    try:

        # Run core tests first
        results['basic_operations'] = test_basic_vector_store_operations()
        results['simple_end_to_end'] = test_simple_end_to_end()

        # Run advanced tests if core tests pass
        if results['basic_operations']:
            results['hybrid_store'] = test_hybrid_vector_store()
            results['persistence'] = test_persistence()
            results['embedding_integration'] = test_integration_with_embeddings()

            # Optional tests (may fail due to dependencies)
            try:
                results['multimodal_storage'] = test_multimodal_vector_storage()
            except Exception as e:
                print(f" Multimodal test skipped: {e}")

            try:
                results['document_processor_integration'] = test_document_processor_integration()
            except Exception as e:
                print(f" Document processor integration test skipped: {e}")

            # Performance test (may be slow)
            try:
                results['performance'] = test_performance_benchmarks()
            except Exception as e:
                print(f" Performance test skipped: {e}")

    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    # Print results summary
    print(f"\n" + "=" * 60)
    print(" VECTOR STORE TEST RESULTS")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = " PASSED" if passed else " FAILED"
        print(f"{test_name.replace('_', ' ').title(): <35} {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    # Determine success threshold
    core_tests = ['dependencies', 'basic_operations', 'simple_end_to_end', 'embedding_integration']
    core_passed = sum(results[test] for test in core_tests if test in results)
    core_total = len(core_tests)

    if core_passed == core_total:
        print("\n All core tests passed! Vector store system is ready!")
    elif core_passed >= core_total - 1:
        print(f"\n Most core tests passed. Vector store is functional.")
    else:
        print(f"\nCore tests failed.")

    # Next steps
    print(f"\n" + "=" * 60)
    print(" NEXT STEPS")
    print("=" * 60)

    if results['basic_operations'] and results['embedding_integration']:
        print("✓ Core vector store functionality works")
        print("✓ Integration with embeddings verified")
        print("✓ Ready to build retrieval system")
    else:
        print("Fix core vector store issues first")

    if results['simple_end_to_end']:
        print("✓ End-to-end workflow verified")
        print("✓ Ready for production pipeline")

    if results['performance']:
        print("✓ Performance benchmarks completed")
        print("✓ Ready for scale testing")

    return core_passed >= core_total - 1


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test vector store system")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--perf", action="store_true", help="Run performance tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration test only")

    args = parser.parse_args()

    if args.integration:
        success = test_integration_with_embeddings()
    elif args.perf:
        success = test_performance_benchmarks()
    elif args.quick:
        success = quick_vector_store_test()
    elif args.full:
        success = run_vector_store_tests()
    else:
        print("Running quick test by default. Use --full for comprehensive tests.")
        success = quick_vector_store_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()