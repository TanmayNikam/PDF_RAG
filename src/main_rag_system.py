"""
Complete Multimodal RAG integration, integrating all 5 modules, document_processor, embeddings, generation, retrieval,
 and vector_store
"""
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import json
from dataclasses import asdict
import traceback

# Import all our custom modules
from src.document_processor import MultimodalDocumentProcessor
from src.embeddings import create_embedder, MultimodalContent
from src.vector_store import create_vector_store, documents_to_vector_documents
from src.retrieval import create_retriever, RetrievalPipeline, Query
from src.generation import create_generation_pipeline, MultimodalGenerationPipeline


class MultimodalRAGSystem:
    """
    Complete Multimodal RAG System that integrates all components

    Supports end-to-end workflow:
    1. Process PDFs with text and images
    2. Generate multimodal embeddings
    3. Store in vector database
    4. Hybrid retrieval (semantic + keyword)
    5. Generate responses with LLM
    """

    def __init__(self, config: Dict = None):
        """Initialize the complete RAG system"""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

        # System state
        self.is_initialized = False
        self.document_count = 0
        self.index_stats = {}

        # Core components (will be initialized)
        self.document_processor = None
        self.embedder = None
        self.vector_store = None
        self.retrieval_system = None
        self.generation_pipeline = None

        # Performance tracking
        self.performance_stats = {
            'documents_processed': 0,
            'queries_answered': 0,
            'total_processing_time': 0.0,
            'total_query_time': 0.0,
            'avg_processing_time': 0.0,
            'avg_query_time': 0.0
        }

        self.logger.info(" Initializing Multimodal RAG System...")
        self._initialize_system()

    def _default_config(self) -> Dict:
        """Default system configuration"""
        return {
            # Document processing config
            'document_processor': {
                'pdf': {
                    'extract_images': True,
                    'min_image_size': (100, 100),
                    'ocr_enabled': True,
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                },
                'image': {
                    'ocr_enabled': True,
                    'enhance_images': True,
                    'target_size': (512, 512),
                },
                'chunking': {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'strategy': 'adaptive',
                }
            },

            # Embedding config
            'embeddings': {
                'type': 'multimodal',  # text, image, multimodal
                'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'image_model': 'openai/clip-vit-base-patch32',
                'fusion_method': 'concatenation',
                'normalize_embeddings': True
            },

            # Vector store config
            'vector_store': {
                'type': 'hybrid',  # faiss, hybrid
                'faiss': {
                    'index_type': 'IndexFlatIP',
                    'metric_type': 'INNER_PRODUCT'
                },
                'enable_caching': True,
                'cache_size': 1000,
                'enable_reranking': True
            },

            # Retrieval config
            'retrieval': {
                'type': 'hybrid',  # dense, sparse, hybrid
                'dense_weight': 0.7,
                'sparse_weight': 0.3,
                'fusion_method': 'weighted_sum',
                'adaptive_weights': True,
                'enable_query_processing': True,
                'enable_reranking': True,
                'enable_diversity': True
            },

            # Generation config
            'generation': {
                'primary_generator': {
                    'provider': 'ollama',
                    'model': 'llama3.2:3b',
                    'config': {
                        'temperature': 0.1,
                        'max_tokens': 2048,
                        'use_chat_model': True
                    }
                },
                'enable_fallback': True,
                'enable_optimization': True,
                'default_template': 'multimodal',
                'optimizer_config': {
                    'enable_citation_extraction': True,
                    'enable_formatting': True,
                    'max_response_length': 2000
                }
            },

            # System config
            'system': {
                'save_processed_docs': True,
                'output_dir': './rag_outputs',
                'log_level': 'INFO',
                'enable_performance_tracking': True
            }
        }

    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # 1. Initialize Document Processor
            self.logger.info(" Initializing Document Processor...")
            self.document_processor = MultimodalDocumentProcessor(
                self.config['document_processor']
            )

            # 2. Initialize Embedder
            self.logger.info(" Initializing Multimodal Embedder...")
            self.embedder = create_embedder(
                self.config['embeddings']['type'],
                config=self.config['embeddings']
            )

            # 3. Initialize Vector Store
            self.logger.info(" Initializing Vector Store...")
            self.vector_store = create_vector_store(
                self.config['vector_store']['type'],
                dimension=self.embedder.get_dimension(),
                config=self.config['vector_store']
            )

            # 4. Initialize Retrieval System
            self.logger.info(" Initializing Retrieval System...")
            retriever = create_retriever(
                self.config['retrieval']['type'],
                vector_store=self.vector_store,
                embedder=self.embedder,
                config=self.config['retrieval']
            )

            self.retrieval_system = RetrievalPipeline(
                retriever=retriever,
                config=self.config['retrieval']
            )

            # 5. Initialize Generation Pipeline
            self.logger.info(" Initializing Generation Pipeline...")
            self.generation_pipeline = create_generation_pipeline(
                self.config['generation']
            )

            self.is_initialized = True
            self.logger.info(" Multimodal RAG System initialized successfully!")

        except Exception as e:
            self.logger.error(f" System initialization failed: {e}")
            raise

    def add_documents(self, file_paths: List[Union[str, Path]],
                      batch_process: bool = True) -> Dict:
        """
        Add documents to the RAG system

        Args:
            file_paths: List of paths to PDF files or images
            batch_process: Whether to process documents in batch

        Returns:
            Processing results with statistics
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")

        start_time = time.time()
        self.logger.info(f" Adding {len(file_paths)} documents to RAG system...")

        try:
            # Step 1: Process documents
            self.logger.info(" Step 1: Processing documents...")
            if batch_process:
                processing_result = self.document_processor.process_batch(
                    file_paths,
                    output_dir=Path(self.config['system']['output_dir']) / 'processed'
                )
            else:
                processing_result = {
                    'documents': {},
                    'successful_documents': 0,
                    'failed_documents': 0
                }

                for file_path in file_paths:
                    try:
                        result = self.document_processor.process_document(file_path)
                        processing_result['documents'][result['document_id']] = result
                        processing_result['successful_documents'] += 1
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path}: {e}")
                        processing_result['failed_documents'] += 1

            processed_documents = list(processing_result['documents'].values())
            self.logger.info(f" Processed {len(processed_documents)} documents")

            # Step 2: Convert to vector documents
            self.logger.info(" Step 2: Generating embeddings...")
            vector_docs = documents_to_vector_documents(processed_documents, self.embedder)
            self.logger.info(f" Generated {len(vector_docs)} vector documents")

            # Step 3: Add to vector store
            self.logger.info(" Step 3: Adding to vector store...")
            doc_ids = self.vector_store.add_documents(vector_docs)
            self.logger.info(f" Added {len(doc_ids)} documents to vector store")

            # Step 4: Update retrieval system if it supports document addition
            if hasattr(self.retrieval_system.retriever, 'add_documents'):
                self.logger.info(" Step 4: Updating retrieval index...")

                # Convert processed documents for sparse retriever
                sparse_docs = []
                for doc in processed_documents:
                    for chunk_type in ['text_chunks', 'image_chunks', 'table_chunks']:
                        chunks = doc.get(chunk_type, [])
                        for chunk in chunks:
                            sparse_docs.append({
                                'id': chunk['chunk_id'],
                                'content': chunk['content'],
                                'metadata': chunk.get('metadata', {})
                            })

                self.retrieval_system.retriever.add_documents(sparse_docs)
                self.logger.info(f" Updated retrieval index with {len(sparse_docs)} chunks")

            # Update system statistics
            self.document_count += len(processed_documents)
            processing_time = time.time() - start_time

            self.performance_stats['documents_processed'] += len(processed_documents)
            self.performance_stats['total_processing_time'] += processing_time
            # self.performance_stats['avg_processing_time'] = (
            #         self.performance_stats['total_processing_time'] /
            #         self.performance_stats['documents_processed']
            # )

            # Prepare result summary
            result = {
                'success': True,
                'documents_processed': len(processed_documents),
                'vector_documents_created': len(vector_docs),
                'documents_indexed': len(doc_ids),
                'processing_time': processing_time,
                'processing_details': processing_result,
                'system_stats': self.get_system_stats()
            }

            self.logger.info(f" Document addition completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f" Document addition failed: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'documents_processed': 0,
                'processing_time': time.time() - start_time
            }

    def query(self, question: str,
              top_k: int = 5,
              query_config: Dict = None) -> Dict:
        """
        Answer a question using the RAG system

        Args:
            question: User question
            top_k: Number of documents to retrieve
            query_config: Optional query configuration

        Returns:
            Complete response with answer, sources, and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")

        if self.document_count == 0:
            return {
                'success': False,
                'error': 'No documents in the system. Please add documents first.',
                'answer': 'I don\'t have any documents to search through. Please add some documents first.',
                'sources': [],
                'metadata': {}
            }

        start_time = time.time()
        query_config = query_config or {}

        self.logger.info(f" Processing query: '{question}'")

        try:
            # Step 1: Retrieve relevant documents
            self.logger.info(" Step 1: Retrieving relevant documents...")
            retrieval_results = self.retrieval_system.search(
                query_text=question,
                top_k=top_k,
                query_metadata=query_config.get('metadata', {})
            )

            self.logger.info(f"✅ Retrieved {len(retrieval_results)} relevant documents")

            # Step 2: Convert retrieval results to generation format
            context_documents = []
            for result in retrieval_results:
                context_doc = {
                    'document_id': result.document_id,
                    'content': result.content,
                    'score': result.score,
                    'metadata': result.metadata,
                    'rank': result.rank
                }
                context_documents.append(context_doc)

            # Step 3: Generate response
            self.logger.info(" Step 2: Generating response...")
            generation_result = self.generation_pipeline.generate(
                query=question,
                context_documents=context_documents,
                generation_config=query_config.get('generation', {})
            )

            # Update system statistics
            query_time = time.time() - start_time
            self.performance_stats['queries_answered'] += 1
            self.performance_stats['total_query_time'] += query_time
            self.performance_stats['avg_query_time'] = (
                    self.performance_stats['total_query_time'] /
                    self.performance_stats['queries_answered']
            )

            # Prepare comprehensive response
            response = {
                'success': True,
                'question': question,
                'answer': generation_result['response'],
                'sources': self._format_sources(retrieval_results),
                'query_time': query_time,
                'metadata': {
                    'retrieval_metadata': {
                        'documents_found': len(retrieval_results),
                        'top_score': retrieval_results[0].score if retrieval_results else 0,
                        'retrieval_method': getattr(retrieval_results[0], 'retrieval_method',
                                                    'unknown') if retrieval_results else 'none'
                    },
                    'generation_metadata': generation_result['metadata'],
                    'system_metadata': {
                        'total_documents': self.document_count,
                        'query_processing_time': query_time,
                        'components_used': {
                            'document_processor': True,
                            'embedder': True,
                            'vector_store': True,
                            'retrieval_system': True,
                            'generation_pipeline': True
                        }
                    }
                }
            }

            self.logger.info(f" Query completed in {query_time:.2f}s")
            return response

        except Exception as e:
            self.logger.error(f" Query processing failed: {e}")
            return {
                'success': False,
                'question': question,
                'error': str(e),
                'answer': f'I apologize, but I encountered an error while processing your question: {str(e)}',
                'sources': [],
                'query_time': time.time() - start_time,
                'metadata': {'error_details': str(e)}
            }

    def _format_sources(self, retrieval_results: List) -> List[Dict]:
        """Format retrieval results as source citations"""
        sources = []

        for i, result in enumerate(retrieval_results):
            source = {
                'rank': i + 1,
                'document_id': result.document_id,
                'relevance_score': float(result.score),
                'content_preview': result.content[:200] + '...' if len(result.content) > 200 else result.content,
                'metadata': {
                    'content_type': result.metadata.get('content_type', 'unknown'),
                    'page_number': result.metadata.get('page_number'),
                    'section': result.metadata.get('section'),
                    'parent_document': result.metadata.get('document_id', result.parent_document_id),
                    'retrieval_method': getattr(result, 'retrieval_method', 'unknown')
                }
            }
            sources.append(source)

        return sources

    def save_system(self, save_path: Union[str, Path]) -> bool:
        """Save the entire RAG system state"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            self.logger.info(f" Saving RAG system to {save_path}")

            # Save vector store
            vector_store_path = save_path / 'vector_store'
            vector_success = self.vector_store.save(str(vector_store_path))

            # Save system configuration and stats
            system_data = {
                'config': self.config,
                'document_count': self.document_count,
                'performance_stats': self.performance_stats,
                'system_info': {
                    'embedder_dimension': self.embedder.get_dimension(),
                    'vector_store_type': self.config['vector_store']['type'],
                    'retrieval_type': self.config['retrieval']['type'],
                    'generation_provider': self.config['generation']['primary_generator']['provider']
                }
            }

            system_file = save_path / 'system_state.json'
            with open(system_file, 'w') as f:
                json.dump(system_data, f, indent=2)

            self.logger.info(f" RAG system saved successfully")
            return vector_success

        except Exception as e:
            self.logger.error(f" Failed to save RAG system: {e}")
            return False

    def load_system(self, load_path: Union[str, Path]) -> bool:
        """Load a previously saved RAG system"""
        load_path = Path(load_path)

        try:
            self.logger.info(f" Loading RAG system from {load_path}")

            # Load system configuration
            system_file = load_path / 'system_state.json'
            if not system_file.exists():
                raise FileNotFoundError(f"System state file not found: {system_file}")

            with open(system_file, 'r') as f:
                system_data = json.load(f)

            # Update configuration
            self.config = system_data['config']
            self.document_count = system_data['document_count']
            self.performance_stats = system_data['performance_stats']

            # Re-initialize system with loaded config
            self._initialize_system()

            # Load vector store
            vector_store_path = load_path / 'vector_store'
            vector_success = self.vector_store.load(str(vector_store_path))

            if vector_success:
                self.logger.info(f" RAG system loaded successfully")
                return True
            else:
                self.logger.error(" Failed to load vector store")
                return False

        except Exception as e:
            self.logger.error(f" Failed to load RAG system: {e}")
            return False

    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            'system_status': {
                'initialized': self.is_initialized,
                'document_count': self.document_count,
                'components_status': {
                    'document_processor': self.document_processor is not None,
                    'embedder': self.embedder is not None,
                    'vector_store': self.vector_store is not None,
                    'retrieval_system': self.retrieval_system is not None,
                    'generation_pipeline': self.generation_pipeline is not None
                }
            },
            'performance_stats': self.performance_stats.copy(),
            'component_stats': {}
        }

        # Add component-specific stats if available
        if self.vector_store:
            stats['component_stats']['vector_store'] = self.vector_store.get_stats()

        if self.retrieval_system and hasattr(self.retrieval_system.retriever, 'get_comprehensive_stats'):
            stats['component_stats']['retrieval'] = self.retrieval_system.retriever.get_comprehensive_stats()

        if self.generation_pipeline:
            stats['component_stats']['generation'] = self.generation_pipeline.get_pipeline_stats()

        return stats

    def health_check(self) -> Dict:
        """Perform system health check"""
        health = {
            'status': 'healthy',
            'components': {},
            'issues': [],
            'recommendations': []
        }

        try:
            # Check system initialization
            if not self.is_initialized:
                health['status'] = 'error'
                health['issues'].append('System not initialized')
                return health

            # Check document count
            if self.document_count == 0:
                health['issues'].append('No documents in system')
                health['recommendations'].append('Add documents using add_documents()')

            # Check vector store
            if self.vector_store:
                vs_stats = self.vector_store.get_stats()
                health['components']['vector_store'] = {
                    'status': 'healthy' if vs_stats['document_count'] > 0 else 'warning',
                    'document_count': vs_stats['document_count']
                }

            # Check embedder
            if self.embedder:
                try:
                    test_embedding = self.embedder.embed("test")
                    health['components']['embedder'] = {
                        'status': 'healthy',
                        'dimension': len(test_embedding.embedding)
                    }
                except Exception as e:
                    health['components']['embedder'] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    health['status'] = 'degraded'

            # Check generation pipeline
            if self.generation_pipeline:
                gen_stats = self.generation_pipeline.get_available_generators()
                available_generators = sum(1 for g in gen_stats.values() if g.get('available', False))

                health['components']['generation'] = {
                    'status': 'healthy' if available_generators > 0 else 'error',
                    'available_generators': available_generators
                }

                if available_generators == 0:
                    health['status'] = 'error'
                    health['issues'].append('No available generators')
                    health['recommendations'].append('Check LLM connectivity (Ollama, etc.)')

            # Overall status
            if health['issues'] and health['status'] == 'healthy':
                health['status'] = 'warning'

        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f'Health check failed: {str(e)}')

        return health

    def interactive_demo(self):
        """Run an interactive demo of the RAG system"""
        print("\n MULTIMODAL RAG SYSTEM - INTERACTIVE DEMO")
        print("=" * 60)

        # Check system health
        health = self.health_check()
        print(f" System Status: {health['status'].upper()}")

        if health['status'] == 'error':
            print(" System has errors:")
            for issue in health['issues']:
                print(f"  - {issue}")
            for rec in health['recommendations']:
                print(f"   {rec}")
            return

        if self.document_count == 0:
            print("️  No documents in system. Add documents first using add_documents()")
            return

        print(f" Documents available: {self.document_count}")
        print("\nType your questions (or 'quit' to exit, 'stats' for statistics):")
        print("-" * 60)

        while True:
            try:
                question = input("\n❓ Your question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print(" Goodbye!")
                    break

                if question.lower() == 'stats':
                    stats = self.get_system_stats()
                    print("\n SYSTEM STATISTICS:")
                    print(f"  Documents: {stats['system_status']['document_count']}")
                    print(f"  Queries answered: {stats['performance_stats']['queries_answered']}")
                    print(f"  Avg query time: {stats['performance_stats']['avg_query_time']:.3f}s")
                    continue

                if not question:
                    continue

                print(f"\n Processing your question...")
                result = self.query(question)

                if result['success']:
                    print(f"\n Answer:")
                    print(f"  {result['answer']}")

                    print(f"\n Sources ({len(result['sources'])}):")
                    for source in result['sources'][:3]:  # Show top 3 sources
                        print(f"  {source['rank']}. Score: {source['relevance_score']:.3f}")
                        print(f"     {source['content_preview']}")
                        print(
                            f"     Type: {source['metadata']['content_type']}, Page: {source['metadata']['page_number']}")

                    print(f"\n  Query time: {result['query_time']:.2f}s")
                else:
                    print(f"\n Error: {result['error']}")

            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")


# Convenience functions for easy system creation
def create_rag_system(config: Dict = None) -> MultimodalRAGSystem:
    """Create a complete RAG system with default or custom configuration"""
    return MultimodalRAGSystem(config)


def quick_rag_setup(pdf_files: List[Union[str, Path]],
                    config: Dict = None) -> MultimodalRAGSystem:
    """
    Quick setup: Create RAG system and add PDF files in one step

    Args:
        pdf_files: List of PDF file paths to process
        config: Optional system configuration

    Returns:
        Initialized RAG system with documents added
    """
    rag_system = create_rag_system(config)

    print(f" Processing {len(pdf_files)} PDF files...")
    result = rag_system.add_documents(pdf_files)

    if result['success']:
        print(f" RAG system ready with {result['documents_processed']} documents!")
    else:
        print(f" Setup failed: {result['error']}")

    return rag_system


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example 1: Create system and add documents
    print(" Creating Multimodal RAG System...")
    rag = create_rag_system()

    # Example 2: Add some PDF files (update paths as needed)
    # pdf_files = [
    #     "path/to/your/document1.pdf",
    #     "path/to/your/document2.pdf"
    # ]
    # rag.add_documents(pdf_files)

    # Example 3: Query the system
    # response = rag.query("What is the main contribution of this research?")
    # print(f"Answer: {response['answer']}")

    # Example 4: Interactive demo
    # rag.interactive_demo()

    print(" RAG System created successfully!")
    print(" Next steps:")
    print("   1. Add documents: rag.add_documents(['path/to/file.pdf'])")
    print("   2. Query system: rag.query('Your question here')")
    print("   3. Run demo: rag.interactive_demo()")