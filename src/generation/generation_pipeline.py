"""
Complete generation pipeline integrating all components with enhanced multimodal support
"""

from typing import Dict, List, Optional, Any, Union
import logging
import time
from dataclasses import asdict

from .base_generator import BaseGenerator, GenerationRequest, GenerationResponse
from .langchain_generator import LangChainGenerator
from .prompt_manager import AdvancedPromptManager
from .response_optimizer import ResponseOptimizer


class MultimodalGenerationPipeline:
    """
    Complete generation pipeline for multimodal RAG system with enhanced capabilities
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.prompt_manager = AdvancedPromptManager(
            self.config.get('prompt_config', {})
        )

        self.response_optimizer = ResponseOptimizer(
            self.config.get('optimizer_config', {})
        )

        # Initialize generators
        self.generators = {}
        self.primary_generator = None
        self._initialize_generators()

        # Pipeline settings
        self.enable_fallback = self.config.get('enable_fallback', True)
        self.enable_optimization = self.config.get('enable_optimization', True)
        self.default_template = self.config.get('default_template', 'multimodal')
        self.enable_context_analysis = self.config.get('enable_context_analysis', True)

        # Multimodal processing settings
        self.multimodal_settings = {
            'auto_detect_content_types': True,
            'prioritize_visual_content': True,
            'enable_cross_modal_references': True,
            'max_context_per_type': 2000  # Max chars per content type
        }

        self.logger.info(" Enhanced Multimodal Generation Pipeline initialized")
        self.logger.info(f"   Available generators: {list(self.generators.keys())}")
        self.logger.info(f"   Primary generator: {self.primary_generator}")
        self.logger.info(f"   Default template: {self.default_template}")

    def _initialize_generators(self):
        """Initialize available generators with enhanced error handling"""

        self.enable_fallback = self.config.get('enable_fallback', True)

        default_primary_config = {
            'provider': 'ollama',
            'model': 'llama3.2:3b',
            'config': {
                'temperature': 0.1,
                'max_tokens': 4096,
                'use_chat_model': True
            }
        }

        # Primary generator configuration
        primary_config = self.config.get('primary_generator', default_primary_config)


        # print("Printing Primary Config: ", primary_config, type(primary_config))
        # for keys in primary_config.keys():
        #     print(keys, primary_config[keys])

        # print("printing config in primary config: ",primary_config['config'])

        try:
            primary_gen = LangChainGenerator(
                provider=primary_config['provider'],
                model_name=primary_config['model'],
                config=primary_config['config']
            )

            generator_key = f"{primary_config['provider']}_{primary_config['model']}"
            self.generators[generator_key] = primary_gen
            self.primary_generator = generator_key

            self.logger.info(f" Primary generator initialized: {generator_key}")

        except Exception as e:
            self.logger.error(f"Failed to initialize primary generator: {e}")
            raise

        # Fallback generators
        if self.enable_fallback:
            fallback_configs = self.config.get('fallback_generators', [])

            for fb_config in fallback_configs:
                try:
                    fb_gen = LangChainGenerator(
                        provider=fb_config['provider'],
                        model_name=fb_config['model'],
                        config=fb_config['config']
                    )

                    fb_key = f"{fb_config['provider']}_{fb_config['model']}"
                    self.generators[fb_key] = fb_gen

                    self.logger.info(f" Fallback generator added: {fb_key}")

                except Exception as e:
                    self.logger.warning(f"Failed to initialize fallback generator {fb_config}: {e}")

    def generate(self,
                 query: str,
                 context_documents: List[Dict],
                 generation_config: Optional[Dict] = None) -> Dict:
        """
        Enhanced generation with multimodal content analysis

        Args:
            query: User question
            context_documents: Retrieved documents from vector store
            generation_config: Optional generation configuration

        Returns:
            Complete response with generation, optimization, and metadata
        """
        start_time = time.time()

        # Prepare generation config
        gen_config = generation_config or {}

        # Enhanced content analysis
        content_analysis = self._analyze_context_content(context_documents)

        # Intelligent template selection
        template_name = self._select_optimal_template(
            context_documents, query, content_analysis, gen_config.get('template')
        )

        # Create enhanced generation request
        request = GenerationRequest(
            query=query,
            context_documents=context_documents,
            generation_config={
                **gen_config,
                'template_type': template_name,
                'content_analysis': content_analysis
            },
            metadata={
                'pipeline_start_time': start_time,
                'template_selected': template_name,
                'content_analysis': content_analysis,
                'multimodal_processing': True
            }
        )

        # Generate response with fallback support
        response = self._generate_with_enhanced_fallback(request)

        # Optimize response if enabled
        if self.enable_optimization:
            optimization_result = self.response_optimizer.optimize_response(
                response.generated_text,
                context_documents,
                query
            )

            # Update response with optimization results
            response.generated_text = optimization_result['optimized_text']
            response.metadata.update({
                'optimization_applied': True,
                'optimization_results': optimization_result
            })

        # Calculate total pipeline time
        total_time = time.time() - start_time

        # Prepare comprehensive final response
        final_response = {
            'response': response.generated_text,
            'metadata': {
                'generation_response': asdict(response),
                'pipeline_time': total_time,
                'template_used': template_name,
                'generator_used': self.primary_generator,
                'context_documents_count': len(context_documents),
                'content_analysis': content_analysis,
                'multimodal_features_used': self._get_multimodal_features_used(content_analysis, template_name)
            }
        }

        self.logger.info(f" Generation completed in {total_time:.2f}s")
        self.logger.info(f"   Template: {template_name}")
        self.logger.info(f"   Content types: {content_analysis.get('content_types', [])}")

        return final_response

    def _analyze_context_content(self, context_documents: List[Dict]) -> Dict:
        """Comprehensive analysis of context content for multimodal processing"""

        analysis = {
            'total_documents': len(context_documents),
            'content_types': {},
            'content_distribution': {},
            'has_images': False,
            'has_tables': False,
            'has_multimodal': False,
            'page_coverage': set(),
            'sections_covered': set(),
            'estimated_tokens': 0,
            'multimodal_elements': []
        }

        if not context_documents:
            return analysis

        # Analyze each document
        for doc in context_documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            content_type = metadata.get('content_type', 'text')

            # Count content types
            analysis['content_types'][content_type] = analysis['content_types'].get(content_type, 0) + 1

            # Track multimodal elements
            if content_type == 'image':
                analysis['has_images'] = True
                analysis['multimodal_elements'].append({
                    'type': 'image',
                    'document_id': doc.get('document_id'),
                    'description': metadata.get('figure_caption', content[:100]),
                    'page': metadata.get('page_number')
                })
            elif content_type == 'table':
                analysis['has_tables'] = True
                analysis['multimodal_elements'].append({
                    'type': 'table',
                    'document_id': doc.get('document_id'),
                    'description': content[:100],
                    'page': metadata.get('page_number')
                })
            elif content_type == 'multimodal':
                analysis['has_multimodal'] = True

            # Track page and section coverage
            if metadata.get('page_number'):
                analysis['page_coverage'].add(metadata['page_number'])
            if metadata.get('section'):
                analysis['sections_covered'].add(metadata['section'])

            # Estimate token count (rough)
            analysis['estimated_tokens'] += len(content.split()) * 1.3

        # Convert sets to lists for JSON serialization
        analysis['page_coverage'] = sorted(list(analysis['page_coverage']))
        analysis['sections_covered'] = list(analysis['sections_covered'])

        # Calculate content distribution percentages
        total_docs = analysis['total_documents']
        for content_type, count in analysis['content_types'].items():
            analysis['content_distribution'][content_type] = count / total_docs

        return analysis

    def _select_optimal_template(self, context_documents: List[Dict], query: str,
                                 content_analysis: Dict, explicit_template: Optional[str] = None) -> str:
        """Enhanced template selection based on content analysis and query type"""

        if explicit_template:
            return explicit_template

        # Check for multimodal content
        if (content_analysis.get('has_images') or
                content_analysis.get('has_tables') or
                content_analysis.get('has_multimodal')):

            # Use PDF analysis for complex multimodal documents
            if (len(content_analysis.get('content_types', {})) > 2 and
                    len(content_analysis.get('page_coverage', [])) > 1):
                return 'pdf_analysis'
            else:
                return 'multimodal'

        # Query-based template selection
        query_lower = query.lower()

        if any(word in query_lower for word in ['cite', 'source', 'reference', 'according']):
            return 'citation_qa'
        elif any(word in query_lower for word in ['define', 'definition', 'what is', 'explain']):
            return 'definition'
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            return 'summary'
        else:
            return 'basic_qa'

    def _generate_with_enhanced_fallback(self, request: GenerationRequest) -> GenerationResponse:
        """Enhanced generation with intelligent fallback"""

        # Try primary generator
        try:
            primary_gen = self.generators[self.primary_generator]
            if primary_gen.is_available():
                response = primary_gen.generate(request)

                # Validate response quality
                if self._validate_response_quality(response, request):
                    return response
                else:
                    self.logger.warning("Primary generator produced low-quality response, trying fallback")
            else:
                self.logger.warning(f"Primary generator {self.primary_generator} not available")
        except Exception as e:
            self.logger.error(f"Primary generator failed: {e}")

        # Try fallback generators
        if self.enable_fallback:
            for gen_name, generator in self.generators.items():
                if gen_name == self.primary_generator:
                    continue

                try:
                    if generator.is_available():
                        self.logger.info(f"Using fallback generator: {gen_name}")
                        response = generator.generate(request)

                        if self._validate_response_quality(response, request):
                            response.metadata['fallback_used'] = gen_name
                            return response
                        else:
                            self.logger.warning(f"Fallback generator {gen_name} also produced low-quality response")
                except Exception as e:
                    self.logger.warning(f"Fallback generator {gen_name} failed: {e}")

        # If all generators fail, return error response
        return GenerationResponse(
            generated_text="I apologize, but I'm unable to generate a response at this time. Please try again later.",
            confidence_score=0.0,
            generation_time=0.0,
            token_count=0,
            model_name="error",
            metadata={'error': 'All generators failed'},
            sources_used=[],
            context_relevance=0.0
        )

    def _validate_response_quality(self, response: GenerationResponse, request: GenerationRequest) -> bool:
        """Validate the quality of generated response"""

        # Basic quality checks
        if not response.generated_text or len(response.generated_text.strip()) < 10:
            return False

        # Check if response is just a repetition
        if response.generated_text.count(response.generated_text.split()[0]) > len(
                response.generated_text.split()) // 2:
            return False

        # Check confidence score
        if response.confidence_score < 0.1:
            return False

        # Check context relevance for multimodal content
        content_analysis = request.metadata.get('content_analysis', {})
        if content_analysis.get('has_images') or content_analysis.get('has_tables'):
            # Response should mention visual elements for multimodal content
            visual_terms = ['figure', 'image', 'chart', 'table', 'diagram', 'visual', 'shown']
            if not any(term in response.generated_text.lower() for term in visual_terms):
                # This might be acceptable, so don't fail completely
                pass

        return True

    def _get_multimodal_features_used(self, content_analysis: Dict, template_name: str) -> Dict:
        """Get information about multimodal features used in generation"""

        features = {
            'template_type': template_name,
            'multimodal_template_used': template_name in ['multimodal', 'pdf_analysis'],
            'content_types_processed': list(content_analysis.get('content_types', {}).keys()),
            'visual_elements_count': len(content_analysis.get('multimodal_elements', [])),
            'cross_modal_processing': False
        }

        # Determine if cross-modal processing was likely used
        if len(features['content_types_processed']) > 1:
            features['cross_modal_processing'] = True

        return features

    def get_available_generators(self) -> Dict:
        """Get detailed status of all generators"""
        status = {}

        for name, generator in self.generators.items():
            try:
                is_available = generator.is_available()
                stats = generator.get_stats()


                status[name] = {
                    'available': is_available,
                    'model_name': generator.model_name,
                    'provider': getattr(generator, 'provider', 'unknown'),
                    'stats': stats,
                    'is_primary': name == self.primary_generator
                }
            except Exception as e:
                status[name] = {
                    'available': False,
                    'error': str(e),
                    'is_primary': name == self.primary_generator
                }

        return status

    def switch_primary_generator(self, generator_name: str):
        """Switch to a different primary generator"""
        if generator_name in self.generators:
            old_primary = self.primary_generator
            self.primary_generator = generator_name
            self.logger.info(f"Switched primary generator from {old_primary} to {generator_name}")
        else:
            raise ValueError(f"Generator {generator_name} not found")

    def add_generator(self, name: str, generator: BaseGenerator):
        """Add a new generator to the pipeline"""
        self.generators[name] = generator
        self.logger.info(f"Added generator: {name}")

        # Set as primary if no primary exists
        if not self.primary_generator:
            self.primary_generator = name
            self.logger.info(f"Set {name} as primary generator")

    def get_pipeline_stats(self) -> Dict:
        """Get comprehensive pipeline statistics"""
        return {
            'generators': self.get_available_generators(),
            'settings': {
                'enable_fallback': self.enable_fallback,
                'enable_optimization': self.enable_optimization,
                'default_template': self.default_template,
                'enable_context_analysis': self.enable_context_analysis
            },
            'multimodal_settings': self.multimodal_settings
        }
