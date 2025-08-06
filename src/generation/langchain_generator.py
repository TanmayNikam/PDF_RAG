"""
LangChain-based generator with support for multiple LLM providers
"""

import time
from typing import List, Dict, Optional, Any, Tuple
import json

# LangChain imports
try:
    from langchain.llms import Ollama
    from langchain.chat_models import ChatOllama
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    from langchain.prompts import (
        PromptTemplate,
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate
    )
    from langchain.chains import LLMChain
    from langchain.callbacks import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available. Install with: pip install langchain")

# Additional LLM providers
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


try:
    # Groq integration
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not available. Install with: pip install langchain-groq")

from .base_generator import BaseGenerator, GenerationRequest, GenerationResponse


class LangChainGenerator(BaseGenerator):
    """
    LangChain-based generator supporting multiple LLM providers

    Supports:
    - Ollama (local LLMs)
    - OpenAI (GPT models)
    - Groq (fast inference)
    - Custom prompt templates
    - Chain-based generation
    """

    def __init__(self, provider: str = "ollama", model_name: str = "llama3.1:8b", config: Dict = None):
        super().__init__(model_name, config)

        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required. Install with: pip install langchain")

        self.provider = provider.lower()
        self.llm = None
        self.chat_model = None

        # Generation configuration
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 1024)
        self.top_p = self.config.get('top_p', 0.9)
        self.use_chat_model = self.config.get('use_chat_model', True)

        # Prompt configuration
        self.system_prompt = self.config.get('system_prompt', self._default_system_prompt())
        self.use_context_compression = self.config.get('use_context_compression', True)
        self.max_context_length = self.config.get('max_context_length', 4000)

        # Initialize LLM
        self._initialize_llm()

        # Initialize prompt templates
        self._setup_prompt_templates()

        self.logger.info(f" Initialized LangChain Generator")
        self.logger.info(f"   Provider: {self.provider}")
        self.logger.info(f"   Model: {self.model_name}")
        self.logger.info(f"   Chat Model: {self.use_chat_model}")

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider"""

        if self.provider == "ollama":
            self._initialize_ollama()
        elif self.provider == "openai":
            self._initialize_openai()
        elif self.provider == "groq":
            self._initialize_groq()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _initialize_ollama(self):
        """Initialize Ollama LLM"""
        try:
            base_url = self.config.get('ollama_base_url', 'http://localhost:11434')

            if self.use_chat_model:
                self.chat_model = ChatOllama(
                    model=self.model_name,
                    base_url=base_url,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
                    top_p=self.top_p,
                )
            else:
                self.llm = Ollama(
                    model=self.model_name,
                    base_url=base_url,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
                    top_p=self.top_p,
                )

            self.logger.info(f" Initialized Ollama with {self.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama: {e}")
            raise

    def _initialize_openai(self):
        """Initialize OpenAI LLM"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")

        try:
            api_key = self.config.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key required")

            if self.use_chat_model:
                self.chat_model = ChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            else:
                self.llm = OpenAI(
                    model_name=self.model_name,
                    openai_api_key=api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )

            self.logger.info(f" Initialized OpenAI with {self.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")
            raise

    def _initialize_groq(self):
        """Initialize Groq LLM"""
        if not GROQ_AVAILABLE:
            raise ImportError("Groq not available. Install with: pip install langchain-groq")

        try:
            api_key = self.config.get('groq_api_key')
            if not api_key:
                raise ValueError("Groq API key required")

            # Groq primarily uses chat models
            self.chat_model = ChatGroq(
                groq_api_key=api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            self.use_chat_model = True

            self.logger.info(f" Initialized Groq with {self.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize Groq: {e}")
            raise

    def _setup_prompt_templates(self):
        """Setup prompt templates for different scenarios"""

        # Standard QA template
        self.qa_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question. Be accurate and concise.

    Context:
    {context}

    Question: {question}

    Answer:"""
        )

        # Chat template for chat models
        if self.use_chat_model:
            self.chat_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template("""Context:
    {context}

    Question: {question}""")
            ])

        # Multimodal template
        self.multimodal_template = PromptTemplate(
            input_variables=["text_context", "image_context", "question"],
            template="""Use the following text and image information to answer the question.

    Text Context:
    {text_context}

    Image Information:
    {image_context}

    Question: {question}

    Answer:"""
        )

        # Summary template
        self.summary_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the provided context, provide a comprehensive answer to the question. 
    Include relevant details and cite specific information from the context when possible.

    Context:
    {context}

    Question: {question}

    Detailed Answer:"""
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response using LangChain"""
        start_time = time.time()

        try:
            # Prepare context
            context = self._prepare_context(request.context_documents)

            # Choose appropriate template
            template_type = request.generation_config.get('template_type', 'qa')

            # Generate response
            if self.use_chat_model and self.chat_model:
                response_text = self._generate_with_chat_model(request, context, template_type)
            elif self.llm:
                response_text = self._generate_with_llm(request, context, template_type)
            else:
                raise ValueError("No LLM initialized")

            # Calculate metrics
            generation_time = time.time() - start_time
            token_count = self._estimate_token_count(response_text)
            confidence_score = self._calculate_confidence(request, response_text)

            # Create response
            response = GenerationResponse(
                generated_text=response_text,
                confidence_score=confidence_score,
                generation_time=generation_time,
                token_count=token_count,
                model_name=self.model_name,
                metadata={
                    'provider': self.provider,
                    'template_type': template_type,
                    'context_length': len(context),
                    'context_documents': len(request.context_documents),
                    'generation_config': request.generation_config
                },
                sources_used=[doc.get('document_id', 'unknown') for doc in request.context_documents],
                context_relevance=self._calculate_context_relevance(request, context)
            )

            # Update statistics
            self.update_stats(response)

            return response

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def _generate_with_chat_model(self, request: GenerationRequest, context: str, template_type: str) -> str:
        """Generate using chat model"""

        # Select template
        if template_type == 'multimodal':
            # Handle multimodal content
            text_context, image_context = self._separate_multimodal_context(request.context_documents)
            prompt = self.multimodal_template.format(
                text_context=text_context,
                image_context=image_context,
                question=request.query
            )
            messages = [HumanMessage(content=prompt)]

        elif template_type == 'summary':
            messages = self.chat_template.format_messages(
                context=context,
                question=request.query
            )

        else:  # Default QA
            messages = self.chat_template.format_messages(
                context=context,
                question=request.query
            )

        # Generate response
        response = self.chat_model(messages)

        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)

    def _generate_with_llm(self, request: GenerationRequest, context: str, template_type: str) -> str:
        """Generate using standard LLM"""

        # Select and format template
        if template_type == 'multimodal':
            text_context, image_context = self._separate_multimodal_context(request.context_documents)
            prompt = self.multimodal_template.format(
                text_context=text_context,
                image_context=image_context,
                question=request.query
            )
        elif template_type == 'summary':
            prompt = self.summary_template.format(
                context=context,
                question=request.query
            )
        else:  # Default QA
            prompt = self.qa_template.format(
                context=context,
                question=request.query
            )

        # Generate response
        response = self.llm(prompt)
        return response

    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prepare context from retrieved documents"""
        if not documents:
            return ""

        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents):
            content = doc.get('content', '')

            # Add document header
            doc_id = doc.get('document_id', f'doc_{i}')
            doc_header = f"Document {i + 1} (ID: {doc_id}):\n"

            # Check length constraints
            if self.use_context_compression and current_length + len(content) > self.max_context_length:
                # Truncate or summarize if too long
                remaining_space = self.max_context_length - current_length - len(doc_header)
                if remaining_space > 100:  # Only add if meaningful space left
                    content = content[:remaining_space] + "..."
                else:
                    break

            doc_section = doc_header + content + "\n\n"
            context_parts.append(doc_section)
            current_length += len(doc_section)

        return "".join(context_parts).strip()

    def _separate_multimodal_context(self, documents: List[Dict]) -> Tuple[str, str]:
        """Separate text and image context for multimodal queries"""
        text_parts = []
        image_parts = []

        for doc in documents:
            content_type = doc.get('metadata', {}).get('content_type', 'text')
            content = doc.get('content', '')

            if content_type == 'image':
                # Extract image description or metadata
                image_desc = doc.get('metadata', {}).get('image_description', content)
                image_parts.append(image_desc)
            else:
                text_parts.append(content)

        text_context = "\n\n".join(text_parts)
        image_context = "\n".join(image_parts) if image_parts else "No image information available."

        return text_context, image_context

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimation: ~1.3 tokens per word for English
        word_count = len(text.split())
        return int(word_count * 1.3)

    def _calculate_confidence(self, request: GenerationRequest, response: str) -> float:
        """Calculate confidence score for the response"""
        # Simple heuristics for confidence
        confidence = 0.5  # Base confidence

        # Boost confidence if response is well-structured
        if len(response) > 50:
            confidence += 0.1

        # Boost if response references context
        context_words = set()
        for doc in request.context_documents:
            context_words.update(doc.get('content', '').lower().split()[:20])

        response_words = set(response.lower().split())
        overlap = len(context_words.intersection(response_words))

        if overlap > 5:
            confidence += 0.2
        elif overlap > 2:
            confidence += 0.1

        # Penalize very short responses
        if len(response) < 20:
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _calculate_context_relevance(self, request: GenerationRequest, context: str) -> float:
        """Calculate how relevant the context is to the query"""
        if not context:
            return 0.0

        query_words = set(request.query.lower().split())
        context_words = set(context.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(context_words))
        relevance = overlap / len(query_words)

        return min(1.0, relevance)

    def _default_system_prompt(self) -> str:
        """Default system prompt for the assistant"""
        return """You are a helpful AI assistant that answers questions based on the provided context. 
    Follow these guidelines:
    1. Use only the information provided in the context
    2. Be accurate and factual
    3. If the context doesn't contain enough information, say so
    4. Provide clear and concise answers
    5. Cite specific information from the context when relevant"""

    def is_available(self) -> bool:
        """Check if the generator is available"""
        try:
            if self.use_chat_model and self.chat_model:
                # Test with a simple message
                test_response = self.chat_model([HumanMessage(content="Hello")])
                return test_response is not None
            elif self.llm:
                # Test with a simple prompt
                test_response = self.llm("Hello")
                return test_response is not None
            return False
        except:
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models for the provider"""
        if self.provider == "ollama":
            try:
                # This would need to call Ollama API to list models
                # For now, return common models
                return [
                    "llama3.1:8b", "llama3.1:70b", "llama3.2:3b",
                    "mistral:7b", "codellama:7b", "vicuna:7b"
                ]
            except:
                return []

        elif self.provider == "openai":
            return [
                "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo",
                "gpt-4o", "gpt-4o-mini"
            ]

        elif self.provider == "groq":
            return [
                "llama-3.1-8b-instant", "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768", "gemma2-9b-it"
            ]

        return []

