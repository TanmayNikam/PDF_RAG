"""
Advanced prompt management and templating system
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import yaml

@dataclass
class PromptExample:
    """Example for few-shot prompting"""
    input: str
    output: str
    metadata: Optional[Dict] = None


class AdvancedPromptManager:
    """Advanced prompt management system with comprehensive templates"""
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.templates = {}
        self.few_shot_examples = {}
        self._load_default_templates()

        # # Load custom templates if specified
        # custom_templates_path = self.config.get('custom_templates_path')
        # if custom_templates_path:
        #     self._load_custom_templates(custom_templates_path)

    def _load_default_templates(self):
        """Load comprehensive set of default prompt templates"""

        # Basic QA template
        self.templates['basic_qa'] = {
            'template': """Answer the question based on the provided context. Be accurate and concise.

    Context:
    {context}

    Question: {question}

    Answer:""",
                'input_variables': ['context', 'question'],
                'description': 'Basic question-answering template'
            }

            # Multimodal template - CORE FOR PDF QA
        self.templates['multimodal'] = {
                'template': """You are analyzing multimodal content that includes both text and visual elements. 
    Use all available information to provide a comprehensive answer.

    Text Content:
    {text_context}

    Visual Content Descriptions:
    {image_context}

    Tables and Data (if any):
    {table_context}

    Question: {question}

    Answer (considering all text, visual, and tabular information):""",
                'input_variables': ['text_context', 'image_context', 'table_context', 'question'],
                'description': 'Comprehensive multimodal template for PDFs with text, images, and tables'
            }

            # PDF-specific analysis template
        self.templates['pdf_analysis'] = {
            'template': """You are analyzing a PDF document that contains multiple types of content. 
    Provide a thorough analysis based on all available information.

    Document Text:
    {text_context}

    Figures and Images:
    {image_context}

    Tables and Charts:
    {table_context}

    Metadata:
    - Document sections analyzed: {sections}
    - Pages covered: {pages}

    Question: {question}

    Comprehensive Analysis:""",
                'input_variables': ['text_context', 'image_context', 'table_context', 'sections', 'pages', 'question'],
                'description': 'Specialized template for comprehensive PDF document analysis'
            }

            # Citation QA template
        self.templates['citation_qa'] = {
                'template': """Answer the question using the provided context. Always cite your sources using [Source: document_id] format.

    Context:
    {context}

    Question: {question}

    Answer with Citations:""",
                'input_variables': ['context', 'question'],
                'description': 'QA template that emphasizes source citations'
            }

            # Summary template
        self.templates['summary'] = {
                'template': """Provide a comprehensive summary based on the context provided.

    Content to Summarize:
    {context}

    Focus Question (if any): {question}

    Summary:""",
                'input_variables': ['context', 'question'],
                'description': 'Template for summarizing document content'
            }
            # Definition template
        self.templates['definition'] = {
                'template': """Provide a clear definition or explanation based on the context.

    Context:
    {context}

    Definition Request: {question}

    Definition and Explanation:""",
                'input_variables': ['context', 'question'],
                'description': 'Template for providing definitions and explanations'
            }

        self.templates['qwen_multimodal'] = {
            'template': """You are analyzing document pages that contain both text and visual elements. 
        Use the provided page images to answer the question accurately.

        Question: {question}

        Visual Content: [IMAGES_PROVIDED]
        Text Context: {text_context}

        Based on both the visual content and text context, provide a comprehensive answer:""",
            'input_variables': ['question', 'text_context'],
            'supports_images': True,
            'description': 'Template for Qwen2.5-VL with visual understanding'
        }

        def get_template(self, template_name: str) -> Optional[Dict]:
            """Get a specific prompt template"""
            return self.templates.get(template_name)

        def list_templates(self) -> List[str]:
            """List available template names"""
            return list(self.templates.keys())

        def add_template(self, name: str, template: str, input_variables: List[str], description: str = ""):
            """Add a new prompt template"""
            self.templates[name] = {
                'template': template,
                'input_variables': input_variables,
                'description': description
            }

    def format_template(self, template_name: str, **kwargs) -> str:
        """Format a template with provided variables"""
        template_data = self.templates.get(template_name)
        if not template_data:
            raise ValueError(f"Template '{template_name}' not found")

        # Fill missing variables with empty strings
        template_vars = template_data['input_variables']
        formatted_kwargs = {}

        for var in template_vars:
            if var in kwargs:
                formatted_kwargs[var] = kwargs[var]
            else:
                formatted_kwargs[var] = ""  # Default to empty string

        return template_data['template'].format(**formatted_kwargs)


    # can we load custom prompt templates or load templates from files?


