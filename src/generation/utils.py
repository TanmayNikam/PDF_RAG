# src/generation/utils.py
"""
Utility functions for the generation module
"""

from typing import Dict, List, Optional, Any, Union
import re
import json
from pathlib import Path


def extract_multimodal_context(context_documents: List[Dict]) -> Dict[str, str]:
    """
    Extract and organize multimodal context for template formatting

    Args:
        context_documents: List of retrieved documents

    Returns:
        Dict with organized context by type
    """

    context = {
        'text_context': '',
        'image_context': '',
        'table_context': '',
        'sections': [],
        'pages': []
    }

    text_parts = []
    image_parts = []
    table_parts = []
    sections = set()
    pages = set()

    for doc in context_documents:
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        content_type = metadata.get('content_type', 'text')

        # Track sections and pages
        if metadata.get('section'):
            sections.add(metadata['section'])
        if metadata.get('page_number'):
            pages.add(metadata['page_number'])

        # Organize by content type
        if content_type == 'text':
            text_parts.append(f"[{doc.get('document_id', 'doc')}] {content}")
        elif content_type == 'image':
            image_description = metadata.get('figure_caption', content)
            image_parts.append(f"[{doc.get('document_id', 'img')}] {image_description}")
        elif content_type == 'table':
            table_caption = metadata.get('table_caption', 'Data table')
            table_parts.append(f"[{doc.get('document_id', 'table')}] {table_caption}: {content}")
        else:
            # Mixed or unknown content - add to text
            text_parts.append(f"[{doc.get('document_id', 'mixed')}] {content}")

    # Join parts
    context['text_context'] = '\n\n'.join(text_parts) if text_parts else 'No text content available.'
    context['image_context'] = '\n\n'.join(image_parts) if image_parts else 'No image content available.'
    context['table_context'] = '\n\n'.join(table_parts) if table_parts else 'No table content available.'
    context['sections'] = ', '.join(sorted(sections)) if sections else 'Unknown'
    context['pages'] = ', '.join(map(str, sorted(pages))) if pages else 'Unknown'

    return context


def validate_generation_config(config: Dict) -> Dict:
    """
    Validate and normalize generation configuration

    Args:
        config: Raw configuration dictionary

    Returns:
        Validated and normalized configuration
    """

    validated = config.copy()

    # Validate primary generator config
    if 'primary_generator' in validated:
        pg_config = validated['primary_generator']

        # Ensure required fields
        if 'provider' not in pg_config:
            pg_config['provider'] = 'ollama'
        if 'model' not in pg_config:
            pg_config['model'] = 'llama3.1:8b'
        if 'config' not in pg_config:
            pg_config['config'] = {}

        # Validate generation parameters
        gen_config = pg_config['config']
        if 'temperature' not in gen_config:
            gen_config['temperature'] = 0.1
        if 'max_tokens' not in gen_config:
            gen_config['max_tokens'] = 1024

        # Clamp temperature
        gen_config['temperature'] = max(0.0, min(2.0, gen_config['temperature']))

        # Clamp max_tokens
        gen_config['max_tokens'] = max(50, min(4000, gen_config['max_tokens']))

    # Validate optimizer config
    if 'optimizer_config' in validated:
        opt_config = validated['optimizer_config']

        # Ensure reasonable response length limits
        if 'max_response_length' in opt_config:
            opt_config['max_response_length'] = max(200, min(5000, opt_config['max_response_length']))

    # Set defaults for missing top-level settings
    validated.setdefault('enable_fallback', True)
    validated.setdefault('enable_optimization', True)
    validated.setdefault('default_template', 'multimodal')

    return validated


def estimate_response_quality(response_text: str, context_documents: List[Dict],
                              query: str) -> Dict[str, float]:
    """
    Estimate the quality of a generated response

    Args:
        response_text: Generated response
        context_documents: Context used for generation
        query: Original query

    Returns:
        Dictionary with quality metrics
    """

    metrics = {}

    # Basic metrics
    word_count = len(response_text.split())
    char_count = len(response_text)
    sentence_count = len([s for s in re.split(r'[.!?]+', response_text) if s.strip()])

    metrics['length_score'] = min(1.0, word_count / 100)  # Prefer 100+ words
    metrics['completeness_score'] = min(1.0, char_count / 500)  # Prefer 500+ chars

    # Structure score
    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
        metrics['structure_score'] = min(1.0, avg_sentence_length / 15)  # Prefer ~15 words/sentence
    else:
        metrics['structure_score'] = 0.0

    # Query relevance
    query_words = set(query.lower().split())
    response_words = set(response_text.lower().split())

    if query_words:
        relevance = len(query_words.intersection(response_words)) / len(query_words)
        metrics['relevance_score'] = relevance
    else:
        metrics['relevance_score'] = 0.5

    # Context utilization
    if context_documents:
        context_words = set()
        for doc in context_documents:
            context_words.update(doc.get('content', '').lower().split()[:50])

        if context_words:
            utilization = len(context_words.intersection(response_words)) / len(context_words)
            metrics['context_score'] = min(1.0, utilization * 2)  # Scale up
        else:
            metrics['context_score'] = 0.0
    else:
        metrics['context_score'] = 0.0

    # Overall quality (weighted average)
    weights = {
        'length_score': 0.2,
        'completeness_score': 0.2,
        'structure_score': 0.2,
        'relevance_score': 0.2,
        'context_score': 0.2
    }

    overall = sum(metrics[k] * weights[k] for k in weights.keys())
    metrics['overall_quality'] = overall

    return metrics


def format_context_for_display(context_documents: List[Dict], max_length: int = 2000) -> str:
    """
    Format context documents for human-readable display

    Args:
        context_documents: List of context documents
        max_length: Maximum length of formatted output

    Returns:
        Formatted context string
    """

    if not context_documents:
        return "No context documents available."

    formatted_parts = []
    current_length = 0

    for i, doc in enumerate(context_documents, 1):
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})

        # Create document header
        doc_id = doc.get('document_id', f'doc_{i}')
        content_type = metadata.get('content_type', 'text')
        page = metadata.get('page_number', 'N/A')

        header = f"Document {i} ({content_type}, Page {page}):"

        # Truncate content if needed
        available_space = max_length - current_length - len(header) - 10
        if available_space <= 0:
            break

        if len(content) > available_space:
            content = content[:available_space] + "..."

        doc_section = f"{header}\n{content}\n"
        formatted_parts.append(doc_section)
        current_length += len(doc_section)

        if current_length >= max_length:
            break

    return "\n".join(formatted_parts)


def save_generation_results(results: Dict, output_path: str) -> bool:
    """
    Save generation results to file for analysis

    Args:
        results: Generation results dictionary
        output_path: Path to save results

    Returns:
        True if successful, False otherwise
    """

    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare serializable results
        serializable_results = {}

        for key, value in results.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)

        # Add timestamp
        from datetime import datetime
        serializable_results['saved_at'] = datetime.now().isoformat()

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Failed to save results: {e}")
        return False


def load_custom_prompts(prompts_file: str) -> Dict[str, Dict]:
    """
    Load custom prompt templates from file

    Args:
        prompts_file: Path to prompts file (JSON or YAML)

    Returns:
        Dictionary of prompt templates
    """

    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        return {}

    try:
        if prompts_path.suffix == '.json':
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif prompts_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return {}
    except Exception as e:
        print(f"Failed to load custom prompts: {e}")
        return {}


class GenerationMetrics:
    """Class for tracking generation performance metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.total_requests = 0
        self.total_time = 0.0
        self.total_tokens = 0
        self.success_count = 0
        self.failure_count = 0
        self.template_usage = {}
        self.model_usage = {}

    def record_generation(self, response_data: Dict):
        """Record metrics from a generation response"""
        self.total_requests += 1

        # Extract metrics from response
        metadata = response_data.get('metadata', {})

        # Time tracking
        if 'pipeline_time' in metadata:
            self.total_time += metadata['pipeline_time']

        # Token tracking
        gen_response = metadata.get('generation_response', {})
        if 'token_count' in gen_response:
            self.total_tokens += gen_response['token_count']

        # Success/failure tracking
        if response_data.get('response') and len(response_data['response'].strip()) > 10:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Template usage
        template = metadata.get('template_used', 'unknown')
        self.template_usage[template] = self.template_usage.get(template, 0) + 1

        # Model usage
        model = gen_response.get('model_name', 'unknown')
        self.model_usage[model] = self.model_usage.get(model, 0) + 1

    def get_summary(self) -> Dict:
        """Get summary of metrics"""
        if self.total_requests == 0:
            return {'error': 'No requests recorded'}

        return {
            'total_requests': self.total_requests,
            'success_rate': self.success_count / self.total_requests,
            'avg_response_time': self.total_time / self.total_requests,
            'avg_tokens_per_request': self.total_tokens / self.total_requests if self.total_tokens > 0 else 0,
            'template_usage': self.template_usage,
            'model_usage': self.model_usage,
            'total_time': self.total_time,
            'total_tokens': self.total_tokens
        }

