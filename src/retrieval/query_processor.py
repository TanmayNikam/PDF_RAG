"""
Query processing
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
import pyinflect

from .base_retriever import Query


class QueryProcessor:
    """
        Query processing using NLP libraries

        Features:
        - SpaCy for NER, POS tagging, dependency parsing
        - TextBlob for sentiment analysis
        - NLTK for WordNet synonyms and query expansion
        - Advanced intent detection using linguistic patterns
        """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Library availability
        self.use_spacy = self.config.get('use_spacy', True)
        self.use_textblob = self.config.get('use_textblob', True)
        self.use_nltk = self.config.get('use_nltk', True)

        # Query processing features
        self.enable_ner = self.config.get('enable_ner', True)  # Named Entity Recognition
        self.enable_expansion = self.config.get('enable_expansion', True)
        self.enable_reformulation = self.config.get('enable_reformulation', True)
        self.enable_intent_detection = self.config.get('enable_intent_detection', True)
        self.enable_sentiment = self.config.get('enable_sentiment', True)

        # Initialize SpaCy patterns if available
        if self.use_spacy and nlp:
            self.matcher = Matcher(nlp.vocab)
            self._setup_spacy_patterns()

        # Enhanced intent patterns using linguistic features
        self.intent_patterns = {
            'definition': {
                'keywords': ['what is', 'define', 'meaning of', 'definition of', 'explain'],
                'pos_patterns': ['WP VBZ', 'VB'],  # What is, Define
                'dependency_patterns': ['nsubj', 'dobj']
            },
            'comparison': {
                'keywords': ['compare', 'difference', 'versus', 'vs', 'better than', 'contrast'],
                'pos_patterns': ['VB', 'JJR'],  # Compare, Better
                'entities': ['ORG', 'PRODUCT', 'PERSON']
            },
            'procedure': {
                'keywords': ['how to', 'steps', 'process', 'procedure', 'method', 'tutorial'],
                'pos_patterns': ['WRB TO VB'],  # How to do
                'dependency_patterns': ['advmod', 'aux']
            },
            'factual': {
                'keywords': ['when', 'where', 'who', 'which', 'name'],
                'pos_patterns': ['WRB', 'WP'],  # When, Who
                'entities': ['DATE', 'PERSON', 'ORG', 'GPE']
            },
            'causal': {
                'keywords': ['why', 'reason', 'because', 'cause', 'due to'],
                'pos_patterns': ['WRB'],  # Why
                'dependency_patterns': ['advmod', 'prep']
            },
            'list': {
                'keywords': ['list', 'examples', 'types of', 'kinds of', 'categories'],
                'pos_patterns': ['VB NNS', 'NNS IN'],  # List items, Types of
            }
        }

        # Domain-specific expansion dictionaries
        self.domain_expansions = {
            'ai_ml': {
                'ai': ['artificial intelligence', 'machine intelligence', 'intelligent systems'],
                'ml': ['machine learning', 'statistical learning'],
                'dl': ['deep learning', 'neural networks'],
                'nlp': ['natural language processing', 'computational linguistics'],
                'cv': ['computer vision', 'image processing'],
                'llm': ['large language model', 'language model'],
                'transformer': ['attention mechanism', 'encoder decoder'],
                'bert': ['bidirectional encoder representations'],
                'gpt': ['generative pre-trained transformer', 'autoregressive model']
            },
            'technical': {
                'api': ['application programming interface', 'programming interface'],
                'db': ['database', 'data storage'],
                'ui': ['user interface', 'interface design'],
                'ux': ['user experience', 'usability'],
                'cpu': ['central processing unit', 'processor'],
                'gpu': ['graphics processing unit', 'graphics card']
            }
        }

        print(f" Query Processor initialized")
        print(f"   SpaCy: {self.use_spacy}")
        print(f"   TextBlob: {self.use_textblob}")
        print(f"   NLTK: {self.use_nltk}")

    def process_query(self, query_text: str, metadata: Dict = None) -> Query:
        """Query processing with NLP libraries"""

        # Create base query
        query = Query(
            text=query_text,
            metadata=metadata or {}
        )

        # SpaCy processing
        if self.use_spacy and nlp:
            spacy_analysis = self._analyze_with_spacy(query_text)
            query.metadata.update(spacy_analysis)

        # TextBlob processing
        if self.use_textblob:
            textblob_analysis = self._analyze_with_textblob(query_text)
            query.metadata.update(textblob_analysis)

        # Enhanced intent detection
        if self.enable_intent_detection:
            intent_info = self._detect_enhanced_intent(query_text, query.metadata)
            query.metadata.update(intent_info)

        # Query type determination
        query.query_type = self._determine_enhanced_query_type(query_text, query.metadata)

        # Advanced query expansion
        if self.enable_expansion:
            expanded_info = self._advanced_query_expansion(query_text, query.metadata)
            query.metadata.update(expanded_info)

        # Intelligent reformulation
        if self.enable_reformulation:
            reformulated = self._intelligent_reformulation(query_text, query.metadata)
            if reformulated != query_text:
                query.metadata['original_query'] = query_text
                query.text = reformulated

        # Extract structured filters
        filters = self._extract_advanced_filters(query_text, query.metadata)
        query.filters.update(filters)

        # Add processing metadata
        query.metadata['enhanced_processing'] = True
        query.metadata['processing_libraries'] = {
            'spacy': self.use_spacy,
            'textblob': self.use_textblob,
            'nltk': self.use_nltk
        }

        return query

    def _analyze_with_spacy(self, text: str) -> Dict:
        """Analyze query using SpaCy's advanced NLP features"""
        if not nlp:
            return {}

        doc = nlp(text)
        analysis = {}

        # Named Entity Recognition
        if self.enable_ner:
            entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
            analysis['entities'] = entities
            analysis['entity_types'] = list(set(ent[1] for ent in entities))

        # POS tags and linguistic features
        analysis['pos_tags'] = [(token.text, token.pos_, token.tag_) for token in doc]
        analysis['lemmatized'] = [token.lemma_ for token in doc if not token.is_stop]
        analysis['key_phrases'] = self._extract_noun_phrases(doc)

        # Dependency parsing for question structure
        analysis['dependencies'] = [(token.text, token.dep_, token.head.text) for token in doc]
        analysis['question_structure'] = self._analyze_question_structure(doc)

        # Language detection and confidence
        analysis['language'] = doc.lang_

        return {'spacy_analysis': analysis}

    def _analyze_with_textblob(self, text: str) -> Dict:
        """Analyze query using TextBlob"""

        blob = TextBlob(text)
        analysis = {}

        # Sentiment analysis
        if self.enable_sentiment:
            sentiment = blob.sentiment
            analysis['sentiment'] = {
                'polarity': sentiment.polarity,  # -1 to 1
                'subjectivity': sentiment.subjectivity  # 0 to 1
            }

        # Alternative POS tagging
        analysis['textblob_pos'] = blob.tags

        # Noun phrases
        analysis['noun_phrases'] = list(blob.noun_phrases)

        return {'textblob_analysis': analysis}

    def _detect_enhanced_intent(self, text: str, metadata: Dict) -> Dict:
        """Enhanced intent detection using linguistic features"""
        text_lower = text.lower()
        detected_intents = []
        confidence_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0.0

            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns['keywords'] if keyword in text_lower)
            score += keyword_matches * 0.3

            # POS pattern matching (if SpaCy available)
            if self.use_spacy and 'spacy_analysis' in metadata:
                pos_sequence = ' '.join([tag[1] for tag in metadata['spacy_analysis']['pos_tags']])
                pos_matches = sum(1 for pattern in patterns.get('pos_patterns', []) if pattern in pos_sequence)
                score += pos_matches * 0.4

            # Entity type matching
            if 'spacy_analysis' in metadata and 'entity_types' in metadata['spacy_analysis']:
                entity_types = metadata['spacy_analysis']['entity_types']
                entity_matches = sum(1 for ent_type in patterns.get('entities', []) if ent_type in entity_types)
                score += entity_matches * 0.3

            if score > 0:
                detected_intents.append(intent)
                confidence_scores[intent] = score

        # Determine primary intent
        primary_intent = max(confidence_scores.items(), key=lambda x: x[1])[0] if confidence_scores else 'general'

        return {
            'intent': primary_intent,
            'all_detected_intents': detected_intents,
            'intent_confidence': confidence_scores,
            'is_question': self._is_enhanced_question(text, metadata)
        }

    def _determine_enhanced_query_type(self, text: str, metadata: Dict) -> str:
        """Enhanced query type determination using NLP features"""

        # Check for multimodal indicators
        multimodal_keywords = ['image', 'picture', 'diagram', 'chart', 'figure', 'visual', 'photo', 'screenshot']
        if any(keyword in text.lower() for keyword in multimodal_keywords):
            return 'multimodal'

        # Use entity information for type detection
        if 'spacy_analysis' in metadata:
            entities = metadata['spacy_analysis'].get('entities', [])
            entity_types = metadata['spacy_analysis'].get('entity_types', [])

            # High entity density suggests keyword search
            if len(entities) > len(text.split()) * 0.3:
                return 'keyword'

            # Specific entity types suggest factual queries
            factual_entities = ['PERSON', 'ORG', 'DATE', 'GPE', 'PRODUCT']
            if any(ent_type in entity_types for ent_type in factual_entities):
                return 'keyword'

        # Use intent for type determination
        intent = metadata.get('intent', 'general')
        if intent in ['definition', 'explanation', 'causal']:
            return 'semantic'
        elif intent in ['factual', 'list']:
            return 'keyword'

        # Use sentiment and subjectivity
        if 'textblob_analysis' in metadata:
            sentiment = metadata['textblob_analysis'].get('sentiment', {})
            subjectivity = sentiment.get('subjectivity', 0.5)

            # Highly subjective queries benefit from semantic search
            if subjectivity > 0.7:
                return 'semantic'

        return 'hybrid'

    def _advanced_query_expansion(self, text: str, metadata: Dict) -> Dict:
        """Advanced query expansion using multiple techniques"""
        expanded_terms = []
        expansion_sources = {}

        # Domain-specific expansion
        text_lower = text.lower()
        for domain, expansions in self.domain_expansions.items():
            for term, synonyms in expansions.items():
                if term in text_lower:
                    expanded_terms.extend(synonyms)
                    expansion_sources[term] = {'domain': domain, 'synonyms': synonyms}

        # WordNet synonyms (if NLTK available)
        if self.use_nltk:
            wordnet_expansions = self._get_wordnet_synonyms(text)
            expanded_terms.extend(wordnet_expansions)
            expansion_sources['wordnet'] = wordnet_expansions

        # Entity-based expansion
        if 'spacy_analysis' in metadata:
            entities = metadata['spacy_analysis'].get('entities', [])
            entity_expansions = self._expand_entities(entities)
            expanded_terms.extend(entity_expansions)
            expansion_sources['entities'] = entity_expansions

        # Lemma-based expansion
        if 'spacy_analysis' in metadata:
            lemmas = metadata['spacy_analysis'].get('lemmatized', [])
            lemma_expansions = self._expand_lemmas(lemmas)
            expanded_terms.extend(lemma_expansions)
            expansion_sources['lemmas'] = lemma_expansions

        return {
            'expanded_terms': list(set(expanded_terms)),
            'expansion_sources': expansion_sources,
            'expansion_count': len(set(expanded_terms))
        }

    def _intelligent_reformulation(self, text: str, metadata: Dict) -> str:
        """Intelligent query reformulation based on linguistic analysis"""
        reformulated = text

        # Intent-based reformulation
        intent = metadata.get('intent', 'general')

        if intent == 'definition':
            # Add definition context
            if not any(word in text.lower() for word in ['define', 'definition', 'meaning']):
                reformulated = f"definition concept {text}"

        elif intent == 'procedure':
            # Add procedural context
            if not any(word in text.lower() for word in ['steps', 'process', 'method']):
                reformulated = f"steps process method {text}"

        elif intent == 'comparison':
            # Add comparison context
            if not any(word in text.lower() for word in ['compare', 'difference', 'versus']):
                reformulated = f"compare contrast difference {text}"

        # Question to statement conversion
        if metadata.get('is_question', False):
            reformulated = self._question_to_statement(reformulated, metadata)

        # Entity enhancement
        if 'spacy_analysis' in metadata:
            entities = metadata['spacy_analysis'].get('entities', [])
            reformulated = self._enhance_with_entities(reformulated, entities)

        return reformulated

    def _extract_advanced_filters(self, text: str, metadata: Dict) -> Dict:
        """Extract filters using NLP analysis"""
        filters = {}

        # Time-based filters using NER
        if 'spacy_analysis' in metadata:
            entities = metadata['spacy_analysis'].get('entities', [])

            for entity_text, entity_type, start, end in entities:
                if entity_type == 'DATE':
                    # Extract year from date entities
                    year_match = re.search(r'\b(19|20)\d{2}\b', entity_text)
                    if year_match:
                        filters['year'] = int(year_match.group())

                elif entity_type == 'ORG':
                    filters['organization'] = entity_text

                elif entity_type == 'PERSON':
                    filters['author'] = entity_text

                elif entity_type == 'PRODUCT':
                    filters['product'] = entity_text

        # Content type filters based on intent and keywords
        intent = metadata.get('intent', 'general')
        content_type_map = {
            'definition': 'educational',
            'procedure': 'tutorial',
            'factual': 'reference',
            'comparison': 'analysis'
        }

        if intent in content_type_map:
            filters['content_category'] = content_type_map[intent]

        # Language and complexity filters
        if 'textblob_analysis' in metadata:
            sentiment = metadata['textblob_analysis'].get('sentiment', {})
            subjectivity = sentiment.get('subjectivity', 0.5)

            if subjectivity < 0.3:
                filters['content_style'] = 'objective'
            elif subjectivity > 0.7:
                filters['content_style'] = 'subjective'

        return filters

    def _setup_spacy_patterns(self):
        """Setup SpaCy matcher patterns for common query structures"""
        if not self.matcher:
            return

        # Definition patterns
        definition_patterns = [
            [{"LOWER": "what"}, {"LOWER": "is"}, {"POS": "DET", "OP": "?"}, {"POS": "NOUN"}],
            [{"LOWER": "define"}, {"POS": "NOUN"}],
            [{"LOWER": "meaning"}, {"LOWER": "of"}, {"POS": "NOUN"}]
        ]
        self.matcher.add("DEFINITION", definition_patterns)

        # Procedure patterns
        procedure_patterns = [
            [{"LOWER": "how"}, {"LOWER": "to"}, {"POS": "VERB"}],
            [{"LOWER": "steps"}, {"LOWER": "to"}, {"POS": "VERB"}],
            [{"LOWER": "process"}, {"LOWER": "of"}, {"POS": "VERB", "OP": "?"}]
        ]
        self.matcher.add("PROCEDURE", procedure_patterns)

    def _extract_noun_phrases(self, doc) -> List[str]:
        """Extract meaningful noun phrases"""
        noun_phrases = []
        for chunk in doc.noun_chunks:
            # Filter out single pronouns and very short phrases
            if len(chunk.text) > 2 and chunk.root.pos_ != 'PRON':
                noun_phrases.append(chunk.text)
        return noun_phrases

    def _analyze_question_structure(self, doc) -> Dict:
        """Analyze the grammatical structure of questions"""
        structure = {
            'has_wh_word': False,
            'question_type': 'unknown',
            'main_verb': None,
            'subject': None
        }

        # Find WH-words
        wh_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which']
        for token in doc:
            if token.text.lower() in wh_words:
                structure['has_wh_word'] = True
                structure['question_type'] = token.text.lower()
                break

        # Find main verb and subject
        for token in doc:
            if token.pos_ == 'VERB' and structure['main_verb'] is None:
                structure['main_verb'] = token.text
            if token.dep_ == 'nsubj':
                structure['subject'] = token.text

        return structure

    def _is_enhanced_question(self, text: str, metadata: Dict) -> bool:
        """Enhanced question detection using linguistic analysis"""
        # Basic question mark check
        if text.endswith('?'):
            return True

        # SpaCy-based question detection
        if 'spacy_analysis' in metadata:
            question_structure = metadata['spacy_analysis'].get('question_structure', {})
            if question_structure.get('has_wh_word', False):
                return True

        # Pattern-based detection
        question_patterns = [
            r'\b(what|who|when|where|why|how|which)\s+',
            r'\b(is|are|was|were|do|does|did|can|could|will|would|should)\s+',
            r'\bcan\s+you\s+',
            r'\bcould\s+you\s+'
        ]

        for pattern in question_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _get_wordnet_synonyms(self, text: str) -> List[str]:
        """Get synonyms using WordNet"""

        synonyms = []
        words = text.split()

        for word in words:
            try:
                synsets = wordnet.synsets(word)
                for synset in synsets[:2]:  # Limit to top 2 synsets
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != word and synonym not in synonyms:
                            synonyms.append(synonym)
            except:
                continue

        return synonyms[:10]  # Limit total synonyms

    def _expand_entities(self, entities: List[Tuple]) -> List[str]:
        """Expand named entities with related terms"""
        expansions = []

        for entity_text, entity_type, start, end in entities:
            if entity_type == 'ORG':
                expansions.extend([f"{entity_text} company", f"{entity_text} organization"])
            elif entity_type == 'PERSON':
                expansions.extend([f"{entity_text} researcher", f"{entity_text} author"])
            elif entity_type == 'PRODUCT':
                expansions.extend([f"{entity_text} technology", f"{entity_text} system"])

        return expansions

    def _expand_lemmas(self, lemmas: List[str]) -> List[str]:
        """Expand lemmatized forms with inflected variants"""

        expansions = []
        for lemma in lemmas:
            try:
                # Get different word forms
                inflections = pyinflect.getAllInflections(lemma)
                for pos, forms in inflections.items():
                    expansions.extend(forms)
            except:
                continue

        return list(set(expansions))[:20]  # Limit and deduplicate

    def _question_to_statement(self, text: str, metadata: Dict) -> str:
        """Convert questions to statements for better semantic search"""

        # Use SpaCy analysis if available
        if 'spacy_analysis' in metadata:
            question_structure = metadata['spacy_analysis'].get('question_structure', {})
            question_type = question_structure.get('question_type', 'unknown')

            # Simple conversion rules based on question type
            if question_type == 'what':
                text = re.sub(r'\bwhat\s+is\s+', '', text, flags=re.IGNORECASE)
            elif question_type == 'how':
                text = re.sub(r'\bhow\s+(do|does|to)\s+', '', text, flags=re.IGNORECASE)
            elif question_type == 'why':
                text = re.sub(r'\bwhy\s+(do|does|is|are)\s+', 'reason ', text, flags=re.IGNORECASE)

        # Remove question mark
        text = text.rstrip('?').strip()

        return text

    def _enhance_with_entities(self, text: str, entities: List[Tuple]) -> str:
        """Enhance query with entity context"""
        enhanced = text

        # Add entity type context
        for entity_text, entity_type, start, end in entities:
            if entity_type == 'ORG' and 'company' not in enhanced.lower():
                enhanced = f"organization {enhanced}"
            elif entity_type == 'PERSON' and 'person' not in enhanced.lower():
                enhanced = f"person {enhanced}"

        return enhanced