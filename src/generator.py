"""
Response Generator for GraphRAG - STEP 4: Natural Language Generation

OVERVIEW:
This module is the final step in the GraphRAG pipeline. It takes the organized
graph context and generates natural language responses using LLM or template-based
approaches with graceful fallbacks.

KEY RESPONSIBILITIES:
1. **Context Integration**: Combine ranked graph paths into coherent context
2. **Prompt Engineering**: Create effective prompts for biomedical LLMs
3. **Response Generation**: Generate natural language using LLM or templates
4. **Reasoning Extraction**: Extract reasoning steps from generated responses
5. **Confidence Estimation**: Assess response quality and reliability

GENERATION PIPELINE:
Organized Context → Prompt Construction → LLM Generation → Post-processing → Final Response

DUAL GENERATION STRATEGY:
1. **Primary: LLM-based Generation**
   - Uses transformer models (GPT-2, DialoGPT, etc.)
   - Sophisticated biomedical reasoning
   - Handles complex queries naturally
   
2. **Fallback: Template-based Generation**
   - Activated when LLM model unavailable
   - Uses structured graph context directly
   - Reliable but simpler responses

BIOMEDICAL SPECIALIZATION:
- Prompts optimized for medical accuracy
- Reasoning chain extraction for transparency
- Entity and relation validation against PrimeKG
- Confidence scoring based on evidence strength

EXAMPLE TRANSFORMATION:
Input (Organized Context):
  - Ranked paths: ["metformin TARGETS AMPK", "AMPK REGULATES glucose_metabolism"]
  - Entity groups: [Drug: metformin], [Process: glucose_metabolism]
  
Output (Generated Response):
  "Metformin works by targeting the AMPK enzyme, which regulates glucose metabolism.
   This mechanism helps control blood sugar levels in type 2 diabetes patients."
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F

from .organizer import OrganizedContext
from .graph_data_source import PrimeKGDataSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os as _os
_env_log_level = _os.getenv('GRAPHRAG_LOG_LEVEL', 'INFO').upper()
try:
    logger.setLevel(getattr(logging, _env_log_level, logging.INFO))
    logging.getLogger('httpx').setLevel(logging.WARNING)
except Exception:
    pass

@dataclass
class GenerationConfig:
    """Configuration for the generation process."""
    max_length: int = 1024
    min_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    num_return_sequences: int = 1
    do_sample: bool = True

@dataclass
class GeneratedResponse:
    """Represents a generated response with explanations."""
    text: str
    evidence: List[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

class PrimeKGGenerator:
    """
    STEP 4 of GraphRAG Pipeline: Natural Language Response Generation
    
    Transforms organized graph context into natural language responses using
    either LLM-based generation or template-based fallbacks.
    
    GENERATION APPROACH:
    1. **Context Preparation**: Convert ranked graph paths to coherent context
    2. **Prompt Engineering**: Create biomedical-optimized prompts
    3. **LLM Generation**: Generate response using transformer model
    4. **Fallback Strategy**: Use template-based generation if LLM fails
    5. **Post-processing**: Extract reasoning steps and confidence scores
    
    BIOMEDICAL FEATURES:
    - Graph-aware generation that respects PrimeKG entity relationships
    - Biological reasoning chain extraction for transparency
    - Evidence integration from multiple graph paths
    - Confidence estimation based on supporting evidence
    - Medical accuracy optimization through specialized prompts
    
    FALLBACK MECHANISM:
    If LLM model loading fails (common in resource-constrained environments):
    - Falls back to template-based response generation
    - Uses organized graph context directly as structured answer
    - Still provides reasoning steps and confidence scores
    - Ensures system always produces usable output
    
    SUPPORTED LLM MODELS:
    - Primary: GPT-2, DialoGPT for general biomedical text
    - Biomedical: BioGPT, ClinicalBERT for medical specialization
    - Fallback: Template generation using graph context
    """
    
    def __init__(self, data_source: PrimeKGDataSource, model_name: str = None):
        """
        Initialize the generator with LLM API client.
        
        Args:
            data_source: PrimeKG data source for knowledge validation
            model_name: Optional model name override
        """
        self.data_source = data_source
        
        # Initialize LLM client instead of local model
        self.llm_client = self._initialize_llm_client(model_name)
        
        # Keep these for backward compatibility, but they won't be used
        self.tokenizer = None
        self.model = None
        
        # Default generation config
        self.default_config = GenerationConfig()
        
        logger.info("PrimeKG Generator initialized with LLM client")
    
    def _initialize_llm_client(self, model_name: str = None):
        """Initialize the LLM client for text generation."""
        # Check if LLM usage is enabled for generation
        use_llm_generation = os.getenv('USE_LLM_GENERATION', 'true').lower() == 'true'
        
        if not use_llm_generation:
            logger.info("LLM generation disabled, will use template-based generation only")
            return None
        
        try:
            from .llm_client import create_llm_client
            
            # Use provided model or default from environment
            client_model = model_name or os.getenv('GENERATION_LLM_MODEL') or os.getenv('DEFAULT_LLM_MODEL')
            
            client = create_llm_client(model=client_model, temperature=0.3)  # Lower temp for factual generation
            
            if type(client).__name__ == "MockLLMClient":
                logger.info("LLM client creation fell back to mock client, will use template generation")
                return None
            else:
                logger.info(f"LLM client initialized for generation: {type(client).__name__}")
                return client
                
        except ImportError:
            logger.warning("LLM client module not available, using template generation")
            return None
    
    def generate_response(
        self,
        query: str,
        context: OrganizedContext,
        config: Optional[GenerationConfig] = None
    ) -> GeneratedResponse:
        """
        Generate a response using PrimeKG knowledge.
        
        Args:
            query: Original query string
            context: Organized context from PrimeKG
            config: Generation configuration (optional)
            
        Returns:
            Generated response with evidence and reasoning
        """
        config = config or self.default_config
        logger.info("Generating response")
        
        try:
            # Step 1: Prepare generation context
            logger.debug(f"Preparing generation context for query: {query[:50]}...")
            generation_context = self._prepare_generation_context(query, context)
            logger.debug(f"Generation context prepared, length: {len(generation_context)} chars")

            # Step 2: Generate initial response
            logger.debug("Calling _generate_text...")
            response_text = self._generate_text(generation_context, config)
            logger.debug(f"Generated response text, length: {len(response_text)} chars")
        
            # Step 3: Extract and validate reasoning steps
            reasoning_steps = self._extract_reasoning_steps(response_text, context)
            
            # Step 4: Collect supporting evidence
            evidence = self._collect_evidence(reasoning_steps, context)
            
            # Step 5: Estimate confidence
            confidence = self._estimate_confidence(
                response_text,
                reasoning_steps,
                evidence,
                context
            )
            
            # Step 6: Collect metadata
            metadata = {
                'generation_config': vars(config),
                'context_coverage': self._calculate_context_coverage(response_text, context),
                'biological_consistency': self._check_biological_consistency(reasoning_steps),
                'evidence_strength': self._evaluate_evidence_strength(evidence)
            }
            
            return GeneratedResponse(
                text=response_text,
                evidence=evidence,
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            # Fallback to template-based generation
            logger.info("Falling back to template-based generation due to error")
            try:
                fallback_text = self._template_based_generation(str(context))
            except Exception as e2:
                logger.error(f"Template generation also failed: {e2}")
                fallback_text = f"Error generating response: {str(e)}"
            return GeneratedResponse(
                text=fallback_text,
                evidence=[],
                reasoning_steps=[],
                confidence=0.3,
                metadata={'error': str(e), 'fallback': True}
            )
    
    def _template_based_generation(self, context: str) -> str:
        """Generate response using template-based approach when LLM fails."""
        logger.info("Using template-based generation")
        
        # Extract ranked text paths from context
        lines = context.split('\n')
        paths = []
        entities = []
        
        for line in lines:
            if '--[' in line and ']-->' in line:
                paths.append(line.strip())
            elif line.strip() and not line.startswith('Context:') and not line.startswith('Based on'):
                entities.append(line.strip())
        
        if not paths:
            return "Based on the available knowledge graph data, I was unable to find sufficient information to answer this question comprehensively."
        
        # Create a structured response from the paths
        response_parts = []
        
        if len(paths) >= 1:
            response_parts.append("Based on the biomedical knowledge graph, here are the key relationships found:")
            
            # Extract main entities and relationships
            drug_disease_relations = [p for p in paths if 'indication' in p]
            target_relations = [p for p in paths if any(word in p.lower() for word in ['targets', 'binds', 'interacts'])]
            
            if drug_disease_relations:
                response_parts.append("\n**Treatment Relationships:**")
                for rel in drug_disease_relations[:3]:
                    # Parse the relationship
                    if '--[indication]-->' in rel:
                        parts = rel.split('--[indication]-->')
                        if len(parts) == 2:
                            source = parts[0].strip()
                            target = parts[1].strip().split('(')[0].strip()
                            response_parts.append(f"• {source} is indicated for treating {target}")
            
            if target_relations:
                response_parts.append("\n**Molecular Targets:**")
                for rel in target_relations[:3]:
                    response_parts.append(f"• {rel}")
            
            if not drug_disease_relations and not target_relations:
                response_parts.append("\n**Key Relationships:**")
                for path in paths[:3]:
                    response_parts.append(f"• {path}")
            
            response_parts.append("\nThis information was extracted from the PrimeKG biomedical knowledge graph.")
        
        return '\n'.join(response_parts) if response_parts else "No specific relationships were found in the knowledge graph for this query."
    
    def _prepare_generation_context(
        self,
        query: str,
        context: OrganizedContext
    ) -> str:
        """Prepare context for generation."""
        # Start with the query
        prompt_parts = [f"Query: {query}\n\n"]
        
        # Add hierarchical context
        if context.hierarchical_view:
            prompt_parts.append("Relevant Biological Context:\n")
            for view_type, view_data in context.hierarchical_view.items():
                if view_data:
                    prompt_parts.append(f"- {view_type.title()} hierarchy:\n")
                    for entity_id, entity_info in view_data.items():
                        prompt_parts.append(
                            f"  * {entity_info['entity']['name']}"
                            f" ({len(entity_info['ancestors'])} ancestors,"
                            f" {len(entity_info['descendants'])} descendants)\n"
                        )
        
        # Add pathway information
        if context.pathway_clusters:
            prompt_parts.append("\nRelevant Biological Pathways:\n")
            for cluster in context.pathway_clusters:
                theme = cluster['biological_theme']
                prompt_parts.append(
                    f"- {theme['primary_focus']} pathways involving"
                    f" {theme['main_relationship']}:\n"
                )
                for pathway in cluster['pathways']:
                    central = pathway['central_pathway']
                    if central:
                        prompt_parts.append(f"  * {central['name']}\n")
        
        # Add entity relationships
        if context.entity_groups:
            prompt_parts.append("\nRelevant Entities and Relationships:\n")
            for entity_type, entities in context.entity_groups.items():
                prompt_parts.append(f"- {entity_type}:\n")
                for entity in entities[:3]:  # Limit to top 3 per type
                    relations = [r['type'] for r in entity.get('relations', [])]
                    prompt_parts.append(
                        f"  * {entity['name']} with"
                        f" {len(relations)} relationships\n"
                    )
        
        # Add evidence chains
        if context.evidence_chains:
            prompt_parts.append("\nEvidence Chains:\n")
            for chain in context.evidence_chains[:3]:  # Limit to top 3 chains
                try:
                    prompt_parts.append("- ")
                    for step in chain:
                        # Defensive: handle missing keys gracefully
                        source_name = step.get('source', {}).get('name', 'Unknown') if isinstance(step.get('source'), dict) else str(step.get('source', 'Unknown'))
                        target_name = step.get('target', {}).get('name', 'Unknown') if isinstance(step.get('target'), dict) else str(step.get('target', 'Unknown'))
                        relation = step.get('relation', 'related_to')
                        prompt_parts.append(
                            f"{source_name}"
                            f" --[{relation}]--> "
                            f"{target_name} "
                        )
                    prompt_parts.append("\n")
                except Exception as e:
                    logger.warning(f"Error processing evidence chain: {e}, skipping chain")
                    continue
        
        # If ranked text paths exist, include the top-k
        if getattr(context, 'ranked_text_paths', None):
            prompt_parts.append("\nRelevant Graph Paths:\n")
            for item in context.ranked_text_paths[:10]:
                prompt_parts.append(f"- {item['text']} (score={item['score']:.2f})\n")

        # Add conversational generation instructions for DialoGPT
        prompt_parts.append(
            "\nHuman: Based on this biomedical knowledge from PrimeKG, can you explain this in detail?\n\n"
            "Assistant: Based on the PrimeKG biomedical knowledge graph, I can provide a comprehensive explanation. "
        )
        
        return "".join(prompt_parts)
    
    def _generate_text(
        self,
        context: str,
        config: GenerationConfig
    ) -> str:
        """Generate response text using LLM API (Claude/OpenAI)."""
        try:
            # Check if we have an LLM client available
            if self.llm_client is None:
                logger.info("No LLM client available, using template-based generation")
                return self._template_based_generation(context)
            
            # Create a focused prompt for biomedical text generation
            generation_prompt = self._create_generation_prompt(context)
            
            # Use the LLM client to generate response
            generated_text = self.llm_client.generate(
                generation_prompt,
                max_tokens=450,
                temperature=0.3
            )
            
            # Clean up the generated text
            generated_text = self._post_process_llm_generated_text(generated_text)
            
            # Quality check
            if len(generated_text.strip()) < 60:
                logger.warning(f"LLM generated text too short ({len(generated_text)} chars), using template fallback")
                return self._template_based_generation(context)
            
            logger.info("Successfully generated response using LLM API")
            return generated_text
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            logger.info("Falling back to template-based generation")
            return self._template_based_generation(context)
    
    def _create_generation_prompt(self, context: str) -> str:
        """Create a focused prompt for LLM generation from the prepared context."""
        prompt = f"""You are a biomedical expert explaining medical mechanisms based on knowledge graph data.

{context}

Please provide a clear, scientifically accurate explanation that:
1. Directly answers the question
2. Explains the biological mechanisms involved
3. Uses the provided graph relationships
4. Maintains medical accuracy
5. Is accessible to a general audience

Keep your response concise but comprehensive (2-3 paragraphs)."""
        
        return prompt
    
    def _post_process_llm_generated_text(self, text: str) -> str:
        """Post-process LLM-generated text for biomedical responses."""
        # Clean up the response
        text = text.strip()
        
        # Remove any instruction artifacts
        if "You are a biomedical expert" in text:
            # Find where the actual response starts
            lines = text.split('\n')
            response_lines = []
            started = False
            for line in lines:
                if not started and any(phrase in line.lower() for phrase in ['based on', 'the', 'metformin', 'diabetes']):
                    started = True
                if started:
                    response_lines.append(line)
            text = '\n'.join(response_lines).strip()
        
        # Ensure proper paragraph formatting
        text = text.replace('\n\n\n', '\n\n')  # Remove excessive line breaks
        
        return text
    
    def _post_process_generated_text(self, text: str, context: str) -> str:
        """Post-process generated text to clean up DialoGPT artifacts."""
        # Remove common DialoGPT artifacts
        text = text.strip()
        
        # Remove any residual "Answer:" prompts
        if text.startswith("Answer:"):
            text = text[7:].strip()
        
        # Remove dialogue artifacts like "User:" or "Assistant:"
        text = text.replace("User:", "").replace("Assistant:", "").replace("Human:", "")
        
        # Clean up repetitive patterns
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("###") and len(line) > 5:
                cleaned_lines.append(line)
        
        # Rejoin and ensure we have substantial content
        cleaned_text = ' '.join(cleaned_lines)
        
        # If post-processing made it too short, try to salvage from original
        if len(cleaned_text) < 20 and len(text) >= 20:
            # Just remove obvious artifacts but keep the content
            cleaned_text = text.replace("Answer:", "").replace("User:", "").replace("Assistant:", "").strip()
        
        return cleaned_text
    
    def _is_text_garbled(self, text: str) -> bool:
        """Check if generated text appears to be garbled or nonsensical."""
        if not text or len(text.strip()) < 10:
            return True
        
        # Check for excessive non-alphabetic characters
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return True
        
        # Check for excessively short "words" (indicating garbled tokens)
        words = text.split()
        if len(words) > 0:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < 2.5:  # Average word too short
                return True
        
        # Check for repetitive patterns that indicate model failure
        words_set = set(words)
        if len(words) > 0 and len(words_set) / len(words) < 0.3:  # Too repetitive
            return True
        
        # Check for nonsensical character combinations
        nonsense_patterns = ['amp amp', 'ets ets', 'com com', '...', '..']
        for pattern in nonsense_patterns:
            if pattern in text.lower():
                return True
        
        return False
    
    def _extract_reasoning_steps(
        self,
        text: str,
        context: OrganizedContext
    ) -> List[Dict[str, Any]]:
        """Extract reasoning steps from generated text."""
        steps = []
        
        # Split text into sentences
        sentences = text.split(". ")
        
        current_step = {
            'statement': '',
            'entities': [],
            'relations': [],
            'evidence_type': 'direct'
        }
        
        for sentence in sentences:
            # Look for biological entities mentioned in the sentence
            entities_found = []
            for entity_type, entities in context.entity_groups.items():
                for entity in entities:
                    if entity['name'].lower() in sentence.lower():
                        entities_found.append(entity)
            
            # Look for biological relationships
            relations_found = []
            for chain in context.evidence_chains:
                for step in chain:
                    rel_text = step.get('relation', '')
                    if rel_text and rel_text.lower() in sentence.lower():
                        relations_found.append(step)

            # Normalize: drop empties and deduplicate by relation label
            if relations_found:
                relations_found = [r for r in relations_found if r.get('relation')]
                seen_rel: set = set()
                unique_relations: list = []
                for r in relations_found:
                    key = r.get('relation')
                    if key not in seen_rel:
                        seen_rel.add(key)
                        unique_relations.append(r)
                relations_found = unique_relations
            
            # Determine if this is a new reasoning step
            if entities_found or relations_found:
                if current_step['statement']:
                    steps.append(current_step)
                    current_step = {
                        'statement': '',
                        'entities': [],
                        'relations': [],
                        'evidence_type': 'direct'
                    }
                
                current_step['statement'] = sentence
                current_step['entities'] = entities_found
                current_step['relations'] = relations_found
                
                # Determine evidence type
                if any(word in sentence.lower() for word in ['suggest', 'may', 'might', 'could']):
                    current_step['evidence_type'] = 'inferential'
                elif any(word in sentence.lower() for word in ['show', 'demonstrate', 'prove']):
                    current_step['evidence_type'] = 'conclusive'
        
        # Add final step if not empty
        if current_step['statement']:
            steps.append(current_step)
        
        return steps
    
    def _collect_evidence(
        self,
        reasoning_steps: List[Dict[str, Any]],
        context: OrganizedContext
    ) -> List[Dict[str, Any]]:
        """Collect supporting evidence for reasoning steps."""
        evidence = []
        
        for step in reasoning_steps:
            # Collect entity evidence
            for entity in step['entities']:
                entity_evidence = self.data_source.get_entity_evidence(entity['id'])
                if entity_evidence:
                    evidence.append({
                        'type': 'entity',
                        'entity': entity,
                        'evidence': entity_evidence,
                        'confidence': self._calculate_evidence_confidence(entity_evidence)
                    })
            
            # Collect relationship evidence
            for relation in step['relations']:
                relation_evidence = self.data_source.get_relation_evidence(
                    relation['source']['id'],
                    relation['target']['id'],
                    relation['relation']
                )
                if relation_evidence:
                    evidence.append({
                        'type': 'relation',
                        'relation': relation,
                        'evidence': relation_evidence,
                        'confidence': self._calculate_evidence_confidence(relation_evidence)
                    })
        
        # Fallback 1: derive evidence from ranked graph paths if empty
        if not evidence and getattr(context, 'ranked_text_paths', None):
            for item in context.ranked_text_paths[:5]:
                text = item.get('text', '')
                if '--[' in text and ']-->' in text:
                    try:
                        lhs, rhs = text.split('--[')
                        rel, tgt = rhs.split(']-->')
                        head = lhs.strip()
                        tail = tgt.strip()
                        rel = rel.strip()
                        # Construct lightweight relation evidence from graph snippet
                        ev = [{
                            'source': 'primekg',
                            'provenance': 'graph_path',
                            'head': head,
                            'predicate': rel,
                            'tail': tail
                        }]
                        evidence.append({
                            'type': 'relation',
                            'relation': {'source': {'id': head}, 'target': {'id': tail}, 'relation': rel},
                            'evidence': ev,
                            'confidence': 0.4
                        })
                    except Exception:
                        continue

        # Fallback 2: if still empty, attach provenance for a few top entities
        if not evidence and context.entity_groups:
            attached = 0
            for _, entities in context.entity_groups.items():
                for ent in entities[:2]:
                    ent_ev = self.data_source.get_entity_evidence(ent['id'])
                    if ent_ev:
                        evidence.append({
                            'type': 'entity',
                            'entity': ent,
                            'evidence': ent_ev,
                            'confidence': self._calculate_evidence_confidence(ent_ev)
                        })
                        attached += 1
                        if attached >= 3:
                            break
                if attached >= 3:
                    break

        return evidence
    
    def _calculate_evidence_confidence(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for evidence."""
        if not evidence:
            return 0.0
        
        scores = []
        for item in evidence:
            # Base score
            score = 0.5
            
            # Adjust based on evidence type
            if item.get('type') == 'experimental':
                score += 0.3
            elif item.get('type') == 'computational':
                score += 0.1
            
            # Adjust based on source quality
            if item.get('source') in ['pubmed', 'clinicaltrials']:
                score += 0.2
            
            # Adjust based on recency
            year = item.get('year', 0)
            if year >= 2020:
                score += 0.2
            elif year >= 2015:
                score += 0.1
            
            scores.append(min(score, 1.0))
        
        return sum(scores) / len(scores)
    
    def _estimate_confidence(
        self,
        text: str,
        reasoning_steps: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        context: OrganizedContext
    ) -> float:
        """Estimate overall confidence in the generated response."""
        confidence_scores = []
        
        # Evidence coverage
        if evidence:
            evidence_confidence = sum(e['confidence'] for e in evidence) / len(evidence)
            confidence_scores.append(evidence_confidence)
        
        # Reasoning coherence
        if reasoning_steps:
            reasoning_confidence = sum(
                1.0 if step['evidence_type'] == 'conclusive' else
                0.7 if step['evidence_type'] == 'direct' else
                0.4  # inferential
                for step in reasoning_steps
            ) / len(reasoning_steps)
            confidence_scores.append(reasoning_confidence)
        
        # Context utilization
        context_coverage = self._calculate_context_coverage(text, context)
        confidence_scores.append(context_coverage)
        
        # Biological consistency
        biological_consistency = self._check_biological_consistency(reasoning_steps)
        confidence_scores.append(biological_consistency)
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    def _calculate_context_coverage(
        self,
        text: str,
        context: OrganizedContext
    ) -> float:
        """Calculate how well the response covers the provided context."""
        mentioned_entities = set()
        total_entities = set()
        
        # Count mentioned entities
        text_lower = text.lower()
        for entity_type, entities in context.entity_groups.items():
            for entity in entities:
                total_entities.add(entity['id'])
                if entity['name'].lower() in text_lower:
                    mentioned_entities.add(entity['id'])
        
        # Calculate coverage
        return len(mentioned_entities) / len(total_entities) if total_entities else 0.0
    
    def _check_biological_consistency(
        self,
        reasoning_steps: List[Dict[str, Any]]
    ) -> float:
        """Check biological consistency of reasoning steps."""
        if not reasoning_steps:
            return 0.0
        
        consistency_scores = []
        
        for i in range(len(reasoning_steps) - 1):
            current = reasoning_steps[i]
            next_step = reasoning_steps[i + 1]
            
            # Check entity consistency
            shared_entities = set(e['id'] for e in current['entities']) & set(e['id'] for e in next_step['entities'])
            if shared_entities:
                consistency_scores.append(0.8)
            
            # Check relation consistency
            current_relations = set(r['relation'] for r in current['relations'])
            next_relations = set(r['relation'] for r in next_step['relations'])
            if current_relations & next_relations:
                consistency_scores.append(1.0)
            elif current_relations or next_relations:
                consistency_scores.append(0.6)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def _evaluate_evidence_strength(
        self,
        evidence: List[Dict[str, Any]]
    ) -> float:
        """Evaluate the strength of supporting evidence."""
        if not evidence:
            return 0.0
        
        strength_scores = []
        
        for item in evidence:
            # Base score
            score = 0.5
            
            # Adjust based on evidence type
            if item['type'] == 'entity':
                score += 0.2
            elif item['type'] == 'relation':
                score += 0.3
            
            # Adjust based on confidence
            score += item['confidence']
            
            strength_scores.append(min(score / 2, 1.0))  # Normalize to [0,1]
        
        return sum(strength_scores) / len(strength_scores)
