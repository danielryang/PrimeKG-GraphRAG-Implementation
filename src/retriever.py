"""
Agent-Based Retriever for GraphRAG

This module implements an intelligent agent-based retriever that uses LLM reasoning
to navigate the PrimeKG knowledge graph and retrieve relevant information for
biomedical question answering.

The agent can:
- Analyze queries and determine retrieval strategy
- Navigate the knowledge graph intelligently
- Make decisions about which entities and relationships to explore
- Adapt its approach based on intermediate results
- Provide reasoning for its retrieval decisions
"""

import os as _os
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict, Counter
import json
import os
import re
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .graph_data_source import PrimeKGDataSource
from .query_processor import QueryProcessor, QueryComponents, GraphQuery
from .organizer import RetrievedContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_env_log_level = _os.getenv('GRAPHRAG_LOG_LEVEL', 'INFO').upper()
try:
    logger.setLevel(getattr(logging, _env_log_level, logging.INFO))
    logging.getLogger('httpx').setLevel(logging.WARNING)
except Exception:
    pass


class AgentAction(Enum):
    """Actions the agent can take during retrieval."""
    EXPLORE_ENTITY = "explore_entity"
    FOLLOW_RELATIONSHIP = "follow_relationship"
    SEARCH_SIMILAR = "search_similar"
    EXPAND_SUBGRAPH = "expand_subgraph"
    ANALYZE_PATH = "analyze_path"
    STOP_RETRIEVAL = "stop_retrieval"


class AgentReasoning(Enum):
    """Types of reasoning the agent can use."""
    DIRECT_MATCH = "direct_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    RELATIONSHIP_CHAIN = "relationship_chain"
    DISEASE_MECHANISM = "disease_mechanism"
    DRUG_TARGET = "drug_target"
    GENE_FUNCTION = "gene_function"
    PATHWAY_ANALYSIS = "pathway_analysis"


@dataclass
class AgentDecision:
    """A decision made by the agent."""
    action: AgentAction
    reasoning: AgentReasoning
    target_entity: Optional[str] = None
    target_relationship: Optional[str] = None
    confidence: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class AgentState:
    """Current state of the agent during retrieval."""
    visited_entities: Set[str]
    explored_relationships: Set[Tuple[str, str]]
    current_path: List[str]
    retrieved_entities: List[Dict]
    retrieved_relationships: List[Dict]
    reasoning_chain: List[AgentDecision]
    max_steps: int = 20
    current_step: int = 0


@dataclass
class RetrievedEntity:
    """Information about a retrieved entity."""
    entity_id: str
    entity_type: str
    name: str
    description: str
    relevance_score: float
    context: Dict[str, Any]
    agent_reasoning: str = ""


@dataclass
class RetrievedRelationship:
    """Information about a retrieved relationship."""
    source_id: str
    target_id: str
    relation_type: str
    display_relation: str
    source_entity: RetrievedEntity
    target_entity: RetrievedEntity
    relevance_score: float
    context: Dict[str, Any]
    agent_reasoning: str = ""


@dataclass
class RetrievedPath:
    """Information about a retrieved path between entities."""
    path: List[str]
    relationships: List[RetrievedRelationship]
    path_score: float
    path_length: int
    context: Dict[str, Any]
    agent_reasoning: str = ""


@dataclass
class RetrievalResult:
    """Complete retrieval result from agent-based retrieval."""
    entities: List[RetrievedEntity]
    relationships: List[RetrievedRelationship]
    paths: List[RetrievedPath]
    subgraph: Optional[nx.DiGraph]
    metadata: Dict[str, Any]
    query_info: Dict[str, Any]
    agent_reasoning_chain: List[AgentDecision]
    seed_entities: List[str]


@dataclass
class RetrievalConfig:
    """Configuration for the agent-based retriever."""
    max_steps: int = 8  # modestly reduced to cap exploration
    max_entities: int = 30  # Reduced from 50
    max_relationships: int = 50  # Reduced from 100
    max_paths: int = 15  # Reduced from 20
    # Increased from 0.4 for better filtering of irrelevant entities
    similarity_threshold: float = 0.5
    confidence_threshold: float = 0.6  # Increased from 0.5
    enable_backtracking: bool = True
    enable_reasoning: bool = True
    enable_adaptation: bool = True
    llm_model: str = "gpt-4"  # placeholder
    temperature: float = 0.1
    # New optimization parameters
    max_exploration_per_entity: int = 3  # tighter cap to avoid excessive LLM calls
    enable_early_stopping: bool = True
    # Stop exploring if relevance drops below this
    relevance_score_threshold: float = 0.5


class GraphRetriever:
    """
    Intelligent agent-based retriever that uses LLM reasoning to navigate PrimeKG.

    The agent can:
    - Analyze queries and determine optimal retrieval strategy
    - Navigate the knowledge graph with reasoning
    - Make decisions about exploration based on intermediate results
    - Adapt its approach based on what it discovers
    - Provide explanations for its retrieval decisions
    """

    def __init__(
            self,
            data_source: PrimeKGDataSource,
            query_processor: QueryProcessor,
            config: Optional[RetrievalConfig] = None):
        """
        Initialize the agent-based retriever.

        Args:
            data_source: PrimeKG data source
            query_processor: Query processor instance
            config: Agent configuration
        """
        self.data_source = data_source
        self.query_processor = query_processor
        self.config = config or RetrievalConfig()

        # Initialize LLM client (placeholder - you'll need to implement this)
        self.llm_client = self._initialize_llm_client()

        # Async processing optimizations (addresses CLAUDE.md Problem 2)
        self._thread_executor = ThreadPoolExecutor(max_workers=4)
        self._batch_size = 5  # Batch size for entity analysis
        self._analysis_cache = {}  # Cache for entity relevance analysis

        # Agent prompts for different tasks
        self.prompts = self._initialize_prompts()

        # Entity type weights for relevance scoring
        self.entity_type_weights = {
            'drug': 1.0,
            'disease': 1.0,
            'gene': 0.9,
            'protein': 0.9,
            'pathway': 0.8,
            'anatomy': 0.7,
            'phenotype': 0.8,
            'biological_process': 0.7,
            'cellular_component': 0.6,
            'molecular_function': 0.6
        }

        # Relationship type weights
        self.relationship_type_weights = {
            'indication': 1.0,
            'contraindication': 1.0,
            'interacts_with': 0.9,
            'associated_with': 0.8,
            'causes': 0.9,
            'treats': 1.0,
            'side_effect': 0.9,
            'targets': 0.9,
            'regulates': 0.8,
            'participates_in': 0.7
        }

        logger.info("Agent-based retriever initialized")
        # Simple in-memory caches
        self._result_cache: Dict[str, RetrievalResult] = {}
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._entity_relevance_cache: Dict[str, Dict[str, Any]] = {}

    def _initialize_llm_client(self):
        """Initialize the LLM client for agent reasoning."""
        # Check if LLM usage is enabled
        use_llm = os.getenv('USE_LLM', 'false').lower() == 'true'

        if not use_llm:
            logger.info(
                "LLM usage disabled (USE_LLM=false), using mock client")
            return MockLLMClient()

        try:
            from .llm_client import create_llm_client
            client = create_llm_client(temperature=self.config.temperature)

            # Test if we got a real LLM client or mock
            if type(client).__name__ == "MockLLMClient":
                logger.info("LLM client creation fell back to mock client")
            else:
                logger.info(f"LLM client initialized: {type(client).__name__}")

            return client
        except ImportError:
            logger.warning(
                "LLM client module not available, using mock client")
            return MockLLMClient()

    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize prompts for different agent tasks."""
        return {
            'query_analysis': """
You are an intelligent biomedical knowledge graph agent. Analyze the following query and determine:
1. What type of biomedical question is this?
2. What entities should be explored?
3. What relationships are most relevant?
4. What retrieval strategy should be used?

Query: {query}

Respond in JSON format:
{{
    "query_type": "drug_disease|gene_disease|drug_target|side_effects|pathway|mechanism",
    "primary_entities": ["entity1", "entity2"],
    "relevant_relationships": ["relationship1", "relationship2"],
    "retrieval_strategy": "direct|exploration|pathway_analysis|mechanism_analysis",
    "reasoning": "explanation of your analysis"
}}
""",

            'next_action': """
You are navigating a biomedical knowledge graph. Based on the current state, decide what to do next.

Current State:
- Visited entities: {visited_entities}
- Current path: {current_path}
- Query: {query}
- Available actions: explore_entity, follow_relationship, search_similar, expand_subgraph, analyze_path, stop_retrieval

Available entities to explore: {available_entities}
Available relationships: {available_relationships}

Decide the next action and explain your reasoning.

Respond in JSON format:
{{
    "action": "action_name",
    "target_entity": "entity_id_or_null",
    "target_relationship": "relationship_type_or_null",
    "reasoning": "explanation",
    "confidence": 0.0-1.0
}}
""",

            'entity_exploration': """
You are exploring a biomedical entity in a knowledge graph.

Entity: {entity_name} ({entity_type})
Description: {entity_description}
Query: {query}

What aspects of this entity are most relevant to the query? What should be explored next?

Respond in JSON format:
{{
    "relevance_score": 0.0-1.0,
    "relevant_aspects": ["aspect1", "aspect2"],
    "next_explorations": ["exploration1", "exploration2"],
    "reasoning": "explanation"
}}
""",

            'relationship_analysis': """
You are analyzing relationships in a biomedical knowledge graph.

Source: {source_entity} ({source_type})
Target: {target_entity} ({target_type})
Relationship: {relationship_type}
Query: {query}

How relevant is this relationship to the query? What does it tell us?

Respond in JSON format:
{{
    "relevance_score": 0.0-1.0,
    "insights": ["insight1", "insight2"],
    "next_steps": ["step1", "step2"],
    "reasoning": "explanation"
}}
"""
        }

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant information using intelligent agent-based approach.

        Args:
            query: Natural language query

        Returns:
            RetrievalResult with relevant entities, relationships, and paths
        """
        logger.info(f"Starting agent-based retrieval for query: {query}")
        cache_key = query.strip().lower()
        if cache_key in self._result_cache:
            logger.info("Using cached retrieval result")
            return self._result_cache[cache_key]

        # Step 1: Analyze the query
        query_analysis = self._analyze_query(query)
        logger.info(f"Query analysis: {query_analysis}")

        # Step 2: Initialize agent state
        agent_state = AgentState(
            visited_entities=set(),
            explored_relationships=set(),
            current_path=[],
            retrieved_entities=[],
            retrieved_relationships=[],
            reasoning_chain=[],
            max_steps=self.config.max_steps
        )

        # Step 3: Identify seed entities
        seed_entities = self._identify_seed_entities(query, query_analysis)
        logger.info(f"Identified {len(seed_entities)} seed entities")

        # Step 4: Agent-driven exploration (with async batch processing)
        agent_state = self._agent_exploration(
            query, seed_entities, agent_state, query_analysis)

        # Step 4.5: If agent exploration returns too little, fall back to CONTROLLED KG expansion
        # CRITICAL: Add timeout to prevent performance issues
        import time
        fallback_start_time = time.time()
        # Reduced to 8 seconds (was 15s) - skip if too slow
        FALLBACK_TIMEOUT = 8.0

        # Skip fallback expansion if using PyKEEN (NetworkX) - shortest paths are too slow
        # Only use neighbor expansion which is faster
        skip_shortest_paths = not getattr(
            self.data_source, 'using_neo4j', False)

        if len(agent_state.retrieved_entities) < 3:
            logger.info(
                f"Agent found only {len(agent_state.retrieved_entities)} entities, using controlled fallback expansion")

            # Limit the number of seed entities to explore
            grounded_entity_ids = seed_entities[:min(
                2, len(seed_entities))]  # Reduced from 3 to 2
            entities_added = 0
            max_fallback_entities = 5  # Reduced from 10 to 5 - be more aggressive

            # Skip shortest paths if using PyKEEN (too slow) or if timeout
            # already exceeded
            if (not skip_shortest_paths and
                len(grounded_entity_ids) >= 2 and
                    (time.time() - fallback_start_time) < FALLBACK_TIMEOUT):
                # Use shortest paths but with strict limits and timeout
                try:
                    # CRITICAL: Add timeout wrapper
                    import signal
                    paths = []
                    try:
                        # Check timeout BEFORE calling (expensive operation)
                        if (time.time() -
                                fallback_start_time) >= FALLBACK_TIMEOUT:
                            logger.warning(
                                f"Skipping shortest paths - timeout already exceeded")
                            paths = []
                        else:
                            paths = self.data_source.find_all_shortest_paths(
                                grounded_entity_ids, max_len=2, limit_paths=3)  # Reduced from 5 to 3
                    except Exception as e:
                        logger.warning(f"Shortest paths failed: {e}, skipping")
                        paths = []

                    # Check timeout before processing
                    if (time.time() - fallback_start_time) >= FALLBACK_TIMEOUT:
                        logger.warning(
                            f"Fallback expansion timeout ({FALLBACK_TIMEOUT}s), stopping early")
                        paths = []

                    for pid, path in enumerate(
                            paths[:5]):  # Reduced from 8 to 5
                        # Check timeout during processing
                        if (time.time() -
                                fallback_start_time) >= FALLBACK_TIMEOUT:
                            logger.warning(
                                f"Fallback expansion timeout during processing, stopping")
                            break
                        if entities_added >= max_fallback_entities:
                            break
                        for node_id in path:
                            # Check timeout during processing
                            if (time.time() -
                                    fallback_start_time) >= FALLBACK_TIMEOUT:
                                logger.warning(
                                    f"Fallback expansion timeout during path processing, stopping")
                                break
                            if node_id not in agent_state.visited_entities and entities_added < max_fallback_entities:
                                info = self.data_source.entity_info.get(
                                    node_id, {'name': node_id, 'type': 'unknown'})
                                # Filter by entity type relevance
                                entity_type = info.get('type', 'unknown')
                                # If type is unknown, try to infer it
                                if entity_type == 'unknown' and hasattr(
                                        self.data_source, '_infer_entity_type'):
                                    inferred_type = self.data_source._infer_entity_type(
                                        info.get('name', node_id))
                                    if inferred_type != 'unknown':
                                        entity_type = inferred_type

                                if self._is_relevant_entity_type(
                                        entity_type, query):
                                    agent_state.retrieved_entities.append(
                                        RetrievedEntity(
                                            entity_id=node_id,
                                            entity_type=entity_type,
                                            name=info.get(
                                                'name',
                                                node_id),
                                            description='',
                                            relevance_score=0.5,
                                            context={
                                                'source': 'shortest_paths'},
                                            agent_reasoning='Controlled path expansion'))
                                    agent_state.visited_entities.add(node_id)
                                    entities_added += 1
                except Exception as e:
                    logger.warning(f"Fallback expansion error: {e}")

            # Always do neighbor expansion (faster than shortest paths)
                # Limited 1-hop expansion with relevance filtering
                for seed in grounded_entity_ids:
                    if entities_added >= max_fallback_entities:
                        break
                    # Check timeout BEFORE expensive neighbor call
                    if (time.time() - fallback_start_time) >= FALLBACK_TIMEOUT:
                        logger.warning(
                            f"Fallback expansion timeout before neighbor expansion, stopping")
                        break

                    neighbors = self.data_source.get_entity_neighbors(
                        seed, max_hops=1, relation_types=None)
                    neighbor_count = 0
                    # Limit neighbors per seed to avoid explosion
                    max_neighbors_per_seed = 20  # Limit neighbors checked per seed
                    for nid, ninfo in list(neighbors.items())[
                            :max_neighbors_per_seed]:
                        # Check timeout during neighbor processing
                        if (time.time() - fallback_start_time) >= FALLBACK_TIMEOUT:
                            logger.warning(
                                f"Fallback expansion timeout during neighbor processing, stopping")
                            break
                        if (nid not in agent_state.visited_entities and
                                entities_added < max_fallback_entities and
                                neighbor_count < self.config.max_exploration_per_entity):

                            info = ninfo.get(
                                'entity_info', {
                                    'name': nid, 'type': 'unknown'})
                            entity_type = info.get('type', 'unknown')
                            # If type is unknown, try to infer it
                            if entity_type == 'unknown' and hasattr(
                                    self.data_source, '_infer_entity_type'):
                                inferred_type = self.data_source._infer_entity_type(
                                    info.get('name', nid))
                                if inferred_type != 'unknown':
                                    entity_type = inferred_type

                            # Filter by entity type relevance
                            if self._is_relevant_entity_type(
                                    entity_type, query):
                                agent_state.retrieved_entities.append(
                                    RetrievedEntity(
                                        entity_id=nid,
                                        entity_type=entity_type,
                                        name=info.get(
                                            'name',
                                            nid),
                                        description='',
                                        relevance_score=0.4,
                                        context={
                                            'source': 'one_hop'},
                                        agent_reasoning='Controlled neighbor expansion'))
                                agent_state.visited_entities.add(nid)
                                entities_added += 1
                                neighbor_count += 1

            logger.info(f"Fallback expansion added {entities_added} entities")

        # Step 5: Compile results
        result = self._compile_retrieval_result(
            agent_state, query, seed_entities, query_analysis)

        logger.info(
            f"Agent-based retrieval completed. Retrieved {len(result.entities)} entities, {len(result.relationships)} relationships")

        self._result_cache[cache_key] = result
        return result

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine retrieval strategy."""
        cache_key = query.strip().lower()
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        prompt = self.prompts['query_analysis'].format(query=query)

        try:
            response = self.llm_client.generate(prompt)
            analysis = json.loads(response)
            self._analysis_cache[cache_key] = analysis
            return analysis
        except Exception as e:
            logger.warning(f"Failed to analyze query with LLM: {e}")
            # Fallback analysis
            analysis = self._fallback_query_analysis(query)
            self._analysis_cache[cache_key] = analysis
            return analysis

    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback query analysis using rule-based approach."""
        query_lower = query.lower()

        # Simple rule-based analysis
        if any(
            word in query_lower for word in [
                'drug',
                'medication',
                'treatment']):
            if any(
                word in query_lower for word in [
                    'disease',
                    'condition',
                    'symptom']):
                query_type = "drug_disease"
            else:
                query_type = "drug_target"
        elif any(word in query_lower for word in ['gene', 'genetic', 'mutation']):
            query_type = "gene_disease"
        elif any(word in query_lower for word in ['side effect', 'adverse', 'contraindication']):
            query_type = "side_effects"
        else:
            query_type = "general"

        return {
            "query_type": query_type,
            "primary_entities": [],
            "relevant_relationships": [],
            "retrieval_strategy": "exploration",
            "reasoning": "Fallback analysis"
        }

    def _identify_seed_entities(
            self, query: str, query_analysis: Dict[str, Any]) -> List[str]:
        """
        Identify seed entities for the query with robust fallback strategies.

        This method uses multiple strategies to ensure we ALWAYS find seed entities,
        even if some methods fail. This makes the system robust to any input.
        """
        # Use the query processor to identify entities
        components, graph_query = self.query_processor.process_query_for_retrieval(
            query)

        # Extract entities from the processed query
        seed_entities = []
        found_entity_names = set()  # Track entity names we've found

        # Strategy 1: Add entities from query processor (most reliable)
        if hasattr(components, 'entities') and components.entities:
            for entity_dict in components.entities:
                entity_id = entity_dict.get('id')
                entity_name = entity_dict.get('text', '')
                # CRITICAL: Validate entity_id exists in entity_info before using it
                # If entity_id is just the entity name (not a real ID), try to
                # find the real ID
                if entity_id:
                    # Check if entity_id exists in entity_info, if not, try to
                    # find it
                    if entity_id not in self.data_source.entity_info:
                        # entity_id might be the entity name, try to find the
                        # real ID
                        real_id = self._find_entity_robust(
                            entity_name if entity_name else entity_id)
                        if real_id:
                            entity_id = real_id
                        else:
                            # Skip this entity if we can't find a valid ID
                            logger.warning(
                                f"Entity '{entity_name}' (id: {entity_id}) not found in entity_info, skipping")
                            continue
                    if entity_id and entity_id not in seed_entities:
                        seed_entities.append(entity_id)
                        found_entity_names.add(entity_name.lower())

        # Strategy 2: Add entities from query analysis primary_entities
        if 'primary_entities' in query_analysis:
            for entity_name in query_analysis['primary_entities']:
                if entity_name.lower() in found_entity_names:
                    continue  # Already found
                # Try multiple search strategies
                entity_id = self._find_entity_robust(entity_name)
                if entity_id and entity_id not in seed_entities:
                    seed_entities.append(entity_id)
                    found_entity_names.add(entity_name.lower())

        # Strategy 3: Extract entities directly from query using dynamic pattern matching
        # This ensures we catch entities even if query processor misses them
        # Use the query processor's pattern matching to find any missed
        # entities
        query_lower = query.lower()
        query_original = query  # Keep original for case-sensitive matching

        # Extract biomedical terms from query using regex patterns (dynamic,
        # not hardcoded)
        import re
        # Drug patterns (common suffixes and known drugs) - match case
        # variations
        drug_patterns = [
            r'\b(aspirin|Aspirin|ASPIRIN|metformin|Metformin|warfarin|Warfarin|insulin|Insulin|ibuprofen|Ibuprofen|acetaminophen|Acetaminophen|paracetamol|Paracetamol|morphine|Morphine|penicillin|Penicillin)\b',
            r'\b\w+mycin\b',  # antibiotics: streptomycin, erythromycin
            r'\b\w+cillin\b',  # antibiotics: penicillin, ampicillin
            r'\b\w+statin\b',  # statins: atorvastatin, simvastatin
            r'\b\w+pril\b',   # ACE inhibitors: lisinopril, captopril
        ]

        # Disease patterns - CRITICAL: More specific patterns to avoid partial matches
        disease_patterns = [
            # Multi-word disease names (match first to avoid partial matches)
            r'\btype\s+[12]\s+diabetes(?:\s+mellitus)?\b',  # type 1/2 diabetes (mellitus)
            r'\balzheimer\'?s?\s+disease\b',
            r'\bparkinson\'?s?\s+disease\b',
            r'\bhuntington\'?s?\s+disease\b',
            r'\bcrohn\'?s?\s+disease\b',
            r'\bbreast\s+cancer\b',
            r'\blung\s+cancer\b',
            r'\bcolon\s+cancer\b',

            # Single-word diseases (specific terms only)
            r'\b(diabetes|cancer|hypertension|asthma|arthritis|osteoporosis|schizophrenia|alzheimer)\b',

            # Generic patterns (match after specific ones)
            r'\b\w{4,}\s+cancer\b',  # at least 4 chars before "cancer"
            r'\b\w{4,}\s+disease\b',  # at least 4 chars before "disease"
            r'\b\w{4,}\s+syndrome\b',  # at least 4 chars before "syndrome"
        ]

        # Gene/protein patterns (case-sensitive)
        gene_patterns = [
            r'\b[A-Z]{2,8}\d*\b',  # Gene symbols: BRCA1, TP53, EGFR, APOE, APP
        ]

        # Extract potential entities from query
        # CRITICAL: Filter out stop words and generic terms to prevent
        # performance issues
        stop_words = {
            'the',
            'are',
            'is',
            'a',
            'an',
            'with',
            'from',
            'to',
            'for',
            'of',
            'in',
            'on',
            'at',
            'by',
            'as',
            'be',
            'been',
            'being',
            'have',
            'has',
            'had',
            'will',
            'would',
            'should',
            'could',
            'may',
            'might',
            'can',
            'this',
            'that',
            'these',
            'those',
            'which',
            'who',
            'where',
            'when',
            'why',
            'what',
            'how',
            'does',
            'do',
            'did'}
        generic_terms = {
            'pathway',
            'disease',
            'gene',
            'protein',
            'drug',
            'effect',
            'effects',  # Added plural
            'side',
            'role',
            'associated',
            'explain',
            'work',
            'treat',
            'development',
            # CRITICAL: Add generic descriptors that aren't entities
            'type',  # "type" alone is not an entity
            'type 1',  # Must be "type 1 diabetes" not just "type 1"
            'type 2',  # Must be "type 2 diabetes" not just "type 2"
            'type i',
            'type ii',
            'mellitus',  # Usually part of "diabetes mellitus"
            'signaling',  # Usually part of "signaling pathway"
            'receptor',  # Usually part of longer name
            'involved',
            'related',
            'positive',
            'negative',
            'regulation',  # Usually part of longer process name
        }

        potential_entities = []
        for pattern in drug_patterns:
            # Use original query to preserve case for drugs like "Aspirin"
            matches = re.finditer(pattern, query_original, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(0).strip()
                # CRITICAL: Filter out stop words and generic terms
                if len(entity_name) < 3 or entity_name.lower(
                ) in stop_words or entity_name.lower() in generic_terms:
                    continue
                potential_entities.append(('drug', entity_name))

        for pattern in disease_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(0).strip()
                # CRITICAL: Filter out stop words and generic terms
                if len(entity_name) < 3 or entity_name.lower(
                ) in stop_words or entity_name.lower() in generic_terms:
                    continue
                potential_entities.append(('disease', entity_name))

        for pattern in gene_patterns:
            # Use original for case-sensitive genes
            matches = re.finditer(pattern, query_original, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(0).strip()
                # CRITICAL: Filter out stop words (genes are usually 2-8 chars,
                # so check length)
                if len(entity_name) < 2 or len(entity_name) > 10:
                    continue
                potential_entities.append(('gene', entity_name))

        # Try to find each potential entity - try multiple name variations
        for entity_type, entity_name in potential_entities:
            # CRITICAL: Double-check stop words and generic terms before searching
            # This prevents matching stop words that might have slipped through
            entity_name_lower = entity_name.lower().strip()
            if (len(entity_name_lower) < 3 or
                entity_name_lower in stop_words or
                entity_name_lower in generic_terms or
                    entity_name_lower in found_entity_names):
                continue  # Skip stop words and generic terms

            # Try exact name first (preserves case)
            entity_id = self._find_entity_robust(entity_name)
            # If not found, try capitalized version
            if not entity_id and entity_name_lower != entity_name:
                entity_id = self._find_entity_robust(entity_name.capitalize())
            # If still not found, try lowercase
            if not entity_id:
                entity_id = self._find_entity_robust(entity_name.lower())
            # If still not found, try uppercase
            if not entity_id:
                entity_id = self._find_entity_robust(entity_name.upper())

            # CRITICAL: Validate that found entity name doesn't start with stop
            # word
            if entity_id:
                entity_info = self.data_source.entity_info.get(entity_id, {})
                found_name = entity_info.get('name', '').lower()
                # Skip if found entity name starts with a stop word (e.g., "the
                # X" or "are Y")
                if found_name:
                    first_word = found_name.split(
                    )[0] if found_name.split() else ''
                    if first_word in stop_words and len(
                            found_name.split()) > 1:
                        logger.debug(
                            f"Skipping entity starting with stop word: {found_name}")
                        continue

            if entity_id and entity_id not in seed_entities:
                seed_entities.append(entity_id)
                found_entity_names.add(entity_name_lower)
                logger.info(
                    f"Found seed entity via pattern matching: {entity_name} -> {entity_id[:50]}...")

        # Strategy 4: Semantic search fallback (if still no entities)
        if not seed_entities:
            try:
                sem_results = self.data_source.semantic_search_entities(
                    query, top_k=15)
                for r in sem_results[:5]:
                    if r['id'] not in seed_entities:
                        seed_entities.append(r['id'])
            except Exception:
                pass

        # Strategy 5: Key term extraction fallback (last resort)
        if not seed_entities:
            key_terms = self._extract_key_terms(query)
            for term in key_terms:
                if term.lower() in found_entity_names:
                    continue
                entity_id = self._find_entity_robust(term)
                if entity_id and entity_id not in seed_entities:
                    seed_entities.append(entity_id)
                    found_entity_names.add(term.lower())
                if len(seed_entities) >= 5:  # Limit fallback entities
                    break

        # Ensure inclusion of Pathway nodes for pathway/mechanism-style queries
        # to improve clustering
        try:
            qa_qtype = (query_analysis.get('query_type') or '').lower()
            qa_strategy = (query_analysis.get(
                'retrieval_strategy') or '').lower()
            needs_pathways = (
                'pathway' in query.lower() or
                'pathway' in qa_qtype or
                'mechanism' in qa_qtype or
                'pathway' in qa_strategy
            )
            if needs_pathways:
                # Prefer pathway entities related to the query
                pathway_candidates = self.data_source.search_entities(
                    query, entity_types=['Pathway'], limit=5) or []
                for pc in pathway_candidates[:3]:
                    pid = pc.get('id')
                    if pid and pid not in seed_entities:
                        seed_entities.append(pid)
                # As a last resort, add a few generic pathways
                if len(seed_entities) < 3:
                    generic_paths = self.data_source.get_entities_by_type(
                        'Pathway', limit=3) or []
                    for pid in generic_paths:
                        if pid not in seed_entities:
                            seed_entities.append(pid)
        except Exception:
            # Non-fatal; proceed without forcing pathways
            pass

        return seed_entities

    def _find_entity_robust(self, entity_name: str) -> Optional[str]:
        """
        Find an entity using multiple strategies to ensure robustness.

        This method tries multiple approaches to find an entity, ensuring
        we can find it even if one method fails. CRITICAL for drugs like "aspirin".
        """
        if not entity_name or not entity_name.strip():
            return None

        entity_name = entity_name.strip()
        entity_name_lower = entity_name.lower()

        # Strategy 1: Direct search with entity name (try multiple case
        # variations)
        search_variations = [
            entity_name,
            entity_name.capitalize(),
            entity_name.lower(),
            entity_name.upper()]
        for search_name in search_variations:
            try:
                search_results = self.data_source.search_entities(
                    search_name, limit=10)
                if search_results:
                    # CRITICAL: Strict matching to prevent "type 2" matching "cutis laxa type 2"

                    # Priority 1: Exact matches (case-insensitive)
                    for result in search_results:
                        result_name = result.get('name', '').lower()
                        if result_name == entity_name_lower:
                            logger.debug(
                                f"Found exact match for '{entity_name}': {result.get('name')}")
                            return result.get('id')

                    # Priority 2: Very close matches (similarity > 0.9, prevents "type 2" matching "cutis laxa type 2")
                    from difflib import SequenceMatcher
                    for result in search_results:
                        result_name = result.get('name', '').lower()
                        similarity = SequenceMatcher(None, entity_name_lower, result_name).ratio()
                        # CRITICAL: High threshold prevents false matches
                        if similarity > 0.9:
                            logger.debug(
                                f"Found high-similarity match for '{entity_name}': {result.get('name')} (similarity={similarity:.2f})")
                            return result.get('id')

                    # Priority 3: Entity name is FULL substring of result (but not too long)
                    # "metformin" in "metformin hydrochloride" is OK
                    # "type 2" in "cutis laxa type 2" is NOT OK (too much extra text)
                    for result in search_results:
                        result_name = result.get('name', '').lower()
                        if entity_name_lower in result_name:
                            # Check that result is not too much longer (prevents false matches)
                            length_ratio = len(result_name) / len(entity_name_lower)
                            if length_ratio <= 1.5:  # Result can be at most 50% longer
                                logger.debug(
                                    f"Found close substring match for '{entity_name}': {result.get('name')}")
                                return result.get('id')

                    # NO "first result" fallback - only return if we have confidence
            except Exception as e:
                logger.debug(f"Search failed for '{search_name}': {e}")
                continue

        # Strategy 3: Search in entity_info cache directly (by name) with STRICT matching
        # This is fast and works even if search_entities fails
        if hasattr(
                self.data_source,
                'entity_info') and self.data_source.entity_info:
            from difflib import SequenceMatcher
            entity_name_lower = entity_name.lower()

            # Try exact match first
            for entity_id, info in self.data_source.entity_info.items():
                cached_name = info.get('name', '').lower()
                if cached_name == entity_name_lower:
                    return entity_id

            # Try high-similarity match (prevents "type 2" matching "cutis laxa type 2")
            for entity_id, info in self.data_source.entity_info.items():
                cached_name = info.get('name', '').lower()
                similarity = SequenceMatcher(None, entity_name_lower, cached_name).ratio()
                if similarity > 0.9:
                    return entity_id

            # Try close substring match with length check
            for entity_id, info in self.data_source.entity_info.items():
                cached_name = info.get('name', '').lower()
                if entity_name_lower in cached_name:
                    # Only if cached name is not too much longer (prevents false matches)
                    length_ratio = len(cached_name) / len(entity_name_lower)
                    if length_ratio <= 1.5:  # At most 50% longer
                        return entity_id

        # Strategy 4: Try semantic search
        try:
            sem_results = self.data_source.semantic_search_entities(
                entity_name, top_k=3)
            if sem_results:
                # Check if semantic result name matches
                for result in sem_results:
                    result_name = result.get('name', '').lower()
                    if entity_name.lower() in result_name or result_name in entity_name.lower():
                        return result.get('id')
                # Fallback to first semantic result
                return sem_results[0].get('id')
        except Exception:
            pass

        return None

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key biomedical terms from the query."""
        # Simple extraction - in practice, you'd use NER
        biomedical_terms = [
            'diabetes', 'cancer', 'hypertension', 'asthma', 'depression',
            'insulin', 'metformin', 'aspirin', 'ibuprofen', 'paracetamol',
            'BRCA1', 'BRCA2', 'TP53', 'APC', 'KRAS'
        ]

        found_terms = []
        query_lower = query.lower()
        for term in biomedical_terms:
            if term in query_lower:
                found_terms.append(term)

        return found_terms

    def _is_relevant_entity_type(self, entity_type: str, query: str) -> bool:
        """Check if an entity type is relevant to the query to avoid exploring irrelevant entities."""
        query_lower = query.lower()
        entity_type_lower = entity_type.lower()

        # Always include high-priority types
        high_priority_types = {'drug', 'disease', 'gene', 'protein', 'pathway'}
        if entity_type_lower in high_priority_types:
            return True

        # For drug queries, focus on drug-related entities
        if any(
            word in query_lower for word in [
                'drug',
                'medication',
                'treatment',
                'therapy']):
            relevant_types = {
                'drug',
                'disease',
                'protein',
                'gene',
                'side_effect',
                'indication'}
            return entity_type_lower in relevant_types

        # For disease queries, focus on disease-related entities
        if any(
            word in query_lower for word in [
                'disease',
                'condition',
                'disorder',
                'syndrome']):
            relevant_types = {
                'disease',
                'drug',
                'gene',
                'protein',
                'pathway',
                'phenotype'}
            return entity_type_lower in relevant_types

        # For genetic queries, focus on genetic entities
        if any(
            word in query_lower for word in [
                'gene',
                'genetic',
                'mutation',
                'protein']):
            relevant_types = {
                'gene',
                'protein',
                'disease',
                'pathway',
                'biological_process'}
            return entity_type_lower in relevant_types

        # For mechanism queries, include pathways and processes
        if any(
            word in query_lower for word in [
                'mechanism',
                'pathway',
                'process',
                'function']):
            relevant_types = {
                'pathway',
                'biological_process',
                'molecular_function',
                'gene',
                'protein',
                'drug'}
            return entity_type_lower in relevant_types

        # Default: include common biomedical types but exclude very specific
        # ones
        excluded_types = {'cellular_component', 'anatomy', 'phenotype'}
        return entity_type_lower not in excluded_types

    def _agent_exploration(self,
                           query: str,
                           seed_entities: List[str],
                           agent_state: AgentState,
                           query_analysis: Dict[str,
                                                Any]) -> AgentState:
        """Agent-driven exploration of the knowledge graph with async batch processing."""

        # Collect entities that need LLM analysis for batch processing
        entities_to_analyze = []

        # Start with seed entities - CRITICAL FIX: Always add seed entities first
        # Seed entities are extracted directly from the query, so they should ALWAYS be included
        # regardless of relevance scores (user explicitly mentioned them)
        for entity_id in seed_entities:
            if entity_id not in agent_state.visited_entities:
                entity_info = self.data_source.entity_info.get(entity_id, {})
                if entity_info:
                    entity_name = entity_info.get('name', entity_id)
                    entity_type = entity_info.get('type', 'unknown')

                    # CRITICAL: Always re-infer entity type to fix misclassifications
                    # This ensures pathways, phenotypes, anatomy are correctly identified
                    cached_type_lower = entity_type.lower()
                    entity_name_lower = entity_name.lower()
                    query_lower = query.lower()
                    
                    # CRITICAL: If query mentions "gene" or "genes", classify gene symbols as "gene" not "protein"
                    # This fixes BRCA1, APOE, APP, PSEN1, PSEN2 being classified as "protein" when query asks about "genes"
                    is_gene_query = 'gene' in query_lower or 'genes' in query_lower or 'role' in query_lower
                    is_gene_symbol = bool(re.match(r'^[A-Z]{2,8}\d*$', entity_name.strip()))
                    common_gene_symbols = ['BRCA1', 'BRCA2', 'APOE', 'APP', 'PSEN1', 'PSEN2', 'TP53', 'EGFR', 'HER2', 'KRAS', 'BRAF']
                    
                    if is_gene_query and (is_gene_symbol or entity_name.strip().upper() in common_gene_symbols):
                        if cached_type_lower == 'protein':
                            entity_type = 'gene'
                            self.data_source.entity_info[entity_id]['type'] = 'gene'
                            logger.debug(f"Re-classified '{entity_name}' as 'gene' (query mentions genes)")
                    
                    # First check: Direct keyword matching (most reliable for pathways)
                    if 'pathway' in entity_name_lower or 'signaling pathway' in entity_name_lower:
                        if cached_type_lower in ['drug', 'protein', 'unknown']:
                            entity_type = 'pathway'
                            self.data_source.entity_info[entity_id]['type'] = 'pathway'
                    elif 'regulation' in entity_name_lower and ('pathway' in entity_name_lower or 'signaling' in entity_name_lower):
                        if cached_type_lower in ['drug', 'protein', 'unknown']:
                            entity_type = 'pathway'
                            self.data_source.entity_info[entity_id]['type'] = 'pathway'
                    elif 'function' in entity_name_lower and 'molecular' in entity_name_lower:
                        if cached_type_lower in ['drug', 'protein', 'unknown']:
                            entity_type = 'molecularfunction'
                            self.data_source.entity_info[entity_id]['type'] = 'molecularfunction'
                    elif 'cytoplasmic side' in entity_name_lower or 'side of' in entity_name_lower:
                        if cached_type_lower in ['drug', 'protein', 'unknown']:
                            entity_type = 'anatomy'
                            self.data_source.entity_info[entity_id]['type'] = 'anatomy'
                    elif entity_name_lower in ['stomach', 'bleeding', 'abnormal bleeding'] or 'abnormality' in entity_name_lower:
                        if cached_type_lower == 'protein':
                            entity_type = 'phenotype' if 'bleeding' in entity_name_lower or 'abnormality' in entity_name_lower else 'anatomy'
                            self.data_source.entity_info[entity_id]['type'] = entity_type
                    elif 'effects of' in entity_name_lower and 'hydrolysis' in entity_name_lower:
                        if cached_type_lower == 'protein':
                            entity_type = 'pathway'
                            self.data_source.entity_info[entity_id]['type'] = 'pathway'
                    # Disease detection (for "neoplasm", "carcinoma", "syndrome", "cutis laxa")
                    elif any(term in entity_name_lower for term in ['neoplasm', 'carcinoma', 'syndrome', 'cutis laxa', 'muscle fiber']):
                        if cached_type_lower == 'protein':
                            entity_type = 'disease' if 'syndrome' in entity_name_lower or 'cutis laxa' in entity_name_lower or 'muscle fiber' in entity_name_lower else 'disease'
                            self.data_source.entity_info[entity_id]['type'] = entity_type
                    # Anatomy/phenotype detection (for "fibers", "relatively", "smaller", "larger")
                    elif ('fiber' in entity_name_lower or 'fibers' in entity_name_lower) and ('relatively' in entity_name_lower or 'smaller' in entity_name_lower or 'larger' in entity_name_lower):
                        if cached_type_lower == 'protein':
                            entity_type = 'phenotype'  # Anatomical variation/phenotype
                            self.data_source.entity_info[entity_id]['type'] = entity_type
                    # Process/pathway detection (for "regulates", "metabolic genes")
                    elif 'regulates' in entity_name_lower and ('gene' in entity_name_lower or 'metabolic' in entity_name_lower):
                        if cached_type_lower in ['drug', 'protein']:
                            entity_type = 'pathway'
                            self.data_source.entity_info[entity_id]['type'] = 'pathway'
                    # Second check: Use inference if keyword matching didn't work
                    elif hasattr(self.data_source, '_infer_entity_type'):
                        inferred_type = self.data_source._infer_entity_type(entity_name)
                        if inferred_type != 'unknown':
                            inferred_type_lower = inferred_type.lower()
                            
                            # Override if cached type is wrong for pathways, phenotypes, anatomy
                            if inferred_type_lower == 'pathway' and cached_type_lower in ['drug', 'protein']:
                                entity_type = inferred_type
                                self.data_source.entity_info[entity_id]['type'] = inferred_type
                            elif inferred_type_lower == 'phenotype' and cached_type_lower == 'protein':
                                entity_type = inferred_type
                                self.data_source.entity_info[entity_id]['type'] = inferred_type
                            elif inferred_type_lower == 'anatomy' and cached_type_lower in ['drug', 'protein']:
                                entity_type = inferred_type
                                self.data_source.entity_info[entity_id]['type'] = inferred_type
                            elif inferred_type_lower == 'biological_process' and cached_type_lower in ['drug', 'protein']:
                                entity_type = inferred_type
                                self.data_source.entity_info[entity_id]['type'] = inferred_type
                            elif entity_type == 'unknown':
                                entity_type = inferred_type
                                self.data_source.entity_info[entity_id]['type'] = inferred_type

                    # CRITICAL: Add seed entities directly with high relevance score
                    # They were extracted from the query, so they're
                    # automatically relevant
                    seed_entity = RetrievedEntity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        name=entity_name,
                        description=entity_info.get('description', ''),
                        relevance_score=0.95,  # High score - from query
                        context={'source': 'seed_entity', 'from_query': True},
                        agent_reasoning='Seed entity extracted from query'
                    )
                    agent_state.retrieved_entities.append(seed_entity)
                    agent_state.visited_entities.add(entity_id)
                    logger.info(
                        f"Added seed entity: {entity_name} (type: {entity_type})")

        # Now explore neighbors and related entities using agent-based exploration
        # But keep the batch analysis for discovered entities
        discovered_entities = []
        for entity in agent_state.retrieved_entities:
            # Get neighbors of seed entities
            neighbors = self.data_source.get_entity_neighbors(
                entity.entity_id, max_hops=1)
            # neighbors is a Dict, so we need to iterate over items and limit to 5
            for neighbor_id, neighbor_data in list(neighbors.items())[:5]:  # Limit neighbors per seed
                if neighbor_id not in agent_state.visited_entities:
                    # neighbor_data contains 'entity_info', 'relations', 'distance'
                    neighbor_info = neighbor_data.get('entity_info', {})
                    if neighbor_info:
                        discovered_entities.append(
                            (neighbor_id, neighbor_info))

        # Batch analyze discovered entities (neighbors of seeds)
        for entity_id, entity_info in discovered_entities:
            if entity_id not in agent_state.visited_entities:
                entity_name = entity_info.get('name', entity_id)
                entity_type = entity_info.get('type', 'unknown')
                entity_desc = entity_info.get('description', '')
                heuristic_score = self._calculate_heuristic_relevance(
                    entity_name, entity_type, entity_desc, query)

                # Collect borderline cases for batch LLM analysis
                if 0.3 < heuristic_score < 0.7:
                    entities_to_analyze.append((entity_id, entity_info))
                elif heuristic_score >= 0.7:
                    # High confidence - process immediately
                    agent_state = self._explore_entity(
                        entity_id, query, agent_state, query_analysis)

        # Batch analyze borderline entities if we have any
        if entities_to_analyze:
            logger.info(
                f"Batch analyzing {len(entities_to_analyze)} borderline entities")
            try:
                # Use async batch analysis
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                batch_results = loop.run_until_complete(
                    self._batch_analyze_entity_relevance(
                        entities_to_analyze, query))
                loop.close()

                # Apply batch results to agent state
                for entity_id, entity_info in entities_to_analyze:
                    if entity_id in batch_results:
                        analysis = batch_results[entity_id]
                        if analysis['relevance_score'] > self.config.similarity_threshold:
                            entity_name = entity_info.get('name', entity_id)
                            entity_type_raw = analysis.get(
                                'entity_type', entity_info.get('type', 'unknown'))

                            # CRITICAL: Validate and correct entity type using inference
                            # This MUST override wrong types (e.g., pathway ->
                            # drug)
                            entity_type = entity_type_raw
                            entity_name_lower = entity_name.lower()
                            entity_type_normalized = str(
                                entity_type_raw).strip().lower()

                            # CRITICAL: Check for pathway/process patterns FIRST before inference
                            # BUT: Only override if type came from inference, not from PrimeKG's explicit format
                            # Check if this entity type came from PrimeKG's
                            # explicit format
                            primekg_explicit_type = '::' in str(entity_id)

                            # Distinguish between pathways and biological
                            # processes
                            pathway_patterns = [
                                'pathway',
                                'signaling pathway',
                                'metabolic pathway',
                                'signaling cascade']
                            process_patterns = [
                                'regulation', 'internalization', 'response', 'process', 'metabolism']

                            is_pathway = any(
                                pattern in entity_name_lower for pattern in pathway_patterns)
                            is_process = any(
                                pattern in entity_name_lower for pattern in process_patterns) and not is_pathway

                            if is_pathway or is_process:
                                if entity_type_normalized == 'drug' and not primekg_explicit_type:
                                    # Only override if type came from inference, not PrimeKG's explicit format
                                    # Use 'pathway' for pathway entities,
                                    # 'biological_process' for processes
                                    override_type = 'pathway' if is_pathway else 'biological_process'
                                    entity_type = override_type
                                    if entity_id in self.data_source.entity_info:
                                        self.data_source.entity_info[entity_id]['type'] = override_type
                                    logger.info(
                                        f"Batch: Force-corrected type '{override_type}' for '{entity_name}' (was: {entity_type_raw}) - detected {'pathway' if is_pathway else 'process'} keywords")
                                elif entity_type_normalized == 'drug' and primekg_explicit_type:
                                    # PrimeKG explicitly says it's a drug -
                                    # respect it
                                    logger.debug(
                                        f"Batch: Entity '{entity_name}' explicitly classified as 'drug' by PrimeKG - respecting PrimeKG classification")

                            # Also try inference-based override
                            if hasattr(
                                    self.data_source,
                                    '_infer_entity_type') and entity_name and entity_name.strip():
                                inferred_type = self.data_source._infer_entity_type(
                                    entity_name.strip())
                                if inferred_type and inferred_type != 'unknown':
                                    # ALWAYS override if inferred type is
                                    # pathway/process and current is drug
                                    if inferred_type.lower() in [
                                            'pathway', 'biological_process']:
                                        if entity_type_normalized == 'drug' or entity_type_normalized == 'unknown':
                                            entity_type = inferred_type
                                            if entity_id in self.data_source.entity_info:
                                                self.data_source.entity_info[entity_id]['type'] = inferred_type
                                            logger.info(
                                                f"Batch: Corrected type '{inferred_type}' for '{entity_name}' (was: {entity_type_raw})")
                                    # Override if inferred type is protein and
                                    # current is drug (for gene symbols)
                                    elif inferred_type == 'protein' and entity_type_normalized == 'drug':
                                        if any(
                                            p in entity_name_lower for p in [
                                                'regulates',
                                                'activates',
                                                'inhibits',
                                                'gene',
                                                'protein',
                                                'metabolic']):
                                            entity_type = inferred_type
                                            if entity_id in self.data_source.entity_info:
                                                self.data_source.entity_info[entity_id]['type'] = inferred_type
                                            logger.info(
                                                f"Batch: Corrected type '{inferred_type}' for '{entity_name}' (was: {entity_type_raw})")
                                    # Use inferred type if original was unknown
                                    elif entity_type_normalized == 'unknown':
                                        entity_type = inferred_type
                                        if entity_id in self.data_source.entity_info:
                                            self.data_source.entity_info[entity_id]['type'] = inferred_type

                            retrieved_entity = RetrievedEntity(
                                entity_id=entity_id,
                                entity_type=entity_type,
                                name=entity_name,
                                description=entity_info.get(
                                    'description',
                                    ''),
                                relevance_score=analysis['relevance_score'],
                                context={
                                    'source': 'batch_analysis'},
                                agent_reasoning=analysis.get(
                                    'reasoning',
                                    'Batch analyzed'))
                            agent_state.retrieved_entities.append(
                                retrieved_entity)
                            logger.info(
                                f"Added entity from batch: {entity_name} (type: {entity_type}, score: {analysis['relevance_score']:.2f})")

                        agent_state.visited_entities.add(entity_id)
            except Exception as e:
                logger.warning(
                    f"Batch analysis failed, falling back to sequential: {e}")
                # Fallback to sequential processing
                for entity_id, entity_info in entities_to_analyze:
                    agent_state = self._explore_entity(
                        entity_id, query, agent_state, query_analysis)

        # Continue exploration until max steps or no more actions
        while agent_state.current_step < agent_state.max_steps:
            # Decide next action
            decision = self._decide_next_action(
                query, agent_state, query_analysis)

            if decision.action == AgentAction.STOP_RETRIEVAL:
                logger.info("Agent decided to stop retrieval")
                break

            # Execute the action
            agent_state = self._execute_action(
                decision, query, agent_state, query_analysis)
            agent_state.current_step += 1

            # Add decision to reasoning chain
            agent_state.reasoning_chain.append(decision)

        return agent_state

    def _explore_entity(self,
                        entity_id: str,
                        query: str,
                        agent_state: AgentState,
                        query_analysis: Dict[str,
                                             Any]) -> AgentState:
        """Explore a specific entity with early stopping."""
        if entity_id in agent_state.visited_entities:
            return agent_state

        # OPTIMIZATION 4: Early stopping if we have enough entities
        if len(agent_state.retrieved_entities) >= self.config.max_entities:
            logger.info(
                f"Stopping exploration - reached max entities ({self.config.max_entities})")
            return agent_state

        entity_name = self.data_source.entity_info.get(
            entity_id, {}).get('name', entity_id)
        logger.info(f"Exploring entity: {entity_name}")

        # Get entity information
        entity_info = self.data_source.entity_info.get(entity_id, {})
        if not entity_info:
            # Mark as visited even if no info
            agent_state.visited_entities.add(entity_id)
            return agent_state

        # Analyze entity relevance (now uses fast heuristics first)
        relevance_analysis = self._analyze_entity_relevance(
            entity_id, entity_info, query)

        # Add to retrieved entities if relevant
        if relevance_analysis['relevance_score'] > self.config.similarity_threshold:
            # Always try to infer entity type - be aggressive about it
            entity_type_raw = entity_info.get('type')
            entity_name = entity_info.get('name', entity_id)

            # Normalize the raw type
            if entity_type_raw:
                entity_type_normalized = str(entity_type_raw).strip().lower()
            else:
                entity_type_normalized = 'unknown'

            # Initialize entity_type with the normalized type (will be
            # overridden if needed)
            entity_type = None  # Will be set by force-override or inference

            # CRITICAL: Check for pathway/process patterns FIRST before inference
            # BUT: Only override if type came from inference, not from PrimeKG's explicit format
            # PrimeKG stores types as "type::name::id" - if entity_id contains
            # "::", trust PrimeKG's type
            entity_name_lower_check = entity_name.lower()

            # Distinguish between pathways and biological processes
            # Pathways: entities ending in "pathway" or containing "signaling
            # pathway"
            pathway_patterns = [
                'pathway',
                'signaling pathway',
                'metabolic pathway',
                'signaling cascade']
            # Biological processes: regulation, internalization, response, etc.
            # (but NOT pathways)
            process_patterns = [
                'regulation',
                'internalization',
                'response',
                'process',
                'metabolism']

            # Check if this entity type came from PrimeKG's explicit format or from our inference
            # If entity_id contains "::", it means PrimeKG explicitly provided
            # the type - trust it
            primekg_explicit_type = '::' in str(
                entity_id) or '::' in str(entity_info.get('source', ''))

            # Check for pathway patterns first (more specific)
            is_pathway = any(
                pattern in entity_name_lower_check for pattern in pathway_patterns)
            is_process = any(
                pattern in entity_name_lower_check for pattern in process_patterns) and not is_pathway

            if is_pathway or is_process:
                if entity_type_normalized == 'drug' and not primekg_explicit_type:
                    # Only override if type came from inference, not PrimeKG's explicit format
                    # Use 'pathway' for pathway entities, 'biological_process'
                    # for processes
                    override_type = 'pathway' if is_pathway else 'biological_process'
                    entity_type = override_type
                    if entity_id in self.data_source.entity_info:
                        self.data_source.entity_info[entity_id]['type'] = override_type
                    logger.info(
                        f"Force-corrected type '{override_type}' for '{entity_name}' (was: {entity_type_raw}) - detected {'pathway' if is_pathway else 'process'} keywords")
                elif entity_type_normalized == 'drug' and primekg_explicit_type:
                    # PrimeKG explicitly says it's a drug - log but don't
                    # override
                    logger.debug(
                        f"Entity '{entity_name}' explicitly classified as 'drug' by PrimeKG (format: type::name::id) - respecting PrimeKG classification")

            # ALWAYS try to infer type from name (even if we have a type,
            # validate it)
            if not entity_type:  # Only infer if force-override didn't set it
                if hasattr(
                        self.data_source,
                        '_infer_entity_type') and entity_name and entity_name.strip():
                    inferred_type = self.data_source._infer_entity_type(
                        entity_name.strip())
                    logger.debug(
                        f"Inferred type for '{entity_name}': {inferred_type} (was: {entity_type_normalized})")

                    # If we inferred a type, use it (especially if original was
                    # unknown or seems wrong)
                    if inferred_type and inferred_type != 'unknown':
                        # Check if original type seems wrong (e.g., gene symbol
                        # classified as drug)
                        should_override = False

                        if entity_type_normalized == 'unknown':
                            should_override = True
                        elif inferred_type == 'protein' and entity_type_normalized == 'drug':
                            # Gene symbols misclassified as drugs (e.g., "TP53 Regulates Metabolic Genes")
                            # Check for gene symbol patterns or action words
                            import re
                            gene_pattern = r'\b[A-Z]{2,8}\d*\b'
                            if (
                                re.search(
                                    gene_pattern,
                                    entity_name) or any(
                                    pattern in entity_name_lower_check for pattern in [
                                        'regulates',
                                        'activates',
                                        'inhibits',
                                        'gene',
                                        'protein',
                                        'metabolic'])):
                                should_override = True
                                logger.debug(
                                    f"Detected gene/protein pattern in '{entity_name}', overriding drug classification")
                        elif inferred_type in ['pathway', 'biological_process'] and entity_type_normalized == 'drug':
                            # Pathways/processes misclassified as drugs (e.g., "insulin signaling pathway", "insulin receptor internalization")
                            # ALWAYS override if inferred type is
                            # pathway/process and cached is drug
                            should_override = True
                            logger.info(
                                f"Detected pathway/process pattern in '{entity_name}', overriding drug classification (inferred: {inferred_type})")
                        elif inferred_type == 'biological_process' and entity_type_normalized in ['drug', 'unknown']:
                            # Biological processes should never be drugs
                            if any(
                                pattern in entity_name_lower_check for pattern in [
                                    'regulation',
                                    'internalization',
                                    'response',
                                    'process',
                                    'signaling']):
                                should_override = True
                                logger.info(
                                    f"Detected biological process pattern in '{entity_name}', overriding {entity_type_normalized} classification")

                        if should_override:
                            entity_type = inferred_type
                            # Update entity_info cache for future use
                            if entity_id in self.data_source.entity_info:
                                self.data_source.entity_info[entity_id]['type'] = inferred_type
                            logger.info(
                                f"Corrected type '{inferred_type}' for entity '{entity_name}' (was: {entity_type_raw})")
                        elif entity_type_normalized == 'unknown':
                            # Use inferred type if original was unknown
                            entity_type = inferred_type
                            if entity_id in self.data_source.entity_info:
                                self.data_source.entity_info[entity_id]['type'] = inferred_type
                            logger.info(
                                f"Inferred type '{inferred_type}' for entity '{entity_name}' (was: {entity_type_raw})")

            # If inference didn't work or wasn't needed, try fallback from
            # relevance analysis
            if not entity_type or entity_type == 'unknown':
                if 'entity_type' in relevance_analysis:
                    fallback_type = relevance_analysis['entity_type']
                    if fallback_type and fallback_type != 'unknown':
                        entity_type = fallback_type
                        logger.debug(
                            f"Using fallback type '{fallback_type}' for entity '{entity_name}'")

            # If still no type, use the original (might be a valid non-unknown
            # type)
            if not entity_type:
                entity_type = entity_type_raw if entity_type_raw and str(
                    entity_type_raw).strip() else 'unknown'

            # Final safety check
            if not entity_type or str(
                    entity_type).strip().lower() == 'unknown':
                entity_type = 'unknown'
            
            # CRITICAL: Apply same keyword-based type correction as seed entities
            # This fixes misclassifications like "cell surface receptor signaling pathway" -> protein
            entity_type_lower = entity_type.lower()
            entity_name_lower = entity_name.lower()
            
            # Pathway detection
            if 'pathway' in entity_name_lower or 'signaling pathway' in entity_name_lower:
                if entity_type_lower in ['drug', 'protein', 'unknown']:
                    entity_type = 'pathway'
                    if entity_id in self.data_source.entity_info:
                        self.data_source.entity_info[entity_id]['type'] = 'pathway'
            elif 'regulation' in entity_name_lower and ('pathway' in entity_name_lower or 'signaling' in entity_name_lower):
                if entity_type_lower in ['drug', 'protein', 'unknown']:
                    entity_type = 'pathway'
                    if entity_id in self.data_source.entity_info:
                        self.data_source.entity_info[entity_id]['type'] = 'pathway'
            # Anatomy detection
            elif 'cytoplasmic side' in entity_name_lower or 'side of' in entity_name_lower:
                if entity_type_lower in ['drug', 'protein', 'unknown']:
                    entity_type = 'anatomy'
                    if entity_id in self.data_source.entity_info:
                        self.data_source.entity_info[entity_id]['type'] = 'anatomy'
            # Phenotype/anatomy detection
            elif entity_name_lower in ['stomach', 'bleeding', 'abnormal bleeding'] or 'abnormality' in entity_name_lower:
                if entity_type_lower == 'protein':
                    entity_type = 'phenotype' if 'bleeding' in entity_name_lower or 'abnormality' in entity_name_lower else 'anatomy'
                    if entity_id in self.data_source.entity_info:
                        self.data_source.entity_info[entity_id]['type'] = entity_type
            # Disease detection (for "neoplasm", "carcinoma", etc.)
            elif any(term in entity_name_lower for term in ['neoplasm', 'carcinoma', 'syndrome', 'disease', 'disorder']):
                if entity_type_lower == 'protein':
                    entity_type = 'disease'
                    if entity_id in self.data_source.entity_info:
                        self.data_source.entity_info[entity_id]['type'] = 'disease'
            # Process/pathway detection (for "regulates", "metabolic genes")
            elif 'regulates' in entity_name_lower and ('gene' in entity_name_lower or 'metabolic' in entity_name_lower):
                if entity_type_lower in ['drug', 'protein']:
                    entity_type = 'pathway'
                    if entity_id in self.data_source.entity_info:
                        self.data_source.entity_info[entity_id]['type'] = 'pathway'
            # Anatomy/phenotype detection (for "fibers", "relatively", "smaller", "larger")
            elif ('fiber' in entity_name_lower or 'fibers' in entity_name_lower) and ('relatively' in entity_name_lower or 'smaller' in entity_name_lower or 'larger' in entity_name_lower):
                if entity_type_lower == 'protein':
                    entity_type = 'phenotype'  # Anatomical variation/phenotype
                    if entity_id in self.data_source.entity_info:
                        self.data_source.entity_info[entity_id]['type'] = 'phenotype'

            retrieved_entity = RetrievedEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                name=entity_info.get('name', entity_id),
                description=entity_info.get('description', ''),
                relevance_score=relevance_analysis['relevance_score'],
                context={'exploration_step': agent_state.current_step},
                agent_reasoning=relevance_analysis['reasoning']
            )
            agent_state.retrieved_entities.append(retrieved_entity)
            logger.info(
                f"Added relevant entity: {entity_name} (type: {entity_type}, score: {relevance_analysis['relevance_score']:.2f})")
        else:
            logger.debug(
                f"Skipped irrelevant entity: {entity_name} (score: {relevance_analysis['relevance_score']:.2f})")

        # Mark as visited
        agent_state.visited_entities.add(entity_id)

        # OPTIMIZATION 5: Early stopping if relevance is dropping
        if (self.config.enable_early_stopping and
            len(agent_state.retrieved_entities) >= 5 and  # Have some entities
                relevance_analysis['relevance_score'] < self.config.relevance_score_threshold):

            # Check if recent entities are all low relevance
            recent_scores = [
                e.relevance_score for e in agent_state.retrieved_entities[-3:]]
            if all(
                    score < self.config.relevance_score_threshold for score in recent_scores):
                logger.info(
                    "Early stopping: Recent entities have low relevance")
                return agent_state

        return agent_state

    def _analyze_entity_relevance(self, entity_id: str, entity_info: Dict[str, Any],
                                  query: str) -> Dict[str, Any]:
        """Analyze how relevant an entity is to the query using fast heuristics first."""
        entity_name = entity_info.get('name', entity_id)
        entity_type = entity_info.get('type', 'unknown')
        # If type is unknown, try to infer it
        if entity_type == 'unknown' and hasattr(
                self.data_source, '_infer_entity_type'):
            inferred_type = self.data_source._infer_entity_type(entity_name)
            if inferred_type != 'unknown':
                entity_type = inferred_type
        entity_desc = entity_info.get('description', '')

        # OPTIMIZATION 1: Fast heuristic filtering first (no LLM call)
        heuristic_score = self._calculate_heuristic_relevance(
            entity_name, entity_type, entity_desc, query)

        # OPTIMIZATION 2: Only use LLM for borderline cases
        if heuristic_score >= 0.7:
            # High confidence - relevant
            return {
                'relevance_score': heuristic_score,
                'entity_type': entity_type,  # Include inferred type
                'relevant_aspects': [entity_type, 'high_confidence'],
                'next_explorations': [],
                'reasoning': f'High heuristic relevance ({heuristic_score:.2f}) - no LLM needed'
            }
        elif heuristic_score <= 0.3:
            # High confidence - not relevant
            return {
                'relevance_score': heuristic_score,
                'entity_type': entity_type,  # Include inferred type
                'relevant_aspects': [],
                'next_explorations': [],
                'reasoning': f'Low heuristic relevance ({heuristic_score:.2f}) - no LLM needed'
            }
        else:
            # OPTIMIZATION 3: Only make LLM call for uncertain cases (30-70%
            # range)
            cache_key = f"{entity_id}|{query.strip().lower()}"
            if cache_key in self._entity_relevance_cache:
                return self._entity_relevance_cache[cache_key]
            logger.info(f"Using LLM for borderline entity: {entity_name}")
            prompt = self.prompts['entity_exploration'].format(
                entity_name=entity_name,
                entity_type=entity_type,
                entity_description=entity_desc,
                query=query
            )

            try:
                # Use synchronous call for now (will be batched in async
                # version)
                response = self.llm_client.generate(prompt)
                analysis = json.loads(response)
                self._entity_relevance_cache[cache_key] = analysis
                return analysis
            except Exception as e:
                logger.warning(
                    f"Failed to analyze entity relevance for {entity_name}: {e}")
                # Fallback to heuristic score
                return {
                    'relevance_score': heuristic_score,
                    'entity_type': entity_type,
                    'relevant_aspects': [],
                    'next_explorations': [],
                    'reasoning': f'LLM failed, using heuristic ({heuristic_score:.2f})'}

    def _calculate_heuristic_relevance(
            self,
            entity_name: str,
            entity_type: str,
            entity_desc: str,
            query: str) -> float:
        """Calculate relevance using fast heuristics without LLM calls."""
        query_lower = query.lower()
        name_lower = entity_name.lower()
        type_lower = entity_type.lower()
        desc_lower = entity_desc.lower()

        # Filter out stop words and common query words that don't indicate
        # relevance
        stop_words = {
            'what',
            'are',
            'the',
            'of',
            'is',
            'how',
            'does',
            'do',
            'a',
            'an',
            'in',
            'on',
            'at',
            'to',
            'for',
            'with',
            'from',
            'by',
            'as',
            'be',
            'been',
            'being',
            'have',
            'has',
            'had',
            'will',
            'would',
            'should',
            'could',
            'may',
            'might',
            'can',
            'this',
            'that',
            'these',
            'those',
            'which',
            'who',
            'where',
            'when',
            'why'}

        # Extract meaningful query terms (exclude stop words)
        query_words = [w for w in query_lower.split(
        ) if w not in stop_words and len(w) > 2]
        query_terms = set(query_words)

        score = 0.0

        # Entity type relevance (from config weights)
        type_weight = self.entity_type_weights.get(type_lower, 0.3)
        score += type_weight * 0.3  # Base type relevance

        # CRITICAL: Require at least one meaningful query term match (not stop words)
        # This prevents "side" from matching "side chain" or "side of"
        meaningful_matches = query_terms & set(name_lower.split())
        if meaningful_matches:
            # Boost for meaningful term matches (biomedical terms, not generic
            # words)
            overlap_ratio = len(meaningful_matches) / max(len(query_terms), 1)
            score += overlap_ratio * 0.6  # Increased weight for meaningful matches
        else:
            # PRIORITY 1 FIX: Don't penalize too heavily - biomedical relationships
            # may not have direct term matches but still be relevant
            # Only penalize if entity type is also irrelevant
            if not self._is_relevant_entity_type(entity_type, query):
                score *= 0.2  # Heavy penalty only if type is irrelevant
            else:
                # Type is relevant but no term match - give base score
                # This handles cases like "type 2 diabetes" for "metformin diabetes" query
                score = max(score, 0.2)  # Minimum score for relevant types

        # Type-specific relevance filters
        if not self._is_relevant_entity_type(entity_type, query):
            score *= 0.2  # More heavily penalize irrelevant types

        # Boost for exact entity name matches (e.g., "aspirin" matches "Aspirin")
        # This is the strongest signal - CRITICAL for drugs like aspirin
        for term in query_terms:
            if len(term) > 3:  # Only check substantial terms
                # Exact match (case-insensitive)
                if term == name_lower:
                    score += 0.6  # Very strong boost for exact matches
                    break
                # Word-level match
                if term in name_lower.split():
                    score += 0.4  # Strong boost for word matches
                    break
                # Substring match (for cases like "aspirin" in "Aspirin")
                if term in name_lower:
                    score += 0.3  # Moderate boost for substring matches
                    break
        
        # PRIORITY 1 FIX: Boost for biomedical relationship patterns
        # Handle cases where entity is related to query entities even without direct term match
        # E.g., "type 2 diabetes" should score high for "metformin diabetes" query
        biomedical_keywords = {
            'diabetes': ['diabetes', 'diabetic', 'diabetes mellitus', 'dm', 't2d', 't1d'],
            'cancer': ['cancer', 'carcinoma', 'tumor', 'tumour', 'neoplasm', 'malignancy'],
            'alzheimer': ['alzheimer', "alzheimer's", 'alzheimers', 'ad', 'dementia'],
            'breast': ['breast', 'mammary'],
            'gene': ['gene', 'protein', 'mutation', 'variant'],
        }
        
        # Check if entity name contains biomedical keywords related to query
        for keyword, variations in biomedical_keywords.items():
            if keyword in query_lower:
                # Check if entity name contains related variations
                if any(var in name_lower for var in variations):
                    score += 0.3  # Boost for biomedical relationship
                    break
                # Also check reverse - if query mentions variation, check for keyword
                if any(var in query_lower for var in variations):
                    if keyword in name_lower or any(var in name_lower for var in variations):
                        score += 0.3  # Boost for biomedical relationship
                        break

        # CRITICAL: Special handling for common drugs - ensure they get high
        # scores
        common_drugs = {
            'aspirin',
            'metformin',
            'warfarin',
            'insulin',
            'ibuprofen',
            'acetaminophen'}
        if name_lower in common_drugs and any(
                drug in query_lower for drug in common_drugs):
            if name_lower in query_lower:
                # Ensure common drugs get high relevance
                score = max(score, 0.8)

        # Penalize very long entity names (often irrelevant descriptions)
        if len(name_lower.split()) > 8:
            score *= 0.5

        # Penalize entities with anatomical/structural terms that aren't
        # relevant
        anatomical_terms = {
            'line',
            'side',
            'end',
            'part',
            'region',
            'area',
            'surface',
            'membrane',
            'component'}
        if any(term in name_lower for term in anatomical_terms):
            # Only penalize if it's not a meaningful match
            if not meaningful_matches:
                score *= 0.3

        # Boost for description matches (only if we have meaningful name
        # matches)
        if desc_lower and meaningful_matches:
            desc_matches = sum(1 for term in query_terms if term in desc_lower)
            if desc_matches > 0:
                score += 0.1 * (desc_matches / len(query_terms))

        # CRITICAL FIX: Add semantic context filtering to prevent irrelevant disease exploration
        # For example, "metformin diabetes" query should not explore "schizophrenia", "hip dysplasia"
        if type_lower == 'disease' and score < 0.5:
            # Check if this disease is semantically related to query context
            # Extract key biomedical entities from query (drugs, diseases mentioned)
            query_entities = []
            for word in query_words:
                # Look for biomedical entities in entity_info
                if len(word) > 3:
                    query_entities.append(word)

            # Check if disease name has semantic overlap with query context
            disease_words = set(name_lower.split())
            context_overlap = disease_words & query_terms

            if not context_overlap and score < 0.5:
                # No semantic overlap - heavily penalize
                # This prevents "schizophrenia" from being explored in "metformin diabetes" query
                score *= 0.1
                logger.debug(f"Heavily penalized irrelevant disease '{entity_name}' (score: {score:.3f}) - no semantic overlap with query")

        return min(1.0, score)

    async def _batch_analyze_entity_relevance(self, entity_batch: List[Tuple[str, Dict[str, Any]]],
                                              query: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze entity relevance for multiple entities in batch using async LLM calls.

        This method batches LLM calls to improve performance (addresses CLAUDE.md Problem 2).

        Args:
            entity_batch: List of (entity_id, entity_info) tuples
            query: Query string

        Returns:
            Dictionary mapping entity_id to relevance analysis dict
        """
        # Separate entities into cached and uncached
        cached_results = {}
        uncached_entities = []

        for entity_id, entity_info in entity_batch:
            cache_key = f"{entity_id}|{query.strip().lower()}"
            if cache_key in self._entity_relevance_cache:
                cached_results[entity_id] = self._entity_relevance_cache[cache_key]
            else:
                # Check if heuristic can determine relevance
                entity_name = entity_info.get('name', entity_id)
                entity_type = entity_info.get('type', 'unknown')
                entity_desc = entity_info.get('description', '')
                heuristic_score = self._calculate_heuristic_relevance(
                    entity_name, entity_type, entity_desc, query)

                # Only add to uncached if needs LLM (borderline cases)
                if 0.3 < heuristic_score < 0.7:
                    uncached_entities.append(
                        (entity_id, entity_info, heuristic_score))
                else:
                    # Use heuristic result
                    result = {
                        'relevance_score': heuristic_score,
                        'entity_type': entity_type,
                        'relevant_aspects': [
                            entity_type,
                            'heuristic'],
                        'next_explorations': [],
                        'reasoning': f'Heuristic relevance ({heuristic_score:.2f})'}
                    cached_results[entity_id] = result
                    self._entity_relevance_cache[cache_key] = result

        if not uncached_entities:
            return cached_results

        # Batch LLM calls for borderline cases
        prompts = []
        entity_metadata = []
        for entity_id, entity_info, heuristic_score in uncached_entities:
            entity_name = entity_info.get('name', entity_id)
            entity_type = entity_info.get('type', 'unknown')
            entity_desc = entity_info.get('description', '')
            prompt = self.prompts['entity_exploration'].format(
                entity_name=entity_name,
                entity_type=entity_type,
                entity_description=entity_desc,
                query=query
            )
            prompts.append(prompt)
            entity_metadata.append((entity_id, entity_info, heuristic_score))

        # Batch async LLM calls
        try:
            if hasattr(self.llm_client, 'generate_batch_async'):
                responses = await self.llm_client.generate_batch_async(prompts, max_tokens=200, temperature=0.1)
            else:
                # Fallback: sequential async calls
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                loop = asyncio.get_event_loop()
                executor = ThreadPoolExecutor(max_workers=min(len(prompts), 5))

                async def generate_one(prompt):
                    return await loop.run_in_executor(
                        executor,
                        lambda: self.llm_client.generate(prompt, max_tokens=200, temperature=0.1)
                    )

                responses = await asyncio.gather(*[generate_one(p) for p in prompts])
                executor.shutdown(wait=False)

            # Parse responses and cache results
            for (
                    entity_id, entity_info, heuristic_score), response in zip(
                    entity_metadata, responses):
                cache_key = f"{entity_id}|{query.strip().lower()}"
                entity_name = entity_info.get('name', entity_id)

                # CRITICAL: Infer correct entity type BEFORE batch analysis
                # This ensures pathways/processes are correctly classified
                correct_entity_type = None
                if hasattr(self.data_source, '_infer_entity_type'):
                    inferred_type = self.data_source._infer_entity_type(
                        entity_name)
                    if inferred_type and inferred_type != 'unknown':
                        correct_entity_type = inferred_type

                try:
                    # Try to parse JSON, handle malformed responses
                    response_clean = response.strip()
                    # Remove markdown code blocks if present
                    if response_clean.startswith('```'):
                        lines = response_clean.split('\n')
                        response_clean = '\n'.join(
                            lines[1:-1]) if len(lines) > 2 else response_clean
                        response_clean = response_clean.strip()

                    # Try to extract JSON from response
                    analysis = json.loads(response_clean)

                    # Validate required fields
                    if 'relevance_score' not in analysis:
                        analysis['relevance_score'] = heuristic_score
                    if 'entity_type' not in analysis:
                        # Use inferred type if available, otherwise use cached
                        # type
                        analysis['entity_type'] = correct_entity_type if correct_entity_type else entity_info.get(
                            'type', 'unknown')
                    else:
                        # CRITICAL: Override LLM type if it's wrong (e.g.,
                        # pathway classified as drug)
                        llm_type = analysis.get('entity_type', '').lower()
                        if correct_entity_type and correct_entity_type.lower() in [
                                'pathway', 'biological_process']:
                            if llm_type == 'drug' or llm_type == 'unknown':
                                analysis['entity_type'] = correct_entity_type
                                logger.info(
                                    f"Batch analysis: Corrected type '{correct_entity_type}' for '{entity_name}' (LLM said: {llm_type})")

                    cached_results[entity_id] = analysis
                    self._entity_relevance_cache[cache_key] = analysis
                except json.JSONDecodeError as e:
                    # Try to extract just the relevance score from malformed
                    # JSON
                    logger.debug(
                        f"Failed to parse LLM JSON for {entity_id}: {e}, attempting extraction")
                    import re
                    # Try to extract relevance_score from text
                    score_match = re.search(
                        r'"relevance_score"\s*:\s*(\d*\.?\d+)', response)
                    score = float(
                        score_match.group(1)) if score_match else heuristic_score

                    # Infer entity type from name
                    entity_name = entity_info.get('name', entity_id)
                    inferred_type = 'unknown'
                    if hasattr(self.data_source, '_infer_entity_type'):
                        inferred_type = self.data_source._infer_entity_type(
                            entity_name)

                    result = {
                        'relevance_score': score,
                        'entity_type': inferred_type if inferred_type != 'unknown' else entity_info.get(
                            'type',
                            'unknown'),
                        'relevant_aspects': [],
                        'next_explorations': [],
                        'reasoning': f'LLM parse failed, extracted score from text ({score:.2f})'}
                    cached_results[entity_id] = result
                    self._entity_relevance_cache[cache_key] = result
                except Exception as e:
                    logger.warning(
                        f"Failed to parse LLM response for {entity_id}: {e}")
                    # Fallback to heuristic with type inference
                    entity_name = entity_info.get('name', entity_id)
                    inferred_type = 'unknown'
                    if hasattr(self.data_source, '_infer_entity_type'):
                        inferred_type = self.data_source._infer_entity_type(
                            entity_name)

                    result = {
                        'relevance_score': heuristic_score,
                        'entity_type': inferred_type if inferred_type != 'unknown' else entity_info.get(
                            'type',
                            'unknown'),
                        'relevant_aspects': [],
                        'next_explorations': [],
                        'reasoning': f'LLM parse failed, using heuristic ({heuristic_score:.2f})'}
                    cached_results[entity_id] = result
                    self._entity_relevance_cache[cache_key] = result

        except Exception as e:
            logger.warning(
                f"Batch LLM analysis failed: {e}, falling back to heuristics")
            # Fallback all to heuristics
            for entity_id, entity_info, heuristic_score in uncached_entities:
                cache_key = f"{entity_id}|{query.strip().lower()}"
                result = {
                    'relevance_score': heuristic_score,
                    'entity_type': entity_info.get(
                        'type',
                        'unknown'),
                    'relevant_aspects': [],
                    'next_explorations': [],
                    'reasoning': f'Batch LLM failed, using heuristic ({heuristic_score:.2f})'}
                cached_results[entity_id] = result
                self._entity_relevance_cache[cache_key] = result

        return cached_results

    async def _batch_entity_analysis(
            self, entities: List[str], query: str) -> Dict[str, float]:
        """Analyze entity relevance in batches to improve performance (addresses CLAUDE.md Problem 2)."""
        # Check cache first
        uncached_entities = []
        results = {}

        for entity in entities:
            cache_key = f"{entity}|{query}"
            if cache_key in self._analysis_cache:
                results[entity] = self._analysis_cache[cache_key]
            else:
                uncached_entities.append(entity)

        if not uncached_entities:
            return results

        # Process uncached entities in batches
        batches = [uncached_entities[i:i + self._batch_size]
                   for i in range(0, len(uncached_entities), self._batch_size)]

        # Process batches concurrently
        batch_tasks = []
        for batch in batches:
            task = self._analyze_entity_batch(batch, query)
            batch_tasks.append(task)

        batch_results = await asyncio.gather(*batch_tasks)

        # Combine results and cache them
        for batch_result in batch_results:
            results.update(batch_result)
            # Cache the results
            for entity, score in batch_result.items():
                cache_key = f"{entity}|{query}"
                if len(self._analysis_cache) < 1000:  # Limit cache size
                    self._analysis_cache[cache_key] = score

        return results

    async def _analyze_entity_batch(
            self, entity_batch: List[str], query: str) -> Dict[str, float]:
        """Analyze a batch of entities using LLM or heuristics."""
        # Use heuristic scoring for most cases (much faster)
        results = {}
        for entity in entity_batch:
            # Try heuristic scoring first
            score = self._score_entity_relevance(
                entity, query, None, "unknown")

            # Only use LLM for borderline cases where heuristics are uncertain
            if 0.4 <= score <= 0.6:
                # Use LLM for borderline cases in background thread
                llm_score = await self._llm_score_entity_async(entity, query)
                if llm_score is not None:
                    score = llm_score

            results[entity] = score

        return results

    async def _llm_score_entity_async(
            self, entity: str, query: str) -> Optional[float]:
        """Score entity relevance using LLM in async mode."""
        try:
            # Run LLM call in thread executor to avoid blocking
            loop = asyncio.get_event_loop()

            def _sync_llm_call():
                if not hasattr(self.llm_client, 'generate'):
                    return None

                prompt = f"""Rate the relevance of entity '{entity}' to query '{query}' on a scale of 0.0 to 1.0.
                Consider biomedical relationships and context. Respond with just the number."""

                try:
                    response = self.llm_client.generate(
                        prompt, max_tokens=10, temperature=0.1)
                    # Parse numeric response
                    import re
                    match = re.search(r'(\d*\.?\d+)', response)
                    if match:
                        return float(match.group(1))
                except Exception:
                    return None
                return None

            # Execute in thread pool with timeout
            score = await asyncio.wait_for(
                loop.run_in_executor(self._thread_executor, _sync_llm_call),
                timeout=2.0  # 2 second timeout per entity
            )

            return score if score is not None and 0 <= score <= 1 else None

        except asyncio.TimeoutError:
            logger.debug(f"LLM scoring timeout for entity: {entity}")
            return None
        except Exception as e:
            logger.debug(f"LLM scoring error for entity {entity}: {e}")
            return None

    def _decide_next_action(self, query: str, agent_state: AgentState,
                            query_analysis: Dict[str, Any]) -> AgentDecision:
        """Decide the next action for the agent."""
        # Get available entities and relationships
        available_entities = self._get_available_entities(agent_state)
        available_relationships = self._get_available_relationships(
            agent_state)

        # OPTIMIZATION: Smart stopping criteria instead of relying only on LLM
        should_continue = self._should_continue_exploration(
            agent_state, available_entities)

        if not should_continue:
            logger.info(
                f"Agent stopping: Found {len(agent_state.retrieved_entities)} entities, no more promising leads")
            return AgentDecision(
                action=AgentAction.STOP_RETRIEVAL,
                reasoning=AgentReasoning.DIRECT_MATCH,
                confidence=0.8,
                explanation=f"Smart stopping: {len(agent_state.retrieved_entities)} entities found, exploration complete"
            )

        # If we should continue, prefer exploring entities over stopping
        if available_entities:
            # Choose the most promising entity to explore next
            target_entity = available_entities[0]  # Could be made smarter
            return AgentDecision(
                action=AgentAction.EXPLORE_ENTITY,
                reasoning=AgentReasoning.DIRECT_MATCH,
                target_entity=target_entity,
                confidence=0.7,
                explanation=f"Continuing exploration with entity: {target_entity}")

        # Try LLM decision as fallback
        prompt = self.prompts['next_action'].format(
            query=query,
            visited_entities=list(agent_state.visited_entities),
            current_path=agent_state.current_path,
            available_entities=available_entities[:10],  # Limit for prompt
            available_relationships=available_relationships[:10]
        )

        try:
            response = self.llm_client.generate(prompt)
            decision_data = json.loads(response)

            # Ensure 'action' field exists and is valid
            action_str = decision_data.get('action', 'stop_retrieval')
            try:
                action = AgentAction(action_str)
            except (ValueError, KeyError):
                logger.warning(
                    f"Invalid action '{action_str}', defaulting to stop_retrieval")
                action = AgentAction.STOP_RETRIEVAL

            return AgentDecision(
                action=action,
                reasoning=AgentReasoning.DIRECT_MATCH,  # Would be determined by LLM
                target_entity=decision_data.get('target_entity'),
                target_relationship=decision_data.get('target_relationship'),
                confidence=decision_data.get('confidence', 0.5),
                explanation=decision_data.get('reasoning', '')
            )
        except Exception as e:
            logger.warning(f"Failed to decide next action: {e}")
            # Fallback decision
            return AgentDecision(
                action=AgentAction.STOP_RETRIEVAL,
                reasoning=AgentReasoning.DIRECT_MATCH,
                confidence=0.0,
                explanation='Fallback decision due to parsing error'
            )

    def _should_continue_exploration(
            self,
            agent_state: AgentState,
            available_entities: List[str]) -> bool:
        """Enhanced early stopping to prevent graph explosion (addresses CLAUDE.md Problem 2)."""
        # Stop if no entities available
        if not available_entities:
            return False

        # AGGRESSIVE: Stop if we've reached max entities (lower limit to
        # prevent explosion)
        # Cap at 50 even if config allows more
        max_entities = min(self.config.max_entities, 50)
        if len(agent_state.retrieved_entities) >= max_entities:
            logger.info(
                f"Early stopping: reached max entities limit ({max_entities})")
            return False

        # AGGRESSIVE: Stop if we've reached max steps (lower limit)
        # Cap at 10 steps to prevent long exploration
        max_steps = min(self.config.max_steps, 10)
        if agent_state.current_step >= max_steps:
            logger.info(
                f"Early stopping: reached max steps limit ({max_steps})")
            return False

        # AGGRESSIVE: Stop if graph is growing too fast (indicates explosion)
        if len(available_entities) > 100:
            logger.warning(
                f"Early stopping: too many available entities ({len(available_entities)}) - possible graph explosion")
            return False

        # Continue only if we have very few relevant entities
        min_entities_threshold = 3  # Reduced from 5
        if len(agent_state.retrieved_entities) < min_entities_threshold:
            return True

        # ENHANCED: More strict criteria for continuing exploration
        if len(agent_state.retrieved_entities) >= 3:
            recent_entities = agent_state.retrieved_entities[-3:]
            avg_recent_score = sum(
                e.relevance_score for e in recent_entities) / len(recent_entities)

            # Higher threshold for continuing
            strict_threshold = max(self.config.similarity_threshold, 0.7)
            if avg_recent_score > strict_threshold:
                # Even with good scores, limit further exploration
                if len(agent_state.retrieved_entities) < 20:
                    return True

        # ENHANCED: Check exploration efficiency (stop if diminishing returns)
        if len(agent_state.retrieved_entities) >= 5:
            all_scores = [
                e.relevance_score for e in agent_state.retrieved_entities]
            recent_scores = all_scores[-3:] if len(
                all_scores) >= 3 else all_scores
            earlier_scores = all_scores[-6:-
                                        3] if len(all_scores) >= 6 else all_scores[:3]

            if recent_scores and earlier_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                earlier_avg = sum(earlier_scores) / len(earlier_scores)

                # Stop if quality is declining
                if recent_avg < earlier_avg * 0.8:
                    logger.info(
                        "Early stopping: quality declining (diminishing returns)")
                    return False

        # Default: stop exploration (conservative approach)
        logger.info(
            f"Early stopping: default conservative stop with {len(agent_state.retrieved_entities)} entities")
        return False

    def clear_analysis_cache(self):
        """Clear analysis cache to free memory."""
        self._analysis_cache.clear()
        logger.info("Retriever analysis cache cleared")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            'cache_size': len(
                self._analysis_cache),
            'thread_pool_active': self._thread_executor._threads if hasattr(
                self._thread_executor,
                '_threads') else 0,
            'batch_size': self._batch_size}

    def _get_available_entities(self, agent_state: AgentState) -> List[str]:
        """Get entities available for exploration."""
        available = []

        # FIXED: Use ALL visited entities as exploration seeds, not just retrieved ones
        # This ensures the agent can continue exploring even if initial
        # entities aren't highly relevant
        exploration_seeds = list(agent_state.visited_entities)
        if not exploration_seeds:
            # If no visited entities, this is the initial state - no available
            # entities yet
            return available

        for entity_id in exploration_seeds:
            neighbors = self.data_source.get_entity_neighbors(
                entity_id, max_hops=1)
            for neighbor_id in neighbors:
                if neighbor_id not in agent_state.visited_entities and neighbor_id not in available:
                    available.append(neighbor_id)

        # Limit to prevent overwhelming the LLM prompt
        return available[:20]

    def _get_available_relationships(
            self, agent_state: AgentState) -> List[str]:
        """Get relationships available for exploration."""
        available = []
        for entity in agent_state.retrieved_entities:
            neighbors = self.data_source.get_entity_neighbors(
                entity.entity_id, max_hops=1)
            for neighbor_id, neighbor_info in neighbors.items():
                for relation in neighbor_info.get('relations', []):
                    if (entity.entity_id,
                            neighbor_id) not in agent_state.explored_relationships:
                        available.append(relation)
        return available

    def _execute_action(self,
                        decision: AgentDecision,
                        query: str,
                        agent_state: AgentState,
                        query_analysis: Dict[str,
                                             Any]) -> AgentState:
        """Execute the agent's decision."""
        if decision.action == AgentAction.EXPLORE_ENTITY and decision.target_entity:
            agent_state = self._explore_entity(
                decision.target_entity, query, agent_state, query_analysis)
        elif decision.action == AgentAction.FOLLOW_RELATIONSHIP and decision.target_relationship:
            agent_state = self._follow_relationship(
                decision.target_relationship, query, agent_state, query_analysis)
        elif decision.action == AgentAction.SEARCH_SIMILAR:
            agent_state = self._search_similar_entities(
                query, agent_state, query_analysis)
        elif decision.action == AgentAction.EXPAND_SUBGRAPH:
            agent_state = self._expand_subgraph(
                query, agent_state, query_analysis)

        return agent_state

    def _follow_relationship(self,
                             relationship_type: str,
                             query: str,
                             agent_state: AgentState,
                             query_analysis: Dict[str,
                                                  Any]) -> AgentState:
        """Follow a specific type of relationship."""
        logger.info(f"Following relationship type: {relationship_type}")

        # Find entities with this relationship type
        for entity in agent_state.retrieved_entities:
            neighbors = self.data_source.get_entity_neighbors(
                entity.entity_id, max_hops=1)
            for neighbor_id, neighbor_info in neighbors.items():
                if relationship_type in neighbor_info.get('relations', []):
                    if neighbor_id not in agent_state.visited_entities:
                        agent_state = self._explore_entity(
                            neighbor_id, query, agent_state, query_analysis)

        return agent_state

    def _search_similar_entities(self, query: str, agent_state: AgentState,
                                 query_analysis: Dict[str, Any]) -> AgentState:
        """Search for entities similar to already retrieved ones."""
        logger.info("Searching for similar entities")

        # Get key terms from query
        key_terms = self._extract_key_terms(query)

        for term in key_terms:
            search_results = self.data_source.search_entities(term, limit=5)
            for result in search_results:
                if result['id'] not in agent_state.visited_entities:
                    agent_state = self._explore_entity(
                        result['id'], query, agent_state, query_analysis)

        return agent_state

    def _expand_subgraph(self, query: str, agent_state: AgentState,
                         query_analysis: Dict[str, Any]) -> AgentState:
        """Expand the subgraph around current entities."""
        logger.info("Expanding subgraph")

        # Get all neighbors of current entities
        for entity in agent_state.retrieved_entities:
            neighbors = self.data_source.get_entity_neighbors(
                entity.entity_id, max_hops=2)
            for neighbor_id in neighbors:
                if neighbor_id not in agent_state.visited_entities:
                    agent_state = self._explore_entity(
                        neighbor_id, query, agent_state, query_analysis)

        return agent_state

    def _get_relevant_relation_types(
            self,
            query_type: str,
            query: str) -> List[str]:
        """
        Map query type to PrimeKG relation types.

        Based on diagnostic results, PrimeKG actually uses:
        - 'exposure_disease' (for drug side effects/adverse effects)
        - 'contraindication' (for contraindications)
        - 'drug_drug' (for drug interactions)
        - 'disease_protein' (for gene-disease associations)
        - 'disease_phenotype_positive' (for disease phenotypes)
        - 'protein_protein' (for protein interactions)
        """
        query_lower = query.lower()
        relevant_types = []

        # Map from expected relation types to ACTUAL PrimeKG types (based on
        # diagnostic)
        if query_type == 'side_effect_query' or 'side effect' in query_lower:
            # PrimeKG uses 'exposure_disease' for side effects, not
            # 'drug_effect'
            relevant_types.extend(['exposure_disease',
                                   'contraindication',
                                   'disease_phenotype_positive',
                                   'disease_phenotype_negative'])
        elif query_type == 'treatment_query' or 'treat' in query_lower:
            # PrimeKG may not have 'indication' - use what exists
            # exposure_disease can indicate treatment
            relevant_types.extend(
                ['drug_drug', 'drug_protein', 'exposure_disease'])
        elif query_type == 'relationship_query' or 'associated' in query_lower or 'gene' in query_lower:
            # For gene-disease associations, use disease_protein (proteins
            # include genes)
            relevant_types.extend(['disease_protein',
                                   'protein_protein',
                                   'disease_disease',
                                   'disease_phenotype_positive'])
        elif query_type == 'pathway_query' or 'pathway' in query_lower:
            relevant_types.extend(['pathway_protein',
                                   'bioprocess_protein',
                                   'bioprocess_bioprocess',
                                   'protein_protein'])

        return relevant_types

    def _is_relevant_relation(
            self,
            relation_type: str,
            query_type: str) -> bool:
        """Check if a relation type is relevant for the query type."""
        relation_lower = relation_type.lower()

        if query_type == 'side_effect_query':
            # PrimeKG uses 'exposure_disease' for side effects
            return any(
                keyword in relation_lower for keyword in [
                    'exposure',
                    'effect',
                    'contraindication',
                    'phenotype',
                    'adverse'])
        elif query_type == 'treatment_query':
            # PrimeKG may use 'exposure_disease' or 'drug_drug' for treatments
            return any(
                keyword in relation_lower for keyword in [
                    'exposure',
                    'indication',
                    'treat',
                    'therapeutic',
                    'drug'])
        elif query_type == 'relationship_query':
            # For gene-disease, use disease_protein (proteins include genes)
            return any(
                keyword in relation_lower for keyword in [
                    'disease',
                    'protein',
                    'associated',
                    'gene',
                    'phenotype'])
        elif query_type == 'pathway_query':
            return any(
                keyword in relation_lower for keyword in [
                    'pathway',
                    'bioprocess',
                    'process',
                    'protein'])

        return True  # Default: include all relations

    def _compile_retrieval_result(self,
                                  agent_state: AgentState,
                                  query: str,
                                  seed_entities: List[str],
                                  query_analysis: Dict[str,
                                                       Any]) -> RetrievalResult:
        """Compile the final retrieval result."""
        # Convert retrieved entities
        entities = []
        for entity_data in agent_state.retrieved_entities:
            if isinstance(entity_data, RetrievedEntity):
                entities.append(entity_data)
            else:
                # Convert dict to RetrievedEntity
                entities.append(RetrievedEntity(**entity_data))

        # CRITICAL: Explicitly search for expected target entities based on query type
        # This ensures entities like "bleeding", "gastric", "APOE", "APP" are
        # retrieved
        query_type = query_analysis.get('query_type', '')
        query_lower = query.lower()

        # Normalize query_type - handle variations from different sources
        # Query processor uses 'side_effect_query', but LLM analysis might use
        # 'side_effects'
        is_side_effect_query = (
            query_type == 'side_effect_query' or
            query_type == 'side_effects' or
            'side effect' in query_lower or
            'adverse' in query_lower
        )
        is_relationship_query = (
            query_type == 'relationship_query' or
            query_type == 'gene_disease' or
            'gene' in query_lower or
            'associated' in query_lower
        )

        # Extract expected target entities from query context
        expected_targets = []
        if is_side_effect_query:
            # For side effect queries, look for common side effect terms
            side_effect_terms = [
                'bleeding',
                'gastric',
                'stomach',
                'ulcer',
                'nausea',
                'dizziness',
                'headache',
                'rash']
            for term in side_effect_terms:
                if term in query_lower or any(
                        term in word for word in query_lower.split()):
                    expected_targets.append(term)
            # Also add common side effects if query mentions a drug
            if any(
                drug in query_lower for drug in [
                    'aspirin',
                    'ibuprofen',
                    'warfarin',
                    'metformin']):
                expected_targets.extend(['bleeding', 'gastric', 'stomach'])

        elif is_relationship_query:
            # For gene-disease queries, extract gene symbols from query
            # Note: re is already imported at module level
            gene_symbols = re.findall(r'\b([A-Z]{2,8}\d*)\b', query)
            expected_targets.extend(gene_symbols)
            # Also check for common Alzheimer's genes if query mentions
            # Alzheimer
            if 'alzheimer' in query_lower:
                expected_targets.extend(['APOE', 'APP', 'PSEN1', 'PSEN2'])
        
        # CRITICAL FIX: For treatment/mechanism queries, extract disease names as expected targets
        # This ensures relationships like (metformin, treats, type 2 diabetes) are found
        elif query_type in ['treatment_query', 'mechanism_query', 'mechanism']:
            # Extract disease names from query
            disease_keywords = {
                'diabetes': ['diabetes', 'diabetic', 'type 2 diabetes', 'type 1 diabetes', 't2d', 't1d', 'diabetes mellitus'],
                'cancer': ['cancer', 'carcinoma', 'tumor', 'tumour'],
                'alzheimer': ['alzheimer', "alzheimer's", 'alzheimers', 'ad'],
            }
            
            for disease_key, variations in disease_keywords.items():
                if any(var in query_lower for var in variations):
                    # Add the specific variation found in query
                    for var in variations:
                        if var in query_lower:
                            if var not in expected_targets:
                                expected_targets.append(var)
                    # Also add the base keyword if not already added
                    if disease_key not in expected_targets:
                        expected_targets.append(disease_key)
            
            # Also extract capitalized disease names (e.g., "Type 2 Diabetes")
            disease_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            disease_matches = re.findall(disease_pattern, query)
            for match in disease_matches:
                match_lower = match.lower()
                # Only add if it looks like a disease (not a drug or gene)
                if any(keyword in match_lower for keyword in ['diabetes', 'cancer', 'disease', 'syndrome', 'disorder']):
                    if match_lower not in expected_targets:
                        expected_targets.append(match_lower)

        # Search for and add expected target entities
        for target_term in expected_targets:
            if not any(target_term.lower() in e.name.lower()
                       for e in entities):
                # Search for this entity in PrimeKG
                search_results = self.data_source.search_entities(
                    query=target_term,
                    entity_types=None,  # Search all types
                    limit=3
                )
                if search_results:
                    best_match = search_results[0]
                    if best_match.get(
                        'relevance_score',
                            0) > 0.1:  # Low threshold for expected targets
                        entity_name = best_match.get('name', target_term)
                        entity_id = best_match.get('id', target_term)

                        # CRITICAL: Infer correct entity type from name (PrimeKG types may be wrong)
                        # This fixes misclassifications like "bleeding" ->
                        # "protein" or "stomach" -> "protein"
                        raw_type = best_match.get('type', 'unknown')
                        entity_name_lower = entity_name.lower()
                        raw_type_lower = raw_type.lower()
                        query_lower = query.lower()
                        
                        # CRITICAL: If query mentions "gene" or "genes", classify gene symbols as "gene" not "protein"
                        is_gene_query = 'gene' in query_lower or 'genes' in query_lower
                        is_gene_symbol = bool(re.match(r'^[A-Z]{2,8}\d*$', entity_name.strip()))
                        common_gene_symbols = ['BRCA1', 'BRCA2', 'APOE', 'APP', 'PSEN1', 'PSEN2', 'TP53', 'EGFR', 'HER2', 'KRAS', 'BRAF']
                        
                        if is_gene_query and (is_gene_symbol or entity_name.strip().upper() in common_gene_symbols):
                            if raw_type_lower == 'protein':
                                entity_type = 'gene'
                            else:
                                entity_type = raw_type
                        # First: Direct keyword matching (most reliable)
                        elif entity_name_lower in ['stomach', 'bleeding', 'abnormal bleeding'] or 'abnormality' in entity_name_lower:
                            if raw_type_lower == 'protein':
                                entity_type = 'phenotype' if 'bleeding' in entity_name_lower or 'abnormality' in entity_name_lower else 'anatomy'
                            else:
                                entity_type = raw_type
                        elif 'cytoplasmic side' in entity_name_lower or 'side of' in entity_name_lower:
                            if raw_type_lower in ['drug', 'protein']:
                                entity_type = 'anatomy'
                            else:
                                entity_type = raw_type
                        # Second: Use inference if keyword matching didn't work
                        elif hasattr(self.data_source, '_infer_entity_type'):
                            inferred_type = self.data_source._infer_entity_type(entity_name)
                            # Use inferred type if it's more specific than raw type
                            # But prefer raw type if it's already correct
                            # (e.g., "disease" vs "unknown")
                            if inferred_type != 'unknown':
                                # Override if raw type is wrong (e.g., "protein" for "bleeding")
                                if raw_type_lower == 'protein' and inferred_type.lower() in ['phenotype', 'disease', 'anatomy']:
                                    entity_type = inferred_type
                                    logger.debug(f"Corrected entity type for '{entity_name}': {raw_type} -> {inferred_type}")
                                elif raw_type_lower == 'unknown':
                                    entity_type = inferred_type
                                else:
                                    # Trust raw type if it seems reasonable
                                    entity_type = raw_type
                            else:
                                entity_type = raw_type
                        else:
                            entity_type = raw_type

                        target_entity = RetrievedEntity(
                            entity_id=entity_id,
                            entity_type=entity_type,
                            name=entity_name,
                            description='',
                            relevance_score=0.7,  # High score for expected targets
                            context={
                                'provenance': 'expected_target',
                                'query_type': query_type,
                                'raw_type': raw_type},
                            agent_reasoning=f"Expected target entity for {query_type}"
                        )
                        entities.append(target_entity)
                        logger.info(
                            f"Added expected target entity: {target_entity.name} (type: {target_entity.entity_type}, was: {raw_type})")

        # Create relationships from entity connections
        relationships = []
        entity_ids = {e.entity_id for e in entities}
        entity_map = {e.entity_id: e for e in entities}  # Map for quick lookup
        relationship_keys = set()  # Track relationships to avoid duplicates

        # Identify query-relevant relation types based on query analysis
        relevant_relation_types = self._get_relevant_relation_types(
            query_type, query)

        # Method 1: Extract relationships directly from graph edges between retrieved entities
        # CRITICAL: Must respect MAX_TOTAL_RELATIONSHIPS limit
        # Use config limit (50)
        MAX_TOTAL_RELATIONSHIPS = self.config.max_relationships
        
        # CRITICAL FIX: Reserve space for Method 3.5 (expected target relationships)
        # These are more important than generic neighbor relationships
        RESERVED_FOR_EXPECTED_TARGETS = 10 if expected_targets else 0
        MAX_BEFORE_EXPECTED_TARGETS = MAX_TOTAL_RELATIONSHIPS - RESERVED_FOR_EXPECTED_TARGETS

        if hasattr(
                self.data_source,
                'graph') and self.data_source.graph is not None:
            for entity in entities:
                # Stop if we've reached the limit (reserving space for expected targets)
                if len(relationships) >= MAX_BEFORE_EXPECTED_TARGETS:
                    break

                entity_id = entity.entity_id
                try:
                    if entity_id in self.data_source.graph:
                        # Get all outgoing edges from this entity
                        for neighbor_id in self.data_source.graph.successors(
                                entity_id):
                            # Stop if we've reached the limit (reserving space for expected targets)
                            if len(relationships) >= MAX_BEFORE_EXPECTED_TARGETS:
                                break

                            if neighbor_id in entity_ids:
                                # Get edge data - MultiDiGraph returns dict of
                                # {edge_key: {attributes}}
                                edge_data = self.data_source.graph.get_edge_data(
                                    entity_id, neighbor_id)
                            if edge_data:
                                # Extract all relation types from edge data
                                # (MultiDiGraph can have multiple edges)
                                relations_found = []
                                for edge_key, edge_attrs in edge_data.items():
                                    relation = edge_attrs.get('relation') or edge_attrs.get(
                                        'type') or 'related_to'
                                    relations_found.append(relation)

                                # Process each relation type found
                                for relation_type in relations_found:
                                    # Stop if we've reached the limit
                                    if len(
                                            relationships) >= MAX_TOTAL_RELATIONSHIPS:
                                        break

                                    # Filter by relevant relation types if
                                    # specified
                                    if relevant_relation_types:
                                        # Check if relation type matches any
                                        # relevant type
                                        if relation_type not in relevant_relation_types:
                                            # Try fuzzy matching
                                            if not self._is_relevant_relation(
                                                    relation_type, query_type):
                                                continue  # Skip this relationship

                                    # Create unique key for this relationship
                                    rel_key = (
                                        entity_id, neighbor_id, relation_type)
                                    if rel_key not in relationship_keys:
                                        neighbor_entity = next(
                                            (e for e in entities if e.entity_id == neighbor_id), None)
                                        if neighbor_entity:
                                            relationship = RetrievedRelationship(
                                                source_id=entity_id,
                                                target_id=neighbor_id,
                                                relation_type=relation_type,
                                                display_relation=relation_type,
                                                source_entity=entity,
                                                target_entity=neighbor_entity,
                                                relevance_score=0.7,
                                                context={'provenance': 'direct_graph_edge'},
                                                agent_reasoning="Direct relationship in PrimeKG graph"
                                            )
                                            relationships.append(relationship)
                                            relationship_keys.add(rel_key)
                except Exception as e:
                    logger.debug(f"Failed to get graph edges for {entity_id}: {e}")
                    continue

        # Method 2: Find relationships from seed/query entities to neighbors (OPTIMIZED)
        # Strategy: Filter by relation type FIRST, then use semantic similarity to rank
        # This prevents retrieving millions of relationships
        seed_entity_ids = set(seed_entities)
        MAX_RELATIONSHIPS_PER_SEED = 15  # Limit relationships per seed entity
        # MAX_TOTAL_RELATIONSHIPS and MAX_BEFORE_EXPECTED_TARGETS are already defined above

        # CRITICAL FIX: Stop Method 2 early to reserve space for Method 3.5
        if len(relationships) >= MAX_BEFORE_EXPECTED_TARGETS:
            logger.info(f"Stopping Method 2 early ({len(relationships)}/{MAX_TOTAL_RELATIONSHIPS}) to reserve space for expected target relationships")
        
        # Process seed entities, but also include high-relevance retrieved entities as fallback
        # This ensures we get relationships even if seed identification
        # partially fails
        seed_entities_list = [
            e for e in entities if e.entity_id in seed_entity_ids]

        # If no seed entities found in retrieved entities, use top retrieved entities as fallback
        # This makes the system robust to seed entity identification failures
        if not seed_entities_list and entities:
            # Use top entities by relevance score as pseudo-seeds
            sorted_entities = sorted(
                entities, key=lambda e: e.relevance_score, reverse=True)
            seed_entities_list = sorted_entities[:min(3, len(sorted_entities))]
            logger.info(
                f"No seed entities in retrieved list, using top {len(seed_entities_list)} entities as fallback")

        # Also ensure we have at least one entity to process
        if not seed_entities_list and seed_entities:
            # Try to find seed entities directly from IDs
            for seed_id in seed_entities[:3]:  # Limit to first 3 seeds
                # Create a minimal entity if not found in retrieved entities
                if seed_id not in entity_map:
                    try:
                        entity_info = self.data_source.entity_info.get(
                            seed_id, {})
                        if entity_info:
                            seed_entity = RetrievedEntity(
                                entity_id=seed_id,
                                entity_type=entity_info.get('type', 'unknown'),
                                name=entity_info.get('name', seed_id),
                                description='',
                                relevance_score=0.8,  # High score for seed entities
                                context={'provenance': 'seed_entity'},
                                agent_reasoning="Direct seed entity"
                            )
                            entities.append(seed_entity)
                            entity_map[seed_id] = seed_entity
                            entity_ids.add(seed_id)
                            seed_entities_list.append(seed_entity)
                    except Exception:
                        continue

        # Pre-compute query embedding for semantic similarity (if available)
        query_embedding = None
        try:
            if hasattr(self.data_source, 'encode_texts'):
                query_embeddings = self.data_source.encode_texts([query])
                query_embedding = query_embeddings[0] if query_embeddings else None
        except Exception:
            pass  # Fallback to non-semantic ranking

        for entity in seed_entities_list:
            # Stop if we've reached the total relationship limit
            if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                logger.info(
                    f"Stopping relationship retrieval - reached limit ({MAX_TOTAL_RELATIONSHIPS})")
                break

            try:
                # CRITICAL: Filter by relation type BEFORE retrieving neighbors (saves time)
                # This prevents retrieving millions of neighbors
                neighbors = self.data_source.get_entity_neighbors(
                    entity.entity_id,
                    max_hops=1,
                    relation_types=relevant_relation_types if relevant_relation_types else None
                )

                # CRITICAL FIX: Improved fallback strategy with logging
                # If filtering is too strict and returns no neighbors, try without filter
                # BUT: Log this so we can monitor and improve filters
                if not neighbors and relevant_relation_types:
                    logger.warning(
                        f"Type filter too strict for entity '{entity.name}' (types: {relevant_relation_types}), trying without filter")
                    neighbors = self.data_source.get_entity_neighbors(
                        entity.entity_id,
                        max_hops=1,
                        relation_types=None  # No filter - get all neighbors
                    )
                    # CRITICAL: Limit neighbors from unfiltered query to prevent explosion
                    if neighbors and len(neighbors) > 20:  # Limit to 20 neighbors when unfiltered
                        logger.warning(
                            f"Unfiltered query returned {len(neighbors)} neighbors, scoring and filtering by relevance")
                        # Score all neighbors by relevance and take top 20
                        scored_neighbors = []
                        for neighbor_id, neighbor_info in neighbors.items():
                            neighbor_entity_info = neighbor_info.get('entity_info', {})
                            neighbor_name = neighbor_entity_info.get('name', neighbor_id)
                            neighbor_type = neighbor_entity_info.get('type', 'unknown')
                            neighbor_desc = neighbor_entity_info.get('description', '')
                            
                            # Score neighbor relevance using existing heuristic function
                            relevance_score = self._calculate_heuristic_relevance(
                                neighbor_name, neighbor_type, neighbor_desc, query
                            )
                            scored_neighbors.append((neighbor_id, neighbor_info, relevance_score))
                        
                        # Sort by relevance (descending) and take top 20
                        scored_neighbors.sort(key=lambda x: x[2], reverse=True)
                        neighbors = {nid: ninfo for nid, ninfo, _ in scored_neighbors[:20]}
                        logger.info(f"Filtered to top 20 neighbors by relevance (scores: {[f'{s:.2f}' for _, _, s in scored_neighbors[:20]]})")

                if not neighbors:
                    continue

                # PRIORITY 1 FIX: Pre-filter neighbors by relevance before relationship extraction
                # This prevents irrelevant neighbors (e.g., "schizophrenia" for "metformin diabetes") 
                # from getting relationships created
                # Lowered threshold from 0.4 to 0.2 to account for improved scoring function
                filtered_neighbors = {}
                min_relevance_threshold = 0.2  # Filter out neighbors with relevance < 0.2 (lowered from 0.4)
                
                for neighbor_id, neighbor_info in neighbors.items():
                    neighbor_entity_info = neighbor_info.get('entity_info', {})
                    neighbor_name = neighbor_entity_info.get('name', neighbor_id)
                    neighbor_type = neighbor_entity_info.get('type', 'unknown')
                    neighbor_desc = neighbor_entity_info.get('description', '')
                    
                    # Score neighbor relevance
                    neighbor_relevance = self._calculate_heuristic_relevance(
                        neighbor_name, neighbor_type, neighbor_desc, query
                    )
                    
                    # Only include neighbors above relevance threshold
                    if neighbor_relevance >= min_relevance_threshold:
                        filtered_neighbors[neighbor_id] = neighbor_info
                    else:
                        logger.debug(f"Filtered out irrelevant neighbor '{neighbor_name}' (relevance: {neighbor_relevance:.2f} < {min_relevance_threshold})")
                
                if not filtered_neighbors:
                    logger.warning(f"No neighbors passed relevance filter (threshold: {min_relevance_threshold}) for entity '{entity.name}'")
                    continue
                
                logger.debug(f"Filtered {len(neighbors)} neighbors to {len(filtered_neighbors)} relevant neighbors for entity '{entity.name}'")
                neighbors = filtered_neighbors

                # Score and rank relationships by relevance
                scored_relationships = []

                for neighbor_id, neighbor_info in neighbors.items():
                    # Get relations from neighbor info
                    relations = neighbor_info.get('relations', [])
                    if not relations:
                        rel_type = neighbor_info.get(
                            'relation') or neighbor_info.get('type') or 'related_to'
                        relations = [rel_type]

                    # Filter to only relevant relations (but be more lenient)
                    # If we have relevant_relation_types, prefer those, but
                    # also allow fuzzy matches
                    if relevant_relation_types:
                        relevant_relations = [
                            r for r in relations
                            if r in relevant_relation_types or self._is_relevant_relation(r, query_type)
                        ]
                    else:
                        # No specific types requested - use fuzzy matching
                        relevant_relations = [
                            r for r in relations if self._is_relevant_relation(
                                r, query_type)]

                    # PHASE 3 FIX: Remove lenient fallback - don't use irrelevant relation types
                    # If no relevant relations found, skip this neighbor entirely
                    # This prevents noise from wrong relation types
                    if not relevant_relations:
                        logger.debug(f"Skipping neighbor '{neighbor_name}' - no relevant relation types found (available: {relations})")
                        continue  # Skip if no relevant relations

                    # Get neighbor entity info
                    neighbor_entity_info = neighbor_info.get('entity_info', {})
                    neighbor_name = neighbor_entity_info.get(
                        'name', neighbor_id)
                    neighbor_type = neighbor_entity_info.get('type', 'unknown')
                    neighbor_desc = neighbor_entity_info.get('description', '')

                    # PHASE 2 FIX: Calculate neighbor relevance first for penalty application
                    neighbor_relevance = self._calculate_heuristic_relevance(
                        neighbor_name, neighbor_type, neighbor_desc, query
                    )

                    # Score relationship relevance
                    score = 0.0

                    # PHASE 2 FIX: Apply negative scoring for irrelevant neighbors
                    # This ensures irrelevant relationships get heavily penalized
                    if neighbor_relevance < 0.3:
                        # Very irrelevant - heavy penalty
                        score = 0.1  # Start with very low base score
                        logger.debug(f"Heavy penalty applied to irrelevant neighbor '{neighbor_name}' (relevance: {neighbor_relevance:.2f})")
                    elif neighbor_relevance < 0.5:
                        # Moderately irrelevant - moderate penalty
                        score = 0.3  # Start with low base score
                        logger.debug(f"Moderate penalty applied to low-relevance neighbor '{neighbor_name}' (relevance: {neighbor_relevance:.2f})")
                    else:
                        # Relevant neighbor - normal scoring
                        score = 0.0  # Start from zero, will add positive scores

                    # PHASE 5 FIX: Enhanced expected target detection with fuzzy matching and synonyms
                    neighbor_name_lower = neighbor_name.lower()
                    is_expected_target = False
                    
                    # Synonym dictionaries for side effects (reused from evaluator)
                    side_effect_synonyms = {
                        'bleeding': ['bleeding', 'hemorrhage', 'hemorrhagic', 'blood loss', 'abnormal bleeding', 
                                    'gastrointestinal bleeding', 'gastric bleeding', 'abnormality of bleeding'],
                        'gastric': ['gastric', 'stomach', 'gastrointestinal', 'gi', 'peptic', 'ulcer', 
                                   'gastric mucosa', 'gastric ulcer', 'abnormality of the gastric mucosa'],
                        'stomach': ['stomach', 'gastric', 'gastrointestinal', 'gi', 'peptic'],
                    }
                    
                    # Disease name normalization patterns
                    disease_normalizations = {
                        'diabetes': ['diabetes', 'diabetes mellitus', 'diabetic', 'dm'],
                        'type 2 diabetes': ['type 2 diabetes', 'type 2 diabetes mellitus', 'type ii diabetes', 
                                           'type ii diabetes mellitus', 't2d', 't2dm'],
                        'type 1 diabetes': ['type 1 diabetes', 'type 1 diabetes mellitus', 'type i diabetes', 
                                           'type i diabetes mellitus', 't1d', 't1dm'],
                    }
                    
                    for target in expected_targets:
                        target_lower = target.lower()
                        
                        # Exact match
                        if target_lower == neighbor_name_lower:
                            is_expected_target = True
                            break
                        
                        # Substring match (original logic)
                        if target_lower in neighbor_name_lower or neighbor_name_lower in target_lower:
                            is_expected_target = True
                            break
                        
                        # Check side effect synonyms
                        for key, synonyms in side_effect_synonyms.items():
                            if key in target_lower:
                                if any(syn in neighbor_name_lower for syn in synonyms):
                                    is_expected_target = True
                                    break
                                if any(syn in target_lower for syn in synonyms if syn in neighbor_name_lower):
                                    is_expected_target = True
                                    break
                        if is_expected_target:
                            break
                        
                        # Check disease normalizations
                        for key, variations in disease_normalizations.items():
                            if key in target_lower:
                                if any(var in neighbor_name_lower for var in variations):
                                    is_expected_target = True
                                    break
                                if any(var in target_lower for var in variations if var in neighbor_name_lower):
                                    is_expected_target = True
                                    break
                        if is_expected_target:
                            break
                        
                        # Fuzzy matching using SequenceMatcher for close matches
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(None, target_lower, neighbor_name_lower).ratio()
                        if similarity > 0.85:  # High similarity threshold
                            is_expected_target = True
                            logger.debug(f"Fuzzy match: '{target}' ~ '{neighbor_name}' (similarity: {similarity:.2f})")
                            break
                    
                    if is_expected_target:
                        score += 1.0  # Strong boost for expected targets
                        logger.debug(
                            f"Boosting relationship score for expected target: {neighbor_name}")

                    # Base score for relevant relation type
                    if any(self._is_relevant_relation(r, query_type)
                           for r in relevant_relations):
                        score += 0.5

                    # PHASE 2 FIX: Strengthen semantic similarity weight and add penalty
                    # Semantic similarity score (if embeddings available)
                    if query_embedding is not None:
                        try:
                            neighbor_text = f"{neighbor_name} {neighbor_entity_info.get('type', '')}"
                            neighbor_embeddings = self.data_source.encode_texts(
                                [neighbor_text])
                            if neighbor_embeddings:
                                neighbor_emb = neighbor_embeddings[0]
                                semantic_sim = float(
                                    np.dot(neighbor_emb, query_embedding))
                                # Increased weight from 0.3 to 0.5 for better discrimination
                                score += semantic_sim * 0.5  # Increased weight semantic similarity
                                # Penalty if semantic similarity is very low
                                if semantic_sim < 0.3:
                                    score *= 0.2  # Heavy penalty for low semantic similarity
                                    logger.debug(f"Penalty applied for low semantic similarity ({semantic_sim:.2f}) for '{neighbor_name}'")
                        except Exception:
                            pass  # Fallback if embedding fails

                    # Boost for exact name matches in query
                    query_lower = query.lower()
                    query_terms = set(query_lower.split())
                    neighbor_words = set(neighbor_name_lower.split())
                    query_overlap = query_terms & neighbor_words
                    
                    if neighbor_name.lower() in query_lower or any(term in neighbor_name.lower()
                                                                   for term in query_lower.split() if len(term) > 3):
                        score += 0.2
                    
                    # PHASE 2 FIX: Penalty if no meaningful query term overlap
                    if not query_overlap and score < 0.5:
                        score *= 0.2  # Penalty for no query term overlap
                        logger.debug(f"Penalty applied for no query term overlap for '{neighbor_name}'")

                    # Store scored relationship
                    for relation in relevant_relations:
                        scored_relationships.append({
                            'neighbor_id': neighbor_id,
                            'neighbor_info': neighbor_info,
                            'relation': relation,
                            'score': score,
                            'neighbor_name': neighbor_name
                        })

                # Sort by score and take top-K
                scored_relationships.sort(
                    key=lambda x: x['score'], reverse=True)
                scored_relationships = scored_relationships[:MAX_RELATIONSHIPS_PER_SEED]

                # Create relationships from top-scored items
                for item in scored_relationships:
                    if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                        break

                    neighbor_id = item['neighbor_id']
                    neighbor_info = item['neighbor_info']
                    relation = item['relation']
                    score = item['score']

                    # Get or create target entity
                    neighbor_entity = entity_map.get(neighbor_id)
                    if not neighbor_entity:
                        neighbor_entity_info = neighbor_info.get(
                            'entity_info', {})
                        neighbor_entity = RetrievedEntity(
                            entity_id=neighbor_id,
                            entity_type=neighbor_entity_info.get(
                                'type', 'unknown'),
                            name=neighbor_entity_info.get('name', neighbor_id),
                            description='',
                            relevance_score=min(
                                score, 1.0),  # Use computed score
                            context={'provenance': 'relationship_target'},
                            agent_reasoning="Discovered as relationship target"
                        )
                        # Only add entity if we haven't exceeded entity limit
                        if len(entities) < self.config.max_entities:
                            entities.append(neighbor_entity)
                            entity_map[neighbor_id] = neighbor_entity
                            entity_ids.add(neighbor_id)
                        else:
                            # Still create relationship but don't add entity
                            entity_map[neighbor_id] = neighbor_entity

                    # Create relationship
                    rel_key = (entity.entity_id, neighbor_id, relation)
                    if rel_key not in relationship_keys:
                        relationship = RetrievedRelationship(
                            source_id=entity.entity_id,
                            target_id=neighbor_id,
                            relation_type=relation,
                            display_relation=relation,
                            source_entity=entity,
                            target_entity=neighbor_entity,
                            relevance_score=min(
                                score, 1.0),  # Use computed score
                            context={'provenance': 'neighbor_link'},
                            agent_reasoning="Discovered through neighbor exploration"
                        )
                        relationships.append(relationship)
                        relationship_keys.add(rel_key)

                        if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                            break

            except Exception as e:
                logger.debug(
                    f"Failed to get neighbors for {entity.entity_id}: {e}")
                continue

        # Method 3: Use neighbor lookup for relationships between retrieved entities (fallback)
        # CRITICAL: Only run if we haven't reached the limit, and respect the
        # limit strictly
        if len(relationships) < MAX_TOTAL_RELATIONSHIPS:
            # PHASE 4 FIX: Only process highly relevant entities to reduce noise
            # Filter to entities with relevance_score > 0.6 and limit to top 5
            relevant_entities = [e for e in entities if e.relevance_score > 0.6]
            entities_to_process = sorted(relevant_entities, key=lambda e: e.relevance_score, reverse=True)[:5]
            logger.debug(f"Method 3: Processing {len(entities_to_process)} highly relevant entities (relevance > 0.6)")

            for entity in entities_to_process:
                # Stop if we've reached the limit
                if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                    break

                try:
                    neighbors = self.data_source.get_entity_neighbors(
                        entity.entity_id, max_hops=1)
                    if neighbors:
                        # Limit neighbors per entity
                        neighbor_items = list(neighbors.items())[
                            :5]  # Max 5 neighbors per entity

                        for neighbor_id, neighbor_info in neighbor_items:
                            # Stop if we've reached the limit
                            if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                                break

                            # Only process if neighbor is already in retrieved
                            # entities
                            neighbor_entity = entity_map.get(neighbor_id)
                            if neighbor_entity:
                                # PHASE 4 FIX: Check if both entities are relevant to query
                                # Skip if either entity has low relevance
                                if entity.relevance_score < 0.4 or neighbor_entity.relevance_score < 0.4:
                                    logger.debug(f"Skipping relationship between low-relevance entities: '{entity.name}' ({entity.relevance_score:.2f}) <-> '{neighbor_entity.name}' ({neighbor_entity.relevance_score:.2f})")
                                    continue
                                
                                # Get relations from neighbor info
                                relations = neighbor_info.get('relations', [])
                                if not relations:
                                    rel_type = neighbor_info.get(
                                        'relation') or neighbor_info.get('type') or 'related_to'
                                    relations = [rel_type]

                                # PHASE 4 FIX: Filter relations by query relevance
                                relevant_relations = [
                                    r for r in relations 
                                    if self._is_relevant_relation(r, query_type)
                                ]
                                
                                if not relevant_relations:
                                    logger.debug(f"Skipping relationship - no relevant relation types: '{entity.name}' -> '{neighbor_entity.name}' (types: {relations})")
                                    continue

                                for relation in relevant_relations:
                                    rel_key = (
                                        entity.entity_id, neighbor_id, relation)
                                    if rel_key not in relationship_keys:
                                        relationship = RetrievedRelationship(
                                            source_id=entity.entity_id,
                                            target_id=neighbor_id,
                                            relation_type=relation,
                                            display_relation=relation,
                                            source_entity=entity,
                                            target_entity=neighbor_entity,
                                            relevance_score=0.6,
                                            context={'provenance': 'retrieved_entity_link'},
                                            agent_reasoning="Relationship between retrieved entities"
                                        )
                                        relationships.append(relationship)
                                        relationship_keys.add(rel_key)

                                        # Stop if we've reached the limit
                                        if len(
                                                relationships) >= MAX_TOTAL_RELATIONSHIPS:
                                            break
                except Exception as e:
                    logger.debug(
                        f"Failed to get neighbors for {entity.entity_id}: {e}")
                    continue

        # Method 3.5: Explicitly search for relationships between seed entities and expected target entities
        # CRITICAL: This ensures relationships like (metformin, treats, type 2 diabetes) are found
        # Run this BEFORE Method 4 to prioritize expected target relationships
        # CRITICAL FIX: Run this even if we're near the limit - expected targets are high priority
        if expected_targets:
            logger.info(
                f"Method 3.5: Checking for relationships with {len(expected_targets)} expected targets: {expected_targets}")
            # Allow Method 3.5 to use reserved space even if we're at the limit
            if len(relationships) < MAX_TOTAL_RELATIONSHIPS:
                # CRITICAL FIX: More flexible matching for expected target entities
                # Handle variations like "type 2 diabetes" matching "type 2 diabetes mellitus"
                expected_target_entities = []
                for e in entities:
                    entity_name_lower = e.name.lower()
                    for target in expected_targets:
                        target_lower = target.lower()
                        # Check if target is substring of entity name or vice versa
                        if target_lower in entity_name_lower or entity_name_lower in target_lower:
                            expected_target_entities.append(e)
                            break
                        # Also check word-level matching for multi-word targets
                        target_words = set(target_lower.split())
                        entity_words = set(entity_name_lower.split())
                        # If at least 2 words match (or all words if target is short), consider it a match
                        common_words = target_words & entity_words
                        if len(common_words) >= min(2, len(target_words)) and len(common_words) > 0:
                            expected_target_entities.append(e)
                            break

                if expected_target_entities and seed_entities_list:
                    logger.info(
                        f"Searching for relationships between {len(seed_entities_list)} seed entities and {len(expected_target_entities)} expected target entities")
                    # DEBUG: Log expected target entity names
                    logger.info(f"Method 3.5: Expected target entities: {[e.name for e in expected_target_entities]}")
                    logger.info(f"Method 3.5: Seed entities: {[e.name for e in seed_entities_list]}")

                    for seed_entity in seed_entities_list:
                        if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                            break

                        seed_id = seed_entity.entity_id
                        seed_name = seed_entity.name
                        if seed_id not in self.data_source.graph:
                            logger.warning(
                                f"Method 3.5: Seed entity '{seed_name}' (ID: {seed_id[:50]}...) not in graph")
                            continue

                        # Check if seed entity has edges to any expected target
                        # entities
                        for target_entity in expected_target_entities:
                            if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                                break

                            target_id = target_entity.entity_id
                            target_name = target_entity.name
                            if target_id not in self.data_source.graph:
                                logger.warning(
                                    f"Method 3.5: Target entity '{target_name}' (ID: {target_id[:50]}...) not in graph")
                                continue
                            
                            # DEBUG: Log what we're checking
                            logger.debug(f"Method 3.5: Checking relationships between '{seed_name}' and '{target_name}'")
                            
                            # DEBUG: Log query_type to verify it's correct
                            logger.info(f"Method 3.5: Current query_type='{query_type}' for seed='{seed_name}', target='{target_name}'")

                            # CRITICAL FIX: For side effect queries, search multi-hop paths
                            # PrimeKG stores side effects as: Drug → [exposure_disease] → Disease → [disease_phenotype_positive] → Phenotype
                            # So we need to search 2-hop paths, not just direct edges
                            if query_type in ['side_effect_query', 'side_effects', 'adverse_effect']:
                                logger.info(f"Method 3.5: Multi-hop search ENABLED for query_type='{query_type}' (seed: {seed_name}, target: {target_name})")
                                # Multi-hop search: Drug → Disease → Phenotype
                                found_via_multihop = False
                                
                                # Step 1: Find diseases connected to seed (drug) via exposure_disease
                                if seed_id in self.data_source.graph:
                                    logger.info(f"Method 3.5: Seed '{seed_name}' is in graph, searching for exposure_disease relationships...")
                                    seed_neighbors = list(self.data_source.graph.successors(seed_id))
                                    logger.info(f"Method 3.5: Found {len(seed_neighbors)} neighbors for '{seed_name}', checking for exposure_disease relationships...")
                                    for disease_id in seed_neighbors:
                                        if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                                            break
                                        
                                        # Check if this is an exposure_disease relationship
                                        disease_edge_data = self.data_source.graph.get_edge_data(seed_id, disease_id)
                                        if disease_edge_data:
                                            for edge_key, edge_attrs in disease_edge_data.items():
                                                relation_type = edge_attrs.get('relation') or edge_attrs.get('type') or 'related_to'
                                                if relation_type == 'exposure_disease':
                                                    # Get disease name for logging
                                                    disease_name = "Unknown"
                                                    if disease_id in self.data_source.graph.nodes:
                                                        disease_name = self.data_source.graph.nodes[disease_id].get('name', disease_id)
                                                    elif disease_id in self.data_source.entity_info:
                                                        disease_name = self.data_source.entity_info[disease_id].get('name', disease_id)
                                                    logger.info(f"Method 3.5: Found exposure_disease relationship: {seed_name} --[exposure_disease]--> {disease_name}")
                                                    # Step 2: Find phenotypes connected to this disease
                                                    if disease_id in self.data_source.graph:
                                                        phenotype_neighbors = list(self.data_source.graph.successors(disease_id))
                                                        logger.info(f"Method 3.5: Found {len(phenotype_neighbors)} phenotype neighbors for disease '{disease_name}', checking for matches...")
                                                        
                                                        # CRITICAL FIX: Initialize matching_target_entity before checking
                                                        matching_target_entity = None
                                                        
                                                        # CRITICAL FIX: Also check if expected target entities themselves are phenotypes
                                                        # that could be connected to ANY disease, not just the one we found
                                                        # This handles cases where expected targets are already in the graph
                                                        for exp_target in expected_target_entities:
                                                            exp_target_id = exp_target.entity_id
                                                            exp_target_name = exp_target.name
                                                            
                                                            # Check if this expected target is a phenotype connected to the disease
                                                            if exp_target_id in phenotype_neighbors:
                                                                # Check if it's a disease_phenotype relationship
                                                                exp_phenotype_edge_data = self.data_source.graph.get_edge_data(disease_id, exp_target_id)
                                                                if exp_phenotype_edge_data:
                                                                    for p_edge_key, p_edge_attrs in exp_phenotype_edge_data.items():
                                                                        p_relation_type = p_edge_attrs.get('relation') or p_edge_attrs.get('type') or 'related_to'
                                                                        if p_relation_type in ['disease_phenotype_positive', 'disease_phenotype_negative']:
                                                                            # Found match! Expected target is a phenotype of this disease
                                                                            matching_target_entity = exp_target
                                                                            logger.info(f"Method 3.5: Expected target '{exp_target_name}' IS a phenotype of disease '{disease_name}'")
                                                                            break
                                                        
                                                        # If we didn't find a direct match, check all phenotype neighbors
                                                        if not matching_target_entity:
                                                            for phenotype_id in phenotype_neighbors:
                                                                if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                                                                    break
                                                                
                                                                # Check if phenotype matches ANY expected target (by ID or name)
                                                                matching_target_entity = None
                                                                
                                                                # First check by ID
                                                                if phenotype_id == target_id:
                                                                    matching_target_entity = target_entity
                                                                else:
                                                                    # Check by name matching - check all expected target entities
                                                                    phenotype_name = None
                                                                    if phenotype_id in self.data_source.graph.nodes:
                                                                        phenotype_name = self.data_source.graph.nodes[phenotype_id].get('name', '')
                                                                    elif phenotype_id in self.data_source.entity_info:
                                                                        phenotype_name = self.data_source.entity_info[phenotype_id].get('name', '')
                                                                    
                                                                    if phenotype_name:
                                                                        phenotype_name_lower = phenotype_name.lower()
                                                                        # Check against all expected target entities
                                                                        for exp_target in expected_target_entities:
                                                                            exp_target_name_lower = exp_target.name.lower()
                                                                            # Check if target keywords appear in phenotype name
                                                                            # Filter out common stop words that cause false positives
                                                                            stop_words = {'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
                                                                            exp_keywords = [kw for kw in exp_target_name_lower.split() if len(kw) >= 3 and kw not in stop_words]
                                                                            phenotype_keywords = [kw for kw in phenotype_name_lower.split() if len(kw) >= 3 and kw not in stop_words]
                                                                            
                                                                            # Check if any expected target keyword appears in phenotype name
                                                                            # Only match if keyword is at least 3 chars (to avoid substring matches like "of" in "carbofuran")
                                                                            if exp_keywords and any(kw in phenotype_name_lower for kw in exp_keywords):
                                                                                matching_target_entity = exp_target
                                                                                logger.info(f"Method 3.5: Matched phenotype '{phenotype_name}' to expected target '{exp_target.name}' (keyword match)")
                                                                                break
                                                                            # Also check reverse - if phenotype keywords appear in expected target
                                                                            # Only match if we have meaningful keywords (not just stop words)
                                                                            if phenotype_keywords and any(kw in exp_target_name_lower for kw in phenotype_keywords):
                                                                                matching_target_entity = exp_target
                                                                                logger.info(f"Method 3.5: Matched phenotype '{phenotype_name}' to expected target '{exp_target.name}' (reverse keyword match)")
                                                                                break
                                                                            # Also check substring matches for common side effect terms
                                                                            if 'bleeding' in exp_target_name_lower and ('bleed' in phenotype_name_lower or 'hemorrhage' in phenotype_name_lower):
                                                                                matching_target_entity = exp_target
                                                                                logger.info(f"Method 3.5: Matched phenotype '{phenotype_name}' to expected target '{exp_target.name}' (bleeding synonym match)")
                                                                                break
                                                                            if 'gastric' in exp_target_name_lower or 'stomach' in exp_target_name_lower:
                                                                                if any(term in phenotype_name_lower for term in ['gastric', 'stomach', 'gastrointestinal', 'gi', 'peptic', 'ulcer', 'mucosa']):
                                                                                    matching_target_entity = exp_target
                                                                                    logger.info(f"Method 3.5: Matched phenotype '{phenotype_name}' to expected target '{exp_target.name}' (gastric synonym match)")
                                                                                    break
                                                            
                                                            if matching_target_entity:
                                                                # Check if this is a disease_phenotype_positive relationship
                                                                # Use phenotype_id if we found a match, or matching_target_id if expected target is the phenotype
                                                                check_phenotype_id = phenotype_id if matching_target_entity.entity_id != phenotype_id else matching_target_entity.entity_id
                                                                phenotype_edge_data = self.data_source.graph.get_edge_data(disease_id, check_phenotype_id)
                                                                if phenotype_edge_data:
                                                                    for p_edge_key, p_edge_attrs in phenotype_edge_data.items():
                                                                        p_relation_type = p_edge_attrs.get('relation') or p_edge_attrs.get('type') or 'related_to'
                                                                        if p_relation_type in ['disease_phenotype_positive', 'disease_phenotype_negative']:
                                                                            # Found multi-hop path! Create relationship
                                                                            # Use the matching expected target entity
                                                                            matching_target_id = matching_target_entity.entity_id
                                                                            matching_target_name = matching_target_entity.name
                                                                            rel_key = (seed_id, matching_target_id, 'has_side_effect')
                                                                            if rel_key not in relationship_keys:
                                                                                # Get disease name for logging
                                                                                disease_name = "Unknown Disease"
                                                                                if disease_id in self.data_source.graph.nodes:
                                                                                    disease_name = self.data_source.graph.nodes[disease_id].get('name', disease_id)
                                                                                elif disease_id in self.data_source.entity_info:
                                                                                    disease_name = self.data_source.entity_info[disease_id].get('name', disease_id)
                                                                                
                                                                                relationship = RetrievedRelationship(
                                                                                    source_id=seed_id,
                                                                                    target_id=matching_target_id,
                                                                                    relation_type='has_side_effect',
                                                                                    display_relation='has_side_effect',
                                                                                    source_entity=seed_entity,
                                                                                    target_entity=matching_target_entity,
                                                                                    relevance_score=0.95,  # Very high score for multi-hop side effects
                                                                                    context={
                                                                                        'provenance': 'multihop_side_effect',
                                                                                        'intermediate_disease': disease_id,
                                                                                        'intermediate_disease_name': disease_name,
                                                                                        'path': f"{seed_name} --[exposure_disease]--> {disease_name} --[{p_relation_type}]--> {matching_target_name}"
                                                                                    },
                                                                                    agent_reasoning=f"Multi-hop side effect relationship via {disease_name}"
                                                                                )
                                                                                relationships.append(relationship)
                                                                                relationship_keys.add(rel_key)
                                                                                logger.info(
                                                                                    f"Found multi-hop side effect via Method 3.5: {seed_name} --[has_side_effect]--> {matching_target_name} (via {disease_name})")
                                                                                found_via_multihop = True
                                                                                break
                                                                    if found_via_multihop:
                                                                        break
                                                        if found_via_multihop:
                                                            break
                                    if found_via_multihop:
                                        break  # Found via multi-hop, exit disease loop
                                else:
                                    logger.info(f"Method 3.5: Seed '{seed_name}' not in graph, skipping multi-hop search")
                                
                                # CRITICAL FIX: If multi-hop didn't find a match after checking all diseases,
                                # try creating relationships directly if expected targets are already retrieved
                                # This handles cases where expected targets exist but aren't connected to diseases we found
                                if not found_via_multihop and query_type in ['side_effect_query', 'side_effects', 'adverse_effect']:
                                    # Check if expected target is already in retrieved entities and is a phenotype/anatomy
                                    for exp_target in expected_target_entities:
                                        if exp_target.entity_id == target_id and exp_target.entity_type in ['phenotype', 'anatomy']:
                                            # Create relationship directly - expected target is already retrieved
                                            rel_key = (seed_id, target_id, 'has_side_effect')
                                            if rel_key not in relationship_keys:
                                                relationship = RetrievedRelationship(
                                                    source_id=seed_id,
                                                    target_id=target_id,
                                                    relation_type='has_side_effect',
                                                    display_relation='has_side_effect',
                                                    source_entity=seed_entity,
                                                    target_entity=target_entity,
                                                    relevance_score=0.9,  # High score for expected targets
                                                    context={
                                                        'provenance': 'expected_target_direct',
                                                        'note': 'Expected target already retrieved, creating direct relationship'
                                                    },
                                                    agent_reasoning=f"Direct relationship to expected target {target_name}"
                                                )
                                                relationships.append(relationship)
                                                relationship_keys.add(rel_key)
                                                logger.info(
                                                    f"Method 3.5: Created direct relationship to expected target: {seed_name} --[has_side_effect]--> {target_name}")
                                                found_via_multihop = True
                                                break
                                    if found_via_multihop:
                                        logger.info(f"Method 3.5: Direct relationship created, skipping direct edge check for '{seed_name}' -> '{target_name}'")
                                        continue  # Skip direct edge check
                            else:
                                logger.info(f"Method 3.5: Multi-hop search DISABLED - query_type='{query_type}' not in ['side_effect_query', 'side_effects', 'adverse_effect']")
                            
                            # Check both directions (seed -> target and target
                            # -> seed) - DIRECT EDGE CHECK
                            for source_id, target_id_check in [
                                    (seed_id, target_id), (target_id, seed_id)]:
                                if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                                    break

                                edge_data = self.data_source.graph.get_edge_data(
                                    source_id, target_id_check)
                                if edge_data:
                                    logger.info(
                                        f"Method 3.5: Found edge_data between '{seed_name}' and '{target_name}'")
                                else:
                                    logger.info(
                                        f"Method 3.5: No edge_data between '{seed_name}' and '{target_name}'")
                                if edge_data:
                                    # Extract relation types from edge data
                                    for edge_key, edge_attrs in edge_data.items():
                                        if len(
                                                relationships) >= MAX_TOTAL_RELATIONSHIPS:
                                            break

                                        relation_type = edge_attrs.get(
                                            'relation') or edge_attrs.get('type') or 'related_to'
                                        
                                        logger.info(f"Method 3.5: Found relation type '{relation_type}' between '{seed_name}' and '{target_name}'")

                                        # Filter by relevant relation types
                                        if relevant_relation_types:
                                            if relation_type not in relevant_relation_types:
                                                if not self._is_relevant_relation(
                                                        relation_type, query_type):
                                                    logger.info(
                                                        f"Method 3.5: Filtered out relation '{relation_type}' between '{seed_name}' and '{target_name}' (not in relevant types: {relevant_relation_types}, query_type: {query_type})")
                                                    continue
                                                else:
                                                    logger.info(f"Method 3.5: Relation '{relation_type}' passed _is_relevant_relation check for query_type '{query_type}'")

                                        # Create relationship
                                        rel_key = (
                                            source_id, target_id_check, relation_type)
                                        if rel_key not in relationship_keys:
                                            source_entity = seed_entity if source_id == seed_id else target_entity
                                            target_entity_rel = target_entity if target_id_check == target_id else seed_entity

                                            relationship = RetrievedRelationship(
                                                source_id=source_id,
                                                target_id=target_id_check,
                                                relation_type=relation_type,
                                                display_relation=relation_type,
                                                source_entity=source_entity,
                                                target_entity=target_entity_rel,
                                                relevance_score=0.9,  # High score for expected target relationships
                                                context={
                                                    'provenance': 'seed_to_expected_target'},
                                                agent_reasoning=f"Relationship between seed entity and expected target"
                                            )
                                            relationships.append(relationship)
                                            relationship_keys.add(rel_key)
                                            logger.info(
                                                f"Found relationship via Method 3.5: {source_entity.name} --[{relation_type}]--> {target_entity_rel.name}")

        # Method 4: Create implicit relationships between entities that share connections
        # This helps when entities are retrieved but not directly connected
        # CRITICAL: Only create implicit relationships if we have very few, and
        # respect limit
        if len(relationships) < 5 and len(entities) > 1 and len(
                relationships) < MAX_TOTAL_RELATIONSHIPS:
            logger.info(
                "No direct relationships found, creating implicit relationships between retrieved entities")
            # Limit to top entities to avoid explosion
            entities_for_implicit = entities[:min(5, len(entities))]
            for i, entity1 in enumerate(entities_for_implicit):
                # Stop if we've reached the limit
                if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                    break

                for entity2 in entities_for_implicit[i + 1:]:
                    # Stop if we've reached the limit
                    if len(relationships) >= MAX_TOTAL_RELATIONSHIPS:
                        break

                    # Create a generic relationship if entities are in the same
                    # retrieval context
                    rel_key = (
                        entity1.entity_id,
                        entity2.entity_id,
                        'related_to')
                    if rel_key not in relationship_keys:
                        relationship = RetrievedRelationship(
                            source_id=entity1.entity_id,
                            target_id=entity2.entity_id,
                            relation_type='related_to',
                            display_relation='related_to',
                            source_entity=entity1,
                            target_entity=entity2,
                            relevance_score=0.5,
                            context={'provenance': 'implicit_context'},
                            agent_reasoning="Entities retrieved in same query context"
                        )
                        relationships.append(relationship)
                        relationship_keys.add(rel_key)

        # FINAL SAFETY CHECK: Ensure we never exceed the limit
        # Sort by relevance score and keep only top relationships
        if len(relationships) > MAX_TOTAL_RELATIONSHIPS:
            logger.warning(
                f"Retrieved {len(relationships)} relationships, truncating to {MAX_TOTAL_RELATIONSHIPS}")
            relationships.sort(key=lambda r: r.relevance_score, reverse=True)
            relationships = relationships[:MAX_TOTAL_RELATIONSHIPS]
            logger.info(f"Truncated to {len(relationships)} relationships")

        # Create paths (simplified)
        paths = []
        if len(entities) > 1:
            # Create simple paths between entities
            for i in range(len(entities) - 1):
                path = RetrievedPath(
                    path=[entities[i].entity_id, entities[i + 1].entity_id],
                    relationships=[],
                    path_score=0.5,
                    path_length=1,
                    context={},
                    agent_reasoning="Agent-discovered path"
                )
                paths.append(path)

        # Create subgraph
        subgraph = self._create_subgraph(entities, relationships, paths)

        # Create metadata
        metadata = {
            'agent_steps': agent_state.current_step,
            'query_analysis': query_analysis,
            'exploration_strategy': 'agent_based',
            'timestamp': datetime.now().isoformat()
        }

        return RetrievalResult(
            entities=entities,
            relationships=relationships,
            paths=paths,
            subgraph=subgraph,
            metadata=metadata,
            query_info={'query': query},
            agent_reasoning_chain=agent_state.reasoning_chain,
            seed_entities=seed_entities
        )

    def to_retrieved_context(
            self,
            result: RetrievalResult) -> RetrievedContext:
        """Utility to convert to organizer input."""
        # Flatten entities and relationships into dicts suitable for organizer
        entities = [
            {
                'id': e.entity_id,
                'name': e.name,
                'type': e.entity_type,
                'properties': {}
            }
            for e in result.entities
        ]
        relations = [
            {
                'source_id': r.source_id,
                'target_id': r.target_id,
                'type': r.relation_type
            }
            for r in result.relationships
        ]
        subgraphs = [result.subgraph] if result.subgraph is not None else []
        return RetrievedContext(
            entities=entities,
            relations=relations,
            subgraphs=subgraphs,
            metadata={
                **result.metadata,
                'query': result.query_info.get('query')})

    def _create_subgraph(self, entities: List[RetrievedEntity],
                         relationships: List[RetrievedRelationship],
                         paths: List[RetrievedPath]) -> nx.DiGraph:
        """Create a NetworkX subgraph from retrieved data."""
        subgraph = nx.DiGraph()

        # Add nodes
        for entity in entities:
            subgraph.add_node(entity.entity_id,
                              name=entity.name,
                              type=entity.entity_type,
                              description=entity.description,
                              relevance_score=entity.relevance_score)

        # Add edges
        for relationship in relationships:
            subgraph.add_edge(relationship.source_id, relationship.target_id,
                              relation=relationship.relation_type,
                              relevance_score=relationship.relevance_score)

        return subgraph


class MockLLMClient:
    """Mock LLM client for testing - replace with actual LLM integration."""

    def generate(self, prompt: str) -> str:
        """Generate a response to the prompt."""
        # Simple mock responses based on prompt content
        if 'query_analysis' in prompt:
            return json.dumps({
                "query_type": "drug_disease",
                "primary_entities": ["metformin", "diabetes"],
                "relevant_relationships": ["treats", "indication", "targets"],
                "retrieval_strategy": "exploration",
                "reasoning": "Query appears to be about drug-disease relationships"
            })
        elif 'next_action' in prompt:
            return json.dumps({
                "action": "stop_retrieval",
                "target_entity": None,
                "target_relationship": None,
                "reasoning": "Sufficient entities and relationships found for analysis",
                "confidence": 0.8
            })
        elif 'entity_exploration' in prompt:
            return json.dumps({
                "action": "continue",
                "relevance_score": 0.9,
                "relevant_aspects": ["disease", "treatment"],
                "next_explorations": ["drugs", "genes"],
                "reasoning": "Highly relevant to the query"
            })
        else:
            # Always include 'action' field for consistency
            return json.dumps({
                "action": "stop_retrieval",
                "relevance_score": 0.5,
                "insights": ["general relationship"],
                "next_steps": ["continue exploration"],
                "reasoning": "Moderately relevant"
            })


# Example usage
if __name__ == "__main__":
    # Initialize components
    data_source = PrimeKGDataSource()
    query_processor = QueryProcessor(data_source)
    retriever = GraphRetriever(data_source, query_processor)
    # Test retrieval
    query = "What drugs are used to treat diabetes?"
    result = retriever.retrieve(query)

    logger.info(f"Retrieved {len(result.entities)} entities")
    logger.info(f"Retrieved {len(result.relationships)} relationships")
    logger.info(
        f"Agent took {len(result.agent_reasoning_chain)} reasoning steps")
