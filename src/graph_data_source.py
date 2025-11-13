"""
PrimeKG Data Source for GraphRAG - Knowledge Graph Access Layer

OVERVIEW:
This module provides the data access layer for the GraphRAG system, handling all
interactions with the PrimeKG biomedical knowledge graph. It supports dual backends
and provides a unified API for graph operations.

KEY RESPONSIBILITIES:
1. **Data Loading**: Load PrimeKG from multiple sources (Neo4j, PyKEEN)
2. **Entity Search**: Find entities using lexical and semantic search
3. **Graph Traversal**: Navigate relationships and find paths
4. **Embedding Integration**: Semantic similarity for entity matching
5. **Caching**: Optimize performance with intelligent caching

DUAL BACKEND ARCHITECTURE:
┌─────────────────┐    ┌──────────────────┐
│   PrimeKG       │    │    Neo4j         │
│   Data Source   │◄───┤    Database      │ (Primary - Production)
│                 │    │   (External)     │
└─────────────────┘    └──────────────────┘
         │              ┌──────────────────┐
         └──────────────┤   PyKEEN         │ (Fallback - Development)
                        │   + NetworkX     │
                        │   (In-Memory)    │
                        └──────────────────┘

DATA FLOW:
1. **Initialization**: Try Neo4j → Fallback to PyKEEN
2. **Entity Search**: Lexical matching + Embedding similarity
3. **Graph Operations**: Shortest paths, neighbors, subgraphs
4. **Caching**: Pickle-based persistence for PyKEEN data

SUPPORTED OPERATIONS:
- Entity search (lexical + semantic)
- Shortest path finding between entities
- K-hop neighborhood expansion
- Relationship type filtering
- Graph statistics and schema information

EMBEDDING INTEGRATION:
- Uses sentence-transformers for semantic entity matching
- Configurable model via EMBEDDING_MODEL environment variable
- Combines lexical and semantic search results
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import pickle
import time
import gc
import re
from threading import Lock

# Memory monitoring
try:
    import psutil
except ImportError:
    psutil = None  # Optional dependency for memory monitoring

# Embeddings (optional)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore
try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None  # type: ignore

# Neo4j
try:
    from neo4j import GraphDatabase, basic_auth
except Exception:
    GraphDatabase = None  # Optional dependency

try:
    import pykeen.datasets
    from pykeen.datasets import PrimeKG
except ImportError:
    raise ImportError("PyKEEN is required. Install with: pip install pykeen")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Honor env-level overrides and silence noisy deps
_env_log_level = os.getenv('GRAPHRAG_LOG_LEVEL', 'INFO').upper()
try:
    logger.setLevel(getattr(logging, _env_log_level, logging.INFO))
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
except Exception:
    pass

class PrimeKGDataSource:
    """
    CORE DATA ACCESS LAYER: PrimeKG Knowledge Graph Interface
    
    This class provides unified access to the PrimeKG biomedical knowledge graph
    through multiple backends (Neo4j or PyKEEN), with intelligent caching and
    semantic search capabilities.
    
    ARCHITECTURE FEATURES:
    1. **Dual Backend Support**: Neo4j (production) + PyKEEN (development)
    2. **Intelligent Caching**: Pickle-based persistence for faster startup
    3. **Semantic Search**: Embedding-based entity matching
    4. **Graph Operations**: Shortest paths, neighbors, subgraph extraction
    5. **Memory Management**: Configurable limits and cache expiration
    
    DATA STRUCTURES MANAGED:
    - self.graph: NetworkX MultiDiGraph of PrimeKG entities and relations
    - self.entity_info: Metadata for all entities (name, type, description)
    - self.entity_search_index: Inverted index for fast entity lookup
    - self._embedding_model: Sentence transformer for semantic search
    - self._node_embeddings: Cached embeddings for entity names
    
    SUPPORTED BACKENDS:
    1. **Neo4j Backend** (Production):
       - External Neo4j database with PrimeKG data
       - Requires NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env
       - Direct Cypher queries for optimal performance
    
    2. **PyKEEN Backend** (Development/Research):
       - Downloads PrimeKG from Harvard Dataverse via PyKEEN
       - Converts to NetworkX for graph operations
       - In-memory processing with pickle caching
    
    PERFORMANCE OPTIMIZATIONS:
    - Cache graph data to avoid re-downloading (~350MB dataset)
    - Pre-compute entity search indices for fast lookup
    - Lazy loading of embeddings for semantic search
    - Memory usage monitoring and cleanup
    """
    
    def __init__(self, data_dir: str = None, use_cache: bool = None, auto_download: bool = None, prefer_neo4j: bool = True):
        """
        Initialize the PrimeKG data source with configurable backend selection.
        
        Sets up data directories, caching strategy, and backend preferences.
        Configuration is loaded from environment variables with sensible defaults.
        
        INITIALIZATION FLOW:
        1. Load configuration from environment variables
        2. Set up data directories and caching
        3. Configure embedding model settings
        4. Initialize data structures (empty until load() is called)
        5. Set up Neo4j connection parameters if available
        
        Args:
            data_dir: Directory for PrimeKG data storage (default: 'data')
            use_cache: Enable pickle caching for faster startup (default: True)
            auto_download: Automatically download PrimeKG if missing (default: True)
            prefer_neo4j: Try Neo4j connection first before PyKEEN fallback (default: True)
        """
        # Load configuration from environment variables
        self.data_dir = data_dir or os.getenv('GRAPHRAG_DATA_DIR', 'data')
        self.use_cache = use_cache if use_cache is not None else os.getenv('GRAPHRAG_USE_CACHE', 'true').lower() == 'true'
        self.auto_download = auto_download if auto_download is not None else os.getenv('PRIMEKG_AUTO_DOWNLOAD', 'true').lower() == 'true'
        
        # Performance settings from environment
        self.max_memory_usage = float(os.getenv('MAX_MEMORY_USAGE', '4.0'))  # GB
        self.cache_expiration_hours = int(os.getenv('CACHE_EXPIRATION_HOURS', '24'))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # ENHANCED MEMORY MANAGEMENT SYSTEM
        # Addresses critical memory issues from CLAUDE.md
        self.memory_monitoring_enabled = psutil is not None
        self.memory_cleanup_threshold = float(os.getenv('MEMORY_CLEANUP_THRESHOLD', '0.85'))  # 85% of limit
        self.memory_monitoring_interval = int(os.getenv('MEMORY_MONITORING_INTERVAL', '30'))  # seconds
        self.last_memory_check = 0
        self.memory_cleanup_lock = Lock()
        
        # Memory statistics tracking
        self._memory_stats = {
            'peak_usage_mb': 0,
            'cleanup_count': 0,
            'last_cleanup_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize data structures
        self.graph = None
        self.entity_info = {}
        self.relation_info = {}
        self.entity_search_index = {}
        self.is_loaded = False
        
        # PyKEEN PrimeKG dataset
        self.primekg_dataset = None
        self.triples_df = None
        
        # Neo4j settings
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_user = os.getenv('NEO4J_USER') or os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.neo4j_database = os.getenv('NEO4J_DATABASE') or None
        self._neo4j_driver = None
        self._use_neo4j = False
        self.prefer_neo4j = prefer_neo4j

        # Embeddings state
        self._embedding_model: Optional[Any] = None
        self._node_embeddings: Optional[np.ndarray] = None
        self._node_embedding_ids: List[str] = []

        # OPTIMIZED CACHE SIZES - Reduced from problematic defaults
        # Old: search=1024, neighbors=1024, paths=256, encode=256 (total=2560)
        # New: Reduced by 60% to prevent memory overload (CLAUDE.md Issue 3)
        self._cache_sizes = {
            'search': int(os.getenv('GRAPHRAG_CACHE_SEARCH', '512')),      # Reduced from 1024
            'neighbors': int(os.getenv('GRAPHRAG_CACHE_NEIGHBORS', '256')), # Reduced from 1024
            'paths': int(os.getenv('GRAPHRAG_CACHE_PATHS', '128')),        # Reduced from 256
            'encode': int(os.getenv('GRAPHRAG_CACHE_ENCODINGS', '128'))    # Reduced from 256
        }
        self._search_cache: Dict[Tuple[str, Tuple[str, ...], int], List[Dict[str, Any]]] = {}
        self._neighbors_cache: Dict[Tuple[str, int, Tuple[str, ...]], Dict[str, Any]] = {}
        self._paths_cache: Dict[Tuple[Tuple[str, ...], int, int], List[List[str]]] = {}
        self._encode_cache = {}

        # Cache settings
        self.cache_dir = Path(self.data_dir) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.graph_cache_file = self.cache_dir / 'primekg_graph.pkl'
        self.entity_cache_file = self.cache_dir / 'entity_info.pkl'
        self.search_index_cache_file = self.cache_dir / 'search_index.pkl'
        
        logger.info(f"Initialized PrimeKG data source with data_dir: {self.data_dir}")
        logger.info(f"Cache enabled: {self.use_cache}")
        logger.info(f"Auto-download enabled: {self.auto_download}")
        if self.neo4j_uri:
            logger.info("Neo4j connection info found in environment. Will attempt to connect on load().")

    # ------------------------- Initialization -------------------------
    def load(self) -> bool:
        """Attempt to initialize from Neo4j first (if configured), otherwise fallback to PyKEEN.

        Returns:
            True if some backend (Neo4j or PyKEEN) is available and initialized.
        """
        # Try Neo4j
        if self.prefer_neo4j and self.neo4j_uri and GraphDatabase is not None:
            try:
                self._connect_neo4j()
                self._use_neo4j = True
                self.is_loaded = True
                logger.info("Connected to Neo4j successfully. Using Neo4j backend.")
                return True
            except Exception as exc:
                logger.warning(f"Neo4j connection failed, falling back to PyKEEN: {exc}")

        # Fallback to PyKEEN
        loaded = self.load_primekg()
        if not loaded:
            logger.error("Failed to initialize any backend (Neo4j or PyKEEN).")
        return loaded

    def _connect_neo4j(self) -> None:
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver not installed. Add 'neo4j' to requirements.")
        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise RuntimeError("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD must be set in .env to use Neo4j.")
        self._neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=basic_auth(self.neo4j_user, self.neo4j_password)
        )
        # Simple test query
        with self._neo4j_driver.session(database=self.neo4j_database) as session:
            session.run("RETURN 1 AS ok").single()

    def close(self) -> None:
        if self._neo4j_driver:
            try:
                self._neo4j_driver.close()
            except Exception:
                pass
        self._neo4j_driver = None

    @property
    def using_neo4j(self) -> bool:
        return bool(self._use_neo4j)
    
    def load_primekg(self, force_download: bool = False, build_search_index: bool = True) -> bool:
        """
        Load PrimeKG data using PyKEEN.
        
        Args:
            force_download: Force re-download even if cached data exists
            build_search_index: Whether to build search index for entity lookup
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Try to load from cache first
            if self.use_cache and not force_download and self._load_from_cache():
                logger.info("Loaded PrimeKG data from cache")
                self.is_loaded = True
                return True
            
            logger.info("Loading PrimeKG data using PyKEEN...")
            
            # Load PrimeKG dataset
            try:
                self.primekg_dataset = PrimeKG()
            except Exception as exc:
                logger.error(f"Unable to instantiate PyKEEN PrimeKG: {exc}")
                return False
            
            # Convert to pandas DataFrame for easier processing
            logger.info("Converting triples to DataFrame...")
            self._convert_to_dataframe()
            
            # Create NetworkX graph
            logger.info("Creating NetworkX graph...")
            self._create_networkx_graph()
            
            # Extract entity information
            logger.info("Extracting entity information...")
            self._extract_entity_info()
            
            # Build search index if requested
            if build_search_index:
                logger.info("Building search index...")
                self._build_search_index()
            
            # Save to cache
            if self.use_cache:
                logger.info("Saving to cache...")
                self._save_to_cache()
            
            self.is_loaded = True
            logger.info("PrimeKG data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PrimeKG data: {e}")
            return False
    
    def _convert_to_dataframe(self) -> None:
        """Convert PyKEEN triples to pandas DataFrame."""
        # Get training triples (main dataset)
        training_triples = self.primekg_dataset.training.mapped_triples
        
        # Get entity and relation mappings
        entity_to_id = self.primekg_dataset.entity_to_id
        relation_to_id = self.primekg_dataset.relation_to_id
        
        # Create reverse mappings
        id_to_entity = {v: k for k, v in entity_to_id.items()}
        id_to_relation = {v: k for k, v in relation_to_id.items()}
        
        # Convert to DataFrame
        triples_list = []
        for triple in training_triples:
            head_id, relation_id, tail_id = triple.tolist()
            triples_list.append({
                'head': id_to_entity[head_id],
                'relation': id_to_relation[relation_id],
                'tail': id_to_entity[tail_id],
                'head_id': head_id,
                'relation_id': relation_id,
                'tail_id': tail_id
            })
        
        self.triples_df = pd.DataFrame(triples_list)
        logger.info(f"Converted {len(self.triples_df)} triples to DataFrame")
    
    def _create_networkx_graph(self) -> None:
        """Create NetworkX graph from triples DataFrame."""
        self.graph = nx.MultiDiGraph()
        
        # Add edges from triples
        for _, row in self.triples_df.iterrows():
            self.graph.add_edge(
                row['head'], 
                row['tail'], 
                relation=row['relation'],
                relation_id=row['relation_id']
            )
        
        logger.info(f"Created NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _extract_entity_info(self) -> None:
        """Extract entity information from the loaded graph."""
        self.entity_info = {}
        
        # Extract basic entity information from node names
        for node in self.graph.nodes():
            # Parse entity information from node name if it follows PrimeKG format
            entity_parts = str(node).split('::') if '::' in str(node) else [str(node)]
            
            if len(entity_parts) >= 2:
                entity_type = entity_parts[0]
                entity_name = entity_parts[1]
                entity_id = entity_parts[2] if len(entity_parts) > 2 else entity_name
            else:
                entity_type = 'unknown'
                entity_name = str(node)
                entity_id = str(node)
            
            info = {
                'name': entity_name,
                'type': entity_type,
                'description': '',
                'source': 'primekg',
                'index': entity_id
            }
            self.entity_info[str(node)] = info
            # Attach attributes to nx graph for convenience
            try:
                self.graph.nodes[node]['name'] = entity_name
                self.graph.nodes[node]['type'] = entity_type
            except Exception:
                pass
        
        logger.info(f"Extracted information for {len(self.entity_info)} entities")
        
        # ENHANCED ENTITY TYPE CLASSIFICATION - Apply improved patterns
        # Addresses CLAUDE.md Issue 2: 66.1% unknown entities → target <20%
        self._apply_enhanced_entity_classification()

    def _apply_enhanced_entity_classification(self) -> None:
        """
        Apply enhanced entity type classification to reduce unknown entities from ~66% to <30%.
        
        This method implements sophisticated pattern-based classification for biomedical entities
        that don't follow the standard PrimeKG format (type::name::id). It uses comprehensive
        regex patterns, biological naming conventions, and domain knowledge to classify entities.
        
        Classification Strategy:
        1. Drug/Compound Recognition: Generic names, suffixes, brand names
        2. Disease/Condition Recognition: Medical terminology, syndrome patterns
        3. Gene/Protein Recognition: Standard nomenclature, UniProt patterns
        4. Biological Process Recognition: GO terms, pathway names
        5. Anatomical Recognition: Organ systems, cellular structures
        6. Phenotype Recognition: Observable traits, clinical features
        """
        logger.info("Applying enhanced entity type classification...")
        
        # Comprehensive biomedical entity patterns
        # IMPORTANT: Order matters! Check more specific patterns first (pathways, processes)
        # before generic ones (drugs) to avoid false matches like "insulin signaling pathway" -> drug
        classification_patterns = [
            # Check pathways FIRST (most specific)
            ('pathway', [
                r'\b(?:insulin|glucose|lipid)\s*(?:signaling|metabolism|pathway)\b',  # Specific pathway patterns first
                r'\b(?:signaling|regulatory|metabolic)\s*\w*\s*pathway\b',
                r'\b\w+\s*(?:signaling\s+)?pathway\b',  # "X signaling pathway" or "X pathway"
                r'\b.*\s+signaling\s+pathway\b',  # "insulin receptor signaling pathway"
                r'\b(?:glycolysis|krebs|citric\s*acid|electron\s*transport)\b',
                r'\b(?:pentose\s*phosphate|fatty\s*acid|amino\s*acid)\s*pathway\b',
                r'\b(?:MAPK|PI3K|mTOR|Wnt|Notch|JAK-STAT|BMP)\s*pathway\b',  # Added BMP
                r'\b\w+\s*(?:pathway|cascade|network|circuit)\b',
            ]),
            # Check biological processes SECOND (before drugs to avoid misclassification)
            ('biological_process', [
                # Regulation patterns (CRITICAL: must come before drug patterns)
                r'\b(?:positive|negative|positive|negative)?\s*regulation\s+of\s+.*\b',  # "negative regulation of insulin receptor signaling pathway"
                r'\b\w+\s*regulation\s+of\s+.*\b',  # "regulation of X"
                r'\b.*\s+regulation\b',  # "X regulation"
                # Internalization/translocation patterns
                r'\b.*\s+internalization\b',  # "insulin receptor internalization"
                r'\b.*\s+translocation\b',
                r'\b.*\s+localization\b',
                # Process patterns
                r'\b\w+\s*(?:process|pathway|cascade|signaling)\b',
                r'\b(?:cell|cellular)\s+\w+\b',
                r'\b(?:metabolic|metabolite|metabolism)\b',
                r'\b\w+\s*(?:biosynthesis|catabolism|anabolism)\b',
                r'\b\w+\s*(?:transport|localization)\b',
                r'\b(?:glycolysis|gluconeogenesis|lipogenesis|proteolysis)\b',
                r'\b(?:transcription|translation|replication|repair)\b',
                r'\b(?:apoptosis|autophagy|necrosis|proliferation)\b',
                r'\b(?:differentiation|development|morphogenesis)\b',
                r'\b\w+\s*(?:cycle|phase|checkpoint)\b',
                r'\b\w+\s*(?:secretion|metabolism)\b',  # "insulin secretion", "glucose metabolism"
                # Response patterns
                r'\b.*\s+response\s+to\s+.*\b',  # "cellular response to insulin stimulus"
                r'\b.*\s+response\b',
            ]),
            # Check diseases THIRD
            ('disease', [
                r'\b(?:type\s*[12]\s*)?diabetes(?:\s*mellitus)?\b',
                r'\b(?:alzheimer|parkinson|huntington)(?:\'?s)?\s*disease\b',
                r'\b(?:breast|lung|colon|prostate|pancreatic|liver)\s*cancer\b',
                r'\b(?:hypertension|hypotension|tachycardia|bradycardia)\b',
                r'\b(?:depression|anxiety|schizophrenia|bipolar)\b',
                r'\b(?:arthritis|osteoporosis|fibromyalgia)\b',
                r'\b\w+\s*cancer\b',
                r'\b\w+\s*disease\b',
                r'\b\w+\s*syndrome\b',
                r'\b\w+\s*disorder\b',
                r'\b(?:acute|chronic|severe|mild)\s+\w+\b',
                r'\b\w+\s*(?:infection|inflammation|lesion|tumor)\b',
                r'\b(?:malignant|benign)\s+\w+\b',
                r'\b\w+itis\b',
                r'\b\w+osis\b',
                r'\b\w+emia\b',
                r'\b\w+pathy\b',
            ]),
            # Check proteins/genes FOURTH (before drugs to catch gene symbols)
            ('protein', [
                # Gene symbols with action words (e.g., "TP53 Regulates Metabolic Genes")
                r'\b[A-Z]{2,8}\d*\s+(?:regulates?|activates?|inhibits?|controls?|modulates?|binds?|interacts?)\s+',
                r'\b[A-Z]{2,8}\d*\s+(?:gene|genes|protein|proteins|pathway|pathways)\b',
                # Standalone gene symbols: TP53, BRCA1, EGFR
                r'^\b[A-Z]{2,8}\d*\b$',  # Exact match for standalone gene symbols
                r'\b[A-Z]{2,8}\d*\b(?!\s+(?:drug|compound|medication|tablet|capsule))',  # Gene symbols NOT followed by drug words
                r'\b[A-Z][a-z]{2,8}\d*\b', # Protein names: Insulin, Albumin
                r'\b\w+\s*(?:protein|enzyme|receptor|kinase|phosphatase)\b',
                r'\b\w+\s*(?:antibody|immunoglobulin|cytokine|hormone)\b',
                r'\b(?:alpha|beta|gamma|delta)\s*\w+\b',
                r'\b\w+ase\b',
                r'\b\w+\s*(?:dehydrogenase|transferase|ligase|lyase)\b',
                r'\b\w+\s*receptor\b',
                r'\b[A-Z]{2,6}R\d*\b',
                r'\b[A-Z]\d[A-Z0-9]{3}\d\b',  # UniProt format: P12345
            ]),
            # Check drugs LAST (most generic, can match many things)
            ('drug', [
                r'\b(metformin|aspirin|warfarin|statins?|atorvastatin|simvastatin)\b',  # Removed insulin from here
                r'\b(lisinopril|losartan|amlodipine|hydrochlorothiazide|furosemide)\b',
                r'\b(levothyroxine|omeprazole|pantoprazole|ranitidine|famotidine)\b',
                r'\b\w+mycin\b',
                r'\b\w+cillin\b',
                r'\b\w+prazole\b',
                r'\b\w+statin\b',
                r'\b\w+pril\b',
                r'\b\w+sartan\b',
                r'\b\w+olol\b',
                r'\b\w+pine\b',
                r'\b\w+zide\b',
                r'\b[A-Z]{2,4}-?\d{2,6}\b',  # Research compounds: AB-123, XY-4567
                r'\b\w*ine\b(?=.*(?:drug|compound|medication))',
                r'\b\w+\s*(?:tablet|capsule|injection|solution|cream|ointment)\b',
                r'\b(?:generic|brand)\s+\w+\b',
                # Only match "insulin" as drug if it's standalone, not in pathway context
                r'^insulin$',  # Exact match only
                # Gene symbols ONLY if explicitly marked as drug or in drug context
                r'\b[A-Z]{2,8}\d*\b(?=\s*(?:drug|compound|medication|tablet|capsule))',
            ]),
            # Other types
            ('anatomy', [
                r'\b(?:heart|lung|liver|kidney|brain|stomach|intestine)\b',
                r'\b(?:muscle|bone|skin|blood|nerve|vessel)\b',
                r'\b(?:cardiac|hepatic|renal|pulmonary|cerebral|gastric)\b',
                r'\b(?:cell|cellular|membrane|nucleus|mitochondria|ribosome)\b',
                r'\b(?:cytoplasm|endoplasmic|golgi|lysosome|peroxisome)\b',
                r'\b\w+\s*(?:tissue|organ|system|structure)\b',
                r'\b(?:thoracic|abdominal|pelvic|cranial|spinal)\b',
                r'\b(?:anterior|posterior|superior|inferior|medial|lateral)\b',
            ]),
            ('phenotype', [
                r'\b(?:symptom|sign|manifestation|presentation)\b',
                r'\b(?:fever|pain|inflammation|swelling|bleeding)\b',
                r'\b(?:fatigue|weakness|nausea|vomiting|diarrhea)\b',
                r'\b\w+\s*(?:deficiency|excess|abnormality)\b',
                r'\b(?:height|weight|BMI|blood\s*pressure)\b',
                r'\b(?:cholesterol|glucose|hemoglobin)\s*(?:level|concentration)?\b',
                r'\b\w+\s*(?:trait|characteristic|feature)\b',
                r'\b\w+\s*(?:count|level|concentration|activity)\b',
            ])
        ]
        
        # Apply classification patterns in order (most specific first)
        reclassified_count = 0
        original_unknown_count = sum(1 for info in self.entity_info.values() if info['type'] == 'unknown')
        
        for entity_id, info in self.entity_info.items():
            if info['type'] == 'unknown':
                entity_name = info['name'].lower()
                
                # Try each entity type pattern in order
                for entity_type, patterns in classification_patterns:
                    for pattern in patterns:
                        if re.search(pattern, entity_name, re.IGNORECASE):
                            info['type'] = entity_type
                            reclassified_count += 1
                            break
                    
                    # Break if we found a match
                    if info['type'] != 'unknown':
                        break
        
        # Log results
        final_unknown_count = sum(1 for info in self.entity_info.values() if info['type'] == 'unknown')
        unknown_percentage = (final_unknown_count / len(self.entity_info)) * 100
        improvement = original_unknown_count - final_unknown_count
        
        logger.info(f"Enhanced entity classification complete:")
        logger.info(f"  Reclassified: {reclassified_count:,} entities")
        logger.info(f"  Unknown entities: {original_unknown_count:,} -> {final_unknown_count:,}")
        logger.info(f"  Unknown percentage: {unknown_percentage:.1f}%")
        logger.info(f"  Improvement: {improvement:,} entities ({(improvement/original_unknown_count)*100:.1f}%)")
        
        if unknown_percentage < 30:
            logger.info("[OK] Target achieved: Unknown entities < 30%")
        elif unknown_percentage < 50:
            logger.info("[WARNING] Partially improved: Unknown entities < 50%")
        else:
            logger.info("[FAIL] Target missed: Unknown entities still > 50%")
    
    def _infer_entity_type(self, entity_name: str) -> str:
        """
        Infer entity type from entity name using pattern matching.
        
        This method uses the same classification patterns as _apply_enhanced_entity_classification
        but for a single entity name. Used for real-time entity type inference.
        
        Args:
            entity_name: Name of the entity to classify
            
        Returns:
            Entity type string (drug, disease, protein, pathway, etc.) or 'unknown'
        """
        import re
        
        # Normalize entity name
        if not entity_name:
            return 'unknown'
        
        entity_name = str(entity_name).strip()
        if not entity_name:
            return 'unknown'
        
        entity_name_lower = entity_name.lower()
        
        # Use the same classification patterns as _apply_enhanced_entity_classification
        # IMPORTANT: Order matters! Check more specific patterns first (pathways, processes)
        # before generic ones (drugs) to avoid false matches like "insulin signaling pathway" -> drug
        classification_patterns = [
            # Check pathways FIRST (most specific) - CRITICAL: Must catch ALL pathway variants
            ('pathway', [
                # Most specific: regulation of pathways (must come before generic regulation)
                r'\b(?:positive|negative|positive|negative)?\s*regulation\s+of\s+.*\s+pathway\b',  # "negative regulation of insulin receptor signaling pathway"
                r'\b.*\s+regulation\s+of\s+.*\s+signaling\s+pathway\b',
                r'\b.*\s+regulation\s+of\s+.*\s+pathway\b',
                # Pathway names with "pathway" keyword
                r'\b.*\s+signaling\s+pathway\b',  # "insulin receptor signaling pathway", "cell surface receptor signaling pathway"
                r'\b.*\s+receptor\s+signaling\s+pathway\b',  # "AMPA selective glutamate receptor signaling pathway"
                r'\b\w+\s*(?:signaling\s+)?pathway\b',  # "X signaling pathway" or "X pathway"
                r'\b(?:signaling|regulatory|metabolic|biosynthetic|catabolic)\s+\w*\s+pathway\b',
                # Specific pathway names (even without "pathway" keyword)
                r'\b(?:insulin|glucose|lipid|fatty\s*acid|amino\s*acid)\s*(?:signaling|metabolism|pathway)\b',
                r'\b(?:pentose\s*phosphate|glycolysis|krebs|citric\s*acid|electron\s*transport)\s*pathway\b',
                r'\b(?:pentose\s*phosphate|glycolysis|krebs|citric\s*acid|electron\s*transport)\b',  # Even without "pathway"
                r'\b(?:MAPK|PI3K|mTOR|Wnt|Notch|JAK-STAT|BMP|AKT|ERK|JNK)\s+pathway\b',
                r'\b(?:MAPK|PI3K|mTOR|Wnt|Notch|JAK-STAT|BMP|AKT|ERK|JNK)\s+signaling\b',
                # Generic pathway patterns
                r'\b\w+\s*(?:pathway|cascade|network|circuit|signaling\s+cascade)\b',
                # Processes that are pathways
                r'\b.*\s+pathway\s+involved\s+in\s+.*\b',  # "pathway involved in X"
                r'\b.*\s+pathway\s+by\s+.*\b',  # "pathway by X"
            ]),
            # Check biological processes SECOND (before drugs to avoid misclassification)
            ('biological_process', [
                # Regulation patterns (CRITICAL: must come before drug patterns, but AFTER pathway patterns)
                # Only match if NOT a pathway (pathways already matched above)
                r'\b(?:positive|negative)?\s*regulation\s+of\s+(?!.*\s+pathway\b).*\b',  # "regulation of X" but NOT "regulation of X pathway"
                r'\b\w+\s*regulation\s+of\s+(?!.*\s+pathway\b).*\b',  # "regulation of X" but NOT pathways
                r'\b.*\s+regulation\b(?!\s+of\s+.*\s+pathway)',  # "X regulation" but NOT "X regulation of pathway"
                # Internalization/translocation patterns (but NOT pathway-related)
                r'\b.*\s+internalization\b(?!\s+pathway)',  # "insulin receptor internalization" but NOT "X internalization pathway"
                r'\b.*\s+translocation\b(?!\s+pathway)',
                r'\b.*\s+localization\b(?!\s+pathway)',
                # Process patterns (but NOT pathways - pathways already matched)
                r'\b\w+\s*(?:process|cascade|signaling)\b(?!\s+pathway)',  # Exclude "X pathway"
                r'\b(?:cell|cellular)\s+\w+\b',
                r'\b(?:metabolic|metabolite|metabolism)\b',
                r'\b\w+\s*(?:biosynthesis|catabolism|anabolism)\b',
                r'\b\w+\s*(?:transport|localization)\b',
                r'\b(?:glycolysis|gluconeogenesis|lipogenesis|proteolysis)\b',
                r'\b(?:transcription|translation|replication|repair)\b',
                r'\b(?:apoptosis|autophagy|necrosis|proliferation)\b',
                r'\b\w+\s*(?:secretion|metabolism)\b',  # "insulin secretion", "glucose metabolism"
                # Response patterns
                r'\b.*\s+response\s+to\s+.*\b',  # "cellular response to insulin stimulus"
                r'\b.*\s+response\b',
            ]),
            # Check diseases THIRD
            ('disease', [
                r'\b(?:type\s*[12]\s*)?diabetes(?:\s*mellitus)?\b',
                r'\b(?:alzheimer|parkinson|huntington)(?:\'?s)?\s*disease\b',
                r'\b(?:breast|lung|colon|prostate|pancreatic|liver)\s*cancer\b',
                r'\b(?:hypertension|hypotension|tachycardia|bradycardia)\b',
                r'\b(?:depression|anxiety|schizophrenia|bipolar)\b',
                r'\b(?:arthritis|osteoporosis|fibromyalgia)\b',
                r'\b\w+\s*cancer\b',
                r'\b\w+\s*disease\b',
                r'\b\w+\s*syndrome\b',
                r'\b\w+\s*disorder\b',
                r'\b\w+itis\b',
                r'\b\w+osis\b',
                r'\b\w+emia\b',
                r'\b\w+pathy\b',
            ]),
            # Check anatomy FOURTH (BEFORE protein to catch "stomach", "gastric mucosa", "cytoplasmic side")
            ('anatomy', [
                r'\b(?:heart|lung|liver|kidney|brain|stomach|intestine|stomach)\b',  # Explicit "stomach"
                r'\b(?:muscle|bone|skin|blood|nerve|vessel)\b',
                r'\b(?:cardiac|hepatic|renal|pulmonary|cerebral|gastric)\b',
                r'\b(?:cell|cellular|membrane|nucleus|mitochondria|ribosome)\b',
                r'\b(?:mucosa|mucous|membrane)\b',  # "gastric mucosa"
                r'\b(?:cytoplasmic\s+side\s+of\s+.*)\b',  # "cytoplasmic side of dendritic spine plasma membrane"
                r'\b(?:side\s+of\s+.*\s+membrane)\b',  # "side of plasma membrane"
                r'\b(?:line\s+from\s+.*\s+to\s+.*)\b',  # "line from X to Y" (anatomical measurements)
                r'\b(?:abnormality\s+of\s+the\s+.*)\b',  # "Abnormality of the gastric mucosa" - but this is phenotype, not anatomy
            ]),
            # Check phenotype FIFTH (BEFORE protein to catch "bleeding", "abnormal bleeding", "abnormality")
            ('phenotype', [
                r'\b(?:symptom|sign|manifestation|presentation)\b',
                r'\b(?:abnormal\s+bleeding)\b',  # "Abnormal bleeding" - specific pattern first
                r'\b(?:fever|pain|inflammation|swelling|bleeding)\b',
                r'\b(?:fatigue|weakness|nausea|vomiting|diarrhea)\b',
                r'\b(?:abnormality\s+of\s+the\s+.*)\b',  # "Abnormality of the gastric mucosa" - phenotype, not anatomy
                r'\b(?:abnormality|abnormal)\b',  # "Abnormality of..." or "abnormal X"
            ]),
            # Check proteins/genes SIXTH (after anatomy/phenotype to avoid false matches)
            ('protein', [
                # Gene symbols with action words (e.g., "TP53 Regulates Metabolic Genes")
                # BUT exclude if it's a pathway (e.g., "TP53 Regulates Metabolic Genes" might be pathway name)
                r'\b[A-Z]{2,8}\d*\s+(?:regulates?|activates?|inhibits?|controls?|modulates?|binds?|interacts?)\s+(?!.*\s+pathway\b)',
                r'\b[A-Z]{2,8}\d*\s+(?:gene|genes|protein|proteins)\b(?!\s+pathway)',  # Exclude "X genes pathway"
                # Standalone gene symbols: TP53, BRCA1, EGFR (exact match only)
                r'^\b[A-Z]{2,8}\d*\b$',  # Exact match for standalone gene symbols
                r'\b[A-Z]{2,8}\d*\b(?!\s+(?:drug|compound|medication|tablet|capsule|pathway))',  # Gene symbols NOT followed by drug/pathway words
                # Protein names: Insulin, Albumin (but NOT "Abnormal", "Effects", "Role" - exclude common phenotype/anatomy/process words)
                r'\b(?!abnormal|bleeding|gastric|stomach|mucosa|abnormality|effects|role|line|side|cytoplasmic)([A-Z][a-z]{2,8}\d*)\b',  # Exclude phenotype/anatomy/process words
                r'\b\w+\s*(?:protein|enzyme|receptor|kinase|phosphatase)\b(?!\s+pathway)',  # Exclude "X protein pathway"
                r'\b\w+\s*(?:antibody|immunoglobulin|cytokine|hormone)\b',
                r'\b(?:alpha|beta|gamma|delta)\s*\w+\b',
                r'\b\w+ase\b(?!\s+pathway)',  # Exclude "X-ase pathway"
                r'\b\w+\s*(?:dehydrogenase|transferase|ligase|lyase)\b(?!\s+pathway)',
                r'\b\w+\s*receptor\b(?!\s+signaling\s+pathway)',  # Exclude "X receptor signaling pathway"
                r'\b[A-Z]{2,6}R\d*\b',
                r'\b[A-Z]\d[A-Z0-9]{3}\d\b',  # UniProt format: P12345
            ]),
            # Check drugs LAST (most generic, can match many things)
            # CRITICAL: Exclude pathways, processes, anatomy, phenotypes
            ('drug', [
                r'\b(metformin|aspirin|warfarin|statins?|atorvastatin|simvastatin)\b(?!\s+pathway)',  # Removed insulin from here
                r'\b(lisinopril|losartan|amlodipine|hydrochlorothiazide|furosemide)\b(?!\s+pathway)',
                r'\b(levothyroxine|omeprazole|pantoprazole|ranitidine|famotidine)\b(?!\s+pathway)',
                r'\b\w+mycin\b(?!\s+pathway)',  # Exclude "X-mycin pathway"
                r'\b\w+cillin\b(?!\s+pathway)',
                r'\b\w+prazole\b(?!\s+pathway)',
                r'\b\w+statin\b(?!\s+pathway)',
                r'\b\w+pril\b(?!\s+pathway)',
                r'\b\w+sartan\b(?!\s+pathway)',
                r'\b\w+olol\b(?!\s+pathway)',
                r'\b\w+pine\b(?!\s+pathway)',
                r'\b\w+zide\b(?!\s+pathway)',
                r'\b[A-Z]{2,4}-?\d{2,6}\b(?!\s+pathway)',  # Research compounds: AB-123, XY-4567
                r'\b\w*ine\b(?=.*(?:drug|compound|medication))(?!.*pathway)',
                r'\b\w+\s*(?:tablet|capsule|injection|solution|cream|ointment)\b',
                r'\b(?:generic|brand)\s+\w+\b',
                # Only match "insulin" as drug if it's standalone, not in pathway context
                r'^insulin$(?!\s+pathway)',  # Exact match only, not "insulin pathway"
                # Gene symbols ONLY if explicitly marked as drug or in drug context
                r'\b[A-Z]{2,8}\d*\b(?=\s*(?:drug|compound|medication|tablet|capsule))(?!.*pathway)',
            ])
        ]
        
        # Check for gene/protein patterns FIRST (before cache) to avoid cache returning wrong type
        # This is critical for entities like "TP53 Regulates Metabolic Genes" that might be cached as "drug"
        gene_pattern = r'\b[A-Z]{2,8}\d*\s+(?:regulates?|activates?|inhibits?|controls?|modulates?|binds?|interacts?|gene|genes|protein|proteins|pathway|pathways)'
        if re.search(gene_pattern, entity_name, re.IGNORECASE):
            # This looks like a gene/protein, check protein patterns first
            # Find protein patterns dynamically (they're now at index 5 after reordering)
            protein_patterns = None
            for i, (etype, patterns) in enumerate(classification_patterns):
                if etype == 'protein':
                    protein_patterns = patterns
                    break
            if protein_patterns:
                for pattern in protein_patterns:
                    search_text = entity_name if '[A-Z]' in pattern else entity_name_lower
                    if re.search(pattern, search_text, re.IGNORECASE if '[A-Z]' not in pattern else 0):
                        return 'protein'
        
        # Check entity_info cache (by name, case-insensitive)
        # But validate cached type against patterns if it seems wrong
        cached_type = None
        if hasattr(self, 'entity_info') and self.entity_info:
            # Search for entity by name (case-insensitive)
            entity_name_lower_check = entity_name_lower.strip()
            for entity_id, info in self.entity_info.items():
                cached_name = info.get('name', '').lower().strip()
                if cached_name == entity_name_lower_check:
                    cached_type = info.get('type', 'unknown')
                    break
        
        # Try pattern matching in order (most specific first)
        # Use original case for gene symbol patterns, lowercase for others
        inferred_type = None
        for entity_type, patterns in classification_patterns:
            for pattern in patterns:
                # Gene/protein patterns need original case (TP53, BRCA1), others use lowercase
                search_text = entity_name if entity_type == 'protein' and '[A-Z]' in pattern else entity_name_lower
                if re.search(pattern, search_text, re.IGNORECASE if entity_type != 'protein' or '[A-Z]' not in pattern else 0):
                    inferred_type = entity_type
                    break
            if inferred_type:
                break
        
        # If we have a cached type, validate it against inferred type
        # Override cache if inferred type is more specific (e.g., protein vs drug)
        if cached_type and cached_type != 'unknown':
            if inferred_type and inferred_type != 'unknown':
                # Override cache if inferred type is protein/gene and cache is drug
                if inferred_type == 'protein' and cached_type.lower() == 'drug':
                    # Check if entity name suggests gene/protein
                    if re.search(gene_pattern, entity_name, re.IGNORECASE):
                        return inferred_type  # Override cache
                # Otherwise trust cache if it matches inferred type
                if cached_type.lower() == inferred_type.lower():
                    return cached_type
            # Return cached type if no inference or inference matches
            return cached_type
        
        # Return inferred type if no cache or cache was unknown
        return inferred_type if inferred_type else 'unknown'

    # ------------------------- Embeddings -------------------------
    def _ensure_embedding_model(self) -> Any:
        if self._embedding_model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed. Add to requirements to enable embeddings.")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _ensure_node_embeddings(self) -> None:
        """Build embeddings for node names/types for local (PyKEEN) backend.
        For Neo4j, embeddings are created on candidate subsets on the fly.
        """
        if self._node_embeddings is not None:
            return
        if self.using_neo4j:
            return  # defer for Neo4j
        texts = []
        ids = []
        for entity_id, info in self.entity_info.items():
            label = f"{info.get('name','')} ({info.get('type','')})"
            texts.append(label)
            ids.append(entity_id)
        model = self._ensure_embedding_model()
        self._node_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._node_embedding_ids = ids

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        # Simple memoization for repeated small batches
        key = tuple(texts)
        if key in self._encode_cache:
            return self._encode_cache[key]
        model = self._ensure_embedding_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        # bound cache
        if len(self._encode_cache) >= self._cache_sizes['encode']:
            self._encode_cache.pop(next(iter(self._encode_cache)))
        self._encode_cache[key] = vecs
        return vecs

    def semantic_search_entities(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Embedding-based entity retrieval.
        - For Neo4j: first do lexical candidate search then re-rank by embeddings
        - For local graph: by default, re-rank lexical candidates by embeddings (fast);
          optionally cosine over prebuilt node embeddings if allowed
        """
        query_vec = self.encode_texts([query])[0]
        results: List[Dict[str, Any]] = []
        if self.using_neo4j:
            # lexical pre-candidates
            candidates = self.search_entities(query, limit=max(100, top_k * 5))
            if not candidates:
                return []
            cand_labels = [f"{c.get('name','')} ({c.get('type','')})" for c in candidates]
            cand_vecs = self.encode_texts(cand_labels)
            sims = (cand_vecs @ query_vec.reshape(-1,))
            scored = list(zip(candidates, sims))
            scored.sort(key=lambda x: x[1], reverse=True)
            for item, score in scored[:top_k]:
                item = item.copy()
                item['relevance_score'] = float(score)
                results.append(item)
            return results
        else:
            # Prefer candidate-only embedding re-ranking to avoid building full graph embeddings
            use_candidates_only = os.getenv('GRAPHRAG_EMBED_CANDIDATES_ONLY', 'true').lower() == 'true'
            candidate_multiplier = int(os.getenv('GRAPHRAG_EMBED_CANDIDATE_MULTIPLIER', '20'))
            if use_candidates_only:
                candidates = self.search_entities(query, limit=max(top_k * candidate_multiplier, 200)) or []
                if not candidates:
                    # Fallback only if we have prebuilt embeddings already
                    if self._node_embeddings is None:
                        return []
                else:
                    cand_labels = [f"{c.get('name','')} ({c.get('type','')})" for c in candidates]
                    cand_vecs = self.encode_texts(cand_labels)
                    sims = (cand_vecs @ query_vec.reshape(-1,))
                    scored = list(zip(candidates, sims))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    for item, score in scored[:top_k]:
                        item = item.copy()
                        item['relevance_score'] = float(score)
                        results.append(item)
                    return results

            # If candidate-only is disabled, allow full-graph embedding search (may be slow on first run)
            self._ensure_node_embeddings()
            if self._node_embeddings is None:
                return []
            sims = self._node_embeddings @ query_vec.reshape(-1,)
            top_idx = np.argsort(-sims)[:top_k]
            for idx in top_idx:
                entity_id = self._node_embedding_ids[idx]
                info = self.entity_info.get(entity_id, {}).copy()
                info['id'] = entity_id
                info['relevance_score'] = float(sims[idx])
                results.append(info)
            return results

    # ------------------------- Search & Grounding -------------------------
    
    def _build_search_index(self) -> None:
        """Build search index for efficient entity lookup."""
        self.entity_search_index = {}
        
        for entity_id, info in self.entity_info.items():
            # Create searchable text
            searchable_text = f"{info['name']} {info['type']} {info['description']}".lower()
            
            # Simple tokenization
            tokens = searchable_text.split()
            
            # Add to index
            for token in tokens:
                if token not in self.entity_search_index:
                    self.entity_search_index[token] = []
                self.entity_search_index[token].append(entity_id)
        
        logger.info(f"Built search index with {len(self.entity_search_index)} terms")
    
    def search_entities(self, query: str, entity_types: Optional[List[str]] = None, 
                       limit: int = 10) -> List[Dict]:
        """
        Search for entities based on query.
        
        Args:
            query: Search query
            entity_types: Filter by entity types (optional)
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries with relevance scores
        """
        # ENHANCED MEMORY MONITORING: Check memory before expensive operations
        self.check_memory_usage()
        
        if not self.is_loaded and not self.graph and not self._neo4j_driver:
            logger.warning("Data source not initialized. Call load() or load_primekg() first.")
            return []
        
        cache_key = (query.lower(), tuple(sorted([t.lower() for t in entity_types]) if entity_types else ()), limit)
        if cache_key in self._search_cache:
            self._memory_stats['cache_hits'] += 1
            return self._search_cache[cache_key]
        
        # Cache miss - increment counter
        self._memory_stats['cache_misses'] += 1
        if self.using_neo4j:
            cypher = """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($q)
              AND ($types IS NULL OR any(l IN labels(n) WHERE l IN $types))
            RETURN n.name AS name, labels(n)[0] AS type, id(n) AS neo4j_id, coalesce(n.uri, toString(id(n))) AS id
            LIMIT $limit
            """
            params = {"q": query, "types": entity_types if entity_types else None, "limit": limit * 3}
            with self._neo4j_driver.session(database=self.neo4j_database) as session:
                rows = session.run(cypher, **params).data()
            results = []
            for row in rows[:limit]:
                results.append({
                    'id': row.get('id') or str(row.get('neo4j_id')),
                    'name': row.get('name',''),
                    'type': row.get('type','unknown'),
                    'relevance_score': 1.0,
                    'source': 'neo4j'
                })
            # cache
            if len(self._search_cache) >= self._cache_sizes['search']:
                self._search_cache.pop(next(iter(self._search_cache)))
            self._search_cache[cache_key] = results
            return results
        else:
            query_tokens = query.lower().split()
            entity_scores = defaultdict(float)
            # Score entities based on token matches
            for token in query_tokens:
                if token in self.entity_search_index:
                    for entity_id in self.entity_search_index[token]:
                        entity_scores[entity_id] += 1.0
            # Also search in entity names directly
            for entity_id, info in self.entity_info.items():
                if query.lower() in info['name'].lower():
                    entity_scores[entity_id] += 2.0
            # Filter by entity types if specified
            if entity_types:
                entity_types_lower = [t.lower() for t in entity_types]
                filtered_scores = {}
                entities_checked = 0
                entities_filtered = 0
                for entity_id, score in entity_scores.items():
                    entities_checked += 1
                    # CRITICAL: Check if entity_id exists in entity_info before accessing
                    if entity_id not in self.entity_info:
                        # This shouldn't happen if entity_search_index is built correctly
                        # But if it does, skip it rather than crashing
                        logger.debug(f"Entity ID '{entity_id[:50]}...' not found in entity_info during type filtering, skipping")
                        continue
                    
                    entity_type = self.entity_info[entity_id].get('type', 'unknown')
                    entity_type_lower = entity_type.lower() if entity_type else 'unknown'
                    
                    # Check if entity type matches (case-insensitive)
                    # Also check if the requested type is in the entity type string (for compound types)
                    type_matches = (
                        entity_type_lower in entity_types_lower or
                        any(et in entity_type_lower for et in entity_types_lower) or
                        any(et in entity_type_lower.split() for et in entity_types_lower)
                    )
                    
                    if type_matches:
                        filtered_scores[entity_id] = score
                    else:
                        entities_filtered += 1
                        logger.debug(f"Entity '{self.entity_info[entity_id].get('name', entity_id)}' type '{entity_type}' doesn't match requested types {entity_types}")
                
                if entities_checked > 0 and len(filtered_scores) == 0:
                    # CRITICAL: If type filtering removes ALL entities, this is likely a type mismatch issue
                    # Try re-inferring types for top entities using improved inference
                    logger.warning(f"Type filtering removed ALL {entities_checked} entities. Requested types: {entity_types}. Attempting type re-inference...")
                    
                    # Re-infer types for top-scoring entities (check more entities for better coverage)
                    top_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:min(50, len(entity_scores))]
                    reclassified_count = 0
                    
                    # Create type mapping for synonyms (e.g., Pathway -> pathway, MolecularFunction -> protein/biological_process)
                    type_synonyms = {
                        'pathway': ['pathway', 'biological_process', 'process'],
                        'molecularfunction': ['protein', 'biological_process', 'function'],
                        'protein': ['protein', 'gene', 'molecularfunction'],
                        'disease': ['disease', 'disorder', 'syndrome'],
                        'drug': ['drug', 'compound', 'medication'],
                        'phenotype': ['phenotype', 'symptom', 'trait'],
                        'anatomy': ['anatomy', 'anatomical', 'structure']
                    }
                    
                    for entity_id, score in top_entities:
                        if entity_id not in self.entity_info:
                            continue
                        
                        entity_name = self.entity_info[entity_id].get('name', entity_id)
                        old_type = self.entity_info[entity_id].get('type', 'unknown')
                        entity_name_lower = entity_name.lower()
                        
                        # FIRST: Check entity name directly for keywords (most reliable, fastest)
                        # This catches pathways even if inference fails
                        inferred_type = None
                        inferred_type_lower = None
                        
                        if 'pathway' in entity_types_lower:
                            # Check for pathway keywords first
                            if 'pathway' in entity_name_lower or 'signaling pathway' in entity_name_lower:
                                inferred_type = 'pathway'
                                inferred_type_lower = 'pathway'
                            elif 'regulation' in entity_name_lower and ('pathway' in entity_name_lower or 'signaling' in entity_name_lower):
                                inferred_type = 'pathway'
                                inferred_type_lower = 'pathway'
                        
                        if 'molecularfunction' in entity_types_lower and not inferred_type:
                            # Check for molecular function keywords
                            if 'function' in entity_name_lower and ('molecular' in entity_name_lower or 'role' in entity_name_lower):
                                inferred_type = 'molecularfunction'
                                inferred_type_lower = 'molecularfunction'
                        
                        # SECOND: If keyword matching didn't work, try inference
                        if not inferred_type or inferred_type == 'unknown':
                            name_to_infer = entity_name if entity_name and entity_name != entity_id else entity_id
                            inferred_type = self._infer_entity_type(name_to_infer)
                            inferred_type_lower = inferred_type.lower() if inferred_type else 'unknown'
                        
                        # THIRD: Final fallback - check entity name directly for pathway keywords if inference failed
                        if (inferred_type == 'unknown' or not inferred_type) and entity_name:
                            if 'pathway' in entity_types_lower:
                                if 'pathway' in entity_name_lower or 'signaling pathway' in entity_name_lower:
                                    inferred_type = 'pathway'
                                    inferred_type_lower = 'pathway'
                            if 'molecularfunction' in entity_types_lower:
                                if 'function' in entity_name_lower and 'molecular' in entity_name_lower:
                                    inferred_type = 'molecularfunction'
                                    inferred_type_lower = 'molecularfunction'
                        
                        # Check if re-inferred type matches requested types (with synonyms)
                        matches = False
                        for requested_type in entity_types_lower:
                            # Direct match
                            if inferred_type_lower == requested_type:
                                matches = True
                                break
                            # Check if inferred type contains requested type or vice versa
                            if requested_type in inferred_type_lower or inferred_type_lower in requested_type:
                                matches = True
                                break
                            # Check synonyms
                            if requested_type in type_synonyms:
                                if inferred_type_lower in type_synonyms[requested_type]:
                                    matches = True
                                    break
                            # Check if inferred type's synonyms match requested type
                            if inferred_type_lower in type_synonyms:
                                if requested_type in type_synonyms[inferred_type_lower]:
                                    matches = True
                                    break
                        
                        # CRITICAL FIX: If inference failed but entity name contains pathway keywords,
                        # and we're looking for pathways, accept it anyway
                        if not matches and 'pathway' in entity_types_lower:
                            name_lower = entity_name.lower()
                            if 'pathway' in name_lower or 'signaling pathway' in name_lower or 'regulation' in name_lower:
                                matches = True
                                inferred_type = 'pathway'
                                inferred_type_lower = 'pathway'
                        # Similar fix for molecular function
                        if not matches and 'molecularfunction' in entity_types_lower:
                            name_lower = entity_name.lower()
                            if 'function' in name_lower and ('molecular' in name_lower or 'role' in name_lower):
                                matches = True
                                inferred_type = 'molecularfunction'
                                inferred_type_lower = 'molecularfunction'
                        
                        if matches:
                            # Update entity_info with corrected type (use requested type format if available)
                            corrected_type = inferred_type
                            # Prefer the requested type format if it's a close match
                            # Find which requested type matched
                            matched_requested_type = None
                            for rt in entity_types_lower:
                                if rt in ['pathway', 'molecularfunction']:
                                    matched_requested_type = rt
                                    break
                            
                            if matched_requested_type == 'pathway':
                                corrected_type = 'Pathway'
                            elif matched_requested_type == 'molecularfunction':
                                corrected_type = 'MolecularFunction'
                            
                            self.entity_info[entity_id]['type'] = corrected_type
                            filtered_scores[entity_id] = score
                            reclassified_count += 1
                            logger.debug(f"Re-classified '{entity_name[:50]}...' from '{old_type}' to '{corrected_type}'")
                    
                    if len(filtered_scores) > 0:
                        logger.info(f"Type re-inference successful: Re-classified {reclassified_count} entities, found {len(filtered_scores)} matches")
                        entity_scores = filtered_scores
                    else:
                        # Final fallback: Return unfiltered results
                        sample_types = []
                        sample_keys = list(entity_scores.keys())[:5]
                        for eid in sample_keys:
                            if eid in self.entity_info:
                                sample_types.append(self.entity_info[eid].get('type', 'unknown'))
                            else:
                                sample_types.append(f"NAME_NOT_ID:{eid[:30]}")
                        logger.warning(f"Type re-inference failed. Sample types: {set(sample_types)}. Returning unfiltered results.")
                        # Don't filter - return original scores
                        # entity_scores remains unchanged
                else:
                    entity_scores = filtered_scores
            sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            results = []
            for entity_id, score in sorted_entities:
                # CRITICAL: Check if entity_id exists in entity_info before accessing
                if entity_id not in self.entity_info:
                    logger.debug(f"Entity ID '{entity_id}' not found in entity_info, skipping result")
                    continue
                entity_info = self.entity_info[entity_id].copy()
                entity_info['id'] = entity_id
                entity_info['relevance_score'] = score
                results.append(entity_info)
            if len(self._search_cache) >= self._cache_sizes['search']:
                self._search_cache.pop(next(iter(self._search_cache)))
            self._search_cache[cache_key] = results
            return results
    
    def get_entity_neighbors(self, entity_id: str, max_hops: int = 1, 
                           relation_types: Optional[List[str]] = None) -> Dict:
        """
        Get neighbors of an entity up to max_hops away.
        
        Args:
            entity_id: ID of the entity
            max_hops: Maximum number of hops to traverse
            relation_types: Optional list of relation types to include
            
        Returns:
            Dictionary of neighbor entities with their info and relations
        """
        # ENHANCED MEMORY MONITORING: Check memory before expensive graph operations
        self.check_memory_usage()
        
        rel_key = tuple(sorted(relation_types)) if relation_types else ()
        cache_key = (entity_id, max_hops, rel_key)
        if cache_key in self._neighbors_cache:
            self._memory_stats['cache_hits'] += 1
            return self._neighbors_cache[cache_key]
        
        # Cache miss - increment counter  
        self._memory_stats['cache_misses'] += 1
        if self.using_neo4j:
            with self._neo4j_driver.session(database=self.neo4j_database) as session:
                cypher = (
                    "MATCH (s {uri:$id})-[r*1.." + str(max_hops) + "]-(t) "
                    "WITH t, [rel IN r | type(rel)] AS rels, length(r) AS d "
                    "RETURN t.name AS name, labels(t)[0] AS type, coalesce(t.uri,toString(id(t))) AS id, rels AS relations, d AS distance"
                )
                rows = session.run(cypher, id=entity_id).data()
            out: Dict[str, Any] = {}
            for row in rows:
                if relation_types and not any(rt in row['relations'] for rt in relation_types):
                    continue
                out[row['id']] = {
                    'entity_info': {'id': row['id'], 'name': row.get('name',''), 'type': row.get('type','unknown')},
                    'relations': row['relations'],
                    'distance': row['distance']
                }
            if len(self._neighbors_cache) >= self._cache_sizes['neighbors']:
                self._neighbors_cache.pop(next(iter(self._neighbors_cache)))
            self._neighbors_cache[cache_key] = out
            return out
        else:
            if not self.is_loaded or self.graph is None:
                logger.warning("PrimeKG data not loaded. Call load() first.")
                return {}
            if entity_id not in self.graph:
                return {}
            neighbors: Dict[str, Any] = {}
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if isinstance(edge_data, dict):
                    relations = []
                    for _, edge_attrs in edge_data.items():
                        relation = edge_attrs.get('relation', 'unknown')
                        if not relation_types or relation in relation_types:
                            relations.append(relation)
                    if relations:
                        neighbors[neighbor] = {
                            'entity_info': self.entity_info.get(neighbor, {}),
                            'relations': relations,
                            'distance': 1
                        }
            if max_hops > 1:
                visited = set([entity_id])
                queue = [(n, 1) for n in neighbors.keys()]
                while queue:
                    current_entity, distance = queue.pop(0)
                    if distance >= max_hops:
                        continue
                    for neighbor in self.graph.neighbors(current_entity):
                        if neighbor not in visited and neighbor != entity_id:
                            edge_data = self.graph.get_edge_data(current_entity, neighbor)
                            if isinstance(edge_data, dict):
                                relations = []
                                for _, edge_attrs in edge_data.items():
                                    relation = edge_attrs.get('relation', 'unknown')
                                    if not relation_types or relation in relation_types:
                                        relations.append(relation)
                                if relations:
                                    neighbors[neighbor] = {
                                        'entity_info': self.entity_info.get(neighbor, {}),
                                        'relations': relations,
                                        'distance': distance + 1
                                    }
                                    visited.add(neighbor)
                                    if distance + 1 < max_hops:
                                        queue.append((neighbor, distance + 1))
            if len(self._neighbors_cache) >= self._cache_sizes['neighbors']:
                self._neighbors_cache.pop(next(iter(self._neighbors_cache)))
            self._neighbors_cache[cache_key] = neighbors
            return neighbors
    
    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[str]:
        """
        Get entities of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            limit: Maximum number of entities to return
            
        Returns:
            List of entity IDs
        """
        if self.using_neo4j:
            cypher = """
            MATCH (n:`%s`) RETURN coalesce(n.uri,toString(id(n))) AS id LIMIT $limit
            """ % (entity_type,)
            with self._neo4j_driver.session(database=self.neo4j_database) as session:
                rows = session.run(cypher, limit=limit).data()
            return [r['id'] for r in rows]
        else:
            if not self.is_loaded:
                logger.warning("PrimeKG data not loaded. Call load() first.")
                return []
            entities = []
            entity_type_lower = entity_type.lower()
            for entity_id, info in self.entity_info.items():
                if info['type'].lower() == entity_type_lower:
                    entities.append(entity_id)
                    if len(entities) >= limit:
                        break
            return entities
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded graph."""
        if self.using_neo4j:
            with self._neo4j_driver.session(database=self.neo4j_database) as session:
                nodes = session.run("MATCH (n) RETURN count(n) AS c").single()[0]
                edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()[0]
                rels = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").data()
            return {
                'total_nodes': nodes,
                'total_edges': edges,
                'relation_types': [r['relationshipType'] for r in rels],
                'is_directed': True,
                'is_multigraph': True
            }
        if not self.is_loaded:
            return {}
        
        # Count entities by type
        entity_type_counts = defaultdict(int)
        for info in self.entity_info.values():
            entity_type_counts[info['type']] += 1
        
        # Count relations by type
        relation_type_counts = defaultdict(int)
        for _, _, edge_data in self.graph.edges(data=True):
            if isinstance(edge_data, dict):
                relation = edge_data.get('relation', 'unknown')
                relation_type_counts[relation] += 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(entity_type_counts),
            'relation_types': dict(relation_type_counts),
            'is_directed': self.graph.is_directed(),
            'is_multigraph': self.graph.is_multigraph()
        }
    
    def get_triples_dataframe(self) -> pd.DataFrame:
        """Get the triples as a pandas DataFrame."""
        if not self.is_loaded:
            logger.warning("PrimeKG data not loaded. Call load_primekg() first.")
            return pd.DataFrame()
        
        return self.triples_df.copy()

    # ------------------------- Additional API for other modules -------------------------
    def get_valid_relations(self) -> List[str]:
        if self.using_neo4j:
            cypher = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS type"
            with self._neo4j_driver.session(database=self.neo4j_database) as session:
                rows = session.run(cypher).data()
            return [r['type'] for r in rows]
        else:
            if self.triples_df is None:
                return []
            return sorted(self.triples_df['relation'].unique().tolist())

    def get_relations_between_types(self, source_type: str, target_type: str, relation_type: Optional[Any] = None) -> List[Dict[str, Any]]:
        if self.using_neo4j:
            rel_filter = ""
            if isinstance(relation_type, list):
                rel_filter = "WHERE type(r) IN $rels"
            elif isinstance(relation_type, str):
                rel_filter = "WHERE type(r) = $rel"
            cypher = (
                "MATCH (s:`%s`)-[r]->(t:`%s`) %s RETURN coalesce(s.uri,toString(id(s))) AS source_id, type(r) AS type, coalesce(t.uri,toString(id(t))) AS target_id"
                % (source_type, target_type, rel_filter)
            )
            params: Dict[str, Any] = {}
            if isinstance(relation_type, list):
                params['rels'] = relation_type
            elif isinstance(relation_type, str):
                params['rel'] = relation_type
            with self._neo4j_driver.session(database=self.neo4j_database) as session:
                rows = session.run(cypher, **params).data()
            return rows
        else:
            if self.triples_df is None:
                return []
            df = self.triples_df.copy()
            # derive types from entity_info
            def _type(eid: str) -> str:
                return self.entity_info.get(eid, {}).get('type', 'unknown')
            out: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                if _type(row['head']) == source_type and _type(row['tail']) == target_type:
                    if relation_type is None or (isinstance(relation_type, list) and row['relation'] in relation_type) or (row['relation'] == relation_type):
                        out.append({'source_id': row['head'], 'target_id': row['tail'], 'type': row['relation']})
            return out

    def get_entity_evidence(self, entity_id: str) -> List[Dict[str, Any]]:
        """Return lightweight provenance for an entity from PrimeKG triples.

        For Neo4j, fetch a few incident relationships. For local PyKEEN/NetworkX,
        sample a few triples from the DataFrame. This is not external literature,
        but provides graph-native provenance to populate evidence fields.
        """
        try:
            if self.using_neo4j:
                with self._neo4j_driver.session(database=self.neo4j_database) as session:
                    cypher = (
                        "MATCH (s)-[r]->(t) "
                        "WHERE coalesce(s.uri,toString(id(s))) = $id OR coalesce(t.uri,toString(id(t))) = $id "
                        "RETURN coalesce(s.uri,toString(id(s))) AS head, type(r) AS relation, coalesce(t.uri,toString(id(t))) AS tail "
                        "LIMIT 5"
                    )
                    rows = session.run(cypher, id=entity_id).data()
                return [
                    {
                        'source': 'primekg',
                        'provenance': 'graph_relationship',
                        'head': r['head'],
                        'predicate': r['relation'],
                        'tail': r['tail']
                    }
                    for r in rows
                ]
            else:
                if not self.is_loaded or self.triples_df is None:
                    return []
                # Sample up to 5 triples involving the entity
                df = self.triples_df
                mask = (df['head'] == entity_id) | (df['tail'] == entity_id)
                sample = df[mask].head(5)
                out: List[Dict[str, Any]] = []
                for _, row in sample.iterrows():
                    out.append({
                        'source': 'primekg',
                        'provenance': 'graph_triple',
                        'head': row['head'],
                        'predicate': row['relation'],
                        'tail': row['tail'],
                        'relation_id': int(row['relation_id']) if 'relation_id' in row and pd.notna(row['relation_id']) else None
                    })
                return out
        except Exception:
            return []

    def get_relation_evidence(self, source_id: str, target_id: str, relation_type: str) -> List[Dict[str, Any]]:
        """Return lightweight provenance for a specific relation occurrence."""
        try:
            if self.using_neo4j:
                with self._neo4j_driver.session(database=self.neo4j_database) as session:
                    cypher = (
                        "MATCH (s)-[r]->(t) "
                        "WHERE coalesce(s.uri,toString(id(s))) = $sid AND coalesce(t.uri,toString(id(t))) = $tid "
                        "AND type(r) = $rel RETURN type(r) AS relation LIMIT 3"
                    )
                    rows = session.run(cypher, sid=source_id, tid=target_id, rel=relation_type).data()
                return [
                    {
                        'source': 'primekg',
                        'provenance': 'graph_relationship',
                        'head': source_id,
                        'predicate': relation_type,
                        'tail': target_id
                    }
                    for _ in rows
                ]
            else:
                if not self.is_loaded or self.triples_df is None:
                    return []
                df = self.triples_df
                mask = (df['head'] == source_id) & (df['tail'] == target_id) & (df['relation'] == relation_type)
                sample = df[mask].head(3)
                out: List[Dict[str, Any]] = []
                for _, row in sample.iterrows():
                    out.append({
                        'source': 'primekg',
                        'provenance': 'graph_triple',
                        'head': row['head'],
                        'predicate': row['relation'],
                        'tail': row['tail'],
                        'relation_id': int(row['relation_id']) if 'relation_id' in row and pd.notna(row['relation_id']) else None
                    })
                return out
        except Exception:
            return []

    # ------------------------- Paths -------------------------
    def find_all_shortest_paths(self, entity_ids: List[str], max_len: int = 4, limit_paths: int = 100) -> List[List[str]]:
        cache_key = (tuple(entity_ids), max_len, limit_paths)
        if cache_key in self._paths_cache:
            return self._paths_cache[cache_key]
        if self.using_neo4j:
            paths: List[List[str]] = []
            if len(entity_ids) < 2:
                return paths
            with self._neo4j_driver.session(database=self.neo4j_database) as session:
                for i in range(len(entity_ids) - 1):
                    sid = entity_ids[i]
                    tid = entity_ids[i + 1]
                    cypher = (
                        "MATCH (s {uri:$sid}), (t {uri:$tid}), p = allShortestPaths((s)-[*..%d]-(t)) "
                        "RETURN [n IN nodes(p) | coalesce(n.uri, toString(id(n)))] AS path "
                    ) % max_len
                    rows = session.run(cypher, sid=sid, tid=tid).data()
                    for r in rows:
                        paths.append(r['path'])
                        if len(paths) >= limit_paths:
                            return paths
            if len(self._paths_cache) >= self._cache_sizes['paths']:
                self._paths_cache.pop(next(iter(self._paths_cache)))
            self._paths_cache[cache_key] = paths
            return paths
        else:
            if self.graph is None:
                return []
            paths: List[List[str]] = []
            undirected = self.graph.to_undirected()
            for i in range(len(entity_ids) - 1):
                s = entity_ids[i]
                t = entity_ids[i + 1]
                if s in undirected and t in undirected:
                    try:
                        for p in nx.all_shortest_paths(undirected, s, t):
                            paths.append(p)
                            if len(paths) >= limit_paths:
                                return paths
                    except nx.NetworkXNoPath:
                        continue
            if len(self._paths_cache) >= self._cache_sizes['paths']:
                self._paths_cache.pop(next(iter(self._paths_cache)))
            self._paths_cache[cache_key] = paths
            return paths
    
    def get_relations_by_entity(self, entity_id: str) -> List[Dict]:
        """Get all relations involving a specific entity."""
        if self.using_neo4j:
            cypher = """
            MATCH (s)-[r]->(t)
            WHERE coalesce(s.uri,toString(id(s))) = $id OR coalesce(t.uri,toString(id(t))) = $id
            RETURN coalesce(s.uri,toString(id(s))) AS head, type(r) AS relation, coalesce(t.uri,toString(id(t))) AS tail,
                   CASE WHEN coalesce(s.uri,toString(id(s))) = $id THEN 'outgoing' ELSE 'incoming' END AS direction
            """
            with self._neo4j_driver.session(database=self.neo4j_database) as session:
                rows = session.run(cypher, id=entity_id).data()
            return [
                {
                    'subject': r['head'],
                    'predicate': r['relation'],
                    'object': r['tail'],
                    'direction': r['direction']
                } for r in rows
            ]
        else:
            if not self.is_loaded:
                logger.warning("PrimeKG data not loaded. Call load_primekg() first.")
                return []
            relations: List[Dict[str, Any]] = []
            head_relations = self.triples_df[self.triples_df['head'] == entity_id]
            for _, row in head_relations.iterrows():
                relations.append({
                    'subject': row['head'],
                    'predicate': row['relation'],
                    'object': row['tail'],
                    'direction': 'outgoing'
                })
            tail_relations = self.triples_df[self.triples_df['tail'] == entity_id]
            for _, row in tail_relations.iterrows():
                relations.append({
                    'subject': row['head'],
                    'predicate': row['relation'],
                    'object': row['tail'],
                    'direction': 'incoming'
                })
            return relations

    def _load_from_cache(self) -> bool:
        """Load data from cache files."""
        try:
            cache_files = [
                self.graph_cache_file,
                self.entity_cache_file,
                self.search_index_cache_file
            ]
            
            # Check if all cache files exist and are recent
            for cache_file in cache_files:
                if not cache_file.exists():
                    return False
                
                # Check cache expiration
                if self._is_cache_expired(cache_file):
                    logger.info(f"Cache file {cache_file.name} is expired")
                    return False
            
            # Load cached data
            with open(self.graph_cache_file, 'rb') as f:
                self.graph = pickle.load(f)
            
            with open(self.entity_cache_file, 'rb') as f:
                self.entity_info = pickle.load(f)
            
            with open(self.search_index_cache_file, 'rb') as f:
                self.entity_search_index = pickle.load(f)
            
            # ALWAYS apply enhanced entity classification to cached data
            # This ensures the latest classification patterns are used even with cached data
            logger.info("Applying enhanced entity classification to cached data...")
            self._apply_enhanced_entity_classification()
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return False
    
    def _is_cache_expired(self, cache_file: Path) -> bool:
        """Check if cache file is expired."""
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return file_age_hours > self.cache_expiration_hours
    
    def clear_entity_cache(self) -> None:
        """Clear the entity cache to force fresh entity classification."""
        try:
            if self.entity_cache_file.exists():
                self.entity_cache_file.unlink()
                logger.info("Entity cache cleared - enhanced classification will run on next load")
        except Exception as e:
            logger.warning(f"Failed to clear entity cache: {e}")
    
    def _save_to_cache(self) -> None:
        """Save data to cache files."""
        try:
            with open(self.graph_cache_file, 'wb') as f:
                pickle.dump(self.graph, f)
            
            with open(self.entity_cache_file, 'wb') as f:
                pickle.dump(self.entity_info, f)
            
            with open(self.search_index_cache_file, 'wb') as f:
                pickle.dump(self.entity_search_index, f)
            
            logger.info("Data saved to cache successfully")
            
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def ensure_required_files(self) -> bool:
        """Ensure PrimeKG data is loaded (compatibility method)."""
        return self.load_primekg()

    # ========================= ENHANCED MEMORY MANAGEMENT =========================
    # Addresses critical memory issues from CLAUDE.md Issue 3:
    # - Memory usage (3867.6MB) exceeds limit (2048MB)
    # - Unbounded cache growth causing system instability
    # - Lack of proper memory monitoring and cleanup
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.memory_monitoring_enabled:
            return 0.0
        
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def check_memory_usage(self, force_check: bool = False) -> Dict[str, Any]:
        """Check memory usage and trigger cleanup if necessary."""
        current_time = time.time()
        
        # Skip if not enough time has passed since last check (unless forced)
        if not force_check and (current_time - self.last_memory_check) < self.memory_monitoring_interval:
            return self._memory_stats.copy()
        
        self.last_memory_check = current_time
        
        if not self.memory_monitoring_enabled:
            return self._memory_stats.copy()
        
        current_memory_mb = self.get_current_memory_usage()
        memory_limit_mb = self.max_memory_usage * 1024  # Convert GB to MB
        cleanup_threshold_mb = memory_limit_mb * self.memory_cleanup_threshold
        
        # Update peak usage
        if current_memory_mb > self._memory_stats['peak_usage_mb']:
            self._memory_stats['peak_usage_mb'] = current_memory_mb
        
        # Check if cleanup is needed
        if current_memory_mb > cleanup_threshold_mb:
            logger.warning(f"Memory usage ({current_memory_mb:.1f}MB) exceeds cleanup threshold ({cleanup_threshold_mb:.1f}MB). Triggering cleanup.")
            self._trigger_memory_cleanup()
        elif current_memory_mb > memory_limit_mb:
            logger.error(f"Memory usage ({current_memory_mb:.1f}MB) exceeds limit ({memory_limit_mb:.1f}MB)! Forcing aggressive cleanup.")
            self._trigger_memory_cleanup(aggressive=True)
        
        return {
            'current_usage_mb': current_memory_mb,
            'limit_mb': memory_limit_mb,
            'cleanup_threshold_mb': cleanup_threshold_mb,
            'usage_percentage': (current_memory_mb / memory_limit_mb) * 100,
            **self._memory_stats
        }
    
    def _trigger_memory_cleanup(self, aggressive: bool = False) -> None:
        """Trigger memory cleanup to free up space."""
        import gc
        import sys
        
        with self.memory_cleanup_lock:
            initial_memory = self.get_current_memory_usage()
            logger.info(f"Starting memory cleanup (aggressive={aggressive}). Initial usage: {initial_memory:.1f}MB")
            
            # Store cache sizes before clearing
            cache_sizes_before = {
                'search': len(self._search_cache),
                'neighbors': len(self._neighbors_cache),
                'paths': len(self._paths_cache),
                'encode': len(self._encode_cache),
            }
            
            # Clear caches based on priority
            cleanup_operations = [
                ("search_cache", lambda: self._search_cache.clear()),
                ("neighbors_cache", lambda: self._neighbors_cache.clear()),
                ("paths_cache", lambda: self._paths_cache.clear()),
                ("encode_cache", lambda: self._encode_cache.clear()),
            ]
            
            if aggressive:
                # In aggressive mode, also clear embeddings and larger objects
                cleanup_operations.extend([
                    ("node_embeddings", lambda: setattr(self, '_node_embeddings', None)),
                    ("embedding_model", lambda: setattr(self, '_embedding_model', None)),
                ])

                # CRITICAL FIX: DO NOT clear entity_info! It's core data, not a cache!
                # entity_info contains the primary entity data loaded from PrimeKG
                # Clearing it breaks all entity lookups and causes 0% entity retrieval
                # Previous bug: entity_info was being cleared in aggressive mode, causing all
                # entity lookups to fail after NER model loading triggered memory cleanup
            
            # Perform cleanup operations
            for operation_name, operation in cleanup_operations:
                try:
                    operation()
                    logger.debug(f"Cleared {operation_name}")
                except Exception as e:
                    logger.warning(f"Failed to clear {operation_name}: {e}")
            
            # Force multiple garbage collection passes for better memory recovery
            total_collected = 0
            for _ in range(3):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
            
            # Update statistics after a brief delay to allow memory to be freed
            import time
            time.sleep(0.5)  # Longer delay to allow memory to be freed by OS
            
            # Force another GC pass after delay
            gc.collect()
            
            final_memory = self.get_current_memory_usage()
            memory_freed = initial_memory - final_memory
            
            # If memory wasn't freed but we collected objects, log it
            if memory_freed <= 0 and total_collected > 0:
                logger.debug(f"GC collected {total_collected} objects but memory not immediately freed (OS may delay release)")
            
            # Log cache clearing statistics
            cache_items_cleared = sum(cache_sizes_before.values())
            logger.debug(f"Cleared {cache_items_cleared} cache items")
            
            self._memory_stats['cleanup_count'] += 1
            self._memory_stats['last_cleanup_time'] = time.time()
            self._memory_stats['total_memory_freed'] = self._memory_stats.get('total_memory_freed', 0) + max(0, memory_freed)
            
            logger.info(f"Memory cleanup completed. Freed {memory_freed:.1f}MB. Current usage: {final_memory:.1f}MB")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache usage statistics."""
        cache_stats = {}
        
        for cache_name, cache_obj in [
            ('search', self._search_cache),
            ('neighbors', self._neighbors_cache),
            ('paths', self._paths_cache),
            ('encode', self._encode_cache)
        ]:
            max_size = self._cache_sizes.get(cache_name, 0)
            current_size = len(cache_obj) if hasattr(cache_obj, '__len__') else 0
            
            cache_stats[f'{cache_name}_cache'] = {
                'current_size': current_size,
                'max_size': max_size,
                'usage_percentage': (current_size / max_size * 100) if max_size > 0 else 0,
                'is_full': current_size >= max_size
            }
        
        return cache_stats
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization."""
        logger.info("Starting comprehensive memory optimization...")
        
        initial_stats = self.check_memory_usage(force_check=True)
        
        # Step 1: Clear least recently used cache entries
        self._optimize_caches()
        
        # Step 2: Cleanup temporary data structures
        if hasattr(self, 'triples_df') and self.triples_df is not None:
            # Clear the DataFrame if graph is already built
            if self.graph is not None:
                self.triples_df = None
                logger.debug("Cleared triples DataFrame")
        
        # Step 3: Optimize NetworkX graph memory usage
        if self.graph is not None:
            # Remove redundant data from nodes/edges if present
            self._optimize_graph_memory()
        
        # Step 4: Force garbage collection
        gc.collect()
        
        final_stats = self.check_memory_usage(force_check=True)
        
        optimization_result = {
            'initial_memory_mb': initial_stats.get('current_usage_mb', 0),
            'final_memory_mb': final_stats.get('current_usage_mb', 0),
            'memory_freed_mb': initial_stats.get('current_usage_mb', 0) - final_stats.get('current_usage_mb', 0),
            'cache_stats': self.get_cache_statistics()
        }
        
        logger.info(f"Memory optimization completed. Freed {optimization_result['memory_freed_mb']:.1f}MB")
        return optimization_result
    
    def _optimize_caches(self) -> None:
        """Optimize cache sizes by removing oldest entries."""
        for cache_name, cache_obj in [
            ('search', self._search_cache),
            ('neighbors', self._neighbors_cache),
            ('paths', self._paths_cache)
        ]:
            max_size = self._cache_sizes.get(cache_name, 0)
            if len(cache_obj) > max_size * 0.8:  # If cache is >80% full
                # Remove oldest 20% of entries
                items_to_remove = max(1, len(cache_obj) // 5)
                keys_to_remove = list(cache_obj.keys())[:items_to_remove]
                for key in keys_to_remove:
                    cache_obj.pop(key, None)
                logger.debug(f"Optimized {cache_name} cache: removed {items_to_remove} entries")
    
    def _optimize_graph_memory(self) -> None:
        """Optimize NetworkX graph memory usage."""
        if self.graph is None:
            return
        
        # Remove unnecessary node/edge attributes that consume memory
        # Keep only essential attributes for functionality
        essential_node_attrs = {'name', 'type', 'description'}
        essential_edge_attrs = {'relation', 'weight'}
        
        nodes_optimized = 0
        edges_optimized = 0
        
        # Optimize node attributes
        for node, data in self.graph.nodes(data=True):
            attrs_to_remove = [attr for attr in data.keys() if attr not in essential_node_attrs]
            for attr in attrs_to_remove:
                data.pop(attr, None)
                nodes_optimized += 1
        
        # Optimize edge attributes  
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            attrs_to_remove = [attr for attr in data.keys() if attr not in essential_edge_attrs]
            for attr in attrs_to_remove:
                data.pop(attr, None)
                edges_optimized += 1
        
        if nodes_optimized > 0 or edges_optimized > 0:
            logger.debug(f"Graph memory optimization: removed {nodes_optimized} node attrs, {edges_optimized} edge attrs")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system and memory statistics."""
        memory_stats = self.check_memory_usage(force_check=True)
        cache_stats = self.get_cache_statistics()
        
        system_stats = {
            'memory': memory_stats,
            'caches': cache_stats,
            'graph': {
                'loaded': self.is_loaded,
                'backend': 'Neo4j' if self._use_neo4j else 'PyKEEN',
                'nodes': self.graph.number_of_nodes() if self.graph else 0,
                'edges': self.graph.number_of_edges() if self.graph else 0,
            },
            'configuration': {
                'max_memory_gb': self.max_memory_usage,
                'cleanup_threshold': self.memory_cleanup_threshold,
                'monitoring_interval': self.memory_monitoring_interval,
                'cache_sizes': self._cache_sizes
            }
        }
        
        return system_stats

# Example usage
if __name__ == "__main__":
    # Initialize data source
    data_source = PrimeKGDataSource(data_dir='data')
    
    # Load PrimeKG data
    if data_source.load_primekg():
        logger.info("PrimeKG loaded successfully!")
        
        # Get statistics
        stats = data_source.get_graph_statistics()
        if stats:  # Check if stats is not empty
            logger.info(f"Graph has {stats['total_nodes']} nodes and {stats['total_edges']} edges")
        else:
            logger.warning("Warning: Graph statistics are empty - data may not be loaded properly")
        
        # Search for entities
        results = data_source.search_entities("diabetes", limit=5)
        logger.info(f"Found {len(results)} entities related to 'diabetes'")
        
        # Get neighbors
        if results:
            entity_id = results[0]['id']
            neighbors = data_source.get_entity_neighbors(entity_id)
            logger.debug(f"Entity {entity_id} has {len(neighbors)} neighbors")
    else:
        logger.error("Failed to load PrimeKG data")
