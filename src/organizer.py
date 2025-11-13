"""
Organizer module for GraphRAG with PrimeKG-specific organization strategies.

This module implements specialized organization strategies for biomedical knowledge
from PrimeKG, including biological hierarchy organization and pathway-based clustering.
"""

import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from .graph_data_source import PrimeKGDataSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os as _os
_env_log_level = _os.getenv('GRAPHRAG_LOG_LEVEL', 'INFO').upper()
try:
    logger.setLevel(getattr(logging, _env_log_level, logging.INFO))
except Exception:
    pass

@dataclass
class RetrievedContext:
    """Represents a retrieved context from the knowledge graph."""
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    subgraphs: List[nx.Graph]
    metadata: Dict[str, Any]

@dataclass
class OrganizedContext:
    """Represents organized context ready for generation."""
    hierarchical_view: Dict[str, Any]
    pathway_clusters: List[Dict[str, Any]]
    entity_groups: Dict[str, List[Dict[str, Any]]]
    evidence_chains: List[List[Dict[str, Any]]]
    # Optional: ranked text paths from graph-to-text conversion
    ranked_text_paths: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class PrimeKGOrganizer:
    """
    Organizes retrieved PrimeKG subgraphs into coherent structures.
    
    Features:
    - Biological hierarchy organization
    - Pathway-based clustering
    - Evidence chain construction
    - Entity relationship grouping
    """
    
    def __init__(self, data_source: PrimeKGDataSource):
        """
        Initialize the organizer.
        
        Args:
            data_source: PrimeKG data source for additional context
        """
        self.data_source = data_source
        
        # Initialize biological hierarchies
        self.hierarchies = {
            'cellular': self._build_cellular_hierarchy(),
            'molecular': self._build_molecular_hierarchy(),
            'phenotype': self._build_phenotype_hierarchy(),
            'pathway': self._build_pathway_hierarchy()
        }
        
        logger.info("PrimeKG Organizer initialized")
    
    def organize_context(self, retrieved_context: RetrievedContext) -> OrganizedContext:
        """
        Organize retrieved context into coherent structures.
        
        Args:
            retrieved_context: Retrieved context from PrimeKG
            
        Returns:
            Organized context ready for generation
        """
        logger.info("Organizing retrieved context")
        
        # Step 1: Build hierarchical view
        hierarchical_view = self._build_hierarchical_view(retrieved_context)
        logger.info("Built hierarchical view")
        
        # Step 2: Cluster pathways
        pathway_clusters = self._cluster_pathways(retrieved_context)
        logger.info(f"Created {len(pathway_clusters)} pathway clusters")
        
        # Step 3: Group entities
        entity_groups = self._group_entities(retrieved_context)
        logger.info(f"Created {len(entity_groups)} entity groups")
        
        # Step 4: Construct evidence chains
        evidence_chains = self._construct_evidence_chains(retrieved_context)
        logger.info(f"Constructed {len(evidence_chains)} evidence chains")
        
        # Step 5: Collect metadata
        metadata = self._collect_metadata(retrieved_context)
        
        # Step 6: Try to build simple text paths if relations present
        text_paths = self._graph_to_text_paths(retrieved_context)

        return OrganizedContext(
            hierarchical_view=hierarchical_view,
            pathway_clusters=pathway_clusters,
            entity_groups=entity_groups,
            evidence_chains=evidence_chains,
            ranked_text_paths=text_paths,
            metadata=metadata
        )
    
    def _build_cellular_hierarchy(self) -> nx.DiGraph:
        """Build cellular component hierarchy from PrimeKG."""
        hierarchy = nx.DiGraph()
        
        # Get cellular components (returns list of entity IDs)
        component_ids = self.data_source.get_entities_by_type('cellular_component')  # Use lowercase
        
        # Build hierarchy from entity IDs
        for component_id in component_ids:
            # Get entity info from data source
            entity_info = self.data_source.entity_info.get(component_id, {})
            hierarchy.add_node(
                component_id,
                name=entity_info.get('name', component_id),
                type='CellularComponent',
                properties={}
            )
        
        # Skip relations for now as method signature is unclear
        # TODO: Implement relation retrieval when method is clarified
        
        return hierarchy
    
    def _build_molecular_hierarchy(self) -> nx.DiGraph:
        """Build molecular function hierarchy from PrimeKG."""
        hierarchy = nx.DiGraph()
        
        # Get molecular functions (returns list of entity IDs)
        function_ids = self.data_source.get_entities_by_type('molecular_function')  # Use lowercase
        
        # Build hierarchy from entity IDs
        for function_id in function_ids:
            # Get entity info from data source
            entity_info = self.data_source.entity_info.get(function_id, {})
            hierarchy.add_node(
                function_id,
                name=entity_info.get('name', function_id),
                type='MolecularFunction',
                properties={}
            )
        
        # Skip relations for now as method signature is unclear
        # TODO: Implement relation retrieval when method is clarified
        
        return hierarchy
    
    def _build_phenotype_hierarchy(self) -> nx.DiGraph:
        """Build phenotype hierarchy from PrimeKG."""
        hierarchy = nx.DiGraph()
        
        # Get phenotypes (returns list of entity IDs)
        phenotype_ids = self.data_source.get_entities_by_type('phenotype')  # Use lowercase as per entity types
        
        # Build hierarchy from entity IDs
        for phenotype_id in phenotype_ids:
            # Get entity info from data source
            entity_info = self.data_source.entity_info.get(phenotype_id, {})
            hierarchy.add_node(
                phenotype_id,
                name=entity_info.get('name', phenotype_id),
                type='Phenotype',
                properties={}
            )
        
        # Get relationships between phenotype entities
        # For now, skip relations as the method signature is unclear
        # TODO: Implement relation retrieval when method is clarified
        
        return hierarchy
    
    def _build_pathway_hierarchy(self) -> nx.DiGraph:
        """Build biological pathway hierarchy from PrimeKG."""
        hierarchy = nx.DiGraph()
        
        # Get pathways (returns list of entity IDs)
        pathway_ids = self.data_source.get_entities_by_type('pathway')  # Use lowercase
        
        # Build hierarchy from entity IDs
        for pathway_id in pathway_ids:
            # Get entity info from data source
            entity_info = self.data_source.entity_info.get(pathway_id, {})
            hierarchy.add_node(
                pathway_id,
                name=entity_info.get('name', pathway_id),
                type='Pathway',
                properties={}
            )
        
        # Skip relations for now as method signature is unclear
        # TODO: Implement relation retrieval when method is clarified
        
        return hierarchy
    
    def _build_hierarchical_view(self, context: RetrievedContext) -> Dict[str, Any]:
        """Build hierarchical view of retrieved context."""
        view = {
            'cellular': {},
            'molecular': {},
            'phenotype': {},
            'pathway': {}
        }
        
        # Process each entity in context
        for entity in context.entities:
            entity_type = entity['type']
            entity_id = entity['id']
            
            # Find relevant hierarchy
            if entity_type == 'CellularComponent':
                hierarchy = self.hierarchies['cellular']
                view_key = 'cellular'
            elif entity_type == 'MolecularFunction':
                hierarchy = self.hierarchies['molecular']
                view_key = 'molecular'
            elif entity_type == 'Phenotype':
                hierarchy = self.hierarchies['phenotype']
                view_key = 'phenotype'
            elif entity_type == 'Pathway':
                hierarchy = self.hierarchies['pathway']
                view_key = 'pathway'
            else:
                continue
            
            # Get ancestors and descendants
            if entity_id in hierarchy:
                ancestors = nx.ancestors(hierarchy, entity_id)
                descendants = nx.descendants(hierarchy, entity_id)
                
                view[view_key][entity_id] = {
                    'entity': entity,
                    'ancestors': [
                        {
                            'id': a,
                            'name': hierarchy.nodes[a]['name'],
                            'type': hierarchy.nodes[a]['type']
                        }
                        for a in ancestors
                    ],
                    'descendants': [
                        {
                            'id': d,
                            'name': hierarchy.nodes[d]['name'],
                            'type': hierarchy.nodes[d]['type']
                        }
                        for d in descendants
                    ]
                }
        
        return view
    
    def _cluster_pathways(self, context: RetrievedContext) -> List[Dict[str, Any]]:
        """Cluster pathways based on shared entities and relations."""
        # Extract pathway subgraphs
        pathway_graphs = []
        pathway_features = []
        
        # Helper function to check if an entity is a pathway
        def is_pathway_entity(entity_id: str, entity_name: str = None) -> bool:
            """Check if entity is a pathway using multiple methods."""
            # Method 1: Check entity_info from data_source (most reliable)
            if hasattr(self.data_source, 'entity_info') and entity_id in self.data_source.entity_info:
                entity_type = self.data_source.entity_info[entity_id].get('type', '').lower()
                if 'pathway' in entity_type:
                    return True
            
            # Method 2: Use improved type inference if available
            if entity_name and hasattr(self.data_source, '_infer_entity_type'):
                inferred_type = self.data_source._infer_entity_type(entity_name)
                if inferred_type.lower() == 'pathway':
                    return True
            
            # Method 3: Check entity name for pathway keywords
            if entity_name:
                name_lower = entity_name.lower()
                pathway_keywords = ['pathway', 'signaling pathway', 'metabolic pathway', 'biosynthetic pathway']
                if any(kw in name_lower for kw in pathway_keywords):
                    return True
            
            return False
        
        for subgraph in context.subgraphs:
            # Find pathway nodes using improved detection
            pathway_nodes = []
            for node_id in subgraph.nodes():
                # Get entity name from subgraph attributes or data_source
                node_attr = subgraph.nodes[node_id]
                entity_name = node_attr.get('name') or node_attr.get('label') or str(node_id)
                
                # Check if this is a pathway entity
                if is_pathway_entity(node_id, entity_name):
                    pathway_nodes.append(node_id)
            
            if pathway_nodes:
                # Extract pathway neighborhood
                for pathway in pathway_nodes:
                    neighborhood = nx.ego_graph(subgraph, pathway, radius=2)
                    pathway_graphs.append(neighborhood)
                    
                    # Create feature vector from neighborhood properties
                    features = self._extract_pathway_features(neighborhood)
                    pathway_features.append(features)
        
        if not pathway_features:
            return []
        
        # Handle single pathway case - AgglomerativeClustering requires at least 2 samples
        if len(pathway_features) == 1:
            # Return single pathway as a single cluster
            return [
                {
                    'pathways': [{
                        'graph': pathway_graphs[0],
                        'central_pathway': self._get_central_pathway(pathway_graphs[0]),
                        'related_entities': self._get_related_entities(pathway_graphs[0])
                    }],
                    'common_entities': [],
                    'biological_theme': self._infer_biological_theme([{
                        'graph': pathway_graphs[0]
                    }])
                }
            ]
        
        # Convert features to matrix
        feature_matrix = np.array(pathway_features)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(feature_matrix)
        
        # Perform hierarchical clustering
        # Use 'metric' instead of deprecated 'affinity' parameter (sklearn 1.2+)
        clustering = AgglomerativeClustering(
            n_clusters=min(len(pathway_features), 5),
            metric='precomputed',
            linkage='complete'
        )
        cluster_labels = clustering.fit_predict(1 - similarity_matrix)
        
        # Organize clusters
        clusters = defaultdict(list)
        for i, (graph, label) in enumerate(zip(pathway_graphs, cluster_labels)):
            clusters[label].append({
                'graph': graph,
                'central_pathway': self._get_central_pathway(graph),
                'related_entities': self._get_related_entities(graph)
            })
        
        return [
            {
                'pathways': cluster,
                'common_entities': self._find_common_entities(cluster),
                'biological_theme': self._infer_biological_theme(cluster)
            }
            for cluster in clusters.values()
        ]
    
    def _extract_pathway_features(self, graph: nx.Graph) -> np.ndarray:
        """Extract numerical features from pathway neighborhood."""
        # Fixed set of entity types (standardize feature vector length)
        standard_entity_types = [
            'drug', 'disease', 'protein', 'pathway', 'phenotype', 
            'anatomy', 'molecularfunction', 'biological_process', 
            'cellularcomponent', 'unknown'
        ]
        
        # Fixed set of common edge types (standardize feature vector length)
        standard_edge_types = [
            'drug_protein', 'protein_protein', 'disease_protein',
            'exposure_disease', 'drug_drug', 'disease_phenotype_positive',
            'contraindication', 'pathway_protein', 'bioprocess_protein',
            'unknown'
        ]
        
        # Node type distribution (fixed size)
        type_counts = defaultdict(int)
        for _, attr in graph.nodes(data=True):
            entity_type = attr.get('type', 'unknown')
            # Normalize type to lowercase for matching
            entity_type_lower = str(entity_type).lower()
            # Map to standard types
            if entity_type_lower in standard_entity_types:
                type_counts[entity_type_lower] += 1
            else:
                type_counts['unknown'] += 1
        
        # Edge type distribution (fixed size)
        edge_counts = defaultdict(int)
        for _, _, attr in graph.edges(data=True):
            edge_type = attr.get('type', 'unknown')
            edge_type_lower = str(edge_type).lower()
            # Map to standard types
            if edge_type_lower in standard_edge_types:
                edge_counts[edge_type_lower] += 1
            else:
                edge_counts['unknown'] += 1
        
        # Build fixed-size feature vector
        features = []
        
        # Entity type counts (fixed order, padded with 0 if missing)
        for entity_type in standard_entity_types:
            features.append(type_counts.get(entity_type, 0))
        
        # Edge type counts (fixed order, padded with 0 if missing)
        for edge_type in standard_edge_types:
            features.append(edge_counts.get(edge_type, 0))
        
        # Graph metrics
        # Handle both directed and undirected graphs
        if isinstance(graph, nx.DiGraph):
            # For directed graphs, use weakly_connected_components
            num_components = len(list(nx.weakly_connected_components(graph)))
            # Convert to undirected for density calculation
            graph_undirected = graph.to_undirected()
            graph_density = nx.density(graph_undirected)
        else:
            # For undirected graphs, use connected_components
            num_components = len(list(nx.connected_components(graph)))
            graph_density = nx.density(graph)
        
        features.extend([
            graph.number_of_nodes(),
            graph.number_of_edges(),
            graph_density,
            num_components
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _get_central_pathway(self, graph: nx.Graph) -> Dict[str, Any]:
        """Identify central pathway in the subgraph."""
        pathway_nodes = [
            (n, attr) for n, attr in graph.nodes(data=True)
            if attr.get('type') == 'Pathway'
        ]
        
        if not pathway_nodes:
            return {}
        
        # Use degree centrality to find most central pathway
        centrality = nx.degree_centrality(graph)
        central_node = max(
            pathway_nodes,
            key=lambda x: centrality[x[0]]
        )
        
        return {
            'id': central_node[0],
            'name': central_node[1].get('name', ''),
            'properties': central_node[1].get('properties', {})
        }
    
    def _get_related_entities(self, graph: nx.Graph) -> Dict[str, List[Dict[str, Any]]]:
        """Get entities related to the pathway."""
        related = defaultdict(list)
        
        for node, attr in graph.nodes(data=True):
            if attr.get('type') != 'Pathway':
                related[attr.get('type', 'unknown')].append({
                    'id': node,
                    'name': attr.get('name', ''),
                    'properties': attr.get('properties', {})
                })
        
        return dict(related)
    
    def _find_common_entities(self, cluster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find entities common to multiple pathways in cluster."""
        if not cluster:
            return []
        
        # Get all entities from each pathway
        pathway_entities = []
        for item in cluster:
            entities = set()
            for node, attr in item['graph'].nodes(data=True):
                if attr.get('type') != 'Pathway':
                    entities.add(node)
            pathway_entities.append(entities)
        
        # Find common entities
        common = set.intersection(*pathway_entities) if pathway_entities else set()
        
        # Get entity details
        return [
            {
                'id': entity_id,
                'name': cluster[0]['graph'].nodes[entity_id].get('name', ''),
                'type': cluster[0]['graph'].nodes[entity_id].get('type', ''),
                'properties': cluster[0]['graph'].nodes[entity_id].get('properties', {})
            }
            for entity_id in common
        ]
    
    def _infer_biological_theme(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer biological theme of the pathway cluster."""
        # Collect all entities and their types
        entity_types = defaultdict(int)
        relation_types = defaultdict(int)
        
        for item in cluster:
            graph = item['graph']
            
            # Count entity types
            for _, attr in graph.nodes(data=True):
                entity_types[attr.get('type', 'unknown')] += 1
            
            # Count relation types
            for _, _, attr in graph.edges(data=True):
                relation_types[attr.get('type', 'unknown')] += 1
        
        # Determine primary biological focus
        # Handle empty dicts - use defaults if no data
        if entity_types:
            primary_entity_type = max(entity_types.items(), key=lambda x: x[1])[0]
        else:
            primary_entity_type = 'unknown'
        
        if relation_types:
            primary_relation_type = max(relation_types.items(), key=lambda x: x[1])[0]
        else:
            primary_relation_type = 'unknown'
        
        return {
            'primary_focus': primary_entity_type,
            'main_relationship': primary_relation_type,
            'entity_distribution': dict(entity_types),
            'relation_distribution': dict(relation_types)
        }
    
    def _group_entities(self, context: RetrievedContext) -> Dict[str, List[Dict[str, Any]]]:
        """Group entities by type and relationships."""
        groups = defaultdict(list)
        
        # Group by entity type
        for entity in context.entities:
            groups[entity['type']].append(entity)
        
        # Add relationship context
        for entity_type, entities in groups.items():
            for entity in entities:
                # Find direct relationships
                relations = [
                    r for r in context.relations
                    if r['source_id'] == entity['id'] or r['target_id'] == entity['id']
                ]
                
                entity['relations'] = relations
                
                # Add supporting evidence
                entity['evidence'] = self.data_source.get_entity_evidence(entity['id'])
        
        return dict(groups)
    
    def _construct_evidence_chains(self, context: RetrievedContext) -> List[List[Dict[str, Any]]]:
        """Construct chains of evidence from relationships."""
        evidence_chains = []
        
        # Convert subgraphs to DiGraph for path finding
        for subgraph in context.subgraphs:
            digraph = nx.DiGraph(subgraph)
            
            # Find all pairs of entities
            entities = [
                n for n, attr in digraph.nodes(data=True)
                if 'type' in attr
            ]
            
            # Find paths between entity pairs
            for i, source in enumerate(entities):
                for target in entities[i+1:]:
                    try:
                        # Find shortest path
                        path = nx.shortest_path(digraph, source, target)
                        
                        # Convert path to evidence chain
                        chain = []
                        for j in range(len(path)-1):
                            source_id = path[j]
                            target_id = path[j+1]
                            
                            # Get edge data
                            edge_data = digraph.edges[source_id, target_id]
                            
                            chain.append({
                                'source': {
                                    'id': source_id,
                                    'name': digraph.nodes[source_id].get('name', ''),
                                    'type': digraph.nodes[source_id].get('type', '')
                                },
                                'target': {
                                    'id': target_id,
                                    'name': digraph.nodes[target_id].get('name', ''),
                                    'type': digraph.nodes[target_id].get('type', '')
                                },
                                # Prefer explicit 'relation' if present; fall back to 'type'
                                'relation': edge_data.get('relation') or edge_data.get('type', ''),
                                'evidence': edge_data.get('evidence', [])
                            })
                        
                        if chain:
                            evidence_chains.append(chain)
                    
                    except nx.NetworkXNoPath:
                        continue
        
        return evidence_chains

    # ---------------- Graph-to-Text and Ranking (simple) ----------------
    def _graph_to_text_paths(self, context: RetrievedContext) -> List[Dict[str, Any]]:
        """Convert subgraphs or relations into simple textual path snippets and rank them.
        This is a lightweight implementation to feed into generation.
        """
        snippets: List[Tuple[str, float]] = []
        
        # Create entity ID to name mapping
        entity_map = {e.get('id'): e.get('name', e.get('id', '')) for e in context.entities}
        
        # Convert explicit relations
        for rel in context.relations:
            s_id = rel.get('subject') or rel.get('source_id') or rel.get('source', {}).get('id')
            t_id = rel.get('object') or rel.get('target_id') or rel.get('target', {}).get('id')
            r = rel.get('predicate') or rel.get('type') or rel.get('relation')
            
            # Get entity names from mapping
            s_name = entity_map.get(s_id, s_id) if s_id else None
            t_name = entity_map.get(t_id, t_id) if t_id else None
            
            if s_name and t_name and r:
                text = f"{s_name} --[{r}]--> {t_name}"
                snippets.append((text, 0.5))
        
        # Convert subgraphs to short walks (length 2)
        for g in context.subgraphs:
            for u, v, attr in g.edges(data=True):
                r = attr.get('type') or attr.get('relation', '')
                # Get entity names
                u_name = g.nodes[u].get('name', entity_map.get(u, u))
                v_name = g.nodes[v].get('name', entity_map.get(v, v))
                if r and u_name and v_name:
                    text = f"{u_name} --[{r}]--> {v_name}"
                    snippets.append((text, 0.45))
        # If we have a query, score by semantic similarity using data source embeddings
        query_text = context.metadata.get('query') if context.metadata else None
        scores: List[float] = []
        if query_text:
            try:
                query_vec = self.data_source.encode_texts([query_text])[0]
                text_vecs = self.data_source.encode_texts([s for s, _ in snippets])
                for i, (s, base) in enumerate(snippets):
                    sim = float(np.dot(text_vecs[i], query_vec))
                    scores.append(0.7 * sim + 0.3 * base)
            except Exception:
                scores = [base for _, base in snippets]
        else:
            scores = [base for _, base in snippets]

        # Simple inverse frequency penalty
        from collections import Counter
        counts = Counter([s for s, _ in snippets])
        ranked = sorted([
            (snippets[i][0], scores[i] / (1.0 + counts[snippets[i][0]]))
            for i in range(len(snippets))
        ], key=lambda x: x[1], reverse=True)

        return [{'text': s, 'score': float(score)} for s, score in ranked[:100]]
    
    def _collect_metadata(self, context: RetrievedContext) -> Dict[str, Any]:
        """Collect metadata about the organized context."""
        return {
            'entity_count': len(context.entities),
            'relation_count': len(context.relations),
            'subgraph_count': len(context.subgraphs),
            'entity_types': list(set(e['type'] for e in context.entities)),
            'relation_types': list(set(r['type'] for r in context.relations)),
            **context.metadata
        }
