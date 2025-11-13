# PrimeKG GraphRAG

Graph-based Retrieval-Augmented Generation for biomedical question answering using PrimeKG knowledge graph (6.48M edges, 129K entities, 20+ curated sources).

## Features


- **Curated Knowledge Graph Foundation**: Operates over PrimeKG (6.48M edges from 20+ sources including DrugBank, DISEASES, GO, Reactome) rather than extracting graphs from unstructured text, ensuring biological accuracy and eliminating hallucination from document parsing errors
- **Interpretable Reasoning**: Agent-based exploration where LLM makes explicit, auditable decisions about which entities and relationships to explore, with complete reasoning chains explaining why each path was chosen
- **Reproducible Evaluation**: Ground-truth benchmarks with quantitative metrics across accuracy, biological consistency, coverage, and quality dimensions, enabling systematic comparison and validation


- **Biomedical Domain Specialization**: Entity recognition using trained scientific NER models (spaCy en_core_sci_md from scispacy) instead of general-purpose extractors, correctly identifying drugs, diseases, genes, and proteins in context
- **Controlled Graph Exploration**: Heuristic pre-filtering + LLM borderline evaluation prevents exponential explosion (stops at relevance thresholds instead of expanding to 1000+ irrelevant entities), solving the graph traversal scalability problem
- **Biological Context Organization**: ML-based pathway clustering and multi-level hierarchy construction (cellular components, molecular functions, phenotypes) that understands biological relationships rather than generic community detection
- **Full Source Provenance**: Every claim links back to specific PrimeKG relationships with original database sources (DrugBank IDs, GO terms, etc.), enabling verification and confidence assessment

- **Validation**: Evaluation framework with benchmark dataset containing expected entities, relationships, and answers for reproducible performance assessment

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/PrimeKG-GraphRAG-Implementation.git
cd PrimeKG-GraphRAG-Implementation

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
python scripts/install_spacy_biomedical.py

cp .env.example .env  # Edit with your API keys if using LLM generation
```

### Basic Usage

**Python API:**
```python
from src.graph_data_source import PrimeKGDataSource
from src.retriever import GraphRetriever
from src.organizer import PrimeKGOrganizer
from src.generator import PrimeKGGenerator

# Initialize
data_source = PrimeKGDataSource()
data_source.load()

retriever = GraphRetriever(data_source)
organizer = PrimeKGOrganizer(data_source)
generator = PrimeKGGenerator(data_source)

# Query
query = "How does metformin treat type 2 diabetes?"
retrieval_result = retriever.retrieve(query)
context = retriever.to_retrieved_context(retrieval_result)
organized = organizer.organize_context(context)
response = generator.generate_response(query, organized)

print(response.answer)
```

**CLI:**
```bash
python primekg_graphrag.py "What are the side effects of aspirin?"
```

**Jupyter Notebook:**
```bash
jupyter notebook notebooks/primekg_graphrag_biomedical_qa_interface.ipynb
```

Interactive demo with visualization, metrics analysis, and adjustable parameters.

## Architecture

Four-stage pipeline over PrimeKG (DrugBank, DISEASES, GO, Reactome, etc.):

| Stage | Function | Key Techniques |
|-------|----------|----------------|
| **Query Processing** | Entity extraction & intent classification | Biomedical NER → lexical → semantic search; 6 query types |
| **Graph Retrieval** | Agent-based exploration | LLM-guided decisions, heuristic pre-filtering, relevance-based early stopping |
| **Context Organization** | Biological structuring | Pathway clustering (agglomerative), hierarchy construction (cellular/molecular/phenotype) |
| **Response Generation** | NL synthesis with provenance | Dual strategy (LLM + template fallback), multi-factor confidence scoring |

**Backend Options:**
- **Neo4j**: Cypher-optimized, sub-second queries, distributed scaling
- **PyKEEN + NetworkX**: In-memory, 350MB cached, zero external dependencies

## Evaluation

```bash
python scripts/run_evaluation.py --dataset data/benchmark_dataset.json --output results/evaluation.json
```

**Metrics:**
- **Accuracy**: Entity extraction, type classification, relationship retrieval, answer correctness
- **Biological Consistency**: Pathway coherence, entity type consistency
- **Performance**: Per-stage and end-to-end latency
- **Coverage**: Retrieved vs expected entities/relationships
- **Quality**: Confidence scoring, reasoning chain coherence, evidence diversity

Includes benchmark dataset with 5+ validated queries.

## Configuration

Key `.env` variables:

```bash
# Backend (optional, defaults to PyKEEN if not provided)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword

# Processing limits
MAX_ENTITIES=50
MAX_RELATIONSHIPS=100
MAX_HOPS=2

# LLM APIs (optional, uses template fallback if not provided)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Performance tuning
SIMILARITY_THRESHOLD=0.3
QUERY_TIMEOUT=30
```

### Neo4j Setup (Optional)

```bash
python setup_neo4j.py

# Or manual Docker
docker run -d --name primekg-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/primekg123 \
  neo4j:latest
```

## Project Structure

```
src/
├── graph_data_source.py    # PrimeKG data access (Neo4j/PyKEEN)
├── retriever.py             # Agent-based graph exploration
├── organizer.py             # Biological hierarchy construction
├── generator.py             # NL response generation
├── evaluator.py             # Evaluation framework
└── llm_client.py            # LLM API integration

scripts/
├── run_evaluation.py                # Benchmark runner
└── install_spacy_biomedical.py      # NER model installer

notebooks/
└── primekg_graphrag_biomedical_qa_interface.ipynb  # Interactive demo

```

## Research Impact

**Core Contribution**: Interpretable biomedical QA through structured graph reasoning with full provenance tracking over curated biomedical knowledge.

**Key Innovations**:
- Multi-stage entity grounding against PrimeKG ontology
- Intelligent exploration with heuristic pre-filtering and LLM borderline decisions
- Biological context organization via ML-based pathway clustering
- Resilient dual-generation strategy with multi-factor confidence scoring
- Standardized evaluation framework for reproducible benchmarking

## Limitations

**Knowledge Graph Constraints**:
- **Coverage Bounds**: Limited to entities and relationships present in PrimeKG; novel drugs, diseases, or recently discovered pathways not yet curated into PrimeKG cannot be retrieved
- **Update Latency**: PrimeKG snapshot represents knowledge at time of curation; latest research findings may not be reflected until database updates

**System Design Trade-offs**:
- **Domain Specificity**: Optimized for biomedical queries; not suitable for general-domain question answering without significant architectural changes
- **Entity Ambiguity**: Homonyms and ambiguous entity names (e.g., "insulin" as drug vs protein) may cause errors despite multi-stage resolution
- **Computational Complexity**: Agent-based exploration with LLM calls introduces latency (2-5s per query); heuristic pre-filtering partially mitigates but cannot eliminate for complex multi-hop queries

**Operational Requirements**:
- **NER Model Dependency**: Requires installation of specialized scispacy models (en_core_sci_md); system falls back to regex patterns with reduced accuracy if unavailable
- **LLM API Costs**: Full agent-based reasoning requires LLM API access; template-based fallback provides basic functionality but loses intelligent exploration capabilities
- **Neo4j Setup Complexity**: Production-scale deployment benefits significantly from Neo4j backend; PyKEEN fallback works but limited to single-machine memory constraints

**Evaluation Scope**:
- **Benchmark Size**: Current evaluation dataset contains 5+ validated queries; larger-scale clinical validation needed for production medical applications
- **Ground Truth Limitations**: Expected entities/relationships manually curated; may not capture all valid biological pathways for complex mechanisms

## References

- **PrimeKG**: Chandak et al. (2023). "Building a knowledge graph to enable precision medicine." *Scientific Data*.
- **Graph RAG Survey**: Peng et al. (2024). "Graph Retrieval-Augmented Generation: A Survey." arXiv:2408.08921.

## License

MIT License - See LICENSE file for details.
