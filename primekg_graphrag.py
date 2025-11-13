#!/usr/bin/env python3
"""
PrimeKG GraphRAG Command Line Interface

Usage:
    python primekg_graphrag.py "How does metformin work to treat type 2 diabetes?"
    python primekg_graphrag.py "What are the side effects of aspirin?"
    python primekg_graphrag.py "What genes are associated with breast cancer?"

Features:
- Complete GraphRAG pipeline execution
- Colored output for better readability
- Progress indicators
- Detailed or simple output modes
- Error handling and graceful fallbacks
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from src.query_processor import QueryProcessor
    from src.retriever import GraphRetriever
    from src.organizer import PrimeKGOrganizer
    from src.generator import PrimeKGGenerator
    from src.graph_data_source import PrimeKGDataSource
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def colored(text, color):
    """Apply color to text if terminal supports it."""
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        return f"{color}{text}{Colors.END}"
    return text

def print_progress(message, color=Colors.CYAN):
    """Print a progress message."""
    print(colored(f"[INFO] {message}", color))

def print_success(message):
    """Print a success message."""
    print(colored(f"[SUCCESS] {message}", Colors.GREEN))

def print_error(message):
    """Print an error message."""
    print(colored(f"[ERROR] {message}", Colors.RED))

def print_warning(message):
    """Print a warning message."""
    print(colored(f"[WARNING] {message}", Colors.YELLOW))

def print_header(title):
    """Print a section header."""
    print("\n" + colored("="*60, Colors.BLUE))
    print(colored(f"{title}", Colors.BLUE + Colors.BOLD))
    print(colored("="*60, Colors.BLUE))

class GraphRAGCLI:
    """Command line interface for PrimeKG GraphRAG."""
    
    def __init__(self, verbose=False):
        """Initialize the CLI with components."""
        self.verbose = verbose
        self.data_source = None
        self.query_processor = None
        self.retriever = None
        self.organizer = None
        self.generator = None
        
    def initialize_components(self):
        """Initialize all GraphRAG components."""
        print_header("INITIALIZING PRIMEKG GRAPHRAG")
        
        start_time = time.time()
        
        try:
            print_progress("Loading PrimeKG data source...")
            self.data_source = PrimeKGDataSource()
            success = self.data_source.load()
            
            if not success:
                print_error("Failed to load PrimeKG data source")
                return False
            
            stats = self.data_source.get_graph_statistics()
            print_success(f"Loaded PrimeKG: {stats.get('total_nodes', 0):,} nodes, {stats.get('total_edges', 0):,} edges")
            
            print_progress("Initializing pipeline components...")
            self.query_processor = QueryProcessor(self.data_source)
            self.retriever = GraphRetriever(self.data_source, self.query_processor)
            self.organizer = PrimeKGOrganizer(self.data_source)
            self.generator = PrimeKGGenerator(self.data_source)
            
            init_time = time.time() - start_time
            print_success(f"Initialization complete in {init_time:.1f}s")
            return True
            
        except Exception as e:
            print_error(f"Initialization failed: {e}")
            return False
    
    def process_query(self, query):
        """Process a single query through the GraphRAG pipeline."""
        if not self.data_source:
            print_error("Components not initialized. Run initialization first.")
            return None
        
        print_header("PROCESSING QUERY")
        print(colored(f'"{query}"', Colors.BOLD))
        
        start_time = time.time()
        
        try:
            # Step 1: Query Processing
            if self.verbose:
                print_progress("Step 1: Query Processing & Entity Recognition")
            
            components, graph_query = self.query_processor.process_query_for_retrieval(query)
            
            if self.verbose:
                print(f"   Query type: {colored(components.query_type, Colors.CYAN)}")
                print(f"   Entities found: {colored(str(len(components.entities)), Colors.GREEN)}")
                for entity in components.entities:
                    print(f"     • {entity['text']} ({entity['type']}) - confidence: {entity['confidence']:.3f}")
                print(f"   Relations: {components.relations}")
            
            # Step 2: Graph Retrieval
            if self.verbose:
                print_progress("Step 2: Graph Retrieval & Path Finding")
            
            retrieval_result = self.retriever.retrieve(query)
            retrieved_context = self.retriever.to_retrieved_context(retrieval_result)
            
            if self.verbose:
                print(f"   Retrieved {colored(str(len(retrieval_result.entities)), Colors.GREEN)} entities")
                print(f"   Found {colored(str(len(retrieval_result.relationships)), Colors.GREEN)} relationships")
                print(f"   Discovered {colored(str(len(retrieval_result.paths)), Colors.GREEN)} connecting paths")
            
            # Step 3: Context Organization
            if self.verbose:
                print_progress("Step 3: Context Organization & Ranking")
            
            organized_context = self.organizer.organize_context(retrieved_context)
            
            if self.verbose:
                print(f"   Created {colored(str(len(organized_context.pathway_clusters)), Colors.GREEN)} pathway clusters")
                print(f"   Found {colored(str(len(organized_context.evidence_chains)), Colors.GREEN)} evidence chains")
                print(f"   Generated {colored(str(len(organized_context.ranked_text_paths)), Colors.GREEN)} ranked text paths")
            
            # Step 4: Response Generation
            if self.verbose:
                print_progress("Step 4: Natural Language Generation")
            
            response = self.generator.generate_response(query, organized_context)
            
            processing_time = time.time() - start_time
            
            if self.verbose:
                print(f"   Response confidence: {colored(f'{response.confidence:.3f}', Colors.GREEN)}")
                print(f"   Processing time: {colored(f'{processing_time:.1f}s', Colors.CYAN)}")
            
            return {
                'query': query,
                'response': response,
                'components': components,
                'retrieval_result': retrieval_result,
                'organized_context': organized_context,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print_error(f"Query processing failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def display_result(self, result):
        """Display the query result in a formatted way."""
        if not result:
            return
        
        response = result['response']
        organized_context = result['organized_context']
        
        print_header("FINAL ANSWER")
        print(response.text)
        
        if self.verbose and organized_context.ranked_text_paths:
            print_header("TOP KNOWLEDGE PATHS")
            for i, path in enumerate(organized_context.ranked_text_paths[:5], 1):
                score_color = Colors.GREEN if path['score'] > 0.3 else Colors.YELLOW
                score_text = f"(score: {path['score']:.3f})"
                print(f"{i:2d}. {path['text']} {colored(score_text, score_color)}")
        
        if self.verbose and response.reasoning_steps:
            print_header("REASONING STEPS")
            for i, step in enumerate(response.reasoning_steps, 1):
                print(f"{i:2d}. {step.get('statement', 'No statement')}")
                if step.get('evidence_type'):
                    print(f"    Evidence: {step['evidence_type']}")
        
        # Summary
        confidence_color = Colors.GREEN if response.confidence > 0.5 else Colors.YELLOW if response.confidence > 0.3 else Colors.RED
        confidence_text = f"{response.confidence:.3f}"
        time_text = f"{result['processing_time']:.1f}s"
        print(f"\n{colored('Confidence:', Colors.BOLD)} {colored(confidence_text, confidence_color)} | "
              f"{colored('Processing time:', Colors.BOLD)} {time_text}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PrimeKG GraphRAG: Biomedical Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python primekg_graphrag.py "How does metformin work to treat type 2 diabetes?"
  python primekg_graphrag.py "What are the side effects of aspirin?" --verbose
  python primekg_graphrag.py "What genes are associated with breast cancer?"
        """
    )
    
    parser.add_argument('query', help='Biomedical query to process')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Show detailed processing steps')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Initialize CLI
    cli = GraphRAGCLI(verbose=args.verbose)
    
    # Initialize components
    if not cli.initialize_components():
        sys.exit(1)
    
    # Process query
    result = cli.process_query(args.query)
    
    if result:
        cli.display_result(result)
        print_success("Query processed successfully!")
    else:
        print_error("Query processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()