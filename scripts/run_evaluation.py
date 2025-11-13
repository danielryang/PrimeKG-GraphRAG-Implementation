"""
Run comprehensive evaluation of GraphRAG pipeline.

Usage:
    python scripts/run_evaluation.py [--dataset data/benchmark_dataset.json] [--output results/evaluation.json]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluator import GraphRAGEvaluator, load_benchmark_dataset
from src import GraphRAG


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG pipeline")
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/benchmark_dataset.json',
        help='Path to benchmark dataset JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation_results.json',
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--initialize',
        action='store_true',
        help='Initialize GraphRAG before evaluation (required for first run)'
    )
    
    args = parser.parse_args()
    
    # Load benchmark dataset
    print(f"Loading benchmark dataset from {args.dataset}...")
    try:
        test_cases = load_benchmark_dataset(args.dataset)
        print(f"Loaded {len(test_cases)} test cases")
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {args.dataset}")
        print("Please create a benchmark dataset file or specify a different path.")
        return 1
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Initialize GraphRAG
    print("\nInitializing GraphRAG pipeline...")
    graphrag = GraphRAG()
    
    if args.initialize or not graphrag.is_initialized:
        print("Loading data and initializing components...")
        if not graphrag.initialize():
            print("Error: Failed to initialize GraphRAG pipeline")
            return 1
        print("GraphRAG initialized successfully")
    else:
        print("Using existing GraphRAG instance")
    
    # Create evaluator
    evaluator = GraphRAGEvaluator(graphrag)
    
    # Run evaluation
    print(f"\nRunning evaluation on {len(test_cases)} test cases...")
    print("=" * 60)
    
    results = evaluator.evaluate_benchmark(test_cases)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    
    agg = results['aggregate_metrics']
    print(f"\nOverall Score: {results['overall_score']:.2%}")
    print(f"\nAccuracy Metrics:")
    print(f"  Entity Extraction: {agg['accuracy']['entity_extraction']:.2%}")
    print(f"  Entity Type: {agg['accuracy']['entity_type']:.2%}")
    print(f"  Relationship Retrieval: {agg['accuracy']['relationship_retrieval']:.2%}")
    print(f"  Answer Accuracy: {agg['accuracy']['answer']:.2%}")
    
    print(f"\nBiological Consistency:")
    print(f"  Overall: {agg['biological_consistency']['overall']:.2%}")
    print(f"  Entity Type Consistency: {agg['biological_consistency']['entity_type_consistency']:.2%}")
    
    print(f"\nPerformance:")
    print(f"  Average Query Time: {agg['performance']['total']:.2f}s")
    print(f"  Retrieval Time: {agg['performance']['retrieval']:.2f}s")
    
    print(f"\nCoverage:")
    print(f"  Average Entities Retrieved: {agg['coverage']['entities']:.1f}")
    print(f"  Average Relationships Retrieved: {agg['coverage']['relationships']:.1f}")
    print(f"  Coverage Score: {agg['coverage']['score']:.2%}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(results, str(output_path))
    
    print(f"\nDetailed results saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())




