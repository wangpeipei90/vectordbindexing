#!/usr/bin/env python3
"""
Simple CLI script to run HNSW optimization experiments
"""

import argparse
import sys
import logging
from pathlib import Path

from experiment_runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(
        description='Run HNSW optimization experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default=None, 
                       help='Dataset file path (if None, generates synthetic data)')
    parser.add_argument('--n_vectors', type=int, default=10000,
                       help='Number of vectors in dataset')
    parser.add_argument('--n_queries', type=int, default=1000,
                       help='Number of query vectors')
    parser.add_argument('--dimension', type=int, default=128,
                       help='Vector dimension')
    parser.add_argument('--n_modalities', type=int, default=5,
                       help='Number of modalities/clusters')
    
    # HNSW parameters
    parser.add_argument('--M', type=int, default=16,
                       help='HNSW connectivity parameter')
    parser.add_argument('--ef_construction', type=int, default=200,
                       help='HNSW construction search width')
    
    # Optimization parameters
    parser.add_argument('--bridge_budget', type=float, default=1e-5,
                       help='Bridge budget ratio (max new edges / total edges)')
    parser.add_argument('--m_values', nargs='+', type=int, default=[2, 4, 8],
                       help='Multi-entry seed values to test')
    parser.add_argument('--ef_search_values', nargs='+', type=int, default=[50, 100, 200, 400],
                       help='ef_search values to test')
    
    # Evaluation parameters
    parser.add_argument('--k_eval', type=int, default=100,
                       help='Number of neighbors for evaluation')
    parser.add_argument('--target_recall', type=float, default=0.90,
                       help='Target recall threshold')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("HNSW Optimization Experiment")
    print("=" * 50)
    print(f"Dataset: {args.n_vectors} vectors, {args.n_queries} queries, {args.dimension}D")
    print(f"Modalities: {args.n_modalities}")
    print(f"HNSW: M={args.M}, ef_construction={args.ef_construction}")
    print(f"Bridge budget: {args.bridge_budget}")
    print(f"Multi-entry seeds: {args.m_values}")
    print(f"ef_search values: {args.ef_search_values}")
    print(f"Target recall: {args.target_recall}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Run experiment
        runner = ExperimentRunner(output_dir=args.output_dir, seed=args.seed)
        runner.run_full_experiment(
            dataset_path=args.dataset,
            n_vectors=args.n_vectors,
            n_queries=args.n_queries,
            dimension=args.dimension,
            n_modalities=args.n_modalities,
            M=args.M,
            ef_construction=args.ef_construction,
            bridge_budget_ratio=args.bridge_budget,
            m_values=args.m_values,
            ef_search_values=args.ef_search_values,
            k_eval=args.k_eval,
            target_recall=args.target_recall
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {output_path.absolute()}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running experiment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
