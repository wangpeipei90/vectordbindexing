#!/usr/bin/env python3
"""
Experiment runner for HNSW optimization evaluation
"""

import numpy as np
import pandas as pd
import logging
import time
import argparse
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import DataLoader, create_toy_dataset
from gt_utils import GroundTruthComputer, benchmark_search_method, find_ef_for_recall, compute_percentiles
from hnsw_baseline import HNSWBaseline, FAISSBaseline
from bridge_builder import BridgeBuilder
from multi_entry_search import MultiEntrySearch, AdaptiveMultiEntrySearch

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner for HNSW optimization evaluation"""

    def __init__(self,
                 output_dir: str = "results",
                 seed: int = 42):
        """
        Initialize experiment runner

        Args:
            output_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.seed = seed

        # Results storage
        self.results = {}

    def load_dataset(self,
                     dataset_path: Optional[str] = None,
                     n_vectors: int = 10000,
                     n_queries: int = 1000,
                     dimension: int = 128,
                     n_modalities: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load or generate dataset"""
        logger.info("Loading dataset...")

        if dataset_path:
            loader = DataLoader(dataset_path)
            X, Q, modalities, query_modalities = loader.load_data()
        else:
            X, Q, modalities, query_modalities = create_toy_dataset(
                n_vectors, n_queries, dimension, n_modalities, self.seed
            )

        logger.info(
            f"Dataset loaded: {len(X)} vectors, {len(Q)} queries, dim={X.shape[1]}")
        return X, Q, modalities, query_modalities

    def compute_ground_truth(self,
                             X: np.ndarray,
                             Q: np.ndarray,
                             k_eval: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ground truth nearest neighbors"""
        logger.info("Computing ground truth...")

        gt_computer = GroundTruthComputer()
        gt_neighbors, gt_distances = gt_computer.compute_ground_truth(
            X, Q, k_eval)

        return gt_neighbors, gt_distances

    def build_baseline_indexes(self,
                               X: np.ndarray,
                               M: int = 16,
                               ef_construction: int = 200) -> Tuple[HNSWBaseline, FAISSBaseline]:
        """Build baseline HNSW and FAISS indexes"""
        logger.info("Building baseline indexes...")

        # Build HNSW baseline
        hnsw_baseline = HNSWBaseline(
            dimension=X.shape[1],
            M=M,
            ef_construction=ef_construction,
            seed=self.seed
        )
        hnsw_baseline.build_index(X)

        # Build FAISS baseline
        faiss_baseline = FAISSBaseline(dimension=X.shape[1])
        faiss_baseline.build_index(X)

        logger.info("Baseline indexes built")
        return hnsw_baseline, faiss_baseline

    def build_optimized_index(self,
                              X: np.ndarray,
                              modalities: np.ndarray,
                              M: int = 16,
                              ef_construction: int = 200,
                              bridge_budget_ratio: float = 1e-5) -> Tuple[HNSWBaseline, BridgeBuilder]:
        """Build optimized HNSW index with bridge edges"""
        logger.info("Building optimized index with bridge edges...")

        # Build HNSW baseline
        hnsw_baseline = HNSWBaseline(
            dimension=X.shape[1],
            M=M,
            ef_construction=ef_construction,
            seed=self.seed
        )
        hnsw_baseline.build_index(X)

        # Build bridge edges
        bridge_builder = BridgeBuilder(
            max_bridge_per_node=2,
            bridge_budget_ratio=bridge_budget_ratio
        )
        bridge_map = bridge_builder.build_bridges(hnsw_baseline, X, modalities)

        logger.info("Optimized index built")
        return hnsw_baseline, bridge_builder

    def run_baseline_evaluation(self,
                                hnsw_baseline: HNSWBaseline,
                                Q: np.ndarray,
                                gt_neighbors: np.ndarray,
                                ef_search_values: List[int] = [
                                    50, 100, 200, 400],
                                k_eval: int = 100,
                                target_recall: float = 0.90) -> Dict[str, Any]:
        """Run baseline HNSW evaluation"""
        logger.info("Running baseline evaluation...")

        def baseline_search_func(query, ef, k):
            return hnsw_baseline.search(query, k, ef)

        # Benchmark baseline
        baseline_results = benchmark_search_method(
            X=None, Q=Q, search_func=baseline_search_func,
            ef_values=ef_search_values, k=k_eval
        )

        # Find ef values that achieve target recall
        query_ef_mapping = find_ef_for_recall(baseline_results, target_recall)

        # Compute cost percentiles for queries that achieve target recall
        valid_queries = [qid for qid,
                         ef in query_ef_mapping.items() if ef is not None]

        if valid_queries:
            costs_at_target_recall = []
            for qid in valid_queries:
                ef = query_ef_mapping[qid]
                cost = baseline_results[ef]['costs'][qid]
                costs_at_target_recall.append(cost)

            percentiles = compute_percentiles(np.array(costs_at_target_recall))
        else:
            percentiles = {}

        baseline_eval = {
            'results': baseline_results,
            'query_ef_mapping': query_ef_mapping,
            'valid_queries': valid_queries,
            'costs_at_target_recall': costs_at_target_recall if valid_queries else [],
            'percentiles': percentiles
        }

        logger.info(
            f"Baseline evaluation completed: {len(valid_queries)} queries achieved target recall")
        return baseline_eval

    def run_optimized_evaluation(self,
                                 hnsw_baseline: HNSWBaseline,
                                 bridge_builder: BridgeBuilder,
                                 Q: np.ndarray,
                                 gt_neighbors: np.ndarray,
                                 m_values: List[int] = [2, 4, 8],
                                 ef_search_values: List[int] = [
                                     50, 100, 200, 400],
                                 k_eval: int = 100) -> Dict[str, Any]:
        """Run optimized HNSW evaluation with multi-entry search"""
        logger.info("Running optimized evaluation...")

        # Create multi-entry search
        multi_search = MultiEntrySearch(hnsw_baseline, bridge_builder)

        optimized_results = {}

        for m in m_values:
            logger.info(f"Testing m={m}")

            def optimized_search_func(query, ef, k):
                return multi_search.multi_entry_search(query, k, m, ef)

            # Benchmark optimized search
            m_results = benchmark_search_method(
                X=None, Q=Q, search_func=optimized_search_func,
                ef_values=ef_search_values, k=k_eval
            )

            optimized_results[f'm{m}'] = m_results

        logger.info("Optimized evaluation completed")
        return optimized_results

    def run_comparison_experiment(self,
                                  baseline_eval: Dict[str, Any],
                                  optimized_results: Dict[str, Any],
                                  Q_eval: List[int],
                                  Q: np.ndarray,
                                  k_eval: int = 100) -> pd.DataFrame:
        """Run comparison experiment on evaluation subset"""
        logger.info("Running comparison experiment...")

        comparison_data = []

        for m_key, m_results in optimized_results.items():
            m = int(m_key[1:])  # Extract m value

            for ef in m_results.keys():
                baseline_ef_results = baseline_eval['results'].get(ef, {})

                for query_id in Q_eval:
                    if query_id < len(Q):
                        # Get baseline results
                        baseline_cost = baseline_ef_results.get(
                            'costs', [0] * len(Q))[query_id]
                        baseline_recall = baseline_ef_results.get(
                            'recalls', [0] * len(Q))[query_id]
                        baseline_latency = baseline_ef_results.get(
                            'latencies', [0] * len(Q))[query_id]

                        # Get optimized results
                        optimized_cost = m_results[ef]['costs'][query_id]
                        optimized_recall = m_results[ef]['recalls'][query_id]
                        optimized_latency = m_results[ef]['latencies'][query_id]

                        comparison_data.append({
                            'query_id': query_id,
                            'm': m,
                            'ef': ef,
                            'baseline_cost': baseline_cost,
                            'optimized_cost': optimized_cost,
                            'baseline_recall': baseline_recall,
                            'optimized_recall': optimized_recall,
                            'baseline_latency': baseline_latency,
                            'optimized_latency': optimized_latency,
                            'cost_ratio': optimized_cost / baseline_cost if baseline_cost > 0 else 1.0,
                            'recall_improvement': optimized_recall - baseline_recall,
                            'latency_ratio': optimized_latency / baseline_latency if baseline_latency > 0 else 1.0
                        })

        comparison_df = pd.DataFrame(comparison_data)

        logger.info(
            f"Comparison experiment completed: {len(comparison_df)} measurements")
        return comparison_df

    def generate_percentile_table(self,
                                  comparison_df: pd.DataFrame,
                                  percentiles: List[int] = [10, 25, 50, 75, 90, 95, 99]) -> pd.DataFrame:
        """Generate percentile comparison table"""
        logger.info("Generating percentile table...")

        # Group by m and ef
        percentile_data = []

        for m in comparison_df['m'].unique():
            for ef in comparison_df['ef'].unique():
                subset = comparison_df[(comparison_df['m'] == m) & (
                    comparison_df['ef'] == ef)]

                if len(subset) == 0:
                    continue

                baseline_costs = subset['baseline_cost'].values
                optimized_costs = subset['optimized_cost'].values

                for p in percentiles:
                    baseline_p = np.percentile(baseline_costs, p)
                    optimized_p = np.percentile(optimized_costs, p)

                    percentile_data.append({
                        'm': m,
                        'ef': ef,
                        'percentile': p,
                        'baseline_cost': baseline_p,
                        'optimized_cost': optimized_p,
                        'ratio': optimized_p / baseline_p if baseline_p > 0 else 1.0
                    })

        percentile_df = pd.DataFrame(percentile_data)

        logger.info("Percentile table generated")
        return percentile_df

    def plot_results(self,
                     comparison_df: pd.DataFrame,
                     percentile_df: pd.DataFrame,
                     save_plots: bool = True):
        """Generate and save result plots"""
        logger.info("Generating plots...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Plot 1: Cost ratio distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Cost ratio by m
        sns.boxplot(data=comparison_df, x='m', y='cost_ratio', ax=axes[0, 0])
        axes[0, 0].set_title('Cost Ratio Distribution by M')
        axes[0, 0].set_ylabel('Optimized Cost / Baseline Cost')
        axes[0, 0].axhline(y=1, color='r', linestyle='--', alpha=0.7)

        # Recall improvement by m
        sns.boxplot(data=comparison_df, x='m',
                    y='recall_improvement', ax=axes[0, 1])
        axes[0, 1].set_title('Recall Improvement by M')
        axes[0, 1].set_ylabel('Optimized Recall - Baseline Recall')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)

        # Latency ratio by m
        sns.boxplot(data=comparison_df, x='m',
                    y='latency_ratio', ax=axes[1, 0])
        axes[1, 0].set_title('Latency Ratio Distribution by M')
        axes[1, 0].set_ylabel('Optimized Latency / Baseline Latency')
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.7)

        # Percentile comparison
        if not percentile_df.empty:
            best_m = percentile_df[percentile_df['percentile']
                                   == 95]['ratio'].idxmin()
            best_m_value = percentile_df.loc[best_m, 'm']
            best_ef = percentile_df.loc[best_m, 'ef']
        else:
            best_m_value = 4
            best_ef = 200

        best_subset = percentile_df[(percentile_df['m'] == best_m_value) & (
            percentile_df['ef'] == best_ef)]
        if not best_subset.empty:
            axes[1, 1].plot(best_subset['percentile'],
                            best_subset['ratio'], 'o-', linewidth=2, markersize=8)
        else:
            # 如果没有数据，绘制一条水平线
            axes[1, 1].axhline(y=1, color='b', linestyle='-', alpha=0.7)
        axes[1, 1].set_title(
            f'Cost Ratio by Percentile (m={best_m_value}, ef={best_ef})')
        axes[1, 1].set_xlabel('Percentile')
        axes[1, 1].set_ylabel('Optimized Cost / Baseline Cost')
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig(self.output_dir / 'comparison_results.png',
                        dpi=300, bbox_inches='tight')

        # Plot 2: Detailed analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Cost vs Recall scatter
        scatter = axes[0].scatter(comparison_df['baseline_cost'], comparison_df['baseline_recall'],
                                  c=comparison_df['cost_ratio'], cmap='RdYlBu_r', alpha=0.6, s=50)
        axes[0].set_xlabel('Baseline Cost')
        axes[0].set_ylabel('Baseline Recall')
        axes[0].set_title('Baseline: Cost vs Recall')
        plt.colorbar(scatter, ax=axes[0], label='Cost Ratio')

        # Optimized vs Baseline recall
        axes[1].scatter(comparison_df['baseline_recall'], comparison_df['optimized_recall'],
                        c=comparison_df['m'], cmap='viridis', alpha=0.6, s=50)
        axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.7)
        axes[1].set_xlabel('Baseline Recall')
        axes[1].set_ylabel('Optimized Recall')
        axes[1].set_title('Recall Comparison')

        # Cost improvement distribution
        cost_improvement = 1 - comparison_df['cost_ratio']
        axes[2].hist(cost_improvement, bins=30, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Cost Improvement (1 - ratio)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Cost Improvement Distribution')
        axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_plots:
            plt.savefig(self.output_dir / 'detailed_analysis.png',
                        dpi=300, bbox_inches='tight')

        logger.info("Plots generated and saved")

    def save_results(self,
                     comparison_df: pd.DataFrame,
                     percentile_df: pd.DataFrame,
                     baseline_eval: Dict[str, Any],
                     optimized_results: Dict[str, Any]):
        """Save all results to files"""
        logger.info("Saving results...")

        # Save DataFrames
        comparison_df.to_csv(
            self.output_dir / 'comparison_results.csv', index=False)
        percentile_df.to_csv(
            self.output_dir / 'percentile_analysis.csv', index=False)

        # Save detailed results
        with open(self.output_dir / 'baseline_evaluation.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            baseline_serializable = {}
            for key, value in baseline_eval.items():
                if isinstance(value, dict):
                    baseline_serializable[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            baseline_serializable[key][subkey] = subvalue.tolist(
                            )
                        else:
                            baseline_serializable[key][subkey] = subvalue
                else:
                    baseline_serializable[key] = value
            json.dump(baseline_serializable, f, indent=2)

        with open(self.output_dir / 'optimized_results.json', 'w') as f:
            optimized_serializable = {}
            for m_key, m_results in optimized_results.items():
                optimized_serializable[m_key] = {}
                for ef_key, ef_results in m_results.items():
                    optimized_serializable[m_key][ef_key] = {}
                    for metric, values in ef_results.items():
                        if isinstance(values, np.ndarray):
                            optimized_serializable[m_key][ef_key][metric] = values.tolist(
                            )
                        else:
                            optimized_serializable[m_key][ef_key][metric] = values
            json.dump(optimized_serializable, f, indent=2)

        logger.info("Results saved")

    def run_full_experiment(self,
                            dataset_path: Optional[str] = None,
                            n_vectors: int = 10000,
                            n_queries: int = 1000,
                            dimension: int = 128,
                            n_modalities: int = 5,
                            M: int = 16,
                            ef_construction: int = 200,
                            bridge_budget_ratio: float = 1e-5,
                            m_values: List[int] = [2, 4, 8],
                            ef_search_values: List[int] = [50, 100, 200, 400],
                            k_eval: int = 100,
                            target_recall: float = 0.90):
        """Run complete experiment pipeline"""
        logger.info("Starting full experiment pipeline...")

        # Load dataset
        X, Q, modalities, query_modalities = self.load_dataset(
            dataset_path, n_vectors, n_queries, dimension, n_modalities
        )

        # Compute ground truth
        gt_neighbors, gt_distances = self.compute_ground_truth(X, Q, k_eval)

        # Build baseline indexes
        hnsw_baseline, faiss_baseline = self.build_baseline_indexes(
            X, M, ef_construction)

        # Run baseline evaluation
        baseline_eval = self.run_baseline_evaluation(
            hnsw_baseline, Q, gt_neighbors, ef_search_values, k_eval, target_recall
        )

        # Build optimized index
        hnsw_optimized, bridge_builder = self.build_optimized_index(
            X, modalities, M, ef_construction, bridge_budget_ratio
        )

        # Run optimized evaluation
        optimized_results = self.run_optimized_evaluation(
            hnsw_optimized, bridge_builder, Q, gt_neighbors, m_values, ef_search_values, k_eval
        )

        # Run comparison experiment
        # Use first 100 valid queries
        Q_eval = baseline_eval['valid_queries'][:100]
        comparison_df = self.run_comparison_experiment(
            baseline_eval, optimized_results, Q_eval, Q, k_eval
        )

        # Generate percentile table
        percentile_df = self.generate_percentile_table(comparison_df)

        # Generate plots
        self.plot_results(comparison_df, percentile_df)

        # Save results
        self.save_results(comparison_df, percentile_df,
                          baseline_eval, optimized_results)

        logger.info("Full experiment pipeline completed!")

        # Print summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)

        print(
            f"Dataset: {len(X)} vectors, {len(Q)} queries, {dimension}D, {n_modalities} modalities")
        print(
            f"Baseline: {len(baseline_eval['valid_queries'])} queries achieved {target_recall} recall")
        print(
            f"Bridge edges: {bridge_builder.get_statistics()['total_bridge_edges']} added")

        print("\nBest configuration:")
        if not percentile_df.empty:
            best_idx = percentile_df[percentile_df['percentile']
                                     == 95]['ratio'].idxmin()
            best_row = percentile_df.loc[best_idx]
            print(f"  m={best_row['m']}, ef={best_row['ef']}")
            print(f"  P95 cost ratio: {best_row['ratio']:.3f}")
        else:
            print("  No percentile data available")

        print(f"\nResults saved to: {self.output_dir}")
        print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='HNSW Optimization Experiment Runner')

    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset file path')
    parser.add_argument('--n_vectors', type=int,
                        default=10000, help='Number of vectors')
    parser.add_argument('--n_queries', type=int,
                        default=1000, help='Number of queries')
    parser.add_argument('--dimension', type=int,
                        default=128, help='Vector dimension')
    parser.add_argument('--n_modalities', type=int,
                        default=5, help='Number of modalities')
    parser.add_argument('--M', type=int, default=16, help='HNSW M parameter')
    parser.add_argument('--ef_construction', type=int,
                        default=200, help='HNSW ef_construction')
    parser.add_argument('--bridge_budget', type=float,
                        default=1e-5, help='Bridge budget ratio')
    parser.add_argument('--m_values', nargs='+', type=int,
                        default=[2, 4, 8], help='M values to test')
    parser.add_argument('--ef_search_values', nargs='+', type=int, default=[50, 100, 200, 400],
                        help='ef_search values to test')
    parser.add_argument('--k_eval', type=int, default=100, help='Evaluation k')
    parser.add_argument('--target_recall', type=float,
                        default=0.90, help='Target recall')
    parser.add_argument('--output_dir', type=str,
                        default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
