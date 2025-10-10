#!/usr/bin/env python3
"""
Ground truth utilities for HNSW optimization experiments
"""

import numpy as np
import faiss
from typing import Tuple, List, Optional
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class GroundTruthComputer:
    """Compute ground truth nearest neighbors using exact search"""

    def __init__(self, metric: str = 'L2'):
        """
        Initialize ground truth computer

        Args:
            metric: Distance metric ('L2' or 'IP')
        """
        self.metric = metric
        self.gt_neighbors: Optional[np.ndarray] = None
        self.gt_distances: Optional[np.ndarray] = None

    def compute_ground_truth(self,
                             X: np.ndarray,
                             Q: np.ndarray,
                             k: int = 100,
                             batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ground truth neighbors using exact search

        Args:
            X: Base vectors (N x D)
            Q: Query vectors (Q x D)
            k: Number of nearest neighbors to find
            batch_size: Batch size for processing (None for no batching)

        Returns:
            Tuple of (neighbors, distances) both of shape (Q x k)
        """
        logger.info(f"Computing ground truth for {len(Q)} queries, k={k}")

        if batch_size is None:
            batch_size = min(1000, len(Q))

        # Build FAISS index
        dimension = X.shape[1]

        if self.metric == 'L2':
            index = faiss.IndexFlatL2(dimension)
        elif self.metric == 'IP':
            index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Add vectors to index
        index.add(X.astype(np.float32))

        # Search in batches
        all_neighbors = []
        all_distances = []

        for i in tqdm(range(0, len(Q), batch_size), desc="Computing GT"):
            batch_end = min(i + batch_size, len(Q))
            query_batch = Q[i:batch_end].astype(np.float32)

            distances, neighbors = index.search(query_batch, k)

            all_neighbors.append(neighbors)
            all_distances.append(distances)

        # Concatenate results
        self.gt_neighbors = np.vstack(all_neighbors)
        self.gt_distances = np.vstack(all_distances)

        logger.info(f"Ground truth computation completed")
        return self.gt_neighbors, self.gt_distances

    def compute_recall(self,
                       predicted_neighbors: np.ndarray,
                       k_eval: Optional[int] = None) -> float:
        """
        Compute recall@k_eval

        Args:
            predicted_neighbors: Predicted neighbors (Q x k_pred)
            k_eval: Evaluation k (if None, use min(k_pred, k_gt))

        Returns:
            Average recall across all queries
        """
        if self.gt_neighbors is None:
            raise ValueError("Ground truth not computed")

        if k_eval is None:
            k_eval = min(
                predicted_neighbors.shape[1], self.gt_neighbors.shape[1])

        recalls = []
        for i in range(len(predicted_neighbors)):
            pred_set = set(predicted_neighbors[i, :k_eval])
            gt_set = set(self.gt_neighbors[i, :k_eval])

            if len(gt_set) > 0:
                recall = len(pred_set.intersection(gt_set)) / len(gt_set)
                recalls.append(recall)

        return np.mean(recalls)

    def compute_recall_per_query(self,
                                 predicted_neighbors: np.ndarray,
                                 k_eval: Optional[int] = None) -> np.ndarray:
        """
        Compute recall@k_eval per query

        Args:
            predicted_neighbors: Predicted neighbors (Q x k_pred)
            k_eval: Evaluation k (if None, use min(k_pred, k_gt))

        Returns:
            Array of recalls, one per query
        """
        if self.gt_neighbors is None:
            raise ValueError("Ground truth not computed")

        if k_eval is None:
            k_eval = min(
                predicted_neighbors.shape[1], self.gt_neighbors.shape[1])

        recalls = np.zeros(len(predicted_neighbors))

        for i in range(len(predicted_neighbors)):
            pred_set = set(predicted_neighbors[i, :k_eval])
            gt_set = set(self.gt_neighbors[i, :k_eval])

            if len(gt_set) > 0:
                recall = len(pred_set.intersection(gt_set)) / len(gt_set)
                recalls[i] = recall

        return recalls


def benchmark_search_method(X: np.ndarray,
                            Q: np.ndarray,
                            search_func,
                            ef_values: List[int],
                            k: int = 100) -> dict:
    """
    Benchmark a search method across different ef values

    Args:
        X: Base vectors
        Q: Query vectors
        search_func: Function that takes (query, ef, k) and returns (neighbors, cost)
        ef_values: List of ef values to test
        k: Number of neighbors to retrieve

    Returns:
        Dictionary with results for each ef value
    """
    results = {}

    for ef in tqdm(ef_values, desc="Benchmarking ef values"):
        costs = []
        recalls = []
        latencies = []

        # Compute ground truth for this ef
        gt_computer = GroundTruthComputer()
        gt_neighbors, _ = gt_computer.compute_ground_truth(X, Q, k)

        for i, query in enumerate(Q):
            start_time = time.time()

            neighbors, cost = search_func(query, ef, k)

            latency = time.time() - start_time

            # Compute recall
            pred_set = set(neighbors)
            gt_set = set(gt_neighbors[i])
            recall = len(pred_set.intersection(gt_set)) / \
                len(gt_set) if len(gt_set) > 0 else 0.0

            costs.append(cost)
            recalls.append(recall)
            latencies.append(latency)

        results[ef] = {
            'costs': np.array(costs),
            'recalls': np.array(recalls),
            'latencies': np.array(latencies),
            'mean_cost': np.mean(costs),
            'mean_recall': np.mean(recalls),
            'mean_latency': np.mean(latencies)
        }

    return results


def find_ef_for_recall(results: dict, target_recall: float = 0.90) -> dict:
    """
    Find ef values that achieve target recall for each query

    Args:
        results: Results from benchmark_search_method
        target_recall: Target recall threshold

    Returns:
        Dictionary mapping query_id to ef value that achieves target recall
    """
    query_ef_mapping = {}

    for query_id in range(len(next(iter(results.values()))['costs'])):
        best_ef = None
        best_cost = float('inf')

        for ef, result in results.items():
            if result['recalls'][query_id] >= target_recall:
                if result['costs'][query_id] < best_cost:
                    best_cost = result['costs'][query_id]
                    best_ef = ef

        query_ef_mapping[query_id] = best_ef

    return query_ef_mapping


def compute_percentiles(costs: np.ndarray, percentiles: List[float] = [10, 25, 50, 75, 90, 95, 99]) -> dict:
    """
    Compute cost percentiles

    Args:
        costs: Array of costs per query
        percentiles: List of percentiles to compute

    Returns:
        Dictionary mapping percentile to value
    """
    return {p: np.percentile(costs, p) for p in percentiles}


if __name__ == "__main__":
    # Test ground truth computation
    from data_loader import create_toy_dataset

    # Create small test dataset
    X, Q, modalities, query_modalities = create_toy_dataset(1000, 100, 64, 3)

    # Compute ground truth
    gt_computer = GroundTruthComputer()
    neighbors, distances = gt_computer.compute_ground_truth(X, Q, k=10)

    print(f"Ground truth shape: {neighbors.shape}")
    print(f"Sample neighbors: {neighbors[0, :5]}")
    print(f"Sample distances: {distances[0, :5]}")

    # Test recall computation
    dummy_pred = neighbors + 1  # Shift by 1 to test recall computation
    recall = gt_computer.compute_recall(dummy_pred, k_eval=5)
    print(f"Recall with shifted neighbors: {recall:.3f}")
