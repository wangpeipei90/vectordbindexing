#!/usr/bin/env python3
"""
Adaptive Multi-entry Seeds implementation for HNSW optimization
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import logging
import heapq
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


class MultiEntrySearch:
    """Adaptive multi-entry seeds search implementation"""

    def __init__(self,
                 hnsw_index,
                 bridge_builder=None,
                 max_workers: int = 4):
        """
        Initialize multi-entry search

        Args:
            hnsw_index: HNSW baseline index
            bridge_builder: Optional bridge builder for bridge edges
            max_workers: Maximum number of parallel workers
        """
        self.hnsw_index = hnsw_index
        self.bridge_builder = bridge_builder
        self.max_workers = max_workers

        # Thread-safe access to hnsw_index
        self._lock = threading.Lock()

    def get_high_layer_candidates(self,
                                  query: np.ndarray,
                                  ef_search: int,
                                  candidate_count: int = 50) -> List[int]:
        """
        Get candidates from high-layer search (level=1)

        Args:
            query: Query vector (D,)
            ef_search: Search parameter
            candidate_count: Number of candidates to collect

        Returns:
            List of candidate node IDs from high layers
        """
        # Since hnswlib doesn't expose intermediate search results,
        # we simulate high-layer search by using a smaller ef_search
        # to get a diverse set of candidates

        with self._lock:
            # Use smaller ef to get diverse candidates
            small_ef = min(ef_search // 4, candidate_count)
            self.hnsw_index.index.set_ef(small_ef)

            # Get more candidates than needed
            neighbors, _ = self.hnsw_index.index.knn_query(
                query.reshape(1, -1).astype(np.float32),
                k=min(candidate_count * 2,
                      len(self.hnsw_index.index.get_ids_list()))
            )

        # Filter to high-layer nodes (using heuristic)
        high_layer_candidates = []
        for node_id in neighbors[0]:
            if self.hnsw_index.get_node_level(node_id) >= 1:
                high_layer_candidates.append(node_id)

        return high_layer_candidates[:candidate_count]

    def beam_search_from_seed(self,
                              query: np.ndarray,
                              seed_node: int,
                              k: int,
                              beam_width: int = 100) -> Tuple[List[int], int]:
        """
        Perform beam search from a seed node on layer 0

        Args:
            query: Query vector (D,)
            seed_node: Starting node ID
            k: Number of neighbors to return
            beam_width: Beam width for search

        Returns:
            Tuple of (neighbors, cost)
        """
        # Get all vectors for distance computation
        # In real implementation, this would be more efficient
        all_ids = self.hnsw_index.index.get_ids_list()

        # Simple beam search simulation
        # Start from seed and expand
        visited = set()
        candidates = [(0.0, seed_node)]  # (negative_distance, node_id)
        results = []

        cost = 0

        while candidates and len(results) < k:
            # Get best candidate
            neg_dist, current_node = heapq.heappop(candidates)

            if current_node in visited:
                continue

            visited.add(current_node)
            cost += 1

            # Compute distance to query
            # In real implementation, this would use cached distances
            if current_node in all_ids:
                node_idx = list(all_ids).index(current_node)
                # This is a simplified distance computation
                # Real implementation would use actual vector data
                distance = abs(node_idx - int(np.mean(query)))  # Placeholder
            else:
                distance = float('inf')

            results.append((distance, current_node))

            # Expand neighbors (simplified)
            # In real implementation, this would use actual adjacency
            if len(candidates) < beam_width:
                for neighbor in range(max(0, current_node - 5),
                                      min(len(all_ids), current_node + 6)):
                    if neighbor != current_node and neighbor not in visited:
                        heapq.heappush(candidates, (neg_dist - 1.0, neighbor))

        # Sort by distance and return top k
        results.sort()
        neighbors = [node_id for _, node_id in results[:k]]

        return neighbors, cost

    def search_with_bridge_edges(self,
                                 query: np.ndarray,
                                 current_node: int,
                                 ef_search: int) -> List[int]:
        """
        Enhanced search that includes bridge edges

        Args:
            query: Query vector (D,)
            current_node: Current node being explored
            ef_search: Search parameter

        Returns:
            List of neighbor nodes including bridges
        """
        # Get regular neighbors from HNSW
        neighbors = []

        # In real implementation, this would extract actual neighbors
        # from hnswlib's internal data structures
        # For now, we simulate this
        all_ids = self.hnsw_index.index.get_ids_list()
        if current_node in all_ids:
            # Simulate regular neighbors
            node_idx = list(all_ids).index(current_node)
            for i in range(max(0, node_idx - 5), min(len(all_ids), node_idx + 6)):
                if i != node_idx:
                    neighbors.append(all_ids[i])

        # Add bridge neighbors if available
        if self.bridge_builder is not None:
            bridge_neighbors = self.bridge_builder.get_bridge_neighbors(
                current_node)
            neighbors.extend(bridge_neighbors)

        return neighbors

    def multi_entry_search(self,
                           query: np.ndarray,
                           k: int,
                           m: int = 4,
                           ef_search: int = 200,
                           beam_width: int = 100,
                           use_bridges: bool = True) -> Tuple[np.ndarray, int]:
        """
        Perform multi-entry search with adaptive seeds

        Args:
            query: Query vector (D,)
            k: Number of neighbors to return
            m: Number of entry seeds to use
            ef_search: Search parameter
            beam_width: Beam width for beam search
            use_bridges: Whether to use bridge edges

        Returns:
            Tuple of (neighbors, total_cost)
        """
        logger.debug(
            f"Multi-entry search: k={k}, m={m}, ef_search={ef_search}")

        # Step A: Get high-layer candidates
        high_candidates = self.get_high_layer_candidates(
            query, ef_search, candidate_count=m*2)

        # Step B: Select top-m seeds
        if len(high_candidates) < m:
            # Fallback: use default entry and some random nodes
            all_ids = self.hnsw_index.index.get_ids_list()
            seeds = [all_ids[0]]  # Default entry
            remaining = [node_id for node_id in all_ids[:m-1]
                         if node_id != all_ids[0]]
            seeds.extend(remaining)
        else:
            # Select top-m from high-layer candidates
            seeds = high_candidates[:m]

        logger.debug(f"Selected seeds: {seeds}")

        # Step C: Parallel beam search from each seed
        all_results = []
        total_cost = 0

        if self.max_workers > 1 and len(seeds) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_seed = {
                    executor.submit(self.beam_search_from_seed, query, seed, k, beam_width): seed
                    for seed in seeds
                }

                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    try:
                        neighbors, cost = future.result()
                        all_results.extend([(n, seed) for n in neighbors])
                        total_cost += cost
                    except Exception as e:
                        logger.warning(f"Search from seed {seed} failed: {e}")
        else:
            # Sequential execution
            for seed in seeds:
                neighbors, cost = self.beam_search_from_seed(
                    query, seed, k, beam_width)
                all_results.extend([(n, seed) for n in neighbors])
                total_cost += cost

        # Step D: Merge and re-rank results
        # Collect unique neighbors with their minimum distances
        neighbor_distances = {}

        for neighbor, seed in all_results:
            # In real implementation, compute actual distance to query
            # For now, use a simplified distance based on node ID and query
            # This is a placeholder - in production, use actual vector distance
            # Use a more stable distance calculation
            query_hash = hash(str(query)) % 1000
            distance = abs(int(neighbor) - int(query_hash)
                           ) % 1000  # Placeholder

            if neighbor not in neighbor_distances or distance < neighbor_distances[neighbor]:
                neighbor_distances[neighbor] = distance

        # Sort by distance and return top k
        sorted_neighbors = sorted(
            neighbor_distances.items(), key=lambda x: x[1])
        final_neighbors = np.array(
            [node_id for node_id, _ in sorted_neighbors[:k]])

        logger.debug(
            f"Multi-entry search completed: {len(final_neighbors)} results, cost={total_cost}")

        return final_neighbors, total_cost

    def compare_with_baseline(self,
                              query: np.ndarray,
                              k: int,
                              ef_search: int = 200,
                              m: int = 4) -> Dict[str, any]:
        """
        Compare multi-entry search with baseline HNSW search

        Args:
            query: Query vector (D,)
            k: Number of neighbors to return
            ef_search: Search parameter
            m: Number of entry seeds

        Returns:
            Dictionary with comparison results
        """
        # Baseline search
        start_time = time.time()
        baseline_neighbors, baseline_cost = self.hnsw_index.search(
            query, k, ef_search)
        baseline_time = time.time() - start_time

        # Multi-entry search
        start_time = time.time()
        multi_neighbors, multi_cost = self.multi_entry_search(
            query, k, m, ef_search, use_bridges=True
        )
        multi_time = time.time() - start_time

        # Compare results
        baseline_set = set(baseline_neighbors)
        multi_set = set(multi_neighbors)
        overlap = len(baseline_set.intersection(multi_set))

        results = {
            'baseline_neighbors': baseline_neighbors,
            'multi_neighbors': multi_neighbors,
            'baseline_cost': baseline_cost,
            'multi_cost': multi_cost,
            'baseline_time': baseline_time,
            'multi_time': multi_time,
            'overlap_count': overlap,
            'overlap_ratio': overlap / k if k > 0 else 0,
            'cost_ratio': multi_cost / baseline_cost if baseline_cost > 0 else 1.0,
            'time_ratio': multi_time / baseline_time if baseline_time > 0 else 1.0
        }

        return results


class AdaptiveMultiEntrySearch(MultiEntrySearch):
    """Enhanced multi-entry search with adaptive seed selection"""

    def __init__(self,
                 hnsw_index,
                 bridge_builder=None,
                 max_workers: int = 4,
                 seed_selection_strategy: str = 'diverse'):
        """
        Initialize adaptive multi-entry search

        Args:
            hnsw_index: HNSW baseline index
            bridge_builder: Optional bridge builder
            max_workers: Maximum number of workers
            seed_selection_strategy: Strategy for seed selection ('diverse', 'top', 'random')
        """
        super().__init__(hnsw_index, bridge_builder, max_workers)
        self.seed_selection_strategy = seed_selection_strategy

    def select_adaptive_seeds(self,
                              query: np.ndarray,
                              candidates: List[int],
                              m: int) -> List[int]:
        """
        Select seeds using adaptive strategy

        Args:
            query: Query vector (D,)
            candidates: List of candidate node IDs
            m: Number of seeds to select

        Returns:
            List of selected seed node IDs
        """
        if len(candidates) <= m:
            return candidates

        if self.seed_selection_strategy == 'top':
            # Select top-m candidates (already sorted)
            return candidates[:m]

        elif self.seed_selection_strategy == 'diverse':
            # Select diverse seeds to maximize coverage
            selected = [candidates[0]]  # Start with best candidate

            for _ in range(m - 1):
                best_candidate = None
                best_diversity_score = -1

                for candidate in candidates[1:]:
                    if candidate in selected:
                        continue

                    # Compute diversity score (distance to already selected)
                    min_distance = float('inf')
                    for selected_seed in selected:
                        distance = abs(candidate - selected_seed)  # Simplified
                        min_distance = min(min_distance, distance)

                    if min_distance > best_diversity_score:
                        best_diversity_score = min_distance
                        best_candidate = candidate

                if best_candidate is not None:
                    selected.append(best_candidate)

            return selected

        elif self.seed_selection_strategy == 'random':
            # Random selection
            return np.random.choice(candidates, size=min(m, len(candidates)), replace=False).tolist()

        else:
            # Default: top-m
            return candidates[:m]


if __name__ == "__main__":
    # Test multi-entry search
    from data_loader import create_toy_dataset
    from hnsw_baseline import HNSWBaseline
    from bridge_builder import BridgeBuilder

    # Create test dataset
    X, Q, modalities, query_modalities = create_toy_dataset(1000, 50, 64, 3)

    # Build HNSW index
    hnsw = HNSWBaseline(dimension=64, M=16, ef_construction=200)
    hnsw.build_index(X)

    # Build bridges
    bridge_builder = BridgeBuilder()
    bridge_builder.build_bridges(hnsw, X, modalities)

    # Test multi-entry search
    multi_search = MultiEntrySearch(hnsw, bridge_builder)

    query = Q[0]
    neighbors, cost = multi_search.multi_entry_search(query, k=10, m=4)

    print(f"Multi-entry search results:")
    print(f"  Neighbors: {neighbors}")
    print(f"  Cost: {cost}")

    # Compare with baseline
    comparison = multi_search.compare_with_baseline(query, k=10, m=4)
    print(f"Comparison with baseline:")
    print(f"  Overlap: {comparison['overlap_count']}/10")
    print(f"  Cost ratio: {comparison['cost_ratio']:.2f}")
    print(f"  Time ratio: {comparison['time_ratio']:.2f}")
