#!/usr/bin/env python3
"""
High-layer Bridge Edges implementation for HNSW optimization
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Tuple, Optional, Set
import logging
from collections import defaultdict
import heapq
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class BridgeBuilder:
    """Build high-layer bridge edges for HNSW optimization"""

    def __init__(self,
                 max_bridge_per_node: int = 2,
                 bridge_budget_ratio: float = 1e-5,
                 scoring_weights: Optional[Dict[str, float]] = None):
        """
        Initialize bridge builder

        Args:
            max_bridge_per_node: Maximum bridge edges per node
            bridge_budget_ratio: Maximum ratio of new edges to total edges (0.001% = 1e-5)
            scoring_weights: Weights for scoring function components
        """
        self.max_bridge_per_node = max_bridge_per_node
        self.bridge_budget_ratio = bridge_budget_ratio

        # Default scoring weights
        if scoring_weights is None:
            scoring_weights = {
                'cross_modality_bonus': 1.0,
                'inverse_distance': 0.5,
                'complementarity': 0.3,
                'cost_penalty': -0.1
            }
        self.scoring_weights = scoring_weights

        # Bridge map: node_id -> list of bridge neighbors
        self.bridge_map: Dict[int, List[int]] = defaultdict(list)

        # Statistics
        self.total_original_edges = 0
        self.total_bridge_edges = 0
        # cluster_id -> representative_node_id
        self.cluster_centers: Dict[int, int] = {}

    def extract_high_layer_nodes(self,
                                 hnsw_index,
                                 min_level: int = 1) -> List[int]:
        """
        Extract nodes that appear in high layers

        Args:
            hnsw_index: HNSW baseline index
            min_level: Minimum level threshold

        Returns:
            List of high-layer node IDs
        """
        high_nodes = []

        # Get all node IDs from the index
        all_ids = hnsw_index.index.get_ids_list()

        for node_id in all_ids:
            # Use heuristic to estimate if node is in high layers
            # In real implementation, this would use actual level information
            estimated_level = hnsw_index.get_node_level(node_id)
            if estimated_level >= min_level:
                high_nodes.append(node_id)

        logger.info(
            f"Extracted {len(high_nodes)} high-layer nodes from {len(all_ids)} total nodes")
        return high_nodes

    def cluster_high_nodes(self,
                           high_nodes: List[int],
                           vectors: np.ndarray,
                           modalities: Optional[np.ndarray] = None,
                           n_clusters: Optional[int] = None) -> Dict[int, List[int]]:
        """
        Cluster high-layer nodes by modality or using KMeans

        Args:
            high_nodes: List of high-layer node IDs
            vectors: All vectors (N x D)
            modalities: Optional modality labels (N,)
            n_clusters: Number of clusters for KMeans (if modalities not provided)

        Returns:
            Dictionary mapping cluster_id to list of node IDs
        """
        if modalities is not None:
            # Use modality labels for clustering
            logger.info("Using modality labels for clustering")
            clusters = defaultdict(list)

            for node_id in high_nodes:
                modality = modalities[node_id]
                clusters[modality].append(node_id)

            # Filter out empty clusters
            clusters = {k: v for k, v in clusters.items() if len(v) > 0}

        else:
            # Use KMeans clustering
            if n_clusters is None:
                n_clusters = min(max(len(high_nodes) // 100, 10), 1024)

            logger.info(f"Using MiniBatchKMeans with {n_clusters} clusters")

            # Extract vectors for high-layer nodes
            high_vectors = vectors[high_nodes]

            # Perform clustering
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=min(1000, len(high_vectors))
            )
            cluster_labels = kmeans.fit_predict(high_vectors)

            # Group nodes by cluster
            clusters = defaultdict(list)
            for node_id, cluster_id in zip(high_nodes, cluster_labels):
                clusters[cluster_id].append(node_id)

        logger.info(f"Created {len(clusters)} clusters")
        for cluster_id, nodes in clusters.items():
            logger.info(f"  Cluster {cluster_id}: {len(nodes)} nodes")

        return dict(clusters)

    def compute_cluster_centers(self,
                                clusters: Dict[int, List[int]],
                                vectors: np.ndarray) -> Dict[int, int]:
        """
        Compute representative nodes for each cluster

        Args:
            clusters: Dictionary mapping cluster_id to list of node IDs
            vectors: All vectors (N x D)

        Returns:
            Dictionary mapping cluster_id to representative node_id
        """
        cluster_centers = {}

        for cluster_id, node_ids in clusters.items():
            if len(node_ids) == 0:
                continue

            # Compute cluster center
            cluster_vectors = vectors[node_ids]
            center = np.mean(cluster_vectors, axis=0)

            # Find node closest to center
            distances = np.linalg.norm(cluster_vectors - center, axis=1)
            closest_idx = np.argmin(distances)
            representative_node = node_ids[closest_idx]

            cluster_centers[cluster_id] = representative_node

        self.cluster_centers = cluster_centers
        logger.info(f"Computed {len(cluster_centers)} cluster centers")
        return cluster_centers

    def compute_bridge_scores(self,
                              cluster_centers: Dict[int, int],
                              vectors: np.ndarray,
                              modalities: Optional[np.ndarray] = None) -> List[Tuple[float, int, int]]:
        """
        Compute scores for candidate bridge edges between cluster centers

        Args:
            cluster_centers: Dictionary mapping cluster_id to representative node_id
            vectors: All vectors (N x D)
            modalities: Optional modality labels

        Returns:
            List of (score, node1, node2) tuples sorted by score (descending)
        """
        cluster_ids = list(cluster_centers.keys())
        candidates = []

        # Generate all cross-cluster pairs
        for i, cluster1 in enumerate(cluster_ids):
            for j, cluster2 in enumerate(cluster_ids[i+1:], i+1):
                node1 = cluster_centers[cluster1]
                node2 = cluster_centers[cluster2]

                score = self._compute_bridge_score(
                    node1, node2, vectors, modalities)
                candidates.append((score, node1, node2))

        # Sort by score (descending)
        candidates.sort(reverse=True)

        logger.info(
            f"Computed scores for {len(candidates)} candidate bridge pairs")
        return candidates

    def _compute_bridge_score(self,
                              node1: int,
                              node2: int,
                              vectors: np.ndarray,
                              modalities: Optional[np.ndarray] = None) -> float:
        """
        Compute scoring function for a bridge edge

        Args:
            node1: First node ID
            node2: Second node ID
            vectors: All vectors (N x D)
            modalities: Optional modality labels

        Returns:
            Bridge score
        """
        w1 = self.scoring_weights['cross_modality_bonus']
        w2 = self.scoring_weights['inverse_distance']
        w3 = self.scoring_weights['complementarity']
        w4 = self.scoring_weights['cost_penalty']

        # Cross-modality bonus
        if modalities is not None:
            cross_modality_bonus = 1.0 if modalities[node1] != modalities[node2] else 0.0
        else:
            # Use cluster-based bonus (always 1.0 for cross-cluster pairs)
            cross_modality_bonus = 1.0

        # Inverse distance
        distance = np.linalg.norm(vectors[node1] - vectors[node2])
        inverse_distance = 1.0 / (1.0 + distance)

        # Complementarity (simplified - based on vector similarity)
        # In full implementation, this would use neighborhood overlap
        norm1 = np.linalg.norm(vectors[node1])
        norm2 = np.linalg.norm(vectors[node2])
        if norm1 > 1e-8 and norm2 > 1e-8:  # Avoid division by very small numbers
            similarity = np.dot(
                vectors[node1], vectors[node2]) / (norm1 * norm2)
            # Clamp similarity to [-1, 1] range to avoid numerical issues
            similarity = np.clip(similarity, -1.0, 1.0)
            complementarity = 1.0 - similarity
        else:
            complementarity = 1.0  # Maximum complementarity for zero vectors

        # Cost penalty (simplified)
        cost_penalty = 1.0  # Normalized expected cost

        # Combined score
        score = (w1 * cross_modality_bonus +
                 w2 * inverse_distance +
                 w3 * complementarity -
                 w4 * cost_penalty)

        return score

    def add_bridge_edges(self,
                         candidates: List[Tuple[float, int, int]],
                         original_edge_count: int) -> Dict[int, List[int]]:
        """
        Add bridge edges with strict budget control

        Args:
            candidates: List of (score, node1, node2) tuples
            original_edge_count: Original total edge count in the graph

        Returns:
            Updated bridge_map
        """
        max_new_edges = int(original_edge_count * self.bridge_budget_ratio)

        logger.info(
            f"Budget: {max_new_edges} new edges (≤{self.bridge_budget_ratio*100:.3f}% of {original_edge_count} original edges)")

        # Track edges per node
        node_edge_counts = defaultdict(int)

        added_edges = 0

        for score, node1, node2 in candidates:
            # Check budget constraints
            if added_edges >= max_new_edges:
                break

            # Check per-node constraints
            if (node_edge_counts[node1] >= self.max_bridge_per_node or
                    node_edge_counts[node2] >= self.max_bridge_per_node):
                continue

            # Add bidirectional edge
            self.bridge_map[node1].append(node2)
            self.bridge_map[node2].append(node1)

            node_edge_counts[node1] += 1
            node_edge_counts[node2] += 1
            added_edges += 1

        self.total_bridge_edges = added_edges
        self.total_original_edges = original_edge_count

        logger.info(
            f"Added {added_edges} bridge edges ({added_edges/original_edge_count*100:.6f}% of original)")
        logger.info(
            f"Bridge map contains {len(self.bridge_map)} nodes with bridges")

        return self.bridge_map

    def build_bridges(self,
                      hnsw_index,
                      vectors: np.ndarray,
                      modalities: Optional[np.ndarray] = None,
                      min_level: int = 1) -> Dict[int, List[int]]:
        """
        Complete bridge building pipeline

        Args:
            hnsw_index: HNSW baseline index
            vectors: All vectors (N x D)
            modalities: Optional modality labels
            min_level: Minimum level for high-layer nodes

        Returns:
            Bridge map dictionary
        """
        logger.info("Starting bridge building pipeline")

        # Step 1: Extract high-layer nodes
        high_nodes = self.extract_high_layer_nodes(hnsw_index, min_level)

        if len(high_nodes) < 2:
            logger.warning("Not enough high-layer nodes for bridge building")
            return {}

        # Step 2: Cluster high-layer nodes
        clusters = self.cluster_high_nodes(high_nodes, vectors, modalities)

        if len(clusters) < 2:
            logger.warning("Not enough clusters for bridge building")
            return {}

        # Step 3: Compute cluster centers
        cluster_centers = self.compute_cluster_centers(clusters, vectors)

        # Step 4: Compute bridge scores
        candidates = self.compute_bridge_scores(
            cluster_centers, vectors, modalities)

        # Step 5: Estimate original edge count (simplified)
        # In real implementation, this would count actual edges in HNSW
        estimated_edges = len(vectors) * 16  # Rough estimate based on M=16

        # Step 6: Add bridge edges
        bridge_map = self.add_bridge_edges(candidates, estimated_edges)

        logger.info("Bridge building pipeline completed")
        return bridge_map

    def get_bridge_neighbors(self, node_id: int) -> List[int]:
        """
        Get bridge neighbors for a node

        Args:
            node_id: Node ID

        Returns:
            List of bridge neighbor IDs
        """
        return self.bridge_map.get(node_id, [])

    def get_statistics(self) -> Dict[str, any]:
        """Get bridge building statistics"""
        return {
            'total_bridge_edges': self.total_bridge_edges,
            'total_original_edges': self.total_original_edges,
            'bridge_ratio': self.total_bridge_edges / max(1, self.total_original_edges),
            'nodes_with_bridges': len(self.bridge_map),
            'max_bridge_per_node': self.max_bridge_per_node,
            'bridge_budget_ratio': self.bridge_budget_ratio,
            'cluster_centers': len(self.cluster_centers)
        }


# TODO: Dynamic insert bridge creation
# When implementing dynamic insertion, the following logic should be added:
#
# 1. When a new vector is inserted:
#   - Determine its level in the HNSW hierarchy
#   - If it's inserted in a high layer (level > 0):
#     - Find its modality/cluster assignment
#     - Identify potential bridge candidates from other clusters
#     - Score and potentially add bridges following the same budget constraints
#
# 2. Bridge validation:
#   - Periodically validate that bridge edges are still beneficial
#   - Remove bridges that become redundant or harmful
#
# 3. Soft edges approach:
#   - Consider bridges as "soft" edges that can be easily removed
#   - Maintain separate bridge data structure that doesn't modify core HNSW
#
# This logic should be integrated into the HNSW insertion process and
# called whenever a new high-layer node is created.


if __name__ == "__main__":
    # Test bridge building
    from data_loader import create_toy_dataset
    from hnsw_baseline import HNSWBaseline

    # Create test dataset
    X, Q, modalities, query_modalities = create_toy_dataset(1000, 50, 64, 3)

    # Build HNSW index
    hnsw = HNSWBaseline(dimension=64, M=16, ef_construction=200)
    hnsw.build_index(X)

    # Build bridges
    bridge_builder = BridgeBuilder(
        max_bridge_per_node=2, bridge_budget_ratio=1e-5)
    bridge_map = bridge_builder.build_bridges(hnsw, X, modalities)

    # Print statistics
    stats = bridge_builder.get_statistics()
    print("Bridge building statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test bridge neighbor retrieval
    if bridge_map:
        test_node = list(bridge_map.keys())[0]
        bridge_neighbors = bridge_builder.get_bridge_neighbors(test_node)
        print(f"Bridge neighbors for node {test_node}: {bridge_neighbors}")
