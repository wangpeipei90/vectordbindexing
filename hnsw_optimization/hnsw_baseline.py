#!/usr/bin/env python3
"""
Baseline HNSW implementation and utilities
"""

import numpy as np
import hnswlib
import faiss
from typing import Tuple, List, Optional, Dict, Any
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class HNSWBaseline:
    """Baseline HNSW implementation with cost tracking"""
    
    def __init__(self, 
                 dimension: int,
                 M: int = 16,
                 ef_construction: int = 200,
                 max_elements: int = 1000000,
                 seed: int = 42):
        """
        Initialize HNSW baseline
        
        Args:
            dimension: Vector dimension
            M: Maximum number of bi-directional links created for every new element
            ef_construction: Size of the dynamic candidate list during construction
            max_elements: Maximum number of elements in the index
            seed: Random seed
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.max_elements = max_elements
        self.seed = seed
        
        # Initialize index
        self.index = hnswlib.Index(space='l2', dim=dimension)
        self.index.init_index(max_elements=max_elements, 
                             M=M, 
                             ef_construction=ef_construction,
                             random_seed=seed)
        
        # Track internal state for analysis
        self.node_levels: Dict[int, int] = {}  # node_id -> max_level
        self.adjacency_lists: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.is_built = False
        
    def add_vectors(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """
        Add vectors to the index
        
        Args:
            vectors: Vectors to add (N x D)
            ids: Optional IDs for vectors (if None, use 0, 1, 2, ...)
        """
        if ids is None:
            ids = np.arange(len(vectors))
            
        self.index.add_items(vectors.astype(np.float32), ids.astype(np.int32))
        
        # Track levels for analysis (approximate)
        # Note: hnswlib doesn't expose exact levels, so we use a heuristic
        for i, vector_id in enumerate(ids):
            # Simple heuristic: assume levels based on position
            # This is not exact but sufficient for analysis
            estimated_level = max(0, int(np.log2(i + 1)) - 1) if i > 0 else 0
            self.node_levels[vector_id] = estimated_level
            
        logger.info(f"Added {len(vectors)} vectors to HNSW index")
        
    def build_index(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """
        Build complete index from vectors
        
        Args:
            vectors: All vectors to index (N x D)
            ids: Optional IDs for vectors
        """
        logger.info(f"Building HNSW index for {len(vectors)} vectors")
        start_time = time.time()
        
        self.add_vectors(vectors, ids)
        
        # Set search parameters
        self.index.set_ef(self.ef_construction)
        
        build_time = time.time() - start_time
        logger.info(f"HNSW index built in {build_time:.2f} seconds")
        
        self.is_built = True
        
    def search(self, 
               query: np.ndarray, 
               k: int, 
               ef_search: int = 200) -> Tuple[np.ndarray, int]:
        """
        Search for nearest neighbors
        
        Args:
            query: Query vector (D,)
            k: Number of neighbors to return
            ef_search: Size of dynamic candidate list during search
            
        Returns:
            Tuple of (neighbors, estimated_cost)
        """
        if not self.is_built:
            raise ValueError("Index not built")
            
        # Set search parameters
        self.index.set_ef(ef_search)
        
        # Perform search
        neighbors, distances = self.index.knn_query(query.reshape(1, -1).astype(np.float32), k=k)
        
        # Estimate cost (simplified - hnswlib doesn't expose exact visited count)
        # Use ef_search as proxy for cost
        estimated_cost = min(ef_search, len(self.index.get_ids_list()))
        
        return neighbors[0], estimated_cost
        
    def get_high_layer_nodes(self, min_level: int = 1) -> List[int]:
        """
        Get nodes that appear in high layers (level >= min_level)
        
        Args:
            min_level: Minimum level threshold
            
        Returns:
            List of node IDs in high layers
        """
        high_nodes = []
        for node_id, level in self.node_levels.items():
            if level >= min_level:
                high_nodes.append(node_id)
                
        return high_nodes
        
    def get_node_level(self, node_id: int) -> int:
        """Get the level of a specific node"""
        return self.node_levels.get(node_id, 0)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.is_built:
            return {}
            
        total_nodes = len(self.node_levels)
        high_nodes = len(self.get_high_layer_nodes())
        
        stats = {
            'total_nodes': total_nodes,
            'high_layer_nodes': high_nodes,
            'high_layer_ratio': high_nodes / total_nodes if total_nodes > 0 else 0,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'dimension': self.dimension,
        }
        
        return stats


class FAISSBaseline:
    """FAISS baseline for comparison"""
    
    def __init__(self, 
                 dimension: int,
                 index_type: str = 'flat',
                 nlist: int = 4096):
        """
        Initialize FAISS baseline
        
        Args:
            dimension: Vector dimension
            index_type: Type of FAISS index ('flat', 'ivf')
            nlist: Number of clusters for IVF index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        
        if index_type == 'flat':
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        self.is_trained = False
        self.is_built = False
        
    def build_index(self, vectors: np.ndarray):
        """
        Build FAISS index
        
        Args:
            vectors: Vectors to index (N x D)
        """
        logger.info(f"Building FAISS {self.index_type} index for {len(vectors)} vectors")
        
        if self.index_type == 'ivf' and not self.is_trained:
            # Train IVF index
            self.index.train(vectors.astype(np.float32))
            self.is_trained = True
            
        # Add vectors
        self.index.add(vectors.astype(np.float32))
        
        if self.index_type == 'ivf':
            self.index.nprobe = min(32, self.nlist // 4)  # Set reasonable nprobe
            
        self.is_built = True
        logger.info("FAISS index built")
        
    def search(self, 
               query: np.ndarray, 
               k: int) -> Tuple[np.ndarray, int]:
        """
        Search for nearest neighbors
        
        Args:
            query: Query vector (D,)
            k: Number of neighbors to return
            
        Returns:
            Tuple of (neighbors, cost)
        """
        if not self.is_built:
            raise ValueError("Index not built")
            
        # Perform search
        distances, neighbors = self.index.search(
            query.reshape(1, -1).astype(np.float32), k
        )
        
        # Cost is roughly proportional to number of vectors examined
        if self.index_type == 'flat':
            cost = self.index.ntotal  # Brute force
        else:  # IVF
            cost = self.index.nprobe * (self.index.ntotal // self.nlist)
            
        return neighbors[0], cost


def benchmark_hnsw_baseline(X: np.ndarray, 
                          Q: np.ndarray,
                          M_values: List[int] = [8, 16, 32],
                          ef_construction_values: List[int] = [100, 200, 400],
                          ef_search_values: List[int] = [50, 100, 200, 400],
                          k: int = 100) -> Dict[str, Any]:
    """
    Benchmark baseline HNSW with different parameters
    
    Args:
        X: Base vectors
        Q: Query vectors
        M_values: M parameter values to test
        ef_construction_values: ef_construction parameter values to test
        ef_search_values: ef_search parameter values to test
        k: Number of neighbors to retrieve
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for M in M_values:
        for ef_construction in ef_construction_values:
            config_key = f"M{M}_efc{ef_construction}"
            results[config_key] = {}
            
            # Build index
            hnsw = HNSWBaseline(
                dimension=X.shape[1],
                M=M,
                ef_construction=ef_construction
            )
            
            build_start = time.time()
            hnsw.build_index(X)
            build_time = time.time() - build_start
            
            results[config_key]['build_time'] = build_time
            results[config_key]['stats'] = hnsw.get_statistics()
            
            # Test different ef_search values
            for ef_search in ef_search_values:
                search_key = f"efs{ef_search}"
                
                costs = []
                latencies = []
                
                for query in Q:
                    start_time = time.time()
                    neighbors, cost = hnsw.search(query, k, ef_search)
                    latency = time.time() - start_time
                    
                    costs.append(cost)
                    latencies.append(latency)
                    
                results[config_key][search_key] = {
                    'mean_cost': np.mean(costs),
                    'mean_latency': np.mean(latencies),
                    'costs': np.array(costs),
                    'latencies': np.array(latencies)
                }
                
    return results


if __name__ == "__main__":
    # Test baseline implementation
    from data_loader import create_toy_dataset
    
    # Create test dataset
    X, Q, modalities, query_modalities = create_toy_dataset(1000, 50, 64, 3)
    
    # Test HNSW baseline
    hnsw = HNSWBaseline(dimension=64, M=16, ef_construction=200)
    hnsw.build_index(X)
    
    # Test search
    query = Q[0]
    neighbors, cost = hnsw.search(query, k=10, ef_search=100)
    
    print(f"HNSW search results:")
    print(f"  Neighbors: {neighbors[:5]}")
    print(f"  Cost: {cost}")
    print(f"  Stats: {hnsw.get_statistics()}")
    
    # Test FAISS baseline
    faiss_baseline = FAISSBaseline(dimension=64)
    faiss_baseline.build_index(X)
    
    neighbors, cost = faiss_baseline.search(query, k=10)
    
    print(f"FAISS search results:")
    print(f"  Neighbors: {neighbors[:5]}")
    print(f"  Cost: {cost}")
