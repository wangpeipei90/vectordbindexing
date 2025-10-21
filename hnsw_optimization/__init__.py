"""
HNSW Optimization: High-layer Bridge Edges and Adaptive Multi-entry Seeds

This package implements two key optimizations for Hierarchical Navigable Small World (HNSW) graphs:
1. High-layer Bridge Edges for improved cross-modal reachability
2. Adaptive Multi-entry Seeds for enhanced search performance

Main components:
- data_loader: Dataset loading and generation utilities
- gt_utils: Ground truth computation and evaluation
- hnsw_with_bridges_optimized: Optimized HNSW with integrated bridges (recommended)
- bridge_builder: Bridge edge construction and management
- multi_entry_search: Multi-entry search implementation
"""

__version__ = "1.0.0"
__author__ = "HNSW Optimization Team"

# Import main classes for easy access
from .data_loader import DataLoader, create_toy_dataset
from .gt_utils import GroundTruthComputer
from .hnsw_with_bridges_optimized import HNSWWithBridgesOptimized

__all__ = [
    'DataLoader',
    'create_toy_dataset',
    'GroundTruthComputer',
    'HNSWWithBridgesOptimized',  # 优化版（使用聚类，推荐）
]
