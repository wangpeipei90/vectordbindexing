#!/usr/bin/env python3
"""
Data loading utilities for HNSW optimization experiments
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage dataset for HNSW experiments"""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            dataset_path: Path to dataset file (optional, can generate synthetic data)
        """
        self.dataset_path = dataset_path
        self.X: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.modalities: Optional[np.ndarray] = None
        self.query_modalities: Optional[np.ndarray] = None
        
    def load_data(self, 
                  n_vectors: int = 10000,
                  n_queries: int = 1000,
                  dimension: int = 128,
                  n_modalities: int = 5,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load or generate dataset
        
        Args:
            n_vectors: Number of base vectors
            n_queries: Number of query vectors
            dimension: Vector dimension
            n_modalities: Number of modalities/clusters
            seed: Random seed
            
        Returns:
            Tuple of (X, Q, modalities, query_modalities)
        """
        np.random.seed(seed)
        
        if self.dataset_path:
            # Load from file (placeholder for real implementation)
            logger.info(f"Loading data from {self.dataset_path}")
            # TODO: Implement actual file loading
            raise NotImplementedError("File loading not implemented yet")
        else:
            # Generate synthetic data
            logger.info(f"Generating synthetic data: {n_vectors} vectors, {n_queries} queries, dim={dimension}")
            self.X, self.modalities = self._generate_synthetic_data(
                n_vectors, dimension, n_modalities, seed
            )
            self.Q, self.query_modalities = self._generate_synthetic_data(
                n_queries, dimension, n_modalities, seed + 1
            )
            
        return self.X, self.Q, self.modalities, self.query_modalities
    
    def _generate_synthetic_data(self, 
                                n_vectors: int, 
                                dimension: int, 
                                n_modalities: int, 
                                seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic dataset with multiple modalities"""
        np.random.seed(seed)
        
        # Generate modality assignments
        modalities = np.random.randint(0, n_modalities, n_vectors)
        
        # Generate vectors with modality-specific centers
        X = np.zeros((n_vectors, dimension))
        
        for modality in range(n_modalities):
            mask = modalities == modality
            n_modality_vectors = np.sum(mask)
            
            if n_modality_vectors > 0:
                # Create modality-specific center
                center = np.random.randn(dimension) * 2.0
                
                # Generate vectors around center with some noise
                X[mask] = center + np.random.randn(n_modality_vectors, dimension) * 0.5
                
        # Normalize vectors
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        return X, modalities
    
    def save_data(self, output_path: str):
        """Save loaded data to file"""
        if self.X is None:
            raise ValueError("No data loaded")
            
        data = {
            'X': self.X,
            'Q': self.Q,
            'modalities': self.modalities,
            'query_modalities': self.query_modalities
        }
        np.savez(output_path, **data)
        logger.info(f"Data saved to {output_path}")
    
    def load_from_file(self, file_path: str):
        """Load data from npz file"""
        data = np.load(file_path)
        self.X = data['X']
        self.Q = data['Q']
        self.modalities = data['modalities']
        self.query_modalities = data['query_modalities']
        logger.info(f"Data loaded from {file_path}")
        
    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        if self.X is None:
            return {}
            
        stats = {
            'n_vectors': len(self.X),
            'n_queries': len(self.Q) if self.Q is not None else 0,
            'dimension': self.X.shape[1],
            'n_modalities': len(np.unique(self.modalities)) if self.modalities is not None else 0,
            'modality_distribution': np.bincount(self.modalities) if self.modalities is not None else None,
        }
        return stats


def create_toy_dataset(n_vectors: int = 1000, 
                      n_queries: int = 100,
                      dimension: int = 128,
                      n_modalities: int = 3,
                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a small toy dataset for testing
    
    Args:
        n_vectors: Number of base vectors
        n_queries: Number of query vectors  
        dimension: Vector dimension
        n_modalities: Number of modalities
        seed: Random seed
        
    Returns:
        Tuple of (X, Q, modalities, query_modalities)
    """
    loader = DataLoader()
    return loader.load_data(n_vectors, n_queries, dimension, n_modalities, seed)


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    X, Q, modalities, query_modalities = loader.load_data()
    
    print("Dataset statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print(f"X shape: {X.shape}")
    print(f"Q shape: {Q.shape}")
    print(f"Modalities shape: {modalities.shape}")
    print(f"Query modalities shape: {query_modalities.shape}")
