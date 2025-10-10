#!/usr/bin/env python3
"""
Setup script for HNSW optimization project
"""

from setuptools import setup, find_packages

setup(
    name="hnsw_optimization",
    version="1.0.0",
    description="HNSW optimizations: High-layer Bridge Edges and Adaptive Multi-entry Seeds",
    author="HNSW Optimization Team",
    packages=find_packages(),
    install_requires=[
        "hnswlib>=0.7.0",
        "faiss-cpu>=1.7.4", 
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.10",
)
