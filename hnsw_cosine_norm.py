# from __future__ import annotations

"""
HNSW Index with Global and Sub-modality Centering + Whitening (ZCA/PCA)

This file implements the data preprocessing technique described in the image:
- Global and Sub-modality Centering + Whitening (ZCA/PCA)
- Eliminates global offset and "stretching" in principal directions
- Makes the space more isotropic and reduces the dominance of "modality axes"
- Compatible with HNSW and cosine similarity search

Key features:
1. One-time offline preprocessing with constant online overhead
2. Sub-modality (Text/Image) mean and covariance calculation
3. Sub-modality whitening with same scale return
4. Optional alignment between modalities
5. Final L2 normalization for cosine similarity compatibility
"""

import math
import random
import queue
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Union
# Removed sklearn dependency - using numpy implementation instead

from simple_sim_hash import SimpleSimHash

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 先离线把向量做 unit-norm，再使用 1 - 内积（= 1 - cos）
    return float(1.0 - np.dot(a, b))

def cosine_distance_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    vec: shape (d,)
    arr: shape (n, d)
    要求 vec 和 arr 已经预先 unit-norm 归一化
    """
    # 一次性计算所有 dot product
    a = a.reshape(1, -1)
    b = b.reshape(-1, a.shape[1])
    dots = np.matmul(a, b.T)  # shape (n,)
    
    # cosine distance = 1 - cos
    distances = 1.0 - dots
    return distances.flatten()

def _unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)

@dataclass
class Node:
    """Represents a single data item in the HNSW index.

    Attributes
    ----------
    vector : np.ndarray
        The preprocessed vector (centered, whitened, and normalized).
    level : int
        The maximum layer to which this node belongs (layer 0 is the base).
    modality : str
        The modality type ('text' or 'image').
    original_id : int
        The original ID before preprocessing.
    """

    vector: np.ndarray
    level: int
    modality: str = "unknown"
    original_id: int = -1

class DataPreprocessor:
    """Data preprocessing class for centering and whitening."""
    
    def __init__(self, use_pca: bool = True, n_components: Optional[int] = None, 
                 use_global_whitening: bool = True, sub_modality_scaling: bool = True):
        """
        Initialize the data preprocessor.
        
        Parameters:
        -----------
        use_pca : bool
            Whether to use PCA (True) or ZCA (False) whitening
        n_components : Optional[int]
            Number of components to keep (None for all)
        use_global_whitening : bool
            Whether to use global whitening + light sub-modality tuning
        sub_modality_scaling : bool
            Whether to apply sub-modality specific scaling
        """
        self.use_pca = use_pca
        self.n_components = n_components
        self.use_global_whitening = use_global_whitening
        self.sub_modality_scaling = sub_modality_scaling
        
        # Global statistics
        self.global_mean = None
        self.global_cov = None
        self.global_whitening_matrix = None
        
        # Sub-modality statistics
        self.text_mean = None
        self.text_cov = None
        self.text_whitening_matrix = None
        self.text_scaling_factor = 1.0
        
        self.image_mean = None
        self.image_cov = None
        self.image_whitening_matrix = None
        self.image_scaling_factor = 1.0
        
        # Fitted flag
        self.is_fitted = False
    
    def fit(self, text_data: np.ndarray, image_data: np.ndarray, 
            sample_size: Optional[int] = None, random_seed: int = 42):
        """
        Fit the preprocessor on text and image data.
        
        Parameters:
        -----------
        text_data : np.ndarray
            Text vectors of shape (n_text, d)
        image_data : np.ndarray
            Image vectors of shape (n_image, d)
        sample_size : Optional[int]
            Number of samples to use for covariance estimation (None for all)
        random_seed : int
            Random seed for sampling
        """
        np.random.seed(random_seed)
        
        # Sample data if needed (for large datasets)
        if sample_size is not None:
            n_text = min(sample_size, len(text_data))
            n_image = min(sample_size, len(image_data))
            
            text_indices = np.random.choice(len(text_data), n_text, replace=False)
            image_indices = np.random.choice(len(image_data), n_image, replace=False)
            
            text_sample = text_data[text_indices]
            image_sample = image_data[image_indices]
        else:
            text_sample = text_data
            image_sample = image_data
        
        print(f"Fitting preprocessor on {len(text_sample)} text samples and {len(image_sample)} image samples")
        
        if self.use_global_whitening:
            # Method 1: Global whitening + light sub-modality tuning
            self._fit_global_whitening(text_sample, image_sample)
        else:
            # Method 2: Sub-modality whitening
            self._fit_sub_modality_whitening(text_sample, image_sample)
        
        self.is_fitted = True
        print("Preprocessor fitting completed")
    
    def _fit_global_whitening(self, text_sample: np.ndarray, image_sample: np.ndarray):
        """Fit global whitening approach."""
        # Combine all data
        all_data = np.vstack([text_sample, image_sample])
        
        # Calculate global mean and covariance
        self.global_mean = np.mean(all_data, axis=0)
        centered_data = all_data - self.global_mean
        
        # Calculate covariance matrix
        self.global_cov = np.cov(centered_data.T)
        
        # Compute whitening matrix using SVD
        U, s, Vt = np.linalg.svd(self.global_cov, full_matrices=False)
        
        # Handle numerical stability
        s = np.maximum(s, 1e-12)
        
        if self.n_components is not None:
            # Keep only top components
            s = s[:self.n_components]
            U = U[:, :self.n_components]
            Vt = Vt[:self.n_components, :]
        
        # Compute whitening matrix
        if self.use_pca:
            # PCA whitening: W = U * diag(1/sqrt(s))
            self.global_whitening_matrix = U @ np.diag(1.0 / np.sqrt(s))
        else:
            # ZCA whitening: W = U * diag(1/sqrt(s)) * U^T
            self.global_whitening_matrix = U @ np.diag(1.0 / np.sqrt(s)) @ U.T
        
        # Skip sub-modality scaling when using dimensionality reduction to avoid dimension mismatch
        # Calculate sub-modality specific scaling factors only when not using dimensionality reduction
        if self.sub_modality_scaling and self.n_components is None:
            self._calculate_sub_modality_scaling(text_sample, image_sample)
        else:
            # Set default scaling factors when using dimensionality reduction
            self.text_scaling_factor = 1.0
            self.image_scaling_factor = 1.0
    
    def _fit_sub_modality_whitening(self, text_data: np.ndarray, image_data: np.ndarray):
        """Fit sub-modality whitening approach."""
        # Text modality
        self.text_mean = np.mean(text_data, axis=0)
        text_centered = text_data - self.text_mean
        self.text_cov = np.cov(text_centered.T)
        
        # Image modality
        self.image_mean = np.mean(image_data, axis=0)
        image_centered = image_data - self.image_mean
        self.image_cov = np.cov(image_centered.T)
        
        # Compute whitening matrices for each modality
        self._compute_whitening_matrix(self.text_cov, "text")
        self._compute_whitening_matrix(self.image_cov, "image")
    
    def _compute_whitening_matrix(self, cov_matrix: np.ndarray, modality: str):
        """Compute whitening matrix for a given covariance matrix."""
        U, s, Vt = np.linalg.svd(cov_matrix, full_matrices=False)
        s = np.maximum(s, 1e-12)
        
        if self.n_components is not None:
            s = s[:self.n_components]
            U = U[:, :self.n_components]
            Vt = Vt[:self.n_components, :]
        
        if self.use_pca:
            whitening_matrix = U @ np.diag(1.0 / np.sqrt(s))
        else:
            whitening_matrix = U @ np.diag(1.0 / np.sqrt(s)) @ U.T
        
        if modality == "text":
            self.text_whitening_matrix = whitening_matrix
        else:
            self.image_whitening_matrix = whitening_matrix
    
    def _calculate_sub_modality_scaling(self, text_data: np.ndarray, image_data: np.ndarray):
        """Calculate sub-modality specific scaling factors."""
        # Apply global whitening to both modalities (direct calculation)
        text_centered = text_data - self.global_mean
        text_whitened = text_centered @ self.global_whitening_matrix.T
        
        image_centered = image_data - self.global_mean
        image_whitened = image_centered @ self.global_whitening_matrix.T
        
        # Calculate scaling factors to bring them to similar scales
        text_scale = np.mean(np.linalg.norm(text_whitened, axis=1))
        image_scale = np.mean(np.linalg.norm(image_whitened, axis=1))
        
        # Normalize scaling factors
        avg_scale = (text_scale + image_scale) / 2
        self.text_scaling_factor = avg_scale / text_scale
        self.image_scaling_factor = avg_scale / image_scale
        
        print(f"Sub-modality scaling factors - Text: {self.text_scaling_factor:.4f}, Image: {self.image_scaling_factor:.4f}")
    
    def transform_batch(self, data: np.ndarray, modality: str) -> np.ndarray:
        """
        Transform a batch of data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data of shape (n, d)
        modality : str
            Modality type ('text' or 'image')
            
        Returns:
        --------
        np.ndarray
            Transformed data of shape (n, d')
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transformation")
        
        if self.use_global_whitening:
            # Global whitening approach
            centered_data = data - self.global_mean
            # Handle dimensionality reduction case
            if self.n_components is not None:
                # For PCA/ZCA with dimensionality reduction, the whitening matrix has shape (d, n_components)
                whitened_data = centered_data @ self.global_whitening_matrix
            else:
                # For full dimensionality, use transpose
                whitened_data = centered_data @ self.global_whitening_matrix.T
            
            # Apply sub-modality scaling
            if self.sub_modality_scaling:
                if modality == "text":
                    whitened_data *= self.text_scaling_factor
                else:
                    whitened_data *= self.image_scaling_factor
            
            # Final L2 normalization
            norms = np.linalg.norm(whitened_data, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            return (whitened_data / norms).astype(np.float32)
        
        else:
            # Sub-modality whitening approach
            if modality == "text":
                mean = self.text_mean
                whitening_matrix = self.text_whitening_matrix
            else:
                mean = self.image_mean
                whitening_matrix = self.image_whitening_matrix
            
            # Apply whitening: x̃m = Σm^(-1/2) (x - µm)
            centered_data = data - mean
            whitened_data = centered_data @ whitening_matrix.T
            
            # Final L2 normalization
            norms = np.linalg.norm(whitened_data, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            return (whitened_data / norms).astype(np.float32)
    
    def transform_single(self, vector: np.ndarray, modality: str) -> np.ndarray:
        """
        Transform a single vector.
        
        Parameters:
        -----------
        vector : np.ndarray
            Input vector of shape (d,)
        modality : str
            Modality type ('text' or 'image')
            
        Returns:
        --------
        np.ndarray
            Transformed vector of shape (d',)
        """
        return self.transform_batch(vector.reshape(1, -1), modality).flatten()

class HNSWIndex:
    """Hierarchical Navigable Small World index with data preprocessing."""

    def __init__(
        self,
        M: int = 8,
        ef_construction: int = 200,
        ef_search: int = 50,
        random_seed: Optional[int] = None,
        distance_fn=cosine_distance,
        preprocessor: Optional[DataPreprocessor] = None,
        max_search_nodes: int = 128,  # 新增：最大搜索节点数限制
    ) -> None:
        self.M = M
        self.ef_construction = max(ef_construction, M)
        self.ef_search = max(ef_search, M)
        self.distance = distance_fn
        self.preprocessor = preprocessor
        self.max_search_nodes = max_search_nodes  # 新增：最大搜索节点数限制

        # map node id -> Node
        self.items: Dict[int, Node] = {}
        # adjacency lists per layer: layer -> id -> list of neighbour ids
        self.neighbours: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        # 新增：边的标记信息，用于跟踪 cross distribution 边
        self.edge_flags: Dict[int, Dict[int, Dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
        # 新增：统计信息
        self.cross_distribution_stats = {
            "total_cross_edges": 0,
            "deleted_cross_edges": 0,
            "cross_edges_by_query": defaultdict(int)
        }
        self.max_level: int = -1
        self.entry_point: Optional[int] = None
        self._id_counter = 0
        if random_seed is not None:
            random.seed(random_seed)

    def _assign_level(self) -> int:
        """Draw a random level for a new node (exponential distribution)."""
        if self.M <= 1:
            return 0
        lam = 1.0 / math.log(self.M)
        r = random.random()
        level = int(-math.log(r) * lam)
        return level

    def add_item(self, vector: np.ndarray, id: Optional[int] = None, 
                 modality: str = "unknown", original_id: int = -1, 
                 preprocessed: bool = False) -> int:
        """Add a new vector with optional preprocessing.
        
        Parameters
        ----------
        vector : np.ndarray
            Continuous vector (already preprocessed if preprocessed=True).
        id : Optional[int], optional
            User-specified id; if None, an auto id is assigned.
        modality : str
            Modality type ('text' or 'image')
        original_id : int
            Original ID before preprocessing
        preprocessed : bool
            Whether the vector is already preprocessed (default: False)
            
        Returns
        -------
        int
            The id assigned to the inserted item.
        """
        # Apply preprocessing only if not already preprocessed
        if preprocessed:
            # Vector is already preprocessed, just ensure it's unit normalized
            vec = _unit_norm(np.asarray(vector, dtype=np.float32))
        elif self.preprocessor is not None and self.preprocessor.is_fitted:
            # Apply preprocessing if available
            vec = self.preprocessor.transform_single(vector, modality)
        else:
            # No preprocessing, just normalize
            vec = _unit_norm(np.asarray(vector, dtype=np.float32))
        
        # assign id and level
        if id is None:
            id = self._id_counter
            self._id_counter += 1
        level = self._assign_level()
        self.items[id] = Node(vector=vec, level=level, modality=modality, original_id=original_id)

        # first element initializes the entry point
        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            return id

        # if new node has higher level than current max, set as new entry point
        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

        current_node = self.entry_point
        # insertion: search down from top level to level+1 greedily
        for l in range(self.max_level, level, -1):
            current_node = self._search_layer_greedy(vec, current_node, l)

        # search and connect neighbours on layers <= level
        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vec, current_node, l, self.ef_construction)
            neighbours = self._select_neighbors(vec, candidates, self.M)
            for nb in neighbours:
                self._add_link(id, nb, l)
                self._add_link(nb, id, l)
            if neighbours:
                current_node = neighbours[0]
        return id

    def add_items_batch(self, vectors: np.ndarray, ids: Optional[List[int]] = None,
                       modalities: Optional[List[str]] = None, original_ids: Optional[List[int]] = None,
                       preprocessed: bool = False, batch_size: int = 1000) -> List[int]:
        """Add multiple vectors in batch for faster construction.
        
        Parameters
        ----------
        vectors : np.ndarray
            Array of vectors of shape (n, d)
        ids : Optional[List[int]]
            List of IDs for each vector
        modalities : Optional[List[str]]
            List of modality types for each vector
        original_ids : Optional[List[int]]
            List of original IDs for each vector
        preprocessed : bool
            Whether the vectors are already preprocessed
        batch_size : int
            Size of batches to process at once
            
        Returns
        -------
        List[int]
            List of assigned IDs
        """
        n_vectors = len(vectors)
        
        # Set default values
        if ids is None:
            ids = list(range(self._id_counter, self._id_counter + n_vectors))
        if modalities is None:
            modalities = ["unknown"] * n_vectors
        if original_ids is None:
            original_ids = [-1] * n_vectors
        
        # Preprocess all vectors at once if needed
        if preprocessed:
            # Vectors are already preprocessed, just ensure unit normalization
            processed_vectors = np.array([_unit_norm(v) for v in vectors])
        elif self.preprocessor is not None and self.preprocessor.is_fitted:
            # Batch preprocessing for efficiency
            processed_vectors = []
            for i, (vec, modality) in enumerate(zip(vectors, modalities)):
                processed_vec = self.preprocessor.transform_single(vec, modality)
                processed_vectors.append(processed_vec)
            processed_vectors = np.array(processed_vectors)
        else:
            # No preprocessing, just normalize
            processed_vectors = np.array([_unit_norm(v) for v in vectors])
        
        # Process in batches to avoid memory issues and improve performance
        added_ids = []
        for start_idx in range(0, n_vectors, batch_size):
            end_idx = min(start_idx + batch_size, n_vectors)
            batch_vectors = processed_vectors[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            batch_modalities = modalities[start_idx:end_idx]
            batch_original_ids = original_ids[start_idx:end_idx]
            
            # Add items in this batch
            for i, (vec, id_val, modality, orig_id) in enumerate(zip(batch_vectors, batch_ids, batch_modalities, batch_original_ids)):
                level = self._assign_level()
                self.items[id_val] = Node(vector=vec, level=level, modality=modality, original_id=orig_id)
                
                # Update entry point if needed
                if self.entry_point is None:
                    self.entry_point = id_val
                    self.max_level = level
                elif level > self.max_level:
                    self.max_level = level
                    self.entry_point = id_val
                
                added_ids.append(id_val)
                self._id_counter = max(self._id_counter, id_val + 1)
            
            # Build connections for items in this batch
            for i, (vec, id_val, level) in enumerate(zip(batch_vectors, batch_ids, [self.items[id_val].level for id_val in batch_ids])):
                current_node = self.entry_point
                
                # Search down from top level to level+1 greedily
                for l in range(self.max_level, level, -1):
                    current_node = self._search_layer_greedy(vec, current_node, l)
                
                # Search and connect neighbours on layers <= level
                for l in range(min(level, self.max_level), -1, -1):
                    candidates = self._search_layer(vec, current_node, l, self.ef_construction)
                    neighbours = self._select_neighbors(vec, candidates, self.M)
                    for nb in neighbours:
                        self._add_link(id_val, nb, l)
                        self._add_link(nb, id_val, l)
                    if neighbours:
                        current_node = neighbours[0]
        
        return added_ids

    def _add_link(self, source: int, dest: int, layer: int) -> None:
        """Add a directed link from ``source`` to ``dest`` on ``layer`` and prune."""
        nbrs = self.neighbours[layer][source]
        if dest not in nbrs:
            nbrs.append(dest)
            if len(nbrs) > self.M:
                # 计算到 source 的距离并保留最近的 M 个
                distances = [(n, self.distance(self.items[source].vector, self.items[n].vector))
                             for n in nbrs]
                distances.sort(key=lambda x: x[1])
                # 记录被删除的 cross distribution 边
                deleted_neighbors = [n for n, _ in distances[self.M:]]
                for deleted_nb in deleted_neighbors:
                    if self.edge_flags[layer][source].get(deleted_nb) == "cross_distribution":
                        self.cross_distribution_stats["deleted_cross_edges"] += 1
                        # 从标记中移除
                        if deleted_nb in self.edge_flags[layer][source]:
                            del self.edge_flags[layer][source][deleted_nb]
                
                self.neighbours[layer][source] = [n for n, _ in distances[:self.M]]

    def _add_cross_distribution_link(self, source: int, dest: int, layer: int, query_id: Optional[int] = None) -> None:
        """Add a cross distribution link with marking."""
        nbrs = self.neighbours[layer][source]
        if dest not in nbrs:
            nbrs.append(dest)
            # 标记为 cross distribution 边
            self.edge_flags[layer][source][dest] = "cross_distribution"
            self.cross_distribution_stats["total_cross_edges"] += 1
            if query_id is not None:
                self.cross_distribution_stats["cross_edges_by_query"][query_id] += 1
            
            if len(nbrs) > self.M:
                # 计算到 source 的距离并保留最近的 M 个
                distances = [(n, self.distance(self.items[source].vector, self.items[n].vector))
                             for n in nbrs]
                distances.sort(key=lambda x: x[1])
                # 记录被删除的 cross distribution 边
                deleted_neighbors = [n for n, _ in distances[self.M:]]
                for deleted_nb in deleted_neighbors:
                    if self.edge_flags[layer][source].get(deleted_nb) == "cross_distribution":
                        self.cross_distribution_stats["deleted_cross_edges"] += 1
                        # 从标记中移除
                        if deleted_nb in self.edge_flags[layer][source]:
                            del self.edge_flags[layer][source][deleted_nb]
                
                self.neighbours[layer][source] = [n for n, _ in distances[:self.M]]

    def _dist(self, u: int, v: int) -> float:
        """Pairwise distance between two data nodes by id."""
        if u not in self.items or v not in self.items:
            return float("inf")

        a = self.items[u].vector
        b = self.items[v].vector

        try:
            return 1.0 - float(a.dot(b))
        except Exception:
            return float(self.distance(a, b))

    def _select_neighbors(self, vec: np.ndarray, candidates: Iterable[int], M: int) -> List[int]:
        """Pick the M closest candidate ids to vec (simple heuristic)."""
        cands = list(candidates)
        cands.sort(key=lambda cid: self.distance(vec, self.items[cid].vector))
        return cands[:M]

    def _search_layer_greedy(self, vec: np.ndarray, entry_id: int, layer: int) -> int:
        """Greedy search for the closest neighbour on a specific layer."""
        current = entry_id
        best_dist = self.distance(vec, self.items[current].vector)
        improved = True
        while improved:
            improved = False
            for nb in self.neighbours[layer][current]:
                dist = self.distance(vec, self.items[nb].vector)
                if dist < best_dist:
                    best_dist = dist
                    current = nb
                    improved = True
        return current

    def _search_layer(self, vec: np.ndarray, entry_id: int, layer: int, ef: int) -> List[int]:
        """Best-first search within one layer (beam width = ef) with search depth limit."""
        import heapq
        visited: set[int] = set()
        candidates: List[Tuple[float, int]] = []   # min-heap by dist
        result: List[Tuple[float, int]] = []       # max-heap via negative dist

        dist_entry = self.distance(vec, self.items[entry_id].vector)
        heapq.heappush(candidates, (dist_entry, entry_id))
        heapq.heappush(result, (-dist_entry, entry_id))
        visited.add(entry_id)

        # 添加搜索节点数限制
        nodes_explored = 0
        
        while candidates and nodes_explored < self.max_search_nodes:
            dist_curr, curr = heapq.heappop(candidates)
            worst_dist = -result[0][0] if result else math.inf
            if dist_curr > worst_dist:
                break
            for nb in self.neighbours[layer][curr]:
                if nb in visited:
                    continue
                visited.add(nb)
                nodes_explored += 1  # 增加探索节点计数
                d = self.distance(vec, self.items[nb].vector)
                if len(result) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(result, (-d, nb))
                    if len(result) > ef:
                        heapq.heappop(result)
                        worst_dist = -result[0][0]
                
                # 如果达到最大搜索节点数，提前退出
                if nodes_explored >= self.max_search_nodes:
                    break
        return [nid for (_, nid) in result]

    def query(self, vector: np.ndarray, k: int, modality: str = "unknown") -> List[Tuple[int, float]]:
        """Return the k nearest neighbours to vector (id, distance)."""
        if not self.items:
            return []
        
        # Apply preprocessing if available
        if self.preprocessor is not None and self.preprocessor.is_fitted:
            vec = self.preprocessor.transform_single(vector, modality)
        else:
            vec = _unit_norm(np.asarray(vector, dtype=np.float32))
        
        curr = self.entry_point
        for l in range(self.max_level, 0, -1):
            curr = self._search_layer_greedy(vec, curr, l)
        candidates = self._search_layer(vec, curr, 0, max(self.ef_search, k))
        dists = [(cid, self.distance(vec, self.items[cid].vector)) for cid in candidates]
        dists.sort(key=lambda x: x[1])
        return dists[:k]

    def build_cross_distribution_edges(
        self,
        query: np.ndarray,
        top_k: int = 10,
        max_new_edges_per_node: int = 4,
        random_seed: int = None,
        modality: str = "unknown"
    ) -> dict:
        """
        基于query在layer 1中构建cross distribution边（与hnsw_cosine_status_high.py相同）
        """
        import random
        
        if random_seed is not None:
            random.seed(random_seed)
        
        if self.max_level < 1 or 1 not in self.neighbours:
            return {"error": "Layer 1 not available"}
        
        # Apply preprocessing to query
        if self.preprocessor is not None and self.preprocessor.is_fitted:
            query_vec = self.preprocessor.transform_single(query, modality)
        else:
            query_vec = _unit_norm(np.asarray(query, dtype=np.float32))
        
        from collections import defaultdict
        added_per_node: Dict[int, int] = defaultdict(int)

        stats = {
            "query_processed": True,
            "layer_1_nodes_total": 0,
            "top_k_selected": 0,
            "pairs_considered": 0,
            "pairs_added": 0,
            "skipped_existing": 0,
            "pruned_by_cap": 0,
            "edges_added": 0,
            "top_k_nodes": [],
            "query_distance": 0.0
        }

        def can_add(u: int, v: int, current_layer: int):
            if (max_new_edges_per_node is not None) and ( (added_per_node[u] >= max_new_edges_per_node) or (added_per_node[v] >= max_new_edges_per_node) ):
                stats["pruned_by_cap"] += 1
                return False
            if v in self.neighbours[current_layer][u]:
                stats["skipped_existing"] += 1
                return False
            return True

        def _add_cross_pair(u: int, v: int, current_layer: int):
            self._add_cross_distribution_link(u, v, current_layer, None)
            self._add_cross_distribution_link(v, u, current_layer, None)
            added_per_node[u] += 1
            added_per_node[v] += 1
            stats["pairs_added"] += 1
            stats["edges_added"] += 1

        # Get layer 1 nodes and find top-k
        layer_1_nodes = list(self.neighbours[1].keys())
        stats["layer_1_nodes_total"] = len(layer_1_nodes)
        
        if len(layer_1_nodes) < 2:
            return {"error": "Not enough nodes in layer 1"}
        
        # Calculate distances to query
        node_distances = []
        for node_id in layer_1_nodes:
            if node_id in self.items:
                dist = self.distance(query_vec, self.items[node_id].vector)
                node_distances.append((dist, node_id))
        
        # Select top-k
        node_distances.sort(key=lambda x: x[0])
        top_k_nodes = [node_id for _, node_id in node_distances[:min(top_k, len(node_distances))]]
        stats["top_k_selected"] = len(top_k_nodes)
        stats["top_k_nodes"] = top_k_nodes
        
        if len(top_k_nodes) < 2:
            return {"error": "Not enough nodes in top-k selection"}
        
        if node_distances:
            stats["query_distance"] = node_distances[0][0]
        
        # Add edges between top-k nodes
        for i in range(len(top_k_nodes)):
            for j in range(i + 1, len(top_k_nodes)):
                u, v = top_k_nodes[i], top_k_nodes[j]
                stats["pairs_considered"] += 1
                
                if can_add(u, v, 1):
                    _add_cross_pair(u, v, 1)

        return stats

    def get_cross_distribution_stats(self) -> dict:
        """获取 cross distribution 边的统计信息"""
        return {
            "total_cross_edges": self.cross_distribution_stats["total_cross_edges"],
            "deleted_cross_edges": self.cross_distribution_stats["deleted_cross_edges"],
            "active_cross_edges": self.cross_distribution_stats["total_cross_edges"] - self.cross_distribution_stats["deleted_cross_edges"],
            "cross_edges_by_query": dict(self.cross_distribution_stats["cross_edges_by_query"])
        }

    def reset_cross_distribution_stats(self) -> None:
        """重置 cross distribution 统计信息"""
        self.cross_distribution_stats = {
            "total_cross_edges": 0,
            "deleted_cross_edges": 0,
            "cross_edges_by_query": defaultdict(int)
        }

    def get_modality_stats(self) -> dict:
        """Get statistics about modalities in the index."""
        modality_counts = defaultdict(int)
        modality_levels = defaultdict(list)
        
        for node_id, node in self.items.items():
            modality_counts[node.modality] += 1
            modality_levels[node.modality].append(node.level)
        
        stats = {}
        for modality, count in modality_counts.items():
            levels = modality_levels[modality]
            stats[modality] = {
                "count": count,
                "avg_level": np.mean(levels) if levels else 0,
                "max_level": max(levels) if levels else 0,
                "min_level": min(levels) if levels else 0
            }
        
        return stats

# Utility functions for loading data
def load_fbin_data(file_path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Load data from .fbin file format.
    
    Parameters:
    -----------
    file_path : str
        Path to the .fbin file
    dtype : np.dtype
        Data type to use
        
    Returns:
    --------
    np.ndarray
        Loaded data array
    """
    with open(file_path, 'rb') as f:
        # Read dimensions
        n_vectors = int.from_bytes(f.read(4), byteorder='little')
        dim = int.from_bytes(f.read(4), byteorder='little')
        
        # Read data
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape(n_vectors, dim)
        
    return data

def create_preprocessed_index(text_data_path: str, image_data_path: str, 
                            query_data_path: Optional[str] = None,
                            use_global_whitening: bool = True,
                            n_components: Optional[int] = None,
                            sample_size: int = 100000,
                            M: int = 8, ef_construction: int = 200) -> Tuple[HNSWIndex, DataPreprocessor]:
    """
    Create a preprocessed HNSW index from text and image data.
    
    Parameters:
    -----------
    text_data_path : str
        Path to text data .fbin file
    image_data_path : str
        Path to image data .fbin file
    query_data_path : Optional[str]
        Path to query data .fbin file (optional)
    use_global_whitening : bool
        Whether to use global whitening approach
    n_components : Optional[int]
        Number of components to keep (None for all)
    sample_size : int
        Number of samples to use for preprocessing fitting
    M : int
        HNSW M parameter
    ef_construction : int
        HNSW ef_construction parameter
        
    Returns:
    --------
    Tuple[HNSWIndex, DataPreprocessor]
        The created index and preprocessor
    """
    print("Loading data...")
    
    # Load data
    text_data = load_fbin_data(text_data_path)
    image_data = load_fbin_data(image_data_path)
    
    print(f"Loaded {len(text_data)} text vectors and {len(image_data)} image vectors")
    print(f"Vector dimension: {text_data.shape[1]}")
    
    # Create and fit preprocessor
    preprocessor = DataPreprocessor(
        use_pca=True,
        n_components=n_components,
        use_global_whitening=use_global_whitening,
        sub_modality_scaling=True
    )
    
    preprocessor.fit(text_data, image_data, sample_size=sample_size)
    
    # Create HNSW index
    index = HNSWIndex(
        M=M,
        ef_construction=ef_construction,
        preprocessor=preprocessor
    )
    
    # Add text data
    print("Adding text data to index...")
    for i, vec in enumerate(text_data):
        index.add_item(vec, id=i, modality="text", original_id=i)
        if (i + 1) % 10000 == 0:
            print(f"Added {i + 1} text vectors")
    
    # Add image data
    print("Adding image data to index...")
    for i, vec in enumerate(image_data):
        index.add_item(vec, id=len(text_data) + i, modality="image", original_id=i)
        if (i + 1) % 10000 == 0:
            print(f"Added {i + 1} image vectors")
    
    print(f"Index created with {len(index.items)} total vectors")
    print(f"Max level: {index.max_level}")
    print(f"Entry point: {index.entry_point}")
    
    # Print modality statistics
    modality_stats = index.get_modality_stats()
    for modality, stats in modality_stats.items():
        print(f"{modality}: {stats['count']} vectors, avg level: {stats['avg_level']:.2f}")
    
    return index, preprocessor
