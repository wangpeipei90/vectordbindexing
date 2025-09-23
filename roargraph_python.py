#!/usr/bin/env python3
"""
RoarGraph Python Implementation

基于 RoarGraph 论文的 Python 实现，包含核心的二分图构建、投影图生成和搜索算法。
"""

import numpy as np
import heapq
import random
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import time


class Neighbor:
    """邻居节点类，用于优先队列"""
    def __init__(self, node_id: int, distance: float, expanded: bool = False):
        self.id = node_id
        self.distance = distance
        self.expanded = expanded
    
    def __lt__(self, other):
        return self.distance < other.distance
    
    def __eq__(self, other):
        return self.distance == other.distance


class NeighborPriorityQueue:
    """邻居优先队列"""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.queue = []
        self.expanded = set()
    
    def insert(self, neighbor: Neighbor):
        """插入邻居节点"""
        if len(self.queue) < self.max_size:
            heapq.heappush(self.queue, neighbor)
        elif neighbor.distance < self.queue[0].distance:
            heapq.heapreplace(self.queue, neighbor)
    
    def has_unexpanded_node(self) -> bool:
        """检查是否有未扩展的节点"""
        return any(not neighbor.expanded for neighbor in self.queue)
    
    def closest_unexpanded(self) -> Neighbor:
        """获取最近的未扩展节点"""
        for neighbor in self.queue:
            if not neighbor.expanded:
                neighbor.expanded = True
                self.expanded.add(neighbor.id)
                return neighbor
        return None
    
    def get_results(self, k: int) -> List[Tuple[int, float]]:
        """获取前k个结果"""
        sorted_queue = sorted(self.queue, key=lambda x: x.distance)
        return [(neighbor.id, neighbor.distance) for neighbor in sorted_queue[:k]]


class RoarGraph:
    """RoarGraph 主类"""
    
    def __init__(self, dimension: int, metric: str = "cosine"):
        """
        初始化 RoarGraph
        
        Args:
            dimension: 向量维度
            metric: 距离度量 ("cosine", "l2", "ip")
        """
        self.dimension = dimension
        self.metric = metric
        self.need_normalize = (metric == "cosine")
        
        # 数据存储
        self.base_data = None  # 基础数据
        self.query_data = None  # 查询数据
        self.n_base = 0
        self.n_query = 0
        
        # 图结构
        self.bipartite_graph = []  # 二分图
        self.projection_graph = []  # 投影图
        self.learn_base_knn = []  # 查询到基础的KNN
        
        # 参数
        self.M_sq = 32  # 查询节点的邻居数
        self.M_pjbp = 32  # 投影图的邻居数
        self.L_pjpq = 32  # 搜索队列大小
        
        # 入口点
        self.projection_ep = 0
        
    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """向量归一化"""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算距离"""
        if self.metric == "cosine":
            return 1.0 - np.dot(a, b)
        elif self.metric == "l2":
            return np.linalg.norm(a - b)
        elif self.metric == "ip":
            return -np.dot(a, b)  # 内积取负值，因为我们要最小化
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _distance_batch(self, query: np.ndarray, data: np.ndarray) -> np.ndarray:
        """批量计算距离"""
        if self.metric == "cosine":
            return 1.0 - np.dot(data, query)
        elif self.metric == "l2":
            return np.linalg.norm(data - query, axis=1)
        elif self.metric == "ip":
            return -np.dot(data, query)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def load_data(self, base_data: np.ndarray, query_data: np.ndarray):
        """
        加载数据
        
        Args:
            base_data: 基础数据，形状 (n_base, dimension)
            query_data: 查询数据，形状 (n_query, dimension)
        """
        self.base_data = base_data.astype(np.float32)
        self.query_data = query_data.astype(np.float32)
        self.n_base = base_data.shape[0]
        self.n_query = query_data.shape[0]
        
        # 归一化
        if self.need_normalize:
            print("Normalizing base data...")
            for i in range(self.n_base):
                self.base_data[i] = self._normalize_vector(self.base_data[i])
            print("Normalizing query data...")
            for i in range(self.n_query):
                self.query_data[i] = self._normalize_vector(self.query_data[i])
        
        # 初始化图结构
        self.bipartite_graph = [[] for _ in range(self.n_base + self.n_query)]
        self.projection_graph = [[] for _ in range(self.n_base)]
        self.learn_base_knn = [[] for _ in range(self.n_query)]
    
    def build_learn_base_knn(self, k: int = 100):
        """构建查询到基础的KNN关系"""
        print("Building learn-base KNN...")
        for i in range(self.n_query):
            query = self.query_data[i]
            distances = self._distance_batch(query, self.base_data)
            knn_indices = np.argsort(distances)[:k]
            self.learn_base_knn[i] = knn_indices.tolist()
    
    def calculate_projection_ep(self):
        """计算投影图的入口点"""
        # 选择第一个基础数据点作为入口点
        self.projection_ep = 0
        print(f"Projection entry point: {self.projection_ep}")
    
    def prune_candidates(self, candidates: List[Neighbor], target_id: int, 
                        max_neighbors: int) -> List[int]:
        """
        修剪候选节点（RoarGraph 的核心投影策略）
        
        Args:
            candidates: 候选邻居列表
            target_id: 目标节点ID
            max_neighbors: 最大邻居数
            
        Returns:
            修剪后的邻居列表
        """
        if len(candidates) <= max_neighbors:
            return [c.id for c in candidates]
        
        # 按距离排序
        candidates.sort(key=lambda x: x.distance)
        
        # 选择最近的邻居作为pivot
        pivot = candidates[0]
        selected = [pivot.id]
        
        # 使用投影策略选择其他邻居
        for candidate in candidates[1:]:
            if len(selected) >= max_neighbors:
                break
            
            # 检查是否与已选择的邻居过于相似（遮挡检查）
            should_add = True
            for selected_id in selected:
                if selected_id == candidate.id:
                    continue
                
                # 计算候选节点与已选择节点的距离
                dist_candidate_selected = self._distance(
                    self.base_data[candidate.id], 
                    self.base_data[selected_id]
                )
                
                # 计算候选节点与pivot的距离
                dist_candidate_pivot = self._distance(
                    self.base_data[candidate.id], 
                    self.base_data[pivot.id]
                )
                
                # 遮挡检查：如果候选节点与已选择节点太近，则跳过
                if dist_candidate_selected <= 1.0 * dist_candidate_pivot:
                    should_add = False
                    break
            
            if should_add:
                selected.append(candidate.id)
        
        return selected
    
    def build_projection_graph(self):
        """构建投影图"""
        print("Building projection graph...")
        
        for sq_id in range(self.n_query):
            if sq_id % 1000 == 0:
                print(f"Processing query {sq_id}/{self.n_query}")
            
            # 获取查询sq_id对应的基础数据邻居
            nn_base = self.learn_base_knn[sq_id]
            if len(nn_base) == 0:
                continue
            
            # 选择第一个邻居作为目标节点
            target_id = nn_base[0]
            
            # 构建候选列表
            candidates = []
            for base_id in nn_base[1:]:  # 跳过目标节点本身
                distance = self._distance(
                    self.base_data[base_id], 
                    self.base_data[target_id]
                )
                candidates.append(Neighbor(base_id, distance, False))
            
            # 修剪候选节点
            pruned_neighbors = self.prune_candidates(candidates, target_id, self.M_pjbp)
            
            # 更新投影图
            self.projection_graph[target_id] = pruned_neighbors
            
            # 添加反向边
            self._add_reverse_edges(target_id, pruned_neighbors)
    
    def _add_reverse_edges(self, src_node: int, neighbors: List[int]):
        """添加反向边"""
        for neighbor_id in neighbors:
            if neighbor_id < len(self.projection_graph):
                if src_node not in self.projection_graph[neighbor_id]:
                    self.projection_graph[neighbor_id].append(src_node)
    
    def build(self, base_data: np.ndarray, query_data: np.ndarray, 
              M_sq: int = 32, M_pjbp: int = 32, L_pjpq: int = 32):
        """
        构建 RoarGraph 索引
        
        Args:
            base_data: 基础数据
            query_data: 查询数据
            M_sq: 查询节点的邻居数
            M_pjbp: 投影图的邻居数
            L_pjpq: 搜索队列大小
        """
        self.M_sq = M_sq
        self.M_pjbp = M_pjbp
        self.L_pjpq = L_pjpq
        
        print("Loading data...")
        self.load_data(base_data, query_data)
        
        print("Building learn-base KNN...")
        self.build_learn_base_knn()
        
        print("Calculating projection entry point...")
        self.calculate_projection_ep()
        
        print("Building projection graph...")
        self.build_projection_graph()
        
        print("RoarGraph construction completed!")
    
    def search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float], int, int]:
        """
        搜索最近邻
        
        Args:
            query: 查询向量
            k: 返回的最近邻数量
            
        Returns:
            (indices, distances, comparisons, hops)
        """
        if self.need_normalize:
            query = self._normalize_vector(query)
        
        # 初始化搜索队列
        search_queue = NeighborPriorityQueue(self.L_pjpq)
        visited = set()
        
        # 从入口点开始
        entry_distance = self._distance(query, self.base_data[self.projection_ep])
        search_queue.insert(Neighbor(self.projection_ep, entry_distance, False))
        visited.add(self.projection_ep)
        
        comparisons = 1
        hops = 0
        
        # 搜索循环
        while search_queue.has_unexpanded_node():
            current_node = search_queue.closest_unexpanded()
            if current_node is None:
                break
            
            current_id = current_node.id
            hops += 1
            
            # 扩展当前节点的邻居
            neighbors = self.projection_graph[current_id]
            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                
                visited.add(neighbor_id)
                distance = self._distance(query, self.base_data[neighbor_id])
                comparisons += 1
                
                search_queue.insert(Neighbor(neighbor_id, distance, False))
        
        # 获取结果
        results = search_queue.get_results(k)
        indices = [result[0] for result in results]
        distances = [result[1] for result in results]
        
        return indices, distances, comparisons, hops
    
    def get_statistics(self) -> Dict:
        """获取图统计信息"""
        projection_degrees = [len(neighbors) for neighbors in self.projection_graph]
        
        return {
            "n_base": self.n_base,
            "n_query": self.n_query,
            "projection_ep": self.projection_ep,
            "avg_projection_degree": np.mean(projection_degrees),
            "max_projection_degree": np.max(projection_degrees),
            "min_projection_degree": np.min(projection_degrees),
            "total_projection_edges": sum(projection_degrees)
        }


def create_sample_data(n_base: int = 10000, n_query: int = 1000, 
                      dimension: int = 128, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """创建示例数据"""
    np.random.seed(seed)
    
    # 生成基础数据
    base_data = np.random.randn(n_base, dimension).astype(np.float32)
    
    # 生成查询数据（与基础数据有一定相关性）
    query_data = np.random.randn(n_query, dimension).astype(np.float32)
    
    return base_data, query_data


def evaluate_recall(predictions: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """计算召回率"""
    total_recall = 0.0
    for pred, gt in zip(predictions, ground_truth):
        intersection = set(pred[:k]) & set(gt[:k])
        recall = len(intersection) / k
        total_recall += recall
    return total_recall / len(predictions)


if __name__ == "__main__":
    # 创建示例数据
    print("Creating sample data...")
    base_data, query_data = create_sample_data(n_base=5000, n_query=500, dimension=128)
    
    # 构建 RoarGraph
    print("Building RoarGraph...")
    roargraph = RoarGraph(dimension=128, metric="cosine")
    roargraph.build(base_data, query_data, M_sq=32, M_pjbp=32, L_pjpq=32)
    
    # 获取统计信息
    stats = roargraph.get_statistics()
    print(f"Statistics: {stats}")
    
    # 测试搜索
    print("Testing search...")
    test_query = query_data[0]
    indices, distances, comparisons, hops = roargraph.search(test_query, k=10)
    
    print(f"Search results: {indices[:5]}...")
    print(f"Distances: {distances[:5]}...")
    print(f"Comparisons: {comparisons}, Hops: {hops}")
    
    print("RoarGraph Python implementation completed!")
