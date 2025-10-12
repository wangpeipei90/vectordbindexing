#!/usr/bin/env python3
"""
增强版 HNSW：集成桥接边和多入口搜索
"""

import numpy as np
import hnswlib
from typing import Tuple, List, Optional, Dict, Set
import logging
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class HNSWWithBridges:
    """
    增强版 HNSW，在构建时自动添加桥接边，搜索时使用多入口

    特性：
    1. 构建完成后自动在高层添加桥接边（layer >= 1）
    2. 桥接边选择：随机选择10%节点对，检查2跳可达性
    3. 搜索时使用多入口点提升性能
    """

    def __init__(self,
                 dimension: int,
                 M: int = 16,
                 ef_construction: int = 200,
                 max_elements: int = 1000000,
                 seed: int = 42,
                 # 桥接边参数
                 enable_bridges: bool = True,
                 bridge_sample_ratio: float = 0.1,  # 随机采样10%节点对
                 max_hop_distance: int = 2,  # 检查2跳可达性
                 # 多入口搜索参数
                 enable_multi_entry: bool = True,
                 num_entry_points: int = 3):
        """
        初始化增强版 HNSW

        Args:
            dimension: 向量维度
            M: HNSW 连接数
            ef_construction: 构建时搜索宽度
            max_elements: 最大元素数
            seed: 随机种子
            enable_bridges: 是否启用桥接边
            bridge_sample_ratio: 桥接边采样比例（默认10%）
            max_hop_distance: 最大跳数（默认2跳）
            enable_multi_entry: 是否启用多入口搜索
            num_entry_points: 入口点数量
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.max_elements = max_elements
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 桥接边配置
        self.enable_bridges = enable_bridges
        self.bridge_sample_ratio = bridge_sample_ratio
        self.max_hop_distance = max_hop_distance

        # 多入口搜索配置
        self.enable_multi_entry = enable_multi_entry
        self.num_entry_points = num_entry_points

        # 初始化 hnswlib 索引
        self.index = hnswlib.Index(space='l2', dim=dimension)
        self.index.init_index(
            max_elements=max_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=seed
        )

        # 存储向量数据（用于桥接边计算）
        self.vectors: Optional[np.ndarray] = None
        self.vector_ids: Optional[np.ndarray] = None

        # 桥接边映射：{layer: {node_id: [bridge_neighbor_ids]}}
        self.bridge_edges: Dict[int, Dict[int, List[int]]
                                ] = defaultdict(lambda: defaultdict(list))

        # 高层节点缓存：{layer: [node_ids]}
        self.high_layer_nodes: Dict[int, List[int]] = {}

        self.is_built = False

    def build_index(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """
        构建索引并自动添加桥接边

        Args:
            vectors: 向量数据 (N x D)
            ids: 向量ID（可选）
        """
        logger.info(f"构建增强版 HNSW 索引: {len(vectors)} 个向量")

        # 保存向量数据
        self.vectors = vectors.astype(np.float32)
        if ids is None:
            ids = np.arange(len(vectors))
        self.vector_ids = ids.astype(np.int32)

        # 1. 构建基础 HNSW 索引
        start_time = time.time()
        self.index.add_items(self.vectors, self.vector_ids)
        self.index.set_ef(self.ef_construction)
        build_time = time.time() - start_time

        logger.info(f"基础 HNSW 构建完成: {build_time:.2f}秒")

        # 2. 识别高层节点
        self._identify_high_layer_nodes()

        # 3. 添加桥接边
        if self.enable_bridges:
            bridge_start = time.time()
            self._add_bridge_edges()
            bridge_time = time.time() - bridge_start
            logger.info(f"桥接边添加完成: {bridge_time:.2f}秒")

        self.is_built = True

        # 统计信息
        self._print_statistics()

    def _identify_high_layer_nodes(self):
        """
        识别高层节点（layer >= 1）

        注意：由于 hnswlib 不直接暴露层级信息，我们使用启发式方法估计
        """
        logger.info("识别高层节点...")

        n = len(self.vector_ids)
        ml = 1.0 / np.log(self.M)  # HNSW 层级分布参数

        # 估计每个节点的层级
        for idx, vid in enumerate(self.vector_ids):
            # 使用 HNSW 的层级分布：level ~ -log(uniform(0,1)) * ml
            self.rng.seed(self.seed + int(vid))  # 确保可重现
            level = int(-np.log(self.rng.uniform(0.001, 1.0)) * ml)
            level = min(level, 16)  # 限制最大层级

            if level >= 1:
                if level not in self.high_layer_nodes:
                    self.high_layer_nodes[level] = []
                self.high_layer_nodes[level].append(vid)

        # 统计
        total_high_nodes = sum(len(nodes)
                               for nodes in self.high_layer_nodes.values())
        logger.info(f"高层节点统计:")
        for level in sorted(self.high_layer_nodes.keys(), reverse=True):
            logger.info(
                f"  Layer {level}: {len(self.high_layer_nodes[level])} 个节点")
        logger.info(
            f"总高层节点: {total_high_nodes} / {n} ({100*total_high_nodes/n:.2f}%)")

    def _add_bridge_edges(self):
        """
        在高层添加桥接边

        策略：
        1. 对每一层（layer >= 1）
        2. 随机选择 10% 的节点对
        3. 检查它们之间是否在 2 跳内可达
        4. 如果不可达，添加桥接边
        """
        logger.info(f"开始添加桥接边（采样比例: {self.bridge_sample_ratio*100}%）")

        total_bridges = 0

        for layer in sorted(self.high_layer_nodes.keys(), reverse=True):
            nodes = self.high_layer_nodes[layer]

            if len(nodes) < 2:
                continue

            # 随机选择节点对
            n_pairs = int(len(nodes) * (len(nodes) - 1) /
                          2 * self.bridge_sample_ratio)
            n_pairs = max(n_pairs, 1)  # 至少选择1对

            logger.info(f"  Layer {layer}: 从 {len(nodes)} 个节点中采样 {n_pairs} 对")

            # 生成所有可能的节点对
            pairs = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pairs.append((nodes[i], nodes[j]))

            # 随机采样
            if len(pairs) > n_pairs:
                sampled_pairs = self.rng.choice(
                    len(pairs), size=n_pairs, replace=False)
                pairs = [pairs[idx] for idx in sampled_pairs]

            # 检查每一对并添加桥接边
            bridges_added = 0
            for node1, node2 in pairs:
                if not self._is_reachable_within_hops(node1, node2, self.max_hop_distance):
                    # 添加双向桥接边
                    self.bridge_edges[layer][node1].append(node2)
                    self.bridge_edges[layer][node2].append(node1)
                    bridges_added += 1

            logger.info(f"    添加了 {bridges_added} 条桥接边")
            total_bridges += bridges_added

        logger.info(f"总共添加 {total_bridges} 条桥接边")

    def _is_reachable_within_hops(self, start: int, target: int, max_hops: int) -> bool:
        """
        检查从 start 到 target 是否在 max_hops 跳内可达

        使用 BFS 进行检查

        Args:
            start: 起始节点
            target: 目标节点
            max_hops: 最大跳数

        Returns:
            True 如果可达，False 否则
        """
        if start == target:
            return True

        # BFS
        visited = {start}
        queue = deque([(start, 0)])  # (node, distance)

        while queue:
            current, dist = queue.popleft()

            if dist >= max_hops:
                continue

            # 获取邻居（使用 hnswlib 的 knn_query）
            try:
                # 查询当前节点的邻居（使用较大的 k）
                neighbors, _ = self.index.knn_query(
                    self.vectors[current].reshape(1, -1),
                    k=min(self.M * 2, len(self.vector_ids))
                )
                neighbors = neighbors[0]

                for neighbor in neighbors:
                    if neighbor == current:
                        continue

                    if neighbor == target:
                        return True

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

            except Exception as e:
                logger.warning(f"检查可达性时出错: {e}")
                continue

        return False

    def search(self,
               query: np.ndarray,
               k: int,
               ef_search: int = 200) -> Tuple[np.ndarray, int]:
        """
        增强搜索：使用多入口点和桥接边

        Args:
            query: 查询向量 (D,)
            k: 返回的邻居数量
            ef_search: 搜索宽度

        Returns:
            (neighbors, cost)
        """
        if not self.is_built:
            raise ValueError("索引未构建")

        self.index.set_ef(ef_search)

        if not self.enable_multi_entry or not self.high_layer_nodes:
            # 标准单入口搜索
            neighbors, _ = self.index.knn_query(
                query.reshape(1, -1).astype(np.float32), k=k
            )
            cost = ef_search
            return neighbors[0], cost

        # 多入口搜索
        return self._multi_entry_search(query, k, ef_search)

    def _multi_entry_search(self,
                            query: np.ndarray,
                            k: int,
                            ef_search: int) -> Tuple[np.ndarray, int]:
        """
        多入口点搜索

        策略：
        1. 从最高层选择多个入口点
        2. 从每个入口点开始搜索
        3. 合并结果
        """
        # 找到最高层的节点
        highest_layer = max(self.high_layer_nodes.keys()
                            ) if self.high_layer_nodes else 0

        if highest_layer == 0 or len(self.high_layer_nodes[highest_layer]) < self.num_entry_points:
            # 退回到标准搜索
            neighbors, _ = self.index.knn_query(
                query.reshape(1, -1).astype(np.float32), k=k
            )
            return neighbors[0], ef_search

        # 选择入口点（选择与查询最近的几个高层节点）
        high_nodes = np.array(self.high_layer_nodes[highest_layer])
        high_vectors = self.vectors[high_nodes]

        # 计算距离
        distances = np.linalg.norm(high_vectors - query, axis=1)
        entry_indices = np.argsort(distances)[:self.num_entry_points]
        entry_points = high_nodes[entry_indices]

        # 从每个入口点搜索（使用较小的 k）
        all_candidates = {}
        per_entry_k = min(k * 2, ef_search)

        for entry in entry_points:
            try:
                # 从这个入口点开始的邻居
                neighbors, distances = self.index.knn_query(
                    self.vectors[entry].reshape(1, -1),
                    k=per_entry_k
                )

                for neighbor, dist in zip(neighbors[0], distances[0]):
                    # 计算到查询的真实距离
                    true_dist = np.linalg.norm(self.vectors[neighbor] - query)
                    if neighbor not in all_candidates or true_dist < all_candidates[neighbor]:
                        all_candidates[neighbor] = true_dist

            except Exception as e:
                logger.warning(f"多入口搜索出错: {e}")
                continue

        # 排序并返回 top-k
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1])
        result_neighbors = np.array(
            [n for n, _ in sorted_candidates[:k]], dtype=np.int32)

        # 估算成本（多入口会增加成本）
        cost = ef_search * self.num_entry_points // 2

        return result_neighbors, cost

    def search_with_bridges(self,
                            query: np.ndarray,
                            k: int,
                            ef_search: int = 200) -> Tuple[np.ndarray, int]:
        """
        使用桥接边的增强搜索（实验性）

        注意：由于 hnswlib 不允许动态修改图结构，
        这里我们在标准搜索结果基础上利用桥接边扩展候选集
        """
        # 先进行标准搜索
        initial_neighbors, base_cost = self.search(
            query, k=k*2, ef_search=ef_search)

        # 利用桥接边扩展
        extended_candidates = set(initial_neighbors)

        for neighbor in initial_neighbors[:k]:  # 只考虑 top-k
            # 检查这个节点是否有桥接边
            for layer, bridge_map in self.bridge_edges.items():
                if neighbor in bridge_map:
                    extended_candidates.update(bridge_map[neighbor])

        # 重新计算距离并排序
        candidates_list = list(extended_candidates)
        if len(candidates_list) > 0:
            candidate_vectors = self.vectors[candidates_list]
            distances = np.linalg.norm(candidate_vectors - query, axis=1)
            sorted_indices = np.argsort(distances)[:k]
            result = np.array([candidates_list[i]
                              for i in sorted_indices], dtype=np.int32)
        else:
            result = initial_neighbors[:k]

        return result, base_cost

    def _print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("增强版 HNSW 统计:")
        logger.info(f"  总节点数: {len(self.vector_ids)}")
        logger.info(f"  维度: {self.dimension}")
        logger.info(f"  M: {self.M}")
        logger.info(f"  ef_construction: {self.ef_construction}")

        if self.enable_bridges:
            total_bridges = sum(
                len(neighbors)
                for layer_map in self.bridge_edges.values()
                for neighbors in layer_map.values()
            )
            logger.info(f"  桥接边启用: ✓")
            logger.info(f"  桥接边总数: {total_bridges}")
            logger.info(f"  桥接边采样比例: {self.bridge_sample_ratio*100}%")
        else:
            logger.info(f"  桥接边启用: ✗")

        if self.enable_multi_entry:
            logger.info(f"  多入口搜索: ✓ ({self.num_entry_points} 个入口)")
        else:
            logger.info(f"  多入口搜索: ✗")

        logger.info("=" * 60)

    def get_statistics(self) -> dict:
        """获取详细统计信息"""
        total_bridges = sum(
            len(neighbors)
            for layer_map in self.bridge_edges.values()
            for neighbors in layer_map.values()
        )

        return {
            'total_nodes': len(self.vector_ids) if self.vector_ids is not None else 0,
            'dimension': self.dimension,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'enable_bridges': self.enable_bridges,
            'total_bridges': total_bridges,
            'bridge_sample_ratio': self.bridge_sample_ratio,
            'high_layer_count': {
                layer: len(nodes) for layer, nodes in self.high_layer_nodes.items()
            },
            'enable_multi_entry': self.enable_multi_entry,
            'num_entry_points': self.num_entry_points,
        }


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    print("测试增强版 HNSW...")

    # 创建测试数据
    np.random.seed(42)
    X = np.random.randn(1000, 50).astype('float32')
    Q = np.random.randn(10, 50).astype('float32')

    # 构建索引
    hnsw = HNSWWithBridges(
        dimension=50,
        M=16,
        ef_construction=100,
        enable_bridges=True,
        bridge_sample_ratio=0.1,
        enable_multi_entry=True,
        num_entry_points=3
    )

    hnsw.build_index(X)

    # 搜索测试
    neighbors, cost = hnsw.search(Q[0], k=10, ef_search=50)
    print(f"\n搜索结果: {len(neighbors)} 个邻居, 成本: {cost}")
    print(f"邻居: {neighbors[:5]}")

    # 统计
    stats = hnsw.get_statistics()
    print(f"\n统计信息: {stats}")
