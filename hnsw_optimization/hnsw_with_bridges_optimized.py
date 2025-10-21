#!/usr/bin/env python3
"""
优化版 HNSW：使用聚类策略快速添加桥接边
"""

import numpy as np
import hnswlib
from typing import Tuple, List, Optional, Dict
import logging
import time
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


class HNSWWithBridgesOptimized:
    """
    优化版 HNSW：使用聚类策略快速添加桥接边

    改进：
    1. 使用 KMeans 聚类识别不同的簇
    2. 在不同簇之间添加桥接边（而不是耗时的2跳检测）
    3. num_entry_points 可以在搜索时动态调整
    """

    def __init__(self,
                 dimension: int,
                 M: int = 32,
                 ef_construction: int = 200,
                 max_elements: int = 1000000,
                 seed: int = 42,
                 # 桥接边参数
                 enable_bridges: bool = True,
                 n_clusters: int = 10,  # 聚类数量
                 bridges_per_cluster_pair: int = 5,  # 每对簇之间的桥接边数
                 # 多入口搜索参数
                 num_entry_points: int = 4):
        """
        初始化优化版 HNSW

        Args:
            dimension: 向量维度
            M: HNSW 连接数
            ef_construction: 构建时搜索宽度
            max_elements: 最大元素数
            seed: 随机种子
            enable_bridges: 是否启用桥接边
            n_clusters: 聚类数量（用于识别不同的簇）
            bridges_per_cluster_pair: 每对簇之间添加的桥接边数
            num_entry_points: 默认入口点数量（可在搜索时调整）
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.max_elements = max_elements
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 桥接边配置
        self.enable_bridges = enable_bridges
        self.n_clusters = n_clusters
        self.bridges_per_cluster_pair = bridges_per_cluster_pair

        # 多入口搜索配置
        self.num_entry_points = num_entry_points

        # 初始化 hnswlib 索引
        self.index = hnswlib.Index(space='l2', dim=dimension)
        self.index.init_index(
            max_elements=max_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=seed
        )

        # 存储向量数据
        self.vectors: Optional[np.ndarray] = None
        self.vector_ids: Optional[np.ndarray] = None

        # 桥接边映射：{node_id: [bridge_neighbor_ids]}
        self.bridge_edges: Dict[int, List[int]] = defaultdict(list)

        # 高层节点缓存：{layer: [node_ids]}
        self.high_layer_nodes: Dict[int, List[int]] = {}

        # 聚类标签
        self.cluster_labels: Optional[np.ndarray] = None

        self.is_built = False

    def build_index(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """
        构建索引并自动添加桥接边（使用快速聚类策略）

        Args:
            vectors: 向量数据 (N x D)
            ids: 向量ID（可选）
        """
        logger.info(f"构建优化版 HNSW 索引: {len(vectors)} 个向量")

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

        # 3. 使用快速聚类策略添加桥接边
        if self.enable_bridges and len(self.high_layer_nodes) > 0:
            bridge_start = time.time()
            self._add_bridge_edges_fast()
            bridge_time = time.time() - bridge_start
            logger.info(f"桥接边添加完成: {bridge_time:.2f}秒")

        self.is_built = True
        self._print_statistics()

    def _identify_high_layer_nodes(self):
        """识别高层节点（layer >= 1）"""
        logger.info("识别高层节点...")

        n = len(self.vector_ids)
        ml = 1.0 / np.log(self.M)

        # 安全检查：数据规模太大时警告
        if n > 1000000:
            logger.warning(f"⚠️ 数据规模很大 ({n})，建议使用 <= 500K 数据进行测试")

        # 批量处理以提高效率
        for idx, vid in enumerate(self.vector_ids):
            if idx % 100000 == 0 and idx > 0:
                logger.info(f"  处理进度: {idx}/{n} ({100*idx/n:.1f}%)")

            self.rng.seed(self.seed + int(vid))
            level = int(-np.log(self.rng.uniform(0.001, 1.0)) * ml)
            level = min(level, 16)

            if level >= 1:
                if level not in self.high_layer_nodes:
                    self.high_layer_nodes[level] = []
                self.high_layer_nodes[level].append(vid)

        total_high_nodes = sum(len(nodes)
                               for nodes in self.high_layer_nodes.values())
        logger.info(
            f"总高层节点: {total_high_nodes} / {n} ({100*total_high_nodes/n:.2f}%)")

    def _add_bridge_edges_fast(self):
        """
        快速添加桥接边（使用聚类策略）

        策略：
        1. 对高层节点进行 KMeans 聚类
        2. 识别每个簇的中心节点
        3. 在不同簇的中心节点之间添加桥接边
        4. 这避免了昂贵的2跳可达性检测
        """
        logger.info(f"使用聚类策略添加桥接边（{self.n_clusters} 个簇）")

        # 收集所有高层节点
        all_high_nodes = []
        for layer_nodes in self.high_layer_nodes.values():
            all_high_nodes.extend(layer_nodes)

        if len(all_high_nodes) < self.n_clusters:
            logger.warning(
                f"高层节点数({len(all_high_nodes)}) < 簇数({self.n_clusters})，跳过桥接边添加")
            return

        # 安全检查：高层节点太多时随机采样
        if len(all_high_nodes) > 50000:
            logger.warning(
                f"⚠️ 高层节点过多 ({len(all_high_nodes)})，随机采样 50000 个以避免内存溢出")
            sampled_indices = self.rng.choice(
                len(all_high_nodes), 50000, replace=False)
            all_high_nodes = [all_high_nodes[i] for i in sampled_indices]

        all_high_nodes = np.array(all_high_nodes)

        # 对高层节点进行聚类
        logger.info(f"对 {len(all_high_nodes)} 个高层节点进行聚类...")
        high_vectors = self.vectors[all_high_nodes]

        # 使用 MiniBatchKMeans 加速（对大数据集）
        n_clusters_actual = min(self.n_clusters, len(all_high_nodes))
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters_actual,
            random_state=self.seed,
            batch_size=min(1000, len(all_high_nodes)),
            n_init=3
        )
        self.cluster_labels = kmeans.fit_predict(high_vectors)

        logger.info(f"聚类完成: {n_clusters_actual} 个簇")

        # 为每个簇找到代表性节点（最接近簇中心的节点）
        cluster_representatives = {}
        for cluster_id in range(n_clusters_actual):
            cluster_mask = self.cluster_labels == cluster_id
            if np.sum(cluster_mask) == 0:
                continue

            cluster_nodes = all_high_nodes[cluster_mask]
            cluster_vectors = high_vectors[cluster_mask]
            centroid = kmeans.cluster_centers_[cluster_id]

            # 找到最接近中心的节点作为代表
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            representative_indices = np.argsort(
                distances)[:self.bridges_per_cluster_pair]
            representatives = cluster_nodes[representative_indices]

            cluster_representatives[cluster_id] = representatives

        # 在不同簇的代表节点之间添加桥接边
        total_bridges = 0
        for cluster_i in range(n_clusters_actual):
            if cluster_i not in cluster_representatives:
                continue

            for cluster_j in range(cluster_i + 1, n_clusters_actual):
                if cluster_j not in cluster_representatives:
                    continue

                # 连接两个簇的代表节点
                reps_i = cluster_representatives[cluster_i]
                reps_j = cluster_representatives[cluster_j]

                for rep_i in reps_i:
                    for rep_j in reps_j:
                        # 添加双向桥接边
                        self.bridge_edges[rep_i].append(rep_j)
                        self.bridge_edges[rep_j].append(rep_i)
                        total_bridges += 1

        logger.info(f"总共添加 {total_bridges} 条桥接边")

        # 显示簇大小分布
        logger.info("簇大小分布:")
        for cluster_id in sorted(cluster_representatives.keys()):
            cluster_size = np.sum(self.cluster_labels == cluster_id)
            n_reps = len(cluster_representatives[cluster_id])
            logger.info(f"  簇 {cluster_id}: {cluster_size} 个节点, {n_reps} 个代表")

    def search(self,
               query: np.ndarray,
               k: int,
               ef_search: int = 200,
               num_entry_points: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        搜索（可动态调整入口点数量）

        Args:
            query: 查询向量 (D,)
            k: 返回的邻居数量
            ef_search: 搜索宽度
            num_entry_points: 入口点数量（None 则使用默认值）

        Returns:
            (neighbors, cost)
        """
        if not self.is_built:
            raise ValueError("索引未构建")

        self.index.set_ef(ef_search)

        # 使用提供的或默认的入口点数量
        n_entries = num_entry_points if num_entry_points is not None else self.num_entry_points

        if n_entries <= 1 or not self.high_layer_nodes:
            # 标准单入口搜索
            neighbors, _ = self.index.knn_query(
                query.reshape(1, -1).astype(np.float32), k=k
            )
            return neighbors[0], ef_search

        # 多入口搜索
        return self._multi_entry_search(query, k, ef_search, n_entries)

    def _multi_entry_search(self,
                            query: np.ndarray,
                            k: int,
                            ef_search: int,
                            num_entry_points: int) -> Tuple[np.ndarray, int]:
        """
        多入口点搜索
        """
        # 找到最高层的节点
        highest_layer = max(self.high_layer_nodes.keys()
                            ) if self.high_layer_nodes else 0

        if highest_layer == 0 or len(self.high_layer_nodes[highest_layer]) < num_entry_points:
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
        entry_indices = np.argsort(distances)[:num_entry_points]
        entry_points = high_nodes[entry_indices]

        # 从每个入口点搜索
        all_candidates = {}
        per_entry_k = min(k * 2, ef_search)

        for entry in entry_points:
            try:
                neighbors, distances = self.index.knn_query(
                    self.vectors[entry].reshape(1, -1),
                    k=per_entry_k
                )

                for neighbor, dist in zip(neighbors[0], distances[0]):
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

        # 估算成本
        cost = ef_search * num_entry_points // 2

        return result_neighbors, cost

    def set_num_entry_points(self, num_entry_points: int):
        """
        动态调整入口点数量（无需重建索引）

        Args:
            num_entry_points: 新的入口点数量
        """
        self.num_entry_points = num_entry_points
        logger.info(f"入口点数量已更新为: {num_entry_points}")

    def _print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("优化版 HNSW 统计:")
        logger.info(f"  总节点数: {len(self.vector_ids)}")
        logger.info(f"  维度: {self.dimension}")
        logger.info(f"  M: {self.M}")
        logger.info(f"  ef_construction: {self.ef_construction}")

        if self.enable_bridges:
            total_bridges = sum(len(neighbors)
                                for neighbors in self.bridge_edges.values())
            logger.info(f"  桥接边启用: ✓")
            logger.info(f"  桥接边总数: {total_bridges}")
            logger.info(f"  聚类数量: {self.n_clusters}")
            logger.info(f"  每对簇桥接边数: {self.bridges_per_cluster_pair}")
        else:
            logger.info(f"  桥接边启用: ✗")

        logger.info(f"  默认入口点数: {self.num_entry_points}")
        logger.info("=" * 60)

    def get_statistics(self) -> dict:
        """获取详细统计信息"""
        total_bridges = sum(len(neighbors)
                            for neighbors in self.bridge_edges.values())

        return {
            'total_nodes': len(self.vector_ids) if self.vector_ids is not None else 0,
            'dimension': self.dimension,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'enable_bridges': self.enable_bridges,
            'total_bridges': total_bridges,
            'n_clusters': self.n_clusters,
            'high_layer_count': {
                layer: len(nodes) for layer, nodes in self.high_layer_nodes.items()
            },
            'num_entry_points': self.num_entry_points,
        }


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    print("测试优化版 HNSW...")

    np.random.seed(42)
    X = np.random.randn(5000, 50).astype('float32')
    Q = np.random.randn(10, 50).astype('float32')

    # 构建索引（只构建一次）
    hnsw = HNSWWithBridgesOptimized(
        dimension=50,
        M=16,
        ef_construction=100,
        enable_bridges=True,
        n_clusters=5,
        bridges_per_cluster_pair=3
    )

    hnsw.build_index(X)

    # 测试不同的入口点数量（无需重建）
    print("\n测试不同的入口点数量:")
    for num_entry in [1, 2, 4, 8]:
        neighbors, cost = hnsw.search(
            Q[0], k=10, ef_search=50, num_entry_points=num_entry)
        print(f"  num_entry={num_entry}: {len(neighbors)} 个邻居, 成本: {cost}")
