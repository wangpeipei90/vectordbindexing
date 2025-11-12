#!/usr/bin/env python3
"""
优化版 HNSW：使用C++核心实现的2层图结构
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging
import time

# 导入C++核心模块
import hnsw_core
HNSW_CORE_AVAILABLE = True

logger = logging.getLogger(__name__)


class HNSWWithBridgesOptimized:
    """
    优化版 HNSW：使用C++核心实现的2层图结构
    
    特性：
    1. C++核心实现（高性能）
    2. 2层图结构（Layer0全部节点，Layer1~3-6%节点，符合标准HNSW）
    3. 固定出度（M0和M1=M0/2）
    4. Layer1搜索到稳定
    5. Layer0多入口并行搜索
    6. 返回avg_visited, mean_latency, recall@10
    """

    def __init__(self,
                 dimension: int,
                 M: int = 32,
                 ef_construction: int = 200,
                 max_elements: int = 1000000,
                 distance_type: str = "l2",
                 seed: int = 42,
                 # 多入口搜索参数
                 num_entry_points: int = 4):
        """
        初始化优化版 HNSW

        Args:
            dimension: 向量维度
            M: HNSW 连接数（第0层出度M0=M，第1层出度M1=M/2）
            ef_construction: 构建时搜索宽度
            max_elements: 最大元素数
            seed: 随机种子
            num_entry_points: 默认入口点数量（可在搜索时调整）
        """
        if not HNSW_CORE_AVAILABLE:
            raise ImportError("hnsw_core C++ module is required but not available")
        
        self.dimension = dimension
        self.M = M
        self.M0 = M  # 第0层出度
        self.M1 = M // 2  # 第1层出度
        self.ef_construction = ef_construction
        self.max_elements = max_elements
        self.distance_type = distance_type
        self.seed = seed

        # 多入口搜索配置
        self.num_entry_points = num_entry_points

        # 初始化 C++ 核心索引
        self.index = hnsw_core.HNSW(
            dimension=dimension,
            M0=self.M0,
            ef_construction=ef_construction,
            max_elements=max_elements,
            distance_type=distance_type,
            seed=seed
        )

        # 存储向量数据（用于计算recall等）
        self.vectors: Optional[np.ndarray] = None
        self.vector_ids: Optional[np.ndarray] = None

        self.is_built = False

    def build_index(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None, 
                    rebuild_graph_from: str = "", load_from_roargraph: str = ""):
        """
        构建索引

        Args:
            vectors: 向量数据 (N x D)
            ids: 向量ID（可选，当前版本使用0到N-1）
            rebuild_graph_from: 图结构文件路径（可选，txt格式）
                - 如果为空：正常构建索引
                - 如果不为空：从txt文件加载图结构
            load_from_roargraph: RoarGraph index文件路径（可选）
                - 如果不为空：从RoarGraph index文件加载第0层
        """
        # 保存向量数据
        self.vectors = vectors.astype(np.float32)
        if ids is None:
            ids = np.arange(len(vectors))
        self.vector_ids = ids.astype(np.int32)

        layer0_loaded = False
        
        if load_from_roargraph:
            # 从RoarGraph index文件加载第0层
            logger.info(f"从RoarGraph文件加载第0层: {load_from_roargraph}")
            start_time = time.time()
            self.load_layer0_from_roargraph(load_from_roargraph)
            load_time = time.time() - start_time
            logger.info(f"第0层加载完成: {load_time:.2f}秒")
            layer0_loaded = True
        elif rebuild_graph_from:
            # 从txt文件加载图结构
            logger.info(f"从txt文件加载图结构: {rebuild_graph_from}")
            start_time = time.time()
            self.load_layer0(rebuild_graph_from)
            load_time = time.time() - start_time
            logger.info(f"图结构加载完成: {load_time:.2f}秒")
            layer0_loaded = True

        if layer0_loaded:
            # 第0层已加载，只构建第1层
            logger.info("开始构建第1层...")
            start_time = time.time()
            self.index.build_layer1_only(self.vectors)
            build_time = time.time() - start_time
            logger.info(f"第1层构建完成: {build_time:.2f}秒")
        else:
            # 正常构建索引（第0层和第1层）
            logger.info(f"构建优化版 HNSW 索引: {len(vectors)} 个向量")
            start_time = time.time()
            self.index.build(self.vectors)
            build_time = time.time() - start_time
            logger.info(f"索引构建完成: {build_time:.2f}秒")

        self.is_built = True
        self._print_statistics()


    def search(self,
               query: np.ndarray,
               k: int,
               ef_search: int = 200,
               num_entry_points: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        搜索（返回邻居和统计信息）

        Args:
            query: 查询向量 (D,)
            k: 返回的邻居数量
            ef_search: 搜索宽度
            num_entry_points: 入口点数量（None 则使用默认值）

        Returns:
            (neighbors, stats)
            - neighbors: 邻居ID数组
            - stats: 统计信息字典，包含:
                - visited_count: 访问的节点数 (avg_visited)
                - latency_us: 搜索延迟（微秒）(mean_latency)
                - layer1_visited: 第1层访问的节点数
                - layer0_visited: 第0层访问的节点数
        """
        if not self.is_built:
            raise ValueError("索引未构建")

        # 使用提供的或默认的入口点数量
        n_entries = num_entry_points if num_entry_points is not None else self.num_entry_points

        # 调用C++搜索
        result = self.index.search(
            query.astype(np.float32),
            k=k,
            ef_search=ef_search,
            num_entry_points=n_entries
        )

        # 返回邻居和统计信息
        neighbors = result['neighbors']
        stats = {
            'visited_count': result['visited_count'],
            'latency_us': result['latency_us'],
            'layer1_visited': result['layer1_visited'],
            'layer0_visited': result['layer0_visited'],
        }

        return neighbors, stats

    def batch_search(self,
                     queries: np.ndarray,
                     k: int,
                     ef_search: int = 200,
                     num_entry_points: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        批量搜索

        Args:
            queries: 查询向量 (N x D)
            k: 返回的邻居数量
            ef_search: 搜索宽度
            num_entry_points: 入口点数量

        Returns:
            (all_neighbors, aggregated_stats)
            - all_neighbors: 所有查询的邻居 (N x k)
            - aggregated_stats: 聚合统计信息
        """
        if not self.is_built:
            raise ValueError("索引未构建")

        n_entries = num_entry_points if num_entry_points is not None else self.num_entry_points

        # 调用C++批量搜索
        results = self.index.batch_search(
            queries.astype(np.float32),
            k=k,
            ef_search=ef_search,
            num_entry_points=n_entries
        )

        # 收集所有邻居
        all_neighbors = np.array([r['neighbors'] for r in results])

        # 聚合统计信息
        visited_counts = [r['visited_count'] for r in results]
        latencies = [r['latency_us'] for r in results]
        layer1_visited = [r['layer1_visited'] for r in results]
        layer0_visited = [r['layer0_visited'] for r in results]

        aggregated_stats = {
            'avg_visited': np.mean(visited_counts),
            'std_visited': np.std(visited_counts),
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'avg_layer1_visited': np.mean(layer1_visited),
            'avg_layer0_visited': np.mean(layer0_visited),
            'all_visited_counts': visited_counts,
            'all_latencies': latencies,
        }

        return all_neighbors, aggregated_stats

    def compute_recall(self,
                       results: np.ndarray,
                       ground_truth: np.ndarray,
                       k: int = 10) -> float:
        """
        计算recall@k

        Args:
            results: 搜索结果 (k,) 或 (N x k)
            ground_truth: ground truth (k,) 或 (N x k)
            k: 计算recall的k值

        Returns:
            recall值
        """
        if results.ndim == 1:
            # 单个查询
            return hnsw_core.HNSW.compute_recall(results, ground_truth, k)
        else:
            # 多个查询
            recalls = []
            for res, gt in zip(results, ground_truth):
                recall = hnsw_core.HNSW.compute_recall(res, gt, k)
                recalls.append(recall)
            return np.mean(recalls)

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
        num_nodes = self.index.get_num_nodes()
        num_layer1 = self.index.get_num_layer1_nodes()
        
        logger.info("=" * 60)
        logger.info("优化版 HNSW 统计:")
        logger.info(f"  总节点数: {num_nodes}")
        logger.info(f"  维度: {self.dimension}")
        logger.info(f"  M0 (第0层出度): {self.M0}")
        logger.info(f"  M1 (第1层出度): {self.M1}")
        logger.info(f"  ef_construction: {self.ef_construction}")
        logger.info(f"  第1层节点数: {num_layer1} ({100*num_layer1/num_nodes:.1f}%)")
        logger.info(f"  理论第1层比例: ~{100/self.M0:.1f}% (P(L>=1)=1/M0)")
        logger.info(f"  默认入口点数: {self.num_entry_points}")
        logger.info(f"  实现方式: C++核心")
        logger.info("=" * 60)

    def get_statistics(self) -> dict:
        """获取详细统计信息"""
        num_nodes = self.index.get_num_nodes()
        num_layer1 = self.index.get_num_layer1_nodes()

        return {
            'total_nodes': num_nodes,
            'dimension': self.dimension,
            'M0': self.M0,
            'M1': self.M1,
            'ef_construction': self.ef_construction,
            'layer1_nodes': num_layer1,
            'layer1_ratio': num_layer1 / num_nodes if num_nodes > 0 else 0,
            'num_entry_points': self.num_entry_points,
            'implementation': 'C++',
        }

    def save_layer0(self, filepath: str):
        """
        保存第0层图结构到文件

        格式：每行一个节点
        id \t vector \t neighbor1,neighbor2,...

        Args:
            filepath: 保存路径
        """
        if not self.is_built:
            raise ValueError("索引未构建，无法保存")

        logger.info(f"保存第0层图结构到: {filepath}")
        start_time = time.time()

        with open(filepath, 'w') as f:
            num_nodes = self.index.get_num_nodes()
            
            for node_id in range(num_nodes):
                # 获取向量
                vector = self.vectors[node_id]
                
                # 获取第0层邻居
                neighbors = self.index.get_layer0_neighbors(node_id)
                
                # 格式化输出
                vector_str = ','.join(map(str, vector))
                neighbors_str = ','.join(map(str, neighbors))
                
                f.write(f"{node_id}\t{vector_str}\t{neighbors_str}\n")

        save_time = time.time() - start_time
        logger.info(f"第0层图结构保存完成: {save_time:.2f}秒")

    def load_layer0(self, filepath: str):
        """
        从文件加载第0层图结构

        Args:
            filepath: 文件路径
        """
        logger.info(f"加载第0层图结构: {filepath}")
        start_time = time.time()

        # 准备数据结构
        node_vectors = []
        node_neighbors = []

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                
                node_id = int(parts[0])
                vector = np.array([float(x) for x in parts[1].split(',')])
                neighbors = [int(x) for x in parts[2].split(',') if x]
                
                node_vectors.append(vector)
                node_neighbors.append(neighbors)

        # 调用C++加载方法
        node_vectors_array = np.array(node_vectors, dtype=np.float32)
        self.index.load_layer0(node_vectors_array, node_neighbors)

        load_time = time.time() - start_time
        logger.info(f"第0层图结构加载完成: {load_time:.2f}秒，共{len(node_vectors)}个节点")

    def load_layer0_from_roargraph(self, filepath: str):
        """
        从RoarGraph index文件加载第0层图结构
        
        注意：RoarGraph index 只存储图结构（邻接表），不存储向量数据
        向量数据必须通过 build_index() 的 vectors 参数提供
        
        Args:
            filepath: RoarGraph index文件路径
        """
        import struct
        
        logger.info(f"从RoarGraph文件加载第0层图结构: {filepath}")
        start_time = time.time()
        
        if self.vectors is None:
            raise ValueError("必须先通过 build_index() 提供向量数据才能加载图结构")
        
        node_neighbors = []
        
        with open(filepath, 'rb') as f:
            # RoarGraph 格式（基于实际文件分析）:
            # - 字节 0-3:  元数据（跳过）
            # - 字节 4-7:  节点总数 ✅
            # - 字节 8开始: 每个节点的邻居列表
            #   - 4字节: 邻居数量
            #   - N*4字节: N个邻居ID
            
            # 读取头部
            metadata = struct.unpack('I', f.read(4))[0]  # 字节0-3: 元数据（跳过）
            num_nodes_in_file = struct.unpack('I', f.read(4))[0]  # 字节4-7: 节点总数
            
            logger.info(f"RoarGraph文件信息: 节点数={num_nodes_in_file:,}, 元数据={metadata:,}")
            
            if num_nodes_in_file != len(self.vectors):
                logger.warning(f"节点数不匹配: 文件中 {num_nodes_in_file}, 向量数据 {len(self.vectors)}")
                # 使用较小的值
                num_nodes = min(num_nodes_in_file, len(self.vectors))
            else:
                num_nodes = num_nodes_in_file
            
            # 读取每个节点的邻居列表
            logger.info(f"开始读取 {num_nodes} 个节点的邻居列表...")
            invalid_neighbor_count = 0
            
            for node_id in range(num_nodes):
                # 读取邻居数量
                num_neighbors_bytes = f.read(4)
                if len(num_neighbors_bytes) < 4:
                    logger.warning(f"节点 {node_id} 读取邻居数量失败，使用空邻居列表")
                    node_neighbors.append([])
                    continue
                    
                num_neighbors = struct.unpack('I', num_neighbors_bytes)[0]
                
                # 读取邻居ID列表
                if num_neighbors > 0:
                    neighbors_bytes = f.read(num_neighbors * 4)
                    if len(neighbors_bytes) < num_neighbors * 4:
                        logger.warning(f"节点 {node_id} 邻居数据不完整")
                        neighbors = []
                    else:
                        raw_neighbors = list(struct.unpack(f'{num_neighbors}I', neighbors_bytes))
                        
                        # 🔧 修复：过滤掉超出范围的邻居ID
                        neighbors = []
                        for nid in raw_neighbors:
                            if nid < num_nodes:
                                neighbors.append(nid)
                            else:
                                invalid_neighbor_count += 1
                else:
                    neighbors = []
                
                node_neighbors.append(neighbors)
                
                if (node_id + 1) % 100000 == 0:
                    logger.info(f"  进度: {node_id + 1}/{num_nodes}")
            
            if invalid_neighbor_count > 0:
                logger.warning(f"过滤掉 {invalid_neighbor_count} 个超出范围的邻居ID")
        
        # 调用C++加载方法（使用已有的向量数据）
        logger.info(f"加载完成，共 {len(node_neighbors)} 个节点的邻接表")
        self.index.load_layer0(self.vectors, node_neighbors)
        
        load_time = time.time() - start_time
        logger.info(f"第0层从RoarGraph加载完成: {load_time:.2f}秒")


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    print("测试优化版 HNSW (C++核心)...")

    np.random.seed(42)
    X = np.random.randn(5000, 50).astype('float32')
    Q = np.random.randn(10, 50).astype('float32')

    # 构建索引（只构建一次）
    hnsw = HNSWWithBridgesOptimized(
        dimension=50,
        M=16,
        ef_construction=100,
        num_entry_points=4
    )

    hnsw.build_index(X)

    # 测试不同的入口点数量（无需重建）
    print("\n测试不同的入口点数量:")
    print(f"{'num_entry':<12} {'avg_visited':<15} {'mean_latency(μs)':<18} {'neighbors':<20}")
    print("-" * 70)
    
    for num_entry in [1, 2, 4, 8]:
        neighbors, stats = hnsw.search(
            Q[0], k=10, ef_search=50, num_entry_points=num_entry)
        print(f"{num_entry:<12} {stats['visited_count']:<15} "
              f"{stats['latency_us']:<18.2f} {len(neighbors):<20}")

    # 批量搜索
    print("\n批量搜索测试:")
    all_neighbors, agg_stats = hnsw.batch_search(
        Q, k=10, ef_search=100, num_entry_points=4
    )
    
    print(f"  查询数: {len(Q)}")
    print(f"  avg_visited: {agg_stats['avg_visited']:.1f} ± {agg_stats['std_visited']:.1f}")
    print(f"  mean_latency: {agg_stats['mean_latency']:.2f} ± {agg_stats['std_latency']:.2f} μs")
    print(f"  avg_layer1_visited: {agg_stats['avg_layer1_visited']:.1f}")
    print(f"  avg_layer0_visited: {agg_stats['avg_layer0_visited']:.1f}")

    print("\n✅ 测试完成！")
