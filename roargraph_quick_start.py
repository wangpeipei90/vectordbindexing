#!/usr/bin/env python3
"""
RoarGraph 快速开始脚本

这个脚本提供了 RoarGraph 的快速使用示例，包括：
1. 基本使用
2. 自定义数据加载
3. 性能评估
4. 结果可视化
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from roargraph_python import RoarGraph, create_sample_data, evaluate_recall


class RoarGraphQuickStart:
    """RoarGraph 快速开始类"""
    
    def __init__(self, dimension: int = 128, metric: str = "cosine"):
        """
        初始化
        
        Args:
            dimension: 向量维度
            metric: 距离度量 ("cosine", "l2", "ip")
        """
        self.dimension = dimension
        self.metric = metric
        self.roargraph = None
        self.base_data = None
        self.query_data = None
        
    def load_custom_data(self, base_data: np.ndarray, query_data: np.ndarray):
        """
        加载自定义数据
        
        Args:
            base_data: 基础数据，形状 (n_base, dimension)
            query_data: 查询数据，形状 (n_query, dimension)
        """
        if base_data.shape[1] != self.dimension:
            raise ValueError(f"Base data dimension {base_data.shape[1]} != expected {self.dimension}")
        if query_data.shape[1] != self.dimension:
            raise ValueError(f"Query data dimension {query_data.shape[1]} != expected {self.dimension}")
        
        self.base_data = base_data.astype(np.float32)
        self.query_data = query_data.astype(np.float32)
        print(f"Loaded custom data: base {base_data.shape}, query {query_data.shape}")
    
    def create_sample_dataset(self, n_base: int = 10000, n_query: int = 1000, 
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建示例数据集
        
        Args:
            n_base: 基础数据数量
            n_query: 查询数据数量
            seed: 随机种子
            
        Returns:
            (base_data, query_data)
        """
        print(f"Creating sample dataset: {n_base} base vectors, {n_query} query vectors")
        base_data, query_data = create_sample_data(n_base, n_query, self.dimension, seed)
        self.base_data = base_data
        self.query_data = query_data
        return base_data, query_data
    
    def build_index(self, M_sq: int = 32, M_pjbp: int = 32, L_pjpq: int = 32) -> float:
        """
        构建索引
        
        Args:
            M_sq: 查询节点的邻居数
            M_pjbp: 投影图的邻居数
            L_pjpq: 搜索队列大小
            
        Returns:
            构建时间（秒）
        """
        if self.base_data is None or self.query_data is None:
            raise ValueError("Please load data first using load_custom_data() or create_sample_dataset()")
        
        print("Building RoarGraph index...")
        start_time = time.time()
        
        self.roargraph = RoarGraph(self.dimension, self.metric)
        self.roargraph.build(
            base_data=self.base_data,
            query_data=self.query_data,
            M_sq=M_sq,
            M_pjbp=M_pjbp,
            L_pjpq=L_pjpq
        )
        
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.2f} seconds")
        
        # 显示统计信息
        stats = self.roargraph.get_statistics()
        print("Index statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return build_time
    
    def search_single(self, query: np.ndarray, k: int = 10) -> Tuple[List[int], List[float], int, int]:
        """
        单个查询搜索
        
        Args:
            query: 查询向量
            k: 返回的最近邻数量
            
        Returns:
            (indices, distances, comparisons, hops)
        """
        if self.roargraph is None:
            raise ValueError("Please build index first using build_index()")
        
        return self.roargraph.search(query, k)
    
    def search_batch(self, queries: np.ndarray, k: int = 10) -> List[Tuple[List[int], List[float], int, int]]:
        """
        批量搜索
        
        Args:
            queries: 查询向量数组
            k: 返回的最近邻数量
            
        Returns:
            搜索结果列表
        """
        if self.roargraph is None:
            raise ValueError("Please build index first using build_index()")
        
        results = []
        for query in queries:
            result = self.roargraph.search(query, k)
            results.append(result)
        
        return results
    
    def evaluate_performance(self, n_test_queries: int = 100, k: int = 10) -> dict:
        """
        评估性能
        
        Args:
            n_test_queries: 测试查询数量
            k: 返回的最近邻数量
            
        Returns:
            性能统计字典
        """
        if self.roargraph is None:
            raise ValueError("Please build index first using build_index()")
        
        # 选择测试查询
        test_queries = self.query_data[:n_test_queries]
        
        print(f"Evaluating performance with {n_test_queries} queries...")
        
        # 执行搜索
        start_time = time.time()
        results = self.search_batch(test_queries, k)
        total_time = time.time() - start_time
        
        # 计算统计信息
        comparisons = [r[2] for r in results]
        hops = [r[3] for r in results]
        
        performance = {
            "total_time": total_time,
            "avg_search_time": total_time / n_test_queries,
            "qps": n_test_queries / total_time,
            "avg_comparisons": np.mean(comparisons),
            "avg_hops": np.mean(hops),
            "max_comparisons": np.max(comparisons),
            "max_hops": np.max(hops),
            "min_comparisons": np.min(comparisons),
            "min_hops": np.min(hops)
        }
        
        print("Performance results:")
        for key, value in performance.items():
            if "time" in key:
                print(f"  {key}: {value*1000:.2f} ms")
            else:
                print(f"  {key}: {value:.2f}")
        
        return performance
    
    def compare_with_brute_force(self, n_test_queries: int = 10, k: int = 10) -> dict:
        """
        与暴力搜索比较
        
        Args:
            n_test_queries: 测试查询数量
            k: 返回的最近邻数量
            
        Returns:
            比较结果字典
        """
        if self.roargraph is None:
            raise ValueError("Please build index first using build_index()")
        
        test_queries = self.query_data[:n_test_queries]
        
        print(f"Comparing with brute force search using {n_test_queries} queries...")
        
        # RoarGraph 搜索
        start_time = time.time()
        roargraph_results = self.search_batch(test_queries, k)
        roargraph_time = time.time() - start_time
        
        # 暴力搜索
        start_time = time.time()
        brute_force_results = []
        for query in test_queries:
            distances = self.roargraph._distance_batch(query, self.base_data)
            indices = np.argsort(distances)[:k]
            brute_force_results.append((indices.tolist(), distances[indices].tolist()))
        brute_force_time = time.time() - start_time
        
        # 计算召回率
        recalls = []
        for i in range(n_test_queries):
            roargraph_indices = set(roargraph_results[i][0])
            brute_force_indices = set(brute_force_results[i][0])
            recall = len(roargraph_indices & brute_force_indices) / k
            recalls.append(recall)
        
        comparison = {
            "roargraph_time": roargraph_time,
            "brute_force_time": brute_force_time,
            "speedup": brute_force_time / roargraph_time,
            "avg_recall": np.mean(recalls),
            "min_recall": np.min(recalls),
            "max_recall": np.max(recalls)
        }
        
        print("Comparison results:")
        print(f"  RoarGraph time: {roargraph_time*1000:.2f} ms")
        print(f"  Brute force time: {brute_force_time*1000:.2f} ms")
        print(f"  Speedup: {comparison['speedup']:.2f}x")
        print(f"  Average recall: {comparison['avg_recall']:.3f}")
        
        return comparison
    
    def visualize_results(self, query_idx: int = 0, k: int = 10, save_path: Optional[str] = None):
        """
        可视化搜索结果
        
        Args:
            query_idx: 查询索引
            k: 返回的最近邻数量
            save_path: 保存路径（可选）
        """
        if self.roargraph is None:
            raise ValueError("Please build index first using build_index()")
        
        # 执行搜索
        query = self.query_data[query_idx]
        indices, distances, comparisons, hops = self.roargraph.search(query, k)
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 距离分布
        ax1.bar(range(len(distances)), distances)
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Distance')
        ax1.set_title(f'Search Results for Query {query_idx}')
        ax1.grid(True, alpha=0.3)
        
        # 性能统计
        stats_text = f"""
        Query Index: {query_idx}
        Comparisons: {comparisons}
        Hops: {hops}
        Search Time: {time.time()*1000:.2f} ms
        """
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Search Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def quick_demo():
    """快速演示"""
    print("=== RoarGraph 快速演示 ===")
    
    # 创建快速开始实例
    rg = RoarGraphQuickStart(dimension=128, metric="cosine")
    
    # 创建示例数据
    rg.create_sample_dataset(n_base=5000, n_query=500)
    
    # 构建索引
    build_time = rg.build_index(M_sq=32, M_pjbp=32, L_pjpq=32)
    
    # 评估性能
    performance = rg.evaluate_performance(n_test_queries=100, k=10)
    
    # 与暴力搜索比较
    comparison = rg.compare_with_brute_force(n_test_queries=10, k=10)
    
    # 单个查询示例
    print("\n=== 单个查询示例 ===")
    query = rg.query_data[0]
    indices, distances, comparisons, hops = rg.search_single(query, k=5)
    print(f"Query: {query[:5]}...")
    print(f"Results: {indices}")
    print(f"Distances: {distances}")
    print(f"Comparisons: {comparisons}, Hops: {hops}")
    
    return rg, performance, comparison


def parameter_sweep_demo():
    """参数扫描演示"""
    print("\n=== 参数扫描演示 ===")
    
    # 创建数据
    rg = RoarGraphQuickStart(dimension=128, metric="cosine")
    rg.create_sample_dataset(n_base=3000, n_query=300)
    
    # 测试不同参数
    param_combinations = [
        {"M_sq": 16, "M_pjbp": 16, "L_pjpq": 16},
        {"M_sq": 32, "M_pjbp": 32, "L_pjpq": 32},
        {"M_sq": 64, "M_pjbp": 64, "L_pjpq": 64},
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n测试参数组合 {i+1}: {params}")
        
        # 构建索引
        build_time = rg.build_index(**params)
        
        # 评估性能
        performance = rg.evaluate_performance(n_test_queries=50, k=10)
        
        results.append({
            "params": params,
            "build_time": build_time,
            "performance": performance
        })
    
    # 显示结果比较
    print("\n=== 参数比较结果 ===")
    print("参数组合\t构建时间(s)\tQPS\t\t平均比较次数\t平均跳数")
    print("-" * 60)
    
    for result in results:
        params = result["params"]
        perf = result["performance"]
        print(f"{params['M_sq']}-{params['M_pjbp']}-{params['L_pjpq']}\t"
              f"{result['build_time']:.2f}\t\t{perf['qps']:.0f}\t\t"
              f"{perf['avg_comparisons']:.1f}\t\t{perf['avg_hops']:.1f}")
    
    return results


if __name__ == "__main__":
    print("RoarGraph 快速开始脚本")
    print("=" * 50)
    
    try:
        # 快速演示
        rg, performance, comparison = quick_demo()
        
        # 参数扫描演示
        param_results = parameter_sweep_demo()
        
        print("\n" + "=" * 50)
        print("快速开始演示完成！")
        print("\n使用说明:")
        print("1. 创建 RoarGraphQuickStart 实例")
        print("2. 加载数据（自定义或示例数据）")
        print("3. 构建索引")
        print("4. 执行搜索或性能评估")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
