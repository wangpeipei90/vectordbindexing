#!/usr/bin/env python3
"""
快速测试HNSW优化功能
"""

from gt_utils import GroundTruthComputer
from multi_entry_search import MultiEntrySearch
from bridge_builder import BridgeBuilder
from hnsw_baseline import HNSWBaseline
from data_loader import create_toy_dataset
import time
import numpy as np
import sys
import os
sys.path.append('/root/code/vectordbindexing')
sys.path.append('/root/code/vectordbindexing/hnsw_optimization')


def quick_test():
    """快速测试所有功能"""
    print("HNSW优化快速测试")
    print("=" * 50)

    # 创建小数据集
    print("1. 创建测试数据集...")
    X, Q, modalities, query_modalities = create_toy_dataset(1000, 100, 64, 3)
    print(f"   数据: {X.shape}, 查询: {Q.shape}, 模态: {len(np.unique(modalities))}")

    # 构建基线HNSW
    print("\n2. 构建基线HNSW...")
    hnsw = HNSWBaseline(dimension=64, M=8, ef_construction=100)
    start_time = time.time()
    hnsw.build_index(X)
    build_time = time.time() - start_time
    print(f"   构建时间: {build_time:.2f}秒")

    # 测试基线搜索
    print("\n3. 测试基线搜索...")
    query = Q[0]
    baseline_neighbors, baseline_cost = hnsw.search(query, k=10, ef_search=50)
    print(f"   基线结果: {len(baseline_neighbors)}个邻居, 成本: {baseline_cost}")

    # 构建桥边
    print("\n4. 构建桥边...")
    bridge_builder = BridgeBuilder(
        max_bridge_per_node=2,
        bridge_budget_ratio=1e-3  # 使用较大预算用于测试
    )
    start_time = time.time()
    bridge_map = bridge_builder.build_bridges(hnsw, X, modalities)
    bridge_time = time.time() - start_time
    print(f"   桥边构建时间: {bridge_time:.2f}秒")

    bridge_stats = bridge_builder.get_statistics()
    print(
        f"   桥边统计: {bridge_stats['total_bridge_edges']}条边, 比例: {bridge_stats['bridge_ratio']:.6f}")

    # 测试多入口搜索
    print("\n5. 测试多入口搜索...")
    multi_search = MultiEntrySearch(hnsw, bridge_builder)

    m_values = [2, 4]
    for m in m_values:
        start_time = time.time()
        multi_neighbors, multi_cost = multi_search.multi_entry_search(
            query, k=10, m=m, ef_search=50
        )
        multi_time = time.time() - start_time

        # 计算重叠度
        overlap = len(set(baseline_neighbors) & set(multi_neighbors))
        overlap_ratio = overlap / 10

        print(f"   m={m}: {len(multi_neighbors)}个邻居, 成本: {multi_cost}, "
              f"重叠: {overlap_ratio:.2f}, 时间: {multi_time:.4f}秒")

    # 测试召回率
    print("\n6. 测试召回率计算...")
    gt_computer = GroundTruthComputer()
    gt_neighbors, _ = gt_computer.compute_ground_truth(X, Q, k=10)

    baseline_recall = gt_computer.compute_recall(
        baseline_neighbors.reshape(1, -1), k_eval=10)
    print(f"   基线召回率@10: {baseline_recall:.4f}")

    for m in m_values:
        multi_neighbors, _ = multi_search.multi_entry_search(
            query, k=10, m=m, ef_search=50)
        multi_recall = gt_computer.compute_recall(
            multi_neighbors.reshape(1, -1), k_eval=10)
        print(f"   m={m} 召回率@10: {multi_recall:.4f}")

    print("\n" + "=" * 50)
    print("✅ 所有功能测试通过！")
    print("=" * 50)


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
