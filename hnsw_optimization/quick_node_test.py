#!/usr/bin/env python3
"""
快速测试节点访问统计功能
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace/vectordbindexing')

from hnsw_with_bridges_optimized import HNSWWithBridgesOptimized
from io_utils import read_fbin, read_ibin

print("=" * 60)
print("快速测试：节点访问统计功能")
print("=" * 60)

# 加载数据
X = read_fbin('/workspace/vectordbindexing/Text2Image/base.10M.fbin')
Q = read_fbin('/workspace/vectordbindexing/Text2Image/query.10k.fbin')
gt_neighbors = read_ibin('/workspace/vectordbindexing/Text2Image/gt.10k.ibin')

print(f"数据集: {X.shape[0]:,} 向量")
print(f"查询集: {Q.shape[0]:,} 查询")

# 创建索引（使用IP距离）
print("\n创建索引（从 RoarGraph 加载）...")
hnsw = HNSWWithBridgesOptimized(
    dimension=200, M=64, ef_construction=200,
    distance_type='ip'
)
hnsw.build_index(X, load_from_roargraph='/workspace/vectordbindexing/Text2Image/t2i_10M_roar.index')

# 测试不同配置的节点访问统计
print("\n" + "=" * 60)
print("测试不同配置的节点访问统计")
print("=" * 60)

entry_point_configs = [2, 4, 8]
ef_search_values = [100, 200, 400]

for num_entries in entry_point_configs:
    print(f"\n🔵 测试 {num_entries} 个入口点:")
    
    for ef_search in ef_search_values:
        # 测试单个查询
        neighbors, stats = hnsw.search(
            Q[0], k=10, ef_search=ef_search, num_entry_points=num_entries)
        recall = hnsw.compute_recall(neighbors[:10].reshape(
            1, -1), gt_neighbors[0:1, :10], k=10)
        
        print(f"  ef={ef_search}: Recall@10={recall:.4f}, 延迟={stats['latency_us']/1000:.2f}ms, "
              f"visited=L1:{stats['layer1_visited']}+L0:{stats['layer0_visited']}={stats['visited_count']}")
        
        # 详细节点访问统计（只在第一个ef_search时显示）
        if ef_search == ef_search_values[0]:
            total_visited = stats['layer1_visited'] + stats['layer0_visited']
            l1_pct = (stats['layer1_visited'] / total_visited * 100) if total_visited > 0 else 0
            l0_pct = (stats['layer0_visited'] / total_visited * 100) if total_visited > 0 else 0
            
            print(f"    └─ Layer1 访问节点: {stats['layer1_visited']} ({l1_pct:.1f}%)")
            print(f"    └─ Layer0 访问节点: {stats['layer0_visited']} ({l0_pct:.1f}%)")
            print(f"    └─ 总访问节点: {total_visited}")

print(f"\n" + "=" * 60)
print("✅ 节点访问统计功能测试完成")
print("=" * 60)
