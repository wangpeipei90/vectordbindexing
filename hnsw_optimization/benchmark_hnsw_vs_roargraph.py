#!/usr/bin/env python3
"""
HNSW vs RoarGraph 性能对比测试脚本

测试不同并行入口点数量（1-8）下的性能表现：
1. 相同latency下不同并行入口的recall准确率对比
2. 相同recall下不同并行入口数的latency对比
"""

from gt_utils import GroundTruthComputer
from hnsw_with_bridges_optimized import HNSWWithBridgesOptimized
from roargraph_python import RoarGraph
from io_utils import read_fbin, read_ibin
import seaborn as sns
import matplotlib.pyplot as plt
import time
import json
import numpy as np
import sys
import os
sys.path.append('/root/code/vectordbindexing')
sys.path.append('/root/code/vectordbindexing/hnsw_optimization')


print("=" * 80)
print("HNSW vs RoarGraph 性能对比测试")
print("=" * 80)

# ==================== 1. 数据加载 ====================
print("\n1. 加载数据集...")
file_path = "/root/code/vectordbindexing/Text2Image/base.1M.fbin"
query_path = "/root/code/vectordbindexing/Text2Image/query.public.100K.fbin"
faiss_top100_path = "/root/code/vectordbindexing/faiss_top100_results.json"

data_vector = read_fbin(file_path)
query_vector = read_fbin(query_path)

n_train = 500000  # 使用 500K 数据
n_query = 100000  # 使用 100K 查询

X = data_vector[:n_train]
Q = query_vector[:n_query]

print(f"   训练数据: {X.shape}")
print(f"   查询数据: {Q.shape}")

# 加载 ground truth
print("\n   加载 Ground Truth...")
gt_eval = GroundTruthComputer()
gt_neighbors = gt_eval.load_ground_truth_from_json(
    faiss_top100_path, n_queries=n_query, k=100)
gt_eval.gt_neighbors = gt_neighbors
print(f"   Ground truth: {gt_neighbors.shape}")

# ==================== 2. 构建 HNSW 索引 ====================
print("\n2. 构建 HNSW 索引...")
hnsw = HNSWWithBridgesOptimized(
    dimension=X.shape[1],
    M=32,
    ef_construction=200,
    enable_bridges=True,
    n_clusters=10,
    bridges_per_cluster_pair=5,
    num_entry_points=4
)

start_time = time.time()
hnsw.build_index(X)
hnsw_build_time = time.time() - start_time

print(f"   ✓ HNSW 构建完成: {hnsw_build_time:.2f}s")
stats = hnsw.get_statistics()
print(f"   总节点数: {stats['total_nodes']}")
print(f"   桥接边数: {stats['total_bridges']}")

# ==================== 3. 构建 RoarGraph 索引 ====================
print("\n3. 构建 RoarGraph 索引...")
# 关键修复：使用L2距离与HNSW保持一致
roargraph = RoarGraph(dimension=X.shape[1], metric="l2")

start_time = time.time()
roargraph.build(
    base_data=X,
    query_data=Q[:10000],  # 使用部分查询数据构建索引
    M_sq=32,
    M_pjbp=32,
    L_pjpq=200
)
roargraph_build_time = time.time() - start_time

print(f"   ✓ RoarGraph 构建完成: {roargraph_build_time:.2f}s")
print(f"   Base数据: {roargraph.n_base}")
print(f"   Query数据: {roargraph.n_query}")

# ==================== 4. 性能测试 ====================
print("\n4. 性能测试...")
n_test_queries = 100
entry_point_values = [1, 2, 3, 4, 5, 6, 7, 8]
hnsw_results = {}
roargraph_results = {}

# 测试 HNSW（不同入口点数）
print("\n   测试 HNSW（不同并行入口数）:")
for num_entries in entry_point_values:
    print(f"      {num_entries} 个入口点: ", end="", flush=True)

    all_neighbors = []
    search_times = []

    for j in range(n_test_queries):
        start = time.time()
        neighbors, _ = hnsw.search(
            Q[j], k=100, ef_search=200, num_entry_points=num_entries
        )
        search_times.append(time.time() - start)
        all_neighbors.append(neighbors)

    all_neighbors = np.array(all_neighbors)
    recall_10 = gt_eval.compute_recall(all_neighbors, k_eval=10)
    recall_100 = gt_eval.compute_recall(all_neighbors, k_eval=100)
    avg_time = np.mean(search_times) * 1000
    std_time = np.std(search_times) * 1000

    hnsw_results[num_entries] = {
        'recall_10': recall_10,
        'recall_100': recall_100,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time
    }

    print(f"Recall@10={recall_10:.4f}, 时间={avg_time:.3f}±{std_time:.3f}ms")

# 测试 RoarGraph
print("\n   测试 RoarGraph:")
print(f"      标准配置: ", end="", flush=True)

roargraph_neighbors = []
roargraph_times = []

for j in range(n_test_queries):
    start = time.time()
    indices, distances, comparisons, hops = roargraph.search(Q[j], k=100)
    roargraph_times.append(time.time() - start)
    roargraph_neighbors.append(indices)

roargraph_neighbors = np.array(roargraph_neighbors)
roargraph_recall_10 = gt_eval.compute_recall(roargraph_neighbors, k_eval=10)
roargraph_recall_100 = gt_eval.compute_recall(roargraph_neighbors, k_eval=100)
roargraph_avg_time = np.mean(roargraph_times) * 1000
roargraph_std_time = np.std(roargraph_times) * 1000

roargraph_results = {
    'recall_10': roargraph_recall_10,
    'recall_100': roargraph_recall_100,
    'avg_time_ms': roargraph_avg_time,
    'std_time_ms': roargraph_std_time
}

print(
    f"Recall@10={roargraph_recall_10:.4f}, 时间={roargraph_avg_time:.3f}±{roargraph_std_time:.3f}ms")

# ==================== 5. 结果对比 ====================
print("\n" + "=" * 80)
print("性能对比总结")
print("=" * 80)

print(f"\n{'方法':<25} {'Recall@10':<12} {'Recall@100':<12} {'延迟(ms)':<15} {'标准差':<12}")
print("-" * 80)

for num_entries in entry_point_values:
    result = hnsw_results[num_entries]
    method_name = f"HNSW ({num_entries} entries)"
    print(f"{method_name:<25} {result['recall_10']:<12.4f} {result['recall_100']:<12.4f} "
          f"{result['avg_time_ms']:<15.3f} ±{result['std_time_ms']:<10.3f}")

print(f"{'RoarGraph':<25} {roargraph_recall_10:<12.4f} {roargraph_recall_100:<12.4f} "
      f"{roargraph_avg_time:<15.3f} ±{roargraph_std_time:<10.3f}")

# ==================== 6. 生成对比图表 ====================
print("\n5. 生成对比图表...")

x_values = entry_point_values
hnsw_recall_10 = [hnsw_results[n]['recall_10'] for n in entry_point_values]
hnsw_recall_100 = [hnsw_results[n]['recall_100'] for n in entry_point_values]
hnsw_latency = [hnsw_results[n]['avg_time_ms'] for n in entry_point_values]

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 图1: Recall@10 对比
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(x_values, hnsw_recall_10, marker='o', linewidth=3, markersize=10,
         label='HNSW (Multi-Entry)', color='#2E86AB', linestyle='-')
ax1.axhline(y=roargraph_recall_10, color='#E63946', linestyle='--', linewidth=3,
            label=f'RoarGraph (Recall@10={roargraph_recall_10:.4f})', alpha=0.8)
ax1.set_xlabel('Number of Parallel Entry Points (HNSW)',
               fontsize=14, fontweight='bold')
ax1.set_ylabel('Recall@10', fontsize=14, fontweight='bold')
ax1.set_title('Recall@10: HNSW (Different Entry Points) vs RoarGraph',
              fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=13, loc='best', frameon=True, shadow=True)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.set_xticks(x_values)
ax1.set_ylim([0, 1.05])
for i, (x, y) in enumerate(zip(x_values, hnsw_recall_10)):
    ax1.text(x, y + 0.03, f'{y:.3f}', ha='center', fontsize=10,
             fontweight='bold', color='#2E86AB')
plt.tight_layout()
plt.savefig('/root/code/vectordbindexing/hnsw_optimization/comparison_recall10_hnsw_vs_roargraph.png',
            dpi=300, bbox_inches='tight')
print("   ✓ 图1已保存: comparison_recall10_hnsw_vs_roargraph.png")
plt.close()

# 图2: Latency 对比
fig2, ax2 = plt.subplots(figsize=(12, 7))
ax2.plot(x_values, hnsw_latency, marker='D', linewidth=3, markersize=10,
         label='HNSW (Multi-Entry)', color='#F18F01', linestyle='-')
ax2.axhline(y=roargraph_avg_time, color='#E63946', linestyle='--', linewidth=3,
            label=f'RoarGraph (Latency={roargraph_avg_time:.3f}ms)', alpha=0.8)
ax2.set_xlabel('Number of Parallel Entry Points (HNSW)',
               fontsize=14, fontweight='bold')
ax2.set_ylabel('Average Query Latency (ms)', fontsize=14, fontweight='bold')
ax2.set_title('Query Latency: HNSW (Different Entry Points) vs RoarGraph',
              fontsize=16, fontweight='bold', pad=20)
ax2.legend(fontsize=13, loc='best', frameon=True, shadow=True)
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.set_xticks(x_values)
for i, (x, y) in enumerate(zip(x_values, hnsw_latency)):
    ax2.text(x, y + max(hnsw_latency)*0.03, f'{y:.2f}', ha='center',
             fontsize=10, fontweight='bold', color='#F18F01')
plt.tight_layout()
plt.savefig('/root/code/vectordbindexing/hnsw_optimization/comparison_latency_hnsw_vs_roargraph.png',
            dpi=300, bbox_inches='tight')
print("   ✓ 图2已保存: comparison_latency_hnsw_vs_roargraph.png")
plt.close()

# 图3: Recall vs Latency 散点图
fig3, ax3 = plt.subplots(figsize=(12, 8))
for i, num_entries in enumerate(entry_point_values):
    ax3.scatter(hnsw_latency[i], hnsw_recall_10[i],
                s=200, alpha=0.7, color='#2E86AB', edgecolors='black', linewidth=2,
                label='HNSW' if i == 0 else None)
    ax3.annotate(f'{num_entries}', (hnsw_latency[i], hnsw_recall_10[i]),
                 ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax3.plot(hnsw_latency, hnsw_recall_10, linestyle='--', linewidth=2,
         color='#2E86AB', alpha=0.5, label='HNSW Trend')
ax3.scatter(roargraph_avg_time, roargraph_recall_10,
            s=300, alpha=0.8, color='#E63946', marker='*', edgecolors='black',
            linewidth=2, label='RoarGraph', zorder=5)
ax3.set_xlabel('Average Query Latency (ms)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Recall@10', fontsize=14, fontweight='bold')
ax3.set_title('Recall vs Latency Trade-off: HNSW vs RoarGraph',
              fontsize=16, fontweight='bold', pad=20)
ax3.legend(fontsize=12, loc='best', frameon=True, shadow=True)
ax3.grid(True, alpha=0.4, linestyle='--')
ax3.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig('/root/code/vectordbindexing/hnsw_optimization/comparison_tradeoff_hnsw_vs_roargraph.png',
            dpi=300, bbox_inches='tight')
print("   ✓ 图3已保存: comparison_tradeoff_hnsw_vs_roargraph.png")
plt.close()

# ==================== 7. 详细分析 ====================
print("\n" + "=" * 80)
print("详细性能分析")
print("=" * 80)

# 构建时间对比
print(f"\n📊 构建时间对比:")
print(f"   HNSW:      {hnsw_build_time:.2f}s")
print(f"   RoarGraph: {roargraph_build_time:.2f}s")
print(f"   比率:       {roargraph_build_time/hnsw_build_time:.2f}x")

# 最佳配置
best_hnsw_recall = max(
    entry_point_values, key=lambda x: hnsw_results[x]['recall_10'])
best_hnsw_latency = min(
    entry_point_values, key=lambda x: hnsw_results[x]['avg_time_ms'])

print(f"\n🏆 最佳配置:")
print(
    f"   HNSW 最佳召回: {best_hnsw_recall} 个入口点 (Recall@10={hnsw_results[best_hnsw_recall]['recall_10']:.4f})")
print(
    f"   HNSW 最低延迟: {best_hnsw_latency} 个入口点 (延迟={hnsw_results[best_hnsw_latency]['avg_time_ms']:.3f}ms)")
print(
    f"   RoarGraph:     Recall@10={roargraph_recall_10:.4f}, 延迟={roargraph_avg_time:.3f}ms")

# 相同Recall下延迟对比
closest_recall_config = min(entry_point_values,
                            key=lambda x: abs(hnsw_results[x]['recall_10'] - roargraph_recall_10))
print(f"\n📈 相同Recall水平下的延迟对比:")
print(
    f"   RoarGraph: Recall@10={roargraph_recall_10:.4f}, 延迟={roargraph_avg_time:.3f}ms")
print(f"   最接近的HNSW ({closest_recall_config}入口): Recall@10={hnsw_results[closest_recall_config]['recall_10']:.4f}, "
      f"延迟={hnsw_results[closest_recall_config]['avg_time_ms']:.3f}ms")
print(
    f"   延迟比率: {hnsw_results[closest_recall_config]['avg_time_ms']/roargraph_avg_time:.2f}x")

# 相同延迟下Recall对比
closest_latency_config = min(entry_point_values,
                             key=lambda x: abs(hnsw_results[x]['avg_time_ms'] - roargraph_avg_time))
print(f"\n📈 相同延迟水平下的Recall对比:")
print(
    f"   RoarGraph: 延迟={roargraph_avg_time:.3f}ms, Recall@10={roargraph_recall_10:.4f}")
print(f"   最接近的HNSW ({closest_latency_config}入口): 延迟={hnsw_results[closest_latency_config]['avg_time_ms']:.3f}ms, "
      f"Recall@10={hnsw_results[closest_latency_config]['recall_10']:.4f}")
print(
    f"   Recall比率: {hnsw_results[closest_latency_config]['recall_10']/roargraph_recall_10:.2f}x")

print("\n" + "=" * 80)
print("✅ 完整对比测试完成！")
print("=" * 80)
