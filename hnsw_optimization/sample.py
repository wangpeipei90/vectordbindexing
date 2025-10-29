# 导入必要的库
import sys
sys.path.append('/workspace/vectordbindexing')
sys.path.append('/workspace/vectordbindexing/hnsw_optimization')

import numpy as np
import time
import matplotlib.pyplot as plt
from io_utils import read_fbin, read_ibin

# 导入我们的优化模块
from hnsw_with_bridges_optimized import HNSWWithBridgesOptimized  # 使用优化版

# 数据路径
file_path = "../Text2Image/base.10M.fbin"
query_path = "../Text2Image/query.10k.fbin"
ground_truth_path = "../Text2Image/gt.10k.ibin"
# 定义保存路径 - layer0 graph
graph_file = '/workspace/vectordbindexing/hnsw_optimization/layer0_graph.txt'
# RoarGraph index 文件路径
roargraph_index_path = '/workspace/vectordbindexing/Text2Image/t2i_10M_roar.index'


print("加载数据集...")

# 读取数据集
data_vector = read_fbin(file_path)
query_vector = read_fbin(query_path)

print(f"完整数据向量形状: {data_vector.shape}")
print(f"完整查询向量形状: {query_vector.shape}")

# ⚠️ 重要：使用较小的数据集进行测试（避免内存溢出）
n_train = len(data_vector)  # 使用 500K 数据
n_query = len(query_vector)    # 使用 1000 个查询

X = data_vector[:n_train]
Q = query_vector[:n_query]

print(f"\n使用训练数据: {X.shape} (500K)")
print(f"使用查询数据: {Q.shape} (1000)")

# 加载 ground truth 结果（从.ibin文件）
print("\n加载 ground truth 结果...")
gt_neighbors_full = read_ibin(ground_truth_path)
print(f"完整 Ground truth 形状: {gt_neighbors_full.shape}")

# 只使用前n_query个查询的ground truth
gt_neighbors = gt_neighbors_full[:n_query]
print(f"使用的 Ground truth 形状: {gt_neighbors.shape}")

print("\n✅ 数据加载完成")

np.random.seed(42)

print("=" * 70)
print("构建优化版 HNSW（C++核心实现）")
print("=" * 70)

# ===== 快速加载选项 =====
# 设置为True可以从保存的文件快速加载图结构
use_saved_graph = False  # 改为True以启用快速加载
use_roargraph = True
graph_file_path = '/workspace/vectordbindexing/hnsw_optimization/layer0_graph_tmp.txt'

# 创建优化版 HNSW（只构建一次！）
hnsw_optimized = HNSWWithBridgesOptimized(
    dimension=X.shape[1],
    M=64,              # M0=64, M1=32
    ef_construction=200,
    num_entry_points=4 # 默认4个入口点
)

print(f"\n快速加载模式: {'✅ 启用' if use_saved_graph else '❌ 关闭'}")
if use_saved_graph:
    print(f"将从文件加载: {graph_file_path}")
elif use_roargraph:
    print(f"将从 RoarGraph 加载: {roargraph_index_path}")

    # 构建索引（支持快速加载）
print("\n开始构建索引...")
start_time = time.time()

# 如果启用快速加载，传入文件路径
if use_saved_graph:
    hnsw_optimized.build_index(X, rebuild_graph_from=graph_file_path)
elif use_roargraph:
    hnsw_optimized.build_index(X, load_from_roargraph=roargraph_index_path)
else:
    hnsw_optimized.build_index(X)

optimized_build_time = time.time() - start_time

if use_saved_graph:
    print(f"\n✅ 从文件快速加载完成，耗时: {optimized_build_time:.2f}秒")
else:
    print(f"\n✅ 完整构建完成，耗时: {optimized_build_time:.2f}秒")

# 统计信息
stats = hnsw_optimized.get_statistics()
print(f"\n统计信息:")
print(f"  总节点数: {stats['total_nodes']}")
print(f"  维度: {stats['dimension']}")
print(f"  M0 (第0层出度): {stats['M0']}")
print(f"  M1 (第1层出度): {stats['M1']}")
print(f"  第1层节点数: {stats['layer1_nodes']} ({stats['layer1_ratio']*100:.2f}%)")
print(f"  理论第1层比例: ~{100/stats['M0']:.2f}% (P(L>=1)=1/M0)")
print(f"  实现方式: {stats['implementation']}")

# 测试单个查询
print("\n测试搜索...")
test_query = Q[0]
test_neighbors, test_stats = hnsw_optimized.search(test_query, k=100, ef_search=200, num_entry_points=4)
print(f"  找到 {len(test_neighbors)} 个邻居")
print(f"  avg_visited: {test_stats['visited_count']}")
print(f"  mean_latency: {test_stats['latency_us']:.2f} μs")
print(f"  前10个邻居: {test_neighbors[:10]}")