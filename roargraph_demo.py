#!/usr/bin/env python3
"""
RoarGraph 快速使用演示脚本

这个脚本展示了如何使用 RoarGraph Python 实现进行向量搜索。
"""

import numpy as np
import time
from roargraph_python import RoarGraph, create_sample_data, evaluate_recall


def demo_basic_usage():
    """基本使用演示"""
    print("=== RoarGraph 基本使用演示 ===")
    
    # 1. 创建示例数据
    print("1. 创建示例数据...")
    base_data, query_data = create_sample_data(
        n_base=10000,    # 基础数据：10K个向量
        n_query=1000,    # 查询数据：1K个向量
        dimension=128,   # 向量维度：128
        seed=42
    )
    print(f"   基础数据形状: {base_data.shape}")
    print(f"   查询数据形状: {query_data.shape}")
    
    # 2. 构建 RoarGraph 索引
    print("\n2. 构建 RoarGraph 索引...")
    start_time = time.time()
    
    roargraph = RoarGraph(dimension=128, metric="cosine")
    roargraph.build(
        base_data=base_data,
        query_data=query_data,
        M_sq=32,      # 查询节点的邻居数
        M_pjbp=32,    # 投影图的邻居数
        L_pjpq=32     # 搜索队列大小
    )
    
    build_time = time.time() - start_time
    print(f"   构建时间: {build_time:.2f} 秒")
    
    # 3. 获取索引统计信息
    print("\n3. 索引统计信息:")
    stats = roargraph.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 4. 执行搜索
    print("\n4. 执行搜索...")
    test_queries = query_data[:10]  # 测试前10个查询
    k = 10  # 返回前10个最近邻
    
    total_search_time = 0
    total_comparisons = 0
    total_hops = 0
    
    for i, query in enumerate(test_queries):
        start_time = time.time()
        indices, distances, comparisons, hops = roargraph.search(query, k)
        search_time = time.time() - start_time
        
        total_search_time += search_time
        total_comparisons += comparisons
        total_hops += hops
        
        if i < 3:  # 只显示前3个查询的结果
            print(f"   查询 {i+1}:")
            print(f"     结果ID: {indices[:5]}...")
            print(f"     距离: {distances[:5]}...")
            print(f"     比较次数: {comparisons}, 跳数: {hops}, 时间: {search_time*1000:.2f}ms")
    
    # 5. 性能统计
    print(f"\n5. 性能统计:")
    print(f"   平均搜索时间: {total_search_time/len(test_queries)*1000:.2f} ms")
    print(f"   平均比较次数: {total_comparisons/len(test_queries):.1f}")
    print(f"   平均跳数: {total_hops/len(test_queries):.1f}")
    print(f"   搜索QPS: {len(test_queries)/total_search_time:.1f}")


def demo_different_metrics():
    """不同距离度量的演示"""
    print("\n=== 不同距离度量演示 ===")
    
    # 创建数据
    base_data, query_data = create_sample_data(n_base=5000, n_query=500, dimension=128)
    
    metrics = ["cosine", "l2", "ip"]
    
    for metric in metrics:
        print(f"\n使用 {metric} 距离度量:")
        
        # 构建索引
        start_time = time.time()
        roargraph = RoarGraph(dimension=128, metric=metric)
        roargraph.build(base_data, query_data, M_sq=16, M_pjbp=16, L_pjpq=16)
        build_time = time.time() - start_time
        
        # 测试搜索
        test_query = query_data[0]
        start_time = time.time()
        indices, distances, comparisons, hops = roargraph.search(test_query, k=10)
        search_time = time.time() - start_time
        
        print(f"  构建时间: {build_time:.2f}s")
        print(f"  搜索时间: {search_time*1000:.2f}ms")
        print(f"  比较次数: {comparisons}, 跳数: {hops}")
        print(f"  前5个结果: {indices[:5]}")


def demo_parameter_tuning():
    """参数调优演示"""
    print("\n=== 参数调优演示 ===")
    
    # 创建数据
    base_data, query_data = create_sample_data(n_base=5000, n_query=500, dimension=128)
    
    # 测试不同的参数组合
    param_combinations = [
        {"M_sq": 16, "M_pjbp": 16, "L_pjpq": 16},
        {"M_sq": 32, "M_pjbp": 32, "L_pjpq": 32},
        {"M_sq": 64, "M_pjbp": 64, "L_pjpq": 64},
    ]
    
    for i, params in enumerate(param_combinations):
        print(f"\n参数组合 {i+1}: {params}")
        
        # 构建索引
        start_time = time.time()
        roargraph = RoarGraph(dimension=128, metric="cosine")
        roargraph.build(base_data, query_data, **params)
        build_time = time.time() - start_time
        
        # 测试搜索性能
        test_queries = query_data[:100]
        total_time = 0
        total_comparisons = 0
        
        for query in test_queries:
            start_time = time.time()
            _, _, comparisons, _ = roargraph.search(query, k=10)
            search_time = time.time() - start_time
            total_time += search_time
            total_comparisons += comparisons
        
        avg_search_time = total_time / len(test_queries)
        avg_comparisons = total_comparisons / len(test_queries)
        qps = len(test_queries) / total_time
        
        print(f"  构建时间: {build_time:.2f}s")
        print(f"  平均搜索时间: {avg_search_time*1000:.2f}ms")
        print(f"  平均比较次数: {avg_comparisons:.1f}")
        print(f"  QPS: {qps:.1f}")


def demo_large_scale():
    """大规模数据演示"""
    print("\n=== 大规模数据演示 ===")
    
    # 创建更大的数据集
    print("创建大规模数据集...")
    base_data, query_data = create_sample_data(
        n_base=50000,   # 50K基础数据
        n_query=1000,   # 1K查询数据
        dimension=256,  # 256维向量
        seed=42
    )
    
    print(f"基础数据: {base_data.shape}, 查询数据: {query_data.shape}")
    
    # 构建索引
    print("构建索引...")
    start_time = time.time()
    roargraph = RoarGraph(dimension=256, metric="cosine")
    roargraph.build(base_data, query_data, M_sq=32, M_pjbp=32, L_pjpq=32)
    build_time = time.time() - start_time
    
    print(f"构建时间: {build_time:.2f} 秒")
    
    # 获取统计信息
    stats = roargraph.get_statistics()
    print(f"索引统计: {stats}")
    
    # 批量搜索测试
    print("执行批量搜索...")
    test_queries = query_data[:100]
    k = 20
    
    start_time = time.time()
    results = []
    for query in test_queries:
        indices, distances, comparisons, hops = roargraph.search(query, k)
        results.append((indices, distances, comparisons, hops))
    total_time = time.time() - start_time
    
    # 计算平均性能
    avg_comparisons = np.mean([r[2] for r in results])
    avg_hops = np.mean([r[3] for r in results])
    qps = len(test_queries) / total_time
    
    print(f"批量搜索结果:")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均比较次数: {avg_comparisons:.1f}")
    print(f"  平均跳数: {avg_hops:.1f}")
    print(f"  QPS: {qps:.1f}")


if __name__ == "__main__":
    print("RoarGraph Python 实现演示")
    print("=" * 50)
    
    try:
        # 基本使用演示
        demo_basic_usage()
        
        # 不同距离度量演示
        demo_different_metrics()
        
        # 参数调优演示
        demo_parameter_tuning()
        
        # 大规模数据演示
        demo_large_scale()
        
        print("\n" + "=" * 50)
        print("所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
