#!/usr/bin/env python3
"""
OOD分层图索引Demo运行脚本
快速测试系统的主要功能
"""

import sys
import os

def main():
    print("🚀 OOD分层图索引Demo")
    print("=" * 50)
    
    # 检查依赖
    try:
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        import faiss
        from sklearn.decomposition import PCA
        print("✅ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请运行: pip install numpy networkx matplotlib faiss-cpu scikit-learn")
        return 1
    
    print("\n📋 项目文件:")
    files = [
        "demo_ood_graph.ipynb",
        "README.md", 
        "step5_perturbation.py",
        "step6_async_maintenance.py",
        "step7_query_testing.py"
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    print("\n🎯 使用方法:")
    print("1. 打开 demo_ood_graph.ipynb")
    print("2. 按顺序执行所有单元格")
    print("3. 查看可视化结果和性能统计")
    
    print("\n📊 系统特性:")
    features = [
        "分层图结构 (核心图 + 边缘OOD图)",
        "OOD-score机制和自适应策略", 
        "在线增量维护",
        "统一查询接口",
        "丰富的可视化分析"
    ]
    
    for feature in features:
        print(f"  • {feature}")
    
    print("\n🔧 技术栈:")
    tech = [
        "Python 3.x",
        "NumPy - 数值计算",
        "NetworkX - 图结构",
        "FAISS - 向量检索",
        "Matplotlib - 可视化",
        "Scikit-learn - 机器学习"
    ]
    
    for tech_item in tech:
        print(f"  • {tech_item}")
    
    print("\n" + "=" * 50)
    print("🎉 项目已准备就绪！")
    print("请打开 demo_ood_graph.ipynb 开始体验")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
