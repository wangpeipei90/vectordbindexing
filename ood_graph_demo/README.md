# OOD分层图索引Demo

## 项目简介

本项目实现了一个"面向OOD的分层图索引 + 在线增量维护"的演示系统，专门用于处理分布外（Out-of-Distribution, OOD）查询的向量检索问题。

## 核心设计理念

### 分层图结构
- **核心图 (CoreGraph)**: 存储常见的ID（In-Distribution）节点，使用高效的图结构进行kNN查询
- **边缘图 (OODGraph)**: 存储OOD节点，通过长边连接到核心图，保证OOD查询的可达性

### OOD-score机制
- 基于局部密度和相似度的OOD-score计算
- 根据OOD-score自适应调整图结构策略
- 动态决定节点的连接策略

### 在线增量维护
- 支持新OOD节点的动态插入
- 异步批量处理图结构更新
- 自适应长边连接优化

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐
│   核心图        │    │   边缘OOD图      │
│   (ID节点)      │◄──►│   (OOD节点)     │
│                 │    │                 │
│  • 1000个向量   │    │  • 动态插入     │
│  • kNN查询      │    │  • 长边连接     │
│  • 余弦相似度   │    │  • OOD-score    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
            ┌─────────────────┐
            │  分层查询接口    │
            │                 │
            │ • 自适应策略     │
            │ • 性能统计       │
            │ • 可视化分析     │
            └─────────────────┘
```

## 主要功能

### 1. 数据生成与预处理
- 生成1000个128维的库向量（模拟ID数据）
- 生成200个查询向量，包含20%的OOD查询
- 使用PCA降维可视化向量分布

### 2. 核心图构建
- 基于余弦相似度构建邻接关系
- 使用FAISS加速kNN计算
- 实现高效的ID查询功能

### 3. 边缘OOD图
- 动态检测和添加OOD节点
- 计算到核心图的长边连接
- 支持在线增量插入

### 4. OOD-score机制
- 增强版OOD-score计算（局部密度+相似度）
- 自适应节点策略决策
- 不同阈值下的OOD检测效果测试

### 5. 分层查询系统
- 统一的查询接口
- 自适应搜索策略选择
- 性能统计和可视化分析

## 技术栈

- **Python 3.x**
- **NumPy**: 数值计算
- **NetworkX**: 图结构构建和算法
- **FAISS**: 高效向量相似度计算
- **Matplotlib**: 数据可视化
- **Scikit-learn**: PCA降维

## 文件结构

```
ood_graph_demo/
├── demo_ood_graph.ipynb          # 主要演示Notebook
├── step5_perturbation.py         # 局部扰动增强实现
├── step6_async_maintenance.py    # 异步增量维护实现
├── step7_query_testing.py        # 查询测试实现
└── README.md                     # 项目说明文档
```

## 使用方法

### 1. 环境准备
```bash
pip install numpy networkx matplotlib faiss-cpu scikit-learn
```

### 2. 运行Demo
打开 `demo_ood_graph.ipynb` 并按顺序执行所有单元格：

1. **Step 0**: 环境准备和库导入
2. **Step 1**: 数据生成与预处理
3. **Step 2**: 构建核心图
4. **Step 3**: 构建边缘OOD图
5. **Step 4**: OOD-score机制
6. **Step 7**: 查询测试与性能验证
7. **Step 8**: 系统总结

### 3. 核心类说明

#### CoreGraph
```python
core_graph = CoreGraph(vectors, k_neighbors=20)
results = core_graph.knn_search(query_vector, k=10)
```

#### EnhancedOODGraph
```python
ood_graph = EnhancedOODGraph(core_graph, max_long_edges=5)
ood_id = ood_graph.add_ood_node_with_strategy(vector)
```

#### HierarchicalGraphQuery
```python
query = HierarchicalGraphQuery(core_graph, ood_graph)
results = query.hierarchical_search(query_vector, k=10)
```

## 实验结果

### 数据规模
- 库向量: 1000个128维向量
- ID查询: 160个
- OOD查询: 40个
- OOD比例: 20%

### 性能指标
- 核心图节点数: 1000
- 核心图边数: ~5000-10000
- 平均查询时间: <0.01秒
- OOD检测准确率: >80%

### 可视化输出
- 向量分布散点图（PCA降维）
- 图结构统计（度分布、边权重分布）
- OOD-score分布分析
- 查询性能对比图表

## 系统特点

### 优势
1. **高效处理OOD查询**: 通过分层结构和长边连接保证OOD查询的可达性
2. **自适应策略**: 根据OOD-score动态调整图结构
3. **在线增量**: 支持新节点的动态插入和更新
4. **性能优化**: 使用FAISS加速计算，查询时间快
5. **可视化丰富**: 多维度数据分析和性能监控

### 可扩展性
1. **OOD-score算法**: 可扩展更复杂的密度估计方法
2. **图结构优化**: 可添加更多图算法和优化策略
3. **异步处理**: 支持大规模数据的后台处理
4. **分布式**: 可扩展为分布式图索引系统

## 未来改进方向

1. **更精确的OOD检测**: 集成深度学习模型进行OOD检测
2. **动态边策略**: 根据查询模式动态调整边的权重和连接
3. **分布式架构**: 支持大规模数据的分布式处理
4. **实时监控**: 添加实时性能监控和告警机制
5. **A/B测试**: 支持不同策略的效果对比测试

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。主要改进方向包括：
- 算法优化
- 性能提升
- 功能扩展
- 文档完善

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

**注意**: 这是一个演示项目，主要用于展示OOD分层图索引的设计思路和实现方法。在生产环境中使用时，建议根据具体需求进行优化和扩展。
