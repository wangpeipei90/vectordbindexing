# HNSW优化：高层桥边和自适应多入口种子

本项目实现并评估了分层导航小世界图（HNSW）的两项关键优化，以改善跨模态检索和分布外（OOD）查询的性能。

## 概述

项目实现了两个互补的优化方案：

1. **高层桥边**：在高层节点之间建立战略性连接，提升跨簇/模态的可达性
2. **自适应多入口种子**：使用多个入口点进行搜索，通过自适应种子选择提升查询性能

## 功能特性

### 高层桥边
- 自动对高层节点进行模态分组或KMeans聚类
- 智能桥边评分函数，综合考虑跨模态奖励、逆距离、互补性和成本惩罚
- 严格的预算控制（≤0.001%的原始边数）
- 桥边映射覆盖方法，兼容hnswlib

### 自适应多入口种子
- 从高层候选节点中选择多个入口点
- 从多个种子并行执行束搜索
- 自适应种子选择策略（多样化、最优、随机）
- 可配置的束宽度和搜索参数

### 综合评估框架
- 使用FAISS进行ground truth计算
- 基线HNSW和FAISS对比
- 成本和召回率测量
- 百分位数分析和统计评估
- 自动化绘图和结果可视化

## 安装

### 环境要求

- Python 3.10+
- hnswlib >= 0.7.0
- faiss-cpu >= 1.7.4
- scikit-learn >= 1.0.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- joblib >= 1.1.0
- tqdm >= 4.62.0

### 安装步骤

1. 克隆或下载项目：
```bash
cd /path/to/hnsw_optimization
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装包（可选）：
```bash
pip install -e .
```

## 快速开始

### 运行实验

主要的实验运行器提供了完整的评估流程：

```bash
python experiment_runner.py --n_vectors 10000 --n_queries 1000 --dimension 128 --n_modalities 5
```

### 快速功能验证

运行快速验证脚本（仿照hnsw_status.ipynb风格）：

```bash
python quick_test.py
```

### 基本使用

```python
from data_loader import create_toy_dataset
from hnsw_baseline import HNSWBaseline
from bridge_builder import BridgeBuilder
from multi_entry_search import MultiEntrySearch

# 创建数据集
X, Q, modalities, query_modalities = create_toy_dataset(1000, 100, 64, 3)

# 构建基线HNSW
hnsw = HNSWBaseline(dimension=64, M=16, ef_construction=200)
hnsw.build_index(X)

# 构建桥边
bridge_builder = BridgeBuilder(bridge_budget_ratio=1e-5)
bridge_map = bridge_builder.build_bridges(hnsw, X, modalities)

# 多入口搜索
multi_search = MultiEntrySearch(hnsw, bridge_builder)
neighbors, cost = multi_search.multi_entry_search(Q[0], k=10, m=4)

print(f"找到 {len(neighbors)} 个邻居，成本 {cost}")
```

## 架构设计

### 核心组件

1. **数据加载** (`data_loader.py`)：
   - 多模态合成数据集生成
   - 可选的文件数据加载
   - 数据集统计和验证

2. **Ground Truth** (`gt_utils.py`)：
   - 使用FAISS进行精确最近邻计算
   - 召回率计算和评估工具
   - 搜索方法的基准测试框架

3. **基线实现** (`hnsw_baseline.py`)：
   - 带成本跟踪的HNSW索引
   - 用于对比的FAISS基线
   - 统计和分析工具

4. **桥边构建器** (`bridge_builder.py`)：
   - 高层节点提取
   - 聚类和模态分组
   - 桥边评分和选择
   - 预算控制的边添加

5. **多入口搜索** (`multi_entry_search.py`)：
   - 高层候选选择
   - 并行束搜索实现
   - 自适应种子选择策略
   - 桥边感知的搜索遍历

6. **实验运行器** (`experiment_runner.py`)：
   - 完整的评估流程
   - 统计分析和绘图
   - 结果导出和可视化

### 关键算法

#### 桥边评分函数

桥边的评分函数结合了多个因素：

```
S(u,v) = w1 × 跨模态奖励 + w2 × 逆距离 + w3 × 互补性 - w4 × 成本惩罚
```

其中：
- `跨模态奖励`：如果节点来自不同模态/簇则为1，否则为0
- `逆距离`：1 / (1 + L2距离)
- `互补性`：1 - 余弦相似度（简化版）
- `成本惩罚`：归一化的预期成本

#### 多入口搜索策略

1. **高层候选选择**：从level ≥ 1收集多样化候选
2. **自适应种子选择**：使用多样性标准选择top-m种子
3. **并行束搜索**：从每个种子同时执行束搜索
4. **结果合并**：合并所有种子的结果并重新排序

## 配置参数

### 默认参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `M` | 16 | HNSW连接参数 |
| `ef_construction` | 200 | 构建搜索宽度 |
| `ef_search` | 200 | 搜索宽度 |
| `bridge_budget_ratio` | 1e-5 | 桥边预算比例 |
| `max_bridge_per_node` | 2 | 每节点最大桥边数 |
| `m` (种子数) | 4 | 入口种子数量 |
| `beam_width` | 100 | 束搜索宽度 |
| `target_recall` | 0.90 | 目标召回率阈值 |

### 命令行选项

```bash
python experiment_runner.py \
    --n_vectors 10000 \
    --n_queries 1000 \
    --dimension 128 \
    --n_modalities 5 \
    --M 16 \
    --ef_construction 200 \
    --bridge_budget 1e-5 \
    --m_values 2 4 8 \
    --ef_search_values 50 100 200 400 \
    --k_eval 100 \
    --target_recall 0.90 \
    --output_dir results
```

## 评估方法

### 阶段1：基线测量
1. 使用FAISS精确搜索计算ground truth
2. 构建基线HNSW索引
3. 调整`ef_search`以达到目标召回率（90%）
4. 记录成本分布并确定评估查询

### 阶段2：优化评估
1. 构建带桥边的HNSW
2. 使用不同`m`值运行多入口搜索
3. 测量相同查询的成本和召回率
4. 比较性能指标

### 阶段3：分析
1. 生成百分位数表（P10, P25, P50, P75, P90, P95, P99）
2. 创建成本vs召回率图
3. 分析加速比和改进分布
4. 导出结果和可视化

## 结果和输出

### 生成文件

- `comparison_results.csv`：详细的查询级对比
- `percentile_analysis.csv`：基于百分位数的性能分析
- `baseline_evaluation.json`：基线HNSW结果
- `optimized_results.json`：优化HNSW结果
- `comparison_results.png`：主要对比图
- `detailed_analysis.png`：详细分析可视化

### 关键指标

- **成本比例**：优化成本 / 基线成本
- **召回率改进**：优化召回率 - 基线召回率
- **延迟比例**：优化延迟 / 基线延迟
- **重叠比例**：共同邻居的比例

## 测试验证

### 快速功能验证

运行快速验证脚本：

```bash
python quick_test.py
```

验证脚本包括：
- 数据加载功能测试
- 基线HNSW构建和搜索测试
- 桥边构建功能测试
- 多入口搜索功能测试
- Ground truth计算和recall评估测试
- 比较分析功能测试

### 完整实验

运行完整实验：

```bash
# 小规模测试
make run-experiment

# 大规模实验
make run-experiment-large
```

## 实现说明

### 桥边实现

由于hnswlib不直接暴露邻接操作，实现使用`bridge_map`覆盖方法：

- 桥边与核心HNSW结构分开存储
- 搜索时将桥边邻居与常规邻居合并
- 这种方法保持兼容性的同时启用桥边功能

### 动态插入（待实现）

当前实现专注于索引构建期间的静态桥边构建。动态插入逻辑标记为TODO注释，包括：

- 新高层节点的自动桥边创建
- 桥边验证和移除
- 软边管理

### 内存考虑

对于大型数据集：
- 使用小批量聚类进行桥边构建
- 流式候选生成
- 考虑内存高效的数据结构

## 限制和未来工作

### 当前限制

1. **hnswlib集成**：有限访问HNSW内部结构（通过桥边映射方法缓解）
2. **距离计算**：某些组件中的简化距离计算
3. **动态插入**：未实现（按要求标记为TODO）
4. **内存使用**：对于非常大的数据集可能需要优化

### 未来增强

1. **原生hnswlib集成**：直接修改HNSW邻接列表
2. **高级评分**：更复杂的桥边评分函数
3. **自适应参数**：自动调整`m`和其他参数
4. **GPU加速**：基于CUDA的距离计算
5. **增量更新**：高效的桥边维护

## 引用

如果您在研究中使用了此实现，请引用：

```bibtex
@misc{hnsw_optimization_2024,
  title={HNSW优化：高层桥边和自适应多入口种子},
  author={HNSW优化团队},
  year={2024},
  url={https://github.com/your-repo/hnsw-optimization}
}
```

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

## 贡献

欢迎贡献！请随时提交问题、功能请求或拉取请求。

## 致谢

- hnswlib团队提供优秀的HNSW实现
- FAISS团队提供高效的相似性搜索
- scikit-learn提供聚类算法
- 向量搜索社区提供灵感和反馈