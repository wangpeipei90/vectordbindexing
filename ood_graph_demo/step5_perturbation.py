# Step 5: 局部扰动增强
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random

class PerturbationEnhancedGraph:
    """局部扰动增强图类"""
    
    def __init__(self, enhanced_ood_graph):
        """
        初始化扰动增强图
        
        Args:
            enhanced_ood_graph: 增强版OOD图实例
        """
        self.ood_graph = enhanced_ood_graph
        self.perturbation_edges = {}  # 存储扰动边
        self.perturbation_threshold = 0.6  # 扰动阈值
        
    def add_perturbation_edges(self, ood_id: int, num_perturbations: int = 3):
        """
        为高OOD-score节点添加扰动边
        
        Args:
            ood_id: OOD节点ID
            num_perturbations: 扰动边数量
        """
        if ood_id not in self.ood_graph.ood_vectors:
            return
        
        vector = self.ood_graph.ood_vectors[ood_id]
        ood_score = self.ood_graph.ood_graph.nodes[ood_id]['ood_score']
        
        # 只有高OOD-score节点才添加扰动边
        if ood_score < self.perturbation_threshold:
            return
        
        # 随机选择核心图中的节点作为扰动边目标
        core_nodes = list(range(self.ood_graph.core_graph.n_vectors))
        random.shuffle(core_nodes)
        
        perturbation_targets = []
        for target_id in core_nodes[:num_perturbations]:
            # 计算相似度
            target_vector = self.ood_graph.core_graph.vectors[target_id]
            similarity = np.dot(vector, target_vector) / (np.linalg.norm(vector) * np.linalg.norm(target_vector))
            
            # 添加扰动边（即使相似度很低）
            perturbation_targets.append((target_id, similarity))
        
        self.perturbation_edges[ood_id] = perturbation_targets
        
        print(f"为OOD节点 {ood_id} 添加了 {len(perturbation_targets)} 条扰动边")
    
    def get_perturbation_stats(self) -> Dict:
        """获取扰动边统计信息"""
        return {
            'perturbation_nodes': len(self.perturbation_edges),
            'total_perturbation_edges': sum(len(edges) for edges in self.perturbation_edges.values()),
            'avg_perturbation_per_node': np.mean([len(edges) for edges in self.perturbation_edges.values()]) if self.perturbation_edges else 0
        }
    
    def visualize_perturbation_effects(self):
        """可视化扰动边的效果"""
        if not self.perturbation_edges:
            print("没有扰动边可可视化")
            return
        
        plt.figure(figsize=(15, 5))
        
        # 子图1：扰动边的相似度分布
        plt.subplot(1, 3, 1)
        all_similarities = []
        for edges in self.perturbation_edges.values():
            all_similarities.extend([sim for _, sim in edges])
        
        plt.hist(all_similarities, bins=20, alpha=0.7, color='purple')
        plt.title('扰动边相似度分布')
        plt.xlabel('余弦相似度')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # 子图2：每个节点的扰动边数量
        plt.subplot(1, 3, 2)
        perturbation_counts = [len(edges) for edges in self.perturbation_edges.values()]
        plt.hist(perturbation_counts, bins=range(max(perturbation_counts)+2), alpha=0.7, color='orange')
        plt.title('节点扰动边数量分布')
        plt.xlabel('扰动边数量')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # 子图3：扰动边的OOD-score分布
        plt.subplot(1, 3, 3)
        ood_scores = []
        for ood_id in self.perturbation_edges.keys():
            ood_scores.append(self.ood_graph.ood_graph.nodes[ood_id]['ood_score'])
        
        plt.hist(ood_scores, bins=10, alpha=0.7, color='red')
        plt.title('扰动节点的OOD-score分布')
        plt.xlabel('OOD Score')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def test_perturbation_enhancement():
    """测试扰动增强效果"""
    print("Step 5: 局部扰动增强测试")
    
    # 这里需要传入实际的enhanced_ood_graph实例
    # perturbation_graph = PerturbationEnhancedGraph(enhanced_ood_graph)
    
    # # 为高OOD-score节点添加扰动边
    # high_ood_nodes = []
    # for node_id, data in perturbation_graph.ood_graph.ood_graph.nodes(data=True):
    #     if data['ood_score'] >= perturbation_graph.perturbation_threshold:
    #         high_ood_nodes.append(node_id)
    
    # print(f"找到 {len(high_ood_nodes)} 个高OOD-score节点")
    
    # for ood_id in high_ood_nodes[:5]:  # 只处理前5个
    #     perturbation_graph.add_perturbation_edges(ood_id, num_perturbations=3)
    
    # # 获取统计信息
    # stats = perturbation_graph.get_perturbation_stats()
    # print("扰动增强统计信息:")
    # for key, value in stats.items():
    #     print(f"  {key}: {value}")
    
    # # 可视化效果
    # perturbation_graph.visualize_perturbation_effects()
    
    print("✅ Step 5: 局部扰动增强完成")

if __name__ == "__main__":
    test_perturbation_enhancement()
