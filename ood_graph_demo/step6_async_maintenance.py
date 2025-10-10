# Step 6: 异步增量维护
import numpy as np
import threading
import time
from typing import List, Dict, Optional
from collections import deque
import queue

class AsyncGraphMaintenance:
    """异步增量维护类"""
    
    def __init__(self, enhanced_ood_graph, batch_size: int = 10, update_interval: float = 1.0):
        """
        初始化异步维护
        
        Args:
            enhanced_ood_graph: 增强版OOD图实例
            batch_size: 批量处理大小
            update_interval: 更新间隔（秒）
        """
        self.ood_graph = enhanced_ood_graph
        self.batch_size = batch_size
        self.update_interval = update_interval
        
        # 异步处理队列
        self.insertion_queue = queue.Queue()
        self.update_queue = queue.Queue()
        
        # 维护线程
        self.maintenance_thread = None
        self.running = False
        
        # 统计信息
        self.stats = {
            'total_insertions': 0,
            'total_updates': 0,
            'pending_insertions': 0,
            'pending_updates': 0
        }
    
    def start_maintenance(self):
        """启动异步维护线程"""
        if self.running:
            return
        
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        print("异步维护线程已启动")
    
    def stop_maintenance(self):
        """停止异步维护线程"""
        self.running = False
        if self.maintenance_thread:
            self.maintenance_thread.join()
        print("异步维护线程已停止")
    
    def async_insert_node(self, vector: np.ndarray, priority: str = "normal") -> str:
        """
        异步插入新节点
        
        Args:
            vector: 新节点向量
            priority: 优先级 ("low", "normal", "high")
            
        Returns:
            任务ID
        """
        task_id = f"insert_{int(time.time() * 1000)}"
        task = {
            'id': task_id,
            'type': 'insert',
            'vector': vector.copy(),
            'priority': priority,
            'timestamp': time.time()
        }
        
        self.insertion_queue.put(task)
        self.stats['pending_insertions'] += 1
        
        print(f"异步插入任务已提交: {task_id}")
        return task_id
    
    def async_update_edges(self, node_id: int, update_type: str = "enhance") -> str:
        """
        异步更新边
        
        Args:
            node_id: 节点ID
            update_type: 更新类型 ("enhance", "optimize")
            
        Returns:
            任务ID
        """
        task_id = f"update_{int(time.time() * 1000)}"
        task = {
            'id': task_id,
            'type': 'update',
            'node_id': node_id,
            'update_type': update_type,
            'timestamp': time.time()
        }
        
        self.update_queue.put(task)
        self.stats['pending_updates'] += 1
        
        print(f"异步更新任务已提交: {task_id}")
        return task_id
    
    def _maintenance_loop(self):
        """维护线程主循环"""
        while self.running:
            try:
                # 处理插入任务
                self._process_insertion_batch()
                
                # 处理更新任务
                self._process_update_batch()
                
                # 休眠
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"维护线程错误: {e}")
                time.sleep(1)
    
    def _process_insertion_batch(self):
        """批量处理插入任务"""
        batch = []
        
        # 收集批量任务
        while len(batch) < self.batch_size and not self.insertion_queue.empty():
            try:
                task = self.insertion_queue.get_nowait()
                batch.append(task)
            except queue.Empty:
                break
        
        if not batch:
            return
        
        # 按优先级排序
        priority_order = {"high": 0, "normal": 1, "low": 2}
        batch.sort(key=lambda x: priority_order.get(x['priority'], 1))
        
        # 批量处理
        for task in batch:
            try:
                result = self.ood_graph.add_ood_node_with_strategy(task['vector'])
                if result:
                    self.stats['total_insertions'] += 1
                    print(f"异步插入完成: {task['id']}, 节点ID: {result['ood_id']}")
                else:
                    print(f"异步插入跳过: {task['id']} (非OOD)")
                
                self.stats['pending_insertions'] -= 1
                
            except Exception as e:
                print(f"插入任务失败 {task['id']}: {e}")
    
    def _process_update_batch(self):
        """批量处理更新任务"""
        batch = []
        
        # 收集批量任务
        while len(batch) < self.batch_size and not self.update_queue.empty():
            try:
                task = self.update_queue.get_nowait()
                batch.append(task)
            except queue.Empty:
                break
        
        if not batch:
            return
        
        # 批量处理
        for task in batch:
            try:
                node_id = task['node_id']
                update_type = task['update_type']
                
                if update_type == "enhance":
                    # 增强连接
                    self._enhance_node_connections(node_id)
                elif update_type == "optimize":
                    # 优化连接
                    self._optimize_node_connections(node_id)
                
                self.stats['total_updates'] += 1
                self.stats['pending_updates'] -= 1
                
                print(f"异步更新完成: {task['id']}")
                
            except Exception as e:
                print(f"更新任务失败 {task['id']}: {e}")
    
    def _enhance_node_connections(self, node_id: int):
        """增强节点连接"""
        if node_id not in self.ood_graph.ood_vectors:
            return
        
        vector = self.ood_graph.ood_vectors[node_id]
        
        # 添加更多长边
        similarities = np.dot(self.ood_graph.core_graph.vectors, vector / np.linalg.norm(vector))
        enhanced_edges = []
        
        for i, similarity in enumerate(similarities):
            if similarity > 0.05:  # 很低的阈值
                enhanced_edges.append((i, similarity))
        
        # 限制边的数量
        enhanced_edges = sorted(enhanced_edges, key=lambda x: x[1], reverse=True)[:15]
        
        # 更新连接
        if node_id in self.ood_graph.ood_to_core_edges:
            self.ood_graph.ood_to_core_edges[node_id] = enhanced_edges
        else:
            self.ood_graph.ood_to_core_edges[node_id] = enhanced_edges
    
    def _optimize_node_connections(self, node_id: int):
        """优化节点连接"""
        if node_id not in self.ood_graph.ood_to_core_edges:
            return
        
        # 移除相似度很低的边
        edges = self.ood_graph.ood_to_core_edges[node_id]
        optimized_edges = [(core_id, sim) for core_id, sim in edges if sim > 0.1]
        
        self.ood_graph.ood_to_core_edges[node_id] = optimized_edges
    
    def get_maintenance_stats(self) -> Dict:
        """获取维护统计信息"""
        return {
            **self.stats,
            'queue_sizes': {
                'insertion_queue': self.insertion_queue.qsize(),
                'update_queue': self.update_queue.qsize()
            },
            'is_running': self.running
        }
    
    def wait_for_completion(self, timeout: float = 10.0):
        """等待所有任务完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if (self.stats['pending_insertions'] == 0 and 
                self.stats['pending_updates'] == 0):
                print("所有异步任务已完成")
                return True
            
            time.sleep(0.1)
        
        print(f"等待超时，仍有 {self.stats['pending_insertions']} 个插入任务和 {self.stats['pending_updates']} 个更新任务")
        return False

def test_async_maintenance():
    """测试异步增量维护"""
    print("Step 6: 异步增量维护测试")
    
    # 这里需要传入实际的enhanced_ood_graph实例
    # async_maintenance = AsyncGraphMaintenance(enhanced_ood_graph, batch_size=5, update_interval=0.5)
    
    # # 启动维护线程
    # async_maintenance.start_maintenance()
    
    # # 提交一些异步任务
    # task_ids = []
    # for i, vector in enumerate(ood_queries[10:20]):  # 使用更多OOD查询
    #     priority = "high" if i % 3 == 0 else "normal"
    #     task_id = async_maintenance.async_insert_node(vector, priority=priority)
    #     task_ids.append(task_id)
    
    # # 提交一些更新任务
    # for ood_id in list(enhanced_ood_graph.ood_vectors.keys())[:3]:
    #     task_id = async_maintenance.async_update_edges(ood_id, "enhance")
    #     task_ids.append(task_id)
    
    # # 等待任务完成
    # async_maintenance.wait_for_completion(timeout=15.0)
    
    # # 获取统计信息
    # stats = async_maintenance.get_maintenance_stats()
    # print("异步维护统计信息:")
    # for key, value in stats.items():
    #     print(f"  {key}: {value}")
    
    # # 停止维护线程
    # async_maintenance.stop_maintenance()
    
    print("✅ Step 6: 异步增量维护完成")

if __name__ == "__main__":
    test_async_maintenance()
