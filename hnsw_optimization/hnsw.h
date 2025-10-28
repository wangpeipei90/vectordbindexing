#pragma once

#include <vector>
#include <unordered_set>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <mutex>

namespace hnsw {

// 距离计算函数（L2距离）
inline float l2_distance(const float* a, const float* b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// 邻居结构
struct Neighbor {
    int id;
    float distance;
    
    Neighbor(int id, float dist) : id(id), distance(dist) {}
    
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
    
    bool operator>(const Neighbor& other) const {
        return distance > other.distance;
    }
};

// 搜索结果结构
struct SearchResult {
    std::vector<int> neighbors;      // 找到的邻居ID
    size_t visited_count;             // 访问的节点总数
    double latency_us;                // 搜索延迟（微秒）
    size_t layer1_visited;            // 第1层访问的节点数
    size_t layer0_visited;            // 第0层访问的节点数
};

// 搜索状态
struct SearchState {
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates;  // min heap
    std::priority_queue<Neighbor> top_k;  // max heap for top-k results
    std::unordered_set<int> visited;
};

/**
 * 2层HNSW索引
 * - 第0层：包含全部数据，固定出度M0
 * - 第1层：随机选择的节点（概率与标准HNSW一致），固定出度M1 = M0/2
 */
class HNSW {
public:
    HNSW(size_t dimension, size_t M0, size_t ef_construction, size_t max_elements, int seed = 42);
    ~HNSW();
    
    // 构建索引
    void build(const float* data, size_t num_vectors);
    
    // 搜索（返回详细结果）
    SearchResult search(const float* query, size_t k, size_t ef_search, size_t num_entry_points = 10);
    
    // 批量搜索并计算recall
    std::vector<SearchResult> batch_search(const float* queries, size_t num_queries, size_t k, 
                                           size_t ef_search, size_t num_entry_points = 10);
    
    // 计算recall@k
    static double compute_recall(const std::vector<int>& results, const std::vector<int>& ground_truth, size_t k);
    
    // 获取统计信息
    size_t get_num_nodes() const { return num_elements_; }
    size_t get_num_layer1_nodes() const { return layer1_nodes_.size(); }
    
    // 调试接口：获取节点的邻居
    std::vector<int> get_neighbors_layer0(int node_id) const {
        if (node_id < 0 || node_id >= static_cast<int>(num_elements_)) {
            return std::vector<int>();
        }
        return nodes_[node_id].neighbors_layer0;
    }
    
    std::vector<int> get_neighbors_layer1(int node_id) const {
        if (node_id < 0 || node_id >= static_cast<int>(num_elements_)) {
            return std::vector<int>();
        }
        return nodes_[node_id].neighbors_layer1;
    }
    
    bool is_in_layer1(int node_id) const {
        if (node_id < 0 || node_id >= static_cast<int>(num_elements_)) {
            return false;
        }
        return nodes_[node_id].in_layer1;
    }
    
private:
    // 图结构
    struct Node {
        std::vector<int> neighbors_layer0;  // 第0层邻居
        std::vector<int> neighbors_layer1;  // 第1层邻居（如果节点在第1层）
        bool in_layer1 = false;
    };
    
    // 参数
    size_t dimension_;
    size_t M0_;  // 第0层出度
    size_t M1_;  // 第1层出度
    size_t ef_construction_;
    size_t max_elements_;
    float ml_;  // 层级选择参数
    
    // 数据
    float* data_;  // 向量数据 [num_elements * dimension]
    std::vector<Node> nodes_;
    std::vector<int> layer1_nodes_;  // 第1层节点ID列表
    size_t num_elements_;
    
    // 随机数生成器
    std::mt19937 rng_;
    std::uniform_real_distribution<float> level_dist_;
    
    // 辅助函数
    int select_layer_randomly();
    void insert_node(int node_id, int max_layer);
    void connect_neighbors_layer0(int node_id, int ef);
    void connect_neighbors_layer1(int node_id, int ef);
    std::vector<int> search_layer(const float* query, const std::vector<int>& entry_points, 
                                   size_t ef, int layer);
    std::vector<int> select_neighbors(const std::vector<Neighbor>& candidates, size_t M);
    float distance(int node_id, const float* query) const;
    
    // 第1层搜索（直到top-k稳定）
    std::vector<int> search_layer1_stable(const float* query, size_t k, size_t ef_search, size_t& visited_count);
    
    // 第0层多入口并行搜索
    std::vector<int> search_layer0_multi_entry(const float* query, const std::vector<int>& entry_points, 
                                                 size_t k, size_t ef_search, size_t& visited_count);
};

} // namespace hnsw

