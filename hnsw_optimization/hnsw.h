#pragma once

#include <vector>
#include <unordered_set>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <mutex>

// SIMD intrinsics
#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace hnsw {

// 距离计算函数（L2距离）- ✅ 优化版本with SIMD
inline float l2_distance(const float* a, const float* b, size_t dim) {
    float dist = 0.0f;
    
#ifdef __AVX__
    // AVX优化（8个float同时计算）
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vdiff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(vdiff, vdiff));
    }
    
    // 水平求和
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    dist = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    
    // 处理剩余元素
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
#elif defined(__SSE__)
    // SSE优化（4个float同时计算）
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vdiff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(vdiff, vdiff));
    }
    
    // 水平求和
    float temp[4];
    _mm_storeu_ps(temp, sum);
    dist = temp[0] + temp[1] + temp[2] + temp[3];
    
    // 处理剩余元素
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
#else
    // 标准实现（带循环展开）
    size_t i = 0;
    // 4-way unrolling
    for (; i + 4 <= dim; i += 4) {
        float diff0 = a[i] - b[i];
        float diff1 = a[i+1] - b[i+1];
        float diff2 = a[i+2] - b[i+2];
        float diff3 = a[i+3] - b[i+3];
        dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }
    // 处理剩余元素
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
#endif
    
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
    
    // 只构建第1层（假设第0层已经加载）
    void build_layer1_only(const float* data);
    
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
    
    // 图结构保存和加载
    std::vector<int> get_layer0_neighbors(int node_id) const;
    void load_layer0(const float* data, size_t num_vectors, const std::vector<std::vector<int>>& neighbors_list);
    
    // 调试接口：获取节点的邻居（适配新的数组结构）
    std::vector<int> get_neighbors_layer0(int node_id) const {
        if (node_id < 0 || node_id >= static_cast<int>(num_elements_)) {
            return std::vector<int>();
        }
        const Node& node = nodes_[node_id];
        std::vector<int> neighbors;
        neighbors.reserve(node.num_neighbors_layer0);
        for (int i = 0; i < node.num_neighbors_layer0; ++i) {
            neighbors.push_back(node.neighbors_layer0[i]);
        }
        return neighbors;
    }
    
    std::vector<int> get_neighbors_layer1(int node_id) const {
        if (node_id < 0 || node_id >= static_cast<int>(num_elements_)) {
            return std::vector<int>();
        }
        const Node& node = nodes_[node_id];
        std::vector<int> neighbors;
        neighbors.reserve(node.num_neighbors_layer1);
        for (int i = 0; i < node.num_neighbors_layer1; ++i) {
            neighbors.push_back(node.neighbors_layer1[i]);
        }
        return neighbors;
    }
    
    bool is_in_layer1(int node_id) const {
        if (node_id < 0 || node_id >= static_cast<int>(num_elements_)) {
            return false;
        }
        return nodes_[node_id].in_layer1;
    }
    
private:
    // 图结构 - ✅ 优化：使用固定大小数组
    struct Node {
        int* neighbors_layer0;  // 第0层邻居（固定大小）
        int* neighbors_layer1;  // 第1层邻居（固定大小，如果在第1层）
        int num_neighbors_layer0;  // 第0层实际邻居数
        int num_neighbors_layer1;  // 第1层实际邻居数
        bool in_layer1 = false;
        
        Node() : neighbors_layer0(nullptr), neighbors_layer1(nullptr),
                 num_neighbors_layer0(0), num_neighbors_layer1(0), in_layer1(false) {}
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
    
    // ✅ 优化：预分配的visited位图（避免unordered_set的hash开销）
    // ⚠️ 重要：使用int而不是char，避免溢出问题
    std::vector<int> visited_bitmap_;
    int visited_version_;  // 版本号，用于快速重置visited
    
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
    
    // ✅ 快速Layer1搜索（用于构建时，不等稳定）
    std::vector<int> search_layer1_fast(const float* query, size_t k, size_t ef_search);
    
    // 第0层多入口并行搜索
    std::vector<int> search_layer0_multi_entry(const float* query, const std::vector<int>& entry_points,
                                               size_t k, size_t ef_search, size_t& visited_count);
};

} // namespace hnsw

