#include "hnsw.h"
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <unordered_set>

namespace hnsw {

HNSW::HNSW(size_t dimension, size_t M0, size_t ef_construction, 
                           size_t max_elements, int seed)
    : dimension_(dimension), M0_(M0), M1_(M0 / 2), ef_construction_(ef_construction),
      max_elements_(max_elements), num_elements_(0), rng_(seed), level_dist_(0.0, 1.0) {
    
    // 计算层级选择参数（与标准HNSW一致）
    // 标准HNSW: ml = 1/ln(M)，使得 P(level >= l) = (1/M)^l
    ml_ = 1.0 / std::log(static_cast<double>(M0_));
    
    // 预分配内存
    nodes_.reserve(max_elements);
    data_ = nullptr;
}

HNSW::~HNSW() {
    if (data_ != nullptr) {
        delete[] data_;
    }
}

int HNSW::select_layer_randomly() {
    // 使用与标准HNSW相同的概率分布选择层级
    // P(layer >= 1) = 1/M0
    float r = level_dist_(rng_);
    int layer = static_cast<int>(-std::log(r) * ml_);
    return (layer >= 1) ? 1 : 0;  // 只有0层和1层
}

void HNSW::build(const float* data, size_t num_vectors) {
    std::cout << "开始构建2层HNSW索引：" << num_vectors << " 个向量" << std::endl;
    
    num_elements_ = num_vectors;
    
    // 复制数据
    data_ = new float[num_elements_ * dimension_];
    std::copy(data, data + num_elements_ * dimension_, data_);
    
    // 初始化节点
    nodes_.resize(num_elements_);
    
    // 第一步：确定哪些节点在第1层
    std::cout << "确定第1层节点..." << std::endl;
    for (size_t i = 0; i < num_elements_; ++i) {
        int layer = select_layer_randomly();
        if (layer >= 1) {
            nodes_[i].in_layer1 = true;
            layer1_nodes_.push_back(i);
        }
        
        if ((i + 1) % 10000 == 0) {
            std::cout << "  进度: " << (i + 1) << "/" << num_elements_ << std::endl;
        }
    }
    
    std::cout << "第1层节点数: " << layer1_nodes_.size() << " / " << num_elements_ 
              << " (" << (100.0 * layer1_nodes_.size() / num_elements_) << "%)" << std::endl;
    
    // 第二步：构建第1层连接
    if (!layer1_nodes_.empty()) {
        std::cout << "构建第1层连接..." << std::endl;
        for (size_t i = 0; i < layer1_nodes_.size(); ++i) {
            int node_id = layer1_nodes_[i];
            connect_neighbors_layer1(node_id, ef_construction_);
            
            if ((i + 1) % 1000 == 0) {
                std::cout << "  进度: " << (i + 1) << "/" << layer1_nodes_.size() << std::endl;
            }
        }
    }
    
    // 第三步：构建第0层连接
    std::cout << "构建第0层连接..." << std::endl;
    for (size_t i = 0; i < num_elements_; ++i) {
        connect_neighbors_layer0(i, ef_construction_);
        
        if ((i + 1) % 10000 == 0) {
            std::cout << "  进度: " << (i + 1) << "/" << num_elements_ << std::endl;
        }
    }
    
    std::cout << "索引构建完成！" << std::endl;
}

void HNSW::connect_neighbors_layer1(int node_id, int ef) {
    if (layer1_nodes_.empty() || layer1_nodes_.size() == 1) {
        return;
    }
    
    const float* query = data_ + node_id * dimension_;
    std::vector<Neighbor> candidates;
    
    // 搜索第1层中已有的节点
    for (int other_id : layer1_nodes_) {
        if (other_id >= node_id) break;  // 只连接已插入的节点
        
        float dist = distance(other_id, query);
        candidates.push_back(Neighbor(other_id, dist));
    }
    
    if (candidates.empty()) {
        return;
    }
    
    // 选择最近的M1个邻居
    std::sort(candidates.begin(), candidates.end());
    size_t num_neighbors = std::min(M1_, candidates.size());
    
    for (size_t i = 0; i < num_neighbors; ++i) {
        int neighbor_id = candidates[i].id;
        
        // 添加双向边
        nodes_[node_id].neighbors_layer1.push_back(neighbor_id);
        nodes_[neighbor_id].neighbors_layer1.push_back(node_id);
        
        // 修剪邻居的连接（保持出度不超过M1）
        if (nodes_[neighbor_id].neighbors_layer1.size() > M1_) {
            // 重新评估邻居的所有连接，保留最好的M1个
            const float* neighbor_vec = data_ + neighbor_id * dimension_;
            std::vector<Neighbor> neighbor_candidates;
            
            for (int nn : nodes_[neighbor_id].neighbors_layer1) {
                float dist = distance(nn, neighbor_vec);
                neighbor_candidates.push_back(Neighbor(nn, dist));
            }
            
            std::sort(neighbor_candidates.begin(), neighbor_candidates.end());
            nodes_[neighbor_id].neighbors_layer1.clear();
            
            for (size_t j = 0; j < M1_ && j < neighbor_candidates.size(); ++j) {
                nodes_[neighbor_id].neighbors_layer1.push_back(neighbor_candidates[j].id);
            }
        }
    }
}

void HNSW::connect_neighbors_layer0(int node_id, int ef) {
    if (node_id == 0) {
        return;  // 第一个节点没有邻居
    }
    
    const float* query = data_ + node_id * dimension_;
    std::vector<Neighbor> candidates;
    
    // 搜索第0层中已有的节点
    for (int other_id = 0; other_id < node_id; ++other_id) {
        float dist = distance(other_id, query);
        candidates.push_back(Neighbor(other_id, dist));
    }
    
    // 选择最近的M0个邻居
    std::sort(candidates.begin(), candidates.end());
    size_t num_neighbors = std::min(M0_, candidates.size());
    
    for (size_t i = 0; i < num_neighbors; ++i) {
        int neighbor_id = candidates[i].id;
        
        // 添加双向边
        nodes_[node_id].neighbors_layer0.push_back(neighbor_id);
        nodes_[neighbor_id].neighbors_layer0.push_back(node_id);
        
        // 修剪邻居的连接（保持出度不超过M0）
        if (nodes_[neighbor_id].neighbors_layer0.size() > M0_) {
            const float* neighbor_vec = data_ + neighbor_id * dimension_;
            std::vector<Neighbor> neighbor_candidates;
            
            for (int nn : nodes_[neighbor_id].neighbors_layer0) {
                float dist = distance(nn, neighbor_vec);
                neighbor_candidates.push_back(Neighbor(nn, dist));
            }
            
            std::sort(neighbor_candidates.begin(), neighbor_candidates.end());
            nodes_[neighbor_id].neighbors_layer0.clear();
            
            for (size_t j = 0; j < M0_ && j < neighbor_candidates.size(); ++j) {
                nodes_[neighbor_id].neighbors_layer0.push_back(neighbor_candidates[j].id);
            }
        }
    }
}

float HNSW::distance(int node_id, const float* query) const {
    return l2_distance(data_ + node_id * dimension_, query, dimension_);
}

std::vector<int> HNSW::search_layer1_stable(const float* query, size_t k, size_t ef_search, size_t& visited_count) {
    visited_count = 0;
    
    // 如果第1层为空，返回空
    if (layer1_nodes_.empty()) {
        return std::vector<int>();
    }
    
    // 初始化搜索状态
    std::priority_queue<Neighbor> top_k;  // max heap
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates;  // min heap
    std::unordered_set<int> visited;
    
    // 随机选择一个第1层节点作为入口
    int entry = layer1_nodes_[rng_() % layer1_nodes_.size()];
    float entry_dist = distance(entry, query);
    
    candidates.push(Neighbor(entry, entry_dist));
    visited.insert(entry);
    top_k.push(Neighbor(entry, entry_dist));
    visited_count++;
    
    std::vector<int> prev_top_k;
    int stable_rounds = 0;
    const int required_stable_rounds = 3;  // 需要连续3轮top-k不变
    
    // 搜索直到top-k稳定
    while (!candidates.empty() && stable_rounds < required_stable_rounds) {
        Neighbor current = candidates.top();
        candidates.pop();
        
        // 如果当前节点比top-k中最远的还远，且候选池足够大，检查是否稳定
        if (top_k.size() >= k && current.distance > top_k.top().distance) {
            // 记录当前top-k
            std::vector<int> current_top_k;
            std::priority_queue<Neighbor> temp_top_k = top_k;
            while (!temp_top_k.empty()) {
                current_top_k.push_back(temp_top_k.top().id);
                temp_top_k.pop();
            }
            std::sort(current_top_k.begin(), current_top_k.end());
            
            // 比较是否与上一轮相同
            if (current_top_k == prev_top_k) {
                stable_rounds++;
            } else {
                stable_rounds = 0;
                prev_top_k = current_top_k;
            }
            
            // 如果不够稳定，继续搜索
            if (stable_rounds < required_stable_rounds && candidates.empty()) {
                // 没有候选了但还不稳定，可能需要扩展搜索
                break;
            }
        }
        
        // 检查邻居
        for (int neighbor_id : nodes_[current.id].neighbors_layer1) {
            if (visited.find(neighbor_id) != visited.end()) {
                continue;
            }
            
            visited.insert(neighbor_id);
            visited_count++;
            float dist = distance(neighbor_id, query);
            
            if (top_k.size() < ef_search || dist < top_k.top().distance) {
                candidates.push(Neighbor(neighbor_id, dist));
                top_k.push(Neighbor(neighbor_id, dist));
                
                if (top_k.size() > ef_search) {
                    top_k.pop();
                }
            }
        }
    }
    
    // 返回top-k作为第0层的入口点
    std::vector<Neighbor> results;
    while (!top_k.empty() && results.size() < k) {
        results.push_back(top_k.top());
        top_k.pop();
    }
    
    std::sort(results.begin(), results.end());
    std::vector<int> entry_points;
    for (const auto& n : results) {
        entry_points.push_back(n.id);
    }
    
    return entry_points;
}

std::vector<int> HNSW::search_layer0_multi_entry(const float* query, 
                                                          const std::vector<int>& entry_points,
                                                          size_t k, size_t ef_search, size_t& visited_count) {
    visited_count = 0;
    
    // 确保ef_search至少为k
    ef_search = std::max(ef_search, k);
    
    // 初始化搜索状态
    // W: 候选集合（最多ef_search个最好的候选），使用max heap（距离大的在top）
    std::priority_queue<Neighbor> W;  // max heap
    // candidates: 待扩展的节点，使用min heap（距离小的优先扩展）
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates;
    std::unordered_set<int> visited;
    
    // 将所有入口点加入候选队列和W
    for (int entry_id : entry_points) {
        if (visited.find(entry_id) != visited.end()) {
            continue;
        }
        
        float dist = distance(entry_id, query);
        candidates.push(Neighbor(entry_id, dist));
        W.push(Neighbor(entry_id, dist));
        visited.insert(entry_id);
        visited_count++;
    }
    
    // 主搜索循环
    while (!candidates.empty()) {
        Neighbor current = candidates.top();
        candidates.pop();
        
        // 停止条件：如果当前节点距离大于W中最远的距离，停止
        // （因为不可能再找到更好的节点了）
        if (current.distance > W.top().distance) {
            break;
        }
        
        // 扩展当前节点的所有邻居
        for (int neighbor_id : nodes_[current.id].neighbors_layer0) {
            if (visited.find(neighbor_id) != visited.end()) {
                continue;
            }
            
            visited.insert(neighbor_id);
            visited_count++;
            float dist = distance(neighbor_id, query);
            
            // 如果W还没满，或者这个邻居比W中最远的还近，就加入W
            if (W.size() < ef_search || dist < W.top().distance) {
                candidates.push(Neighbor(neighbor_id, dist));
                W.push(Neighbor(neighbor_id, dist));
                
                // 保持W的大小不超过ef_search
                if (W.size() > ef_search) {
                    W.pop();
                }
            }
        }
    }
    
    // 从W中取出最好的k个结果
    std::vector<Neighbor> results;
    while (!W.empty()) {
        results.push_back(W.top());
        W.pop();
    }
    
    // 按距离排序（从近到远）
    std::sort(results.begin(), results.end());
    
    // 只返回前k个
    std::vector<int> result_ids;
    for (size_t i = 0; i < std::min(k, results.size()); ++i) {
        result_ids.push_back(results[i].id);
    }
    
    return result_ids;
}

SearchResult HNSW::search(const float* query, size_t k, size_t ef_search, 
                                      size_t num_entry_points) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SearchResult result;
    result.layer1_visited = 0;
    result.layer0_visited = 0;
    
    if (num_elements_ == 0) {
        result.visited_count = 0;
        result.latency_us = 0.0;
        return result;
    }
    
    // 阶段1：在第1层搜索，获取多个入口点
    std::vector<int> entry_points;
    
    if (!layer1_nodes_.empty() && num_entry_points > 1) {
        entry_points = search_layer1_stable(query, num_entry_points, ef_search, result.layer1_visited);
        #ifdef DEBUG
        std::cout << "Layer1搜索完成，找到" << entry_points.size() << "个入口点: ";
        for (int ep : entry_points) std::cout << ep << " ";
        std::cout << std::endl;
        #endif
    }
    
    // 如果第1层没有找到足够的入口点，使用第0层的随机节点作为入口
    if (entry_points.empty()) {
        // 使用全局最优的策略：从第0层选择一个随机节点作为入口
        // 更好的策略：选择多个随机节点
        for (size_t i = 0; i < std::min(num_entry_points, num_elements_); ++i) {
            entry_points.push_back(rng_() % num_elements_);
        }
    }
    
    // 阶段2：在第0层从多个入口点开始并行搜索
    result.neighbors = search_layer0_multi_entry(query, entry_points, k, ef_search, result.layer0_visited);
    result.visited_count = result.layer1_visited + result.layer0_visited;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.latency_us = duration.count();
    
    return result;
}

std::vector<SearchResult> HNSW::batch_search(const float* queries, size_t num_queries, size_t k, 
                                           size_t ef_search, size_t num_entry_points) {
    std::vector<SearchResult> results;
    results.reserve(num_queries);
    
    for (size_t i = 0; i < num_queries; ++i) {
        const float* query = queries + i * dimension_;
        results.push_back(search(query, k, ef_search, num_entry_points));
    }
    
    return results;
}

double HNSW::compute_recall(const std::vector<int>& results, const std::vector<int>& ground_truth, size_t k) {
    if (results.empty() || ground_truth.empty()) {
        return 0.0;
    }
    
    // 取前k个结果
    size_t actual_k = std::min(k, std::min(results.size(), ground_truth.size()));
    
    std::unordered_set<int> gt_set;
    for (size_t i = 0; i < actual_k; ++i) {
        gt_set.insert(ground_truth[i]);
    }
    
    size_t matches = 0;
    for (size_t i = 0; i < actual_k && i < results.size(); ++i) {
        if (gt_set.find(results[i]) != gt_set.end()) {
            matches++;
        }
    }
    
    return static_cast<double>(matches) / actual_k;
}

} // namespace hnsw

