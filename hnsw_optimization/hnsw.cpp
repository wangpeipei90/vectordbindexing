#include "hnsw.h"
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <unordered_set>

namespace hnsw {

HNSW::HNSW(size_t dimension, size_t M0, size_t ef_construction, 
                           size_t max_elements, DistanceType distance_type, int seed)
    : dimension_(dimension), M0_(M0), M1_(M0 / 2), ef_construction_(ef_construction),
      max_elements_(max_elements), distance_type_(distance_type), num_elements_(0), 
      rng_(seed), level_dist_(0.0, 1.0), visited_version_(1) {
    
    // 计算层级选择参数（与标准HNSW一致）
    // 标准HNSW: ml = 1/ln(M)，使得 P(level >= l) = (1/M)^l
    ml_ = 1.0 / std::log(static_cast<double>(M0_));
    
    // 预分配内存
    nodes_.reserve(max_elements);
    data_ = nullptr;
    
    // ✅ 优化：预分配visited位图
    visited_bitmap_.resize(max_elements, 0);
}

HNSW::~HNSW() {
    if (data_ != nullptr) {
        delete[] data_;
    }
    
    // ✅ 释放邻居数组内存
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].neighbors_layer0 != nullptr) {
            delete[] nodes_[i].neighbors_layer0;
        }
        if (nodes_[i].neighbors_layer1 != nullptr) {
            delete[] nodes_[i].neighbors_layer1;
        }
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
    
    // ✅ 优化：为所有节点预分配固定大小的邻居数组（避免动态扩容）
    // 使用Mmax = 2*M作为安全边界（标准HNSW做法）
    size_t Mmax0 = 2 * M0_;
    size_t Mmax1 = 2 * M1_;
    
    // 第一步：确定哪些节点在第1层并分配内存
    std::cout << "确定第1层节点并分配内存..." << std::endl;
    for (size_t i = 0; i < num_elements_; ++i) {
        int layer = select_layer_randomly();
        
        // 为第0层分配邻居数组
        nodes_[i].neighbors_layer0 = new int[Mmax0];
        nodes_[i].num_neighbors_layer0 = 0;
        
        if (layer >= 1) {
            nodes_[i].in_layer1 = true;
            layer1_nodes_.push_back(i);
            // 为第1层分配邻居数组
            nodes_[i].neighbors_layer1 = new int[Mmax1];
            nodes_[i].num_neighbors_layer1 = 0;
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
    
    // 安全检查：节点必须在第1层且邻居数组已分配
    if (!nodes_[node_id].in_layer1 || nodes_[node_id].neighbors_layer1 == nullptr) {
        return;
    }
    
    const float* query = data_ + node_id * dimension_;
    
    // ✅ 优化1：使用visited位图替代unordered_set
    size_t current_version = ++visited_version_;
    
    // ✅ 优化2：使用固定大小的候选数组而不是priority_queue
    std::priority_queue<Neighbor> W;  // max heap - 候选集
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates_heap;  // min heap
    
    // ✅ 优化3：找入口点 - 使用最后一个Layer1节点（已经排序）
    int entry = -1;
    if (!layer1_nodes_.empty() && layer1_nodes_.back() < node_id) {
        entry = layer1_nodes_.back();
    } else if (layer1_nodes_.size() > 1) {
        // 找到第一个小于node_id的
        for (int i = layer1_nodes_.size() - 1; i >= 0; --i) {
            if (layer1_nodes_[i] < node_id) {
                entry = layer1_nodes_[i];
                break;
            }
        }
    }
    
    if (entry == -1) {
        return;  // 没有可用的入口点
    }
    
    // 安全检查：entry 必须有效且在第1层
    if (entry < 0 || entry >= static_cast<int>(num_elements_) || 
        !nodes_[entry].in_layer1 || nodes_[entry].neighbors_layer1 == nullptr) {
        return;
    }
    
    float entry_dist = distance(entry, query);
    candidates_heap.push(Neighbor(entry, entry_dist));
    W.push(Neighbor(entry, entry_dist));
    
    // 安全访问 visited_bitmap_
    if (entry < static_cast<int>(visited_bitmap_.size())) {
        visited_bitmap_[entry] = current_version;
    }
    
    // 贪心搜索Layer1图
    while (!candidates_heap.empty()) {
        Neighbor current = candidates_heap.top();
        candidates_heap.pop();
        
        // 安全检查：current.id 必须有效、在第1层且邻居数组已分配
        if (current.id < 0 || current.id >= static_cast<int>(num_elements_) ||
            !nodes_[current.id].in_layer1 || nodes_[current.id].neighbors_layer1 == nullptr) {
            continue;
        }
        
        // 提前终止条件
        if (current.distance > W.top().distance) {
            break;
        }
        
        // 扩展邻居
        int* neighbors = nodes_[current.id].neighbors_layer1;
        int num_neighbors = nodes_[current.id].num_neighbors_layer1;
        
        // 安全检查：邻居数必须在合理范围
        if (num_neighbors < 0 || num_neighbors > (int)(2 * M1_)) {
            num_neighbors = std::min(std::max(0, num_neighbors), (int)(2 * M1_));
        }
        
        for (int i = 0; i < num_neighbors; ++i) {
            int neighbor_id = neighbors[i];
            
            // 🔧 安全检查：跳过无效的邻居ID
            if (neighbor_id < 0 || neighbor_id >= static_cast<int>(num_elements_)) {
                continue;
            }
            
            if (neighbor_id >= node_id) continue;  // 只访问已插入的节点
            if (visited_bitmap_[neighbor_id] == current_version) continue;  // O(1)查重
            
            visited_bitmap_[neighbor_id] = current_version;
            float dist = distance(neighbor_id, query);
            
            if (W.size() < (size_t)ef || dist < W.top().distance) {
                candidates_heap.push(Neighbor(neighbor_id, dist));
                W.push(Neighbor(neighbor_id, dist));
                
                if (W.size() > (size_t)ef) {
                    W.pop();
                }
            }
        }
    }
    
    // 从W中提取候选并排序
    std::vector<Neighbor> candidates;
    candidates.reserve(W.size());
    while (!W.empty()) {
        candidates.push_back(W.top());
        W.pop();
    }
    std::sort(candidates.begin(), candidates.end());
    
    // 选择最近的M1个邻居
    size_t num_neighbors_to_add = std::min(M1_, candidates.size());
    size_t Mmax1 = 2 * M1_;  // 数组容量 = 64
    
    for (size_t i = 0; i < num_neighbors_to_add; ++i) {
        // 安全检查：确保索引在范围内
        if (i >= candidates.size()) {
            break;
        }
        
        int neighbor_id = candidates[i].id;
        
        // 检查 neighbor_id 范围和第1层状态
        if (neighbor_id < 0 || neighbor_id >= static_cast<int>(num_elements_) ||
            !nodes_[neighbor_id].in_layer1 || nodes_[neighbor_id].neighbors_layer1 == nullptr) {
            continue;
        }
        
        Node& neighbor_node = nodes_[neighbor_id];
        
        // 添加正向边到当前节点（检查容量）
        if (nodes_[node_id].num_neighbors_layer1 >= (int)Mmax1) {
            continue;  // 数组满，跳过
        }
        nodes_[node_id].neighbors_layer1[nodes_[node_id].num_neighbors_layer1++] = neighbor_id;
        
        // 添加反向边：先修剪后添加策略
        if (neighbor_node.num_neighbors_layer1 >= (int)M1_) {
            // 已经有 >= M1_ 个邻居，收集所有候选并选择最近的 M1_ 个
            const float* neighbor_vec = data_ + neighbor_id * dimension_;
            
            std::vector<Neighbor> all_candidates;
            int safe_count = std::min(neighbor_node.num_neighbors_layer1, (int)Mmax1);
            all_candidates.reserve(safe_count + 1);
            
            // 添加现有邻居
            for (int j = 0; j < safe_count; ++j) {
                int nn = neighbor_node.neighbors_layer1[j];
                if (nn >= 0 && nn < static_cast<int>(num_elements_)) {
                    float dist = distance(nn, neighbor_vec);
                    all_candidates.push_back(Neighbor(nn, dist));
                }
            }
            
            // 添加新邻居
            float dist_to_new = distance(node_id, neighbor_vec);
            all_candidates.push_back(Neighbor(node_id, dist_to_new));
            
            // 选择最近的 M1_ 个
            if (!all_candidates.empty()) {
                size_t keep = std::min(M1_, all_candidates.size());
                if (keep < all_candidates.size()) {
                    std::nth_element(all_candidates.begin(), 
                                    all_candidates.begin() + keep,
                                    all_candidates.end());
                }
                
                // 更新邻居列表
                neighbor_node.num_neighbors_layer1 = keep;
                for (size_t j = 0; j < keep; ++j) {
                    neighbor_node.neighbors_layer1[j] = all_candidates[j].id;
                }
            }
        } else {
            // 邻居数 < M1_，直接添加
            neighbor_node.neighbors_layer1[neighbor_node.num_neighbors_layer1++] = node_id;
        }
    }
}

void HNSW::connect_neighbors_layer0(int node_id, int ef) {
    if (node_id == 0) {
        return;  // 第一个节点没有邻居
    }
    
    const float* query = data_ + node_id * dimension_;
    
    // ✅ 优化1：使用visited位图替代unordered_set
    size_t current_version = ++visited_version_;
    
    std::priority_queue<Neighbor> W;  // max heap
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates_heap;
    
    // 策略1：如果Layer1已经有节点，先在Layer1搜索找到好的入口点
    std::vector<int> entry_points;
    if (!layer1_nodes_.empty()) {
        // 在Layer1中快速搜索
        size_t layer1_size = 0;
        for (int l1_node : layer1_nodes_) {
            if (l1_node < node_id) layer1_size++;
            else break;
        }
        
        if (layer1_size > 0) {
            entry_points = search_layer1_fast(query, std::min((size_t)10, layer1_size), 
                                             std::min((size_t)50, layer1_size));
            
            // 使用Layer1搜索结果作为Layer0的入口点
            for (int ep : entry_points) {
                if (ep >= node_id) continue;
                float dist = distance(ep, query);
                candidates_heap.push(Neighbor(ep, dist));
                W.push(Neighbor(ep, dist));
                visited_bitmap_[ep] = current_version;
            }
        }
    }
    
    // 策略2：如果还没有入口点，使用少量随机入口 + 图遍历
    if (W.empty()) {
        // ✅ 关键优化：即使是早期节点，也使用图遍历而不是暴力
        // 使用更多随机入口点来保证覆盖率
        int num_random = node_id < 100 ? std::min(20, node_id) :  // 早期多一点
                        node_id < 1000 ? std::min(15, node_id) :  // 中期
                        std::min(10, node_id);  // 后期
        
        for (int i = 0; i < num_random; ++i) {
            int entry = rng_() % node_id;
            if (visited_bitmap_[entry] == current_version) continue;  // O(1)查重
            
            float entry_dist = distance(entry, query);
            candidates_heap.push(Neighbor(entry, entry_dist));
            W.push(Neighbor(entry, entry_dist));
            visited_bitmap_[entry] = current_version;
            
            if (W.size() > (size_t)ef) {
                W.pop();
            }
        }
    }
    
    // ✅ 优化3：受限的贪心搜索，严格控制搜索宽度
    // 使用ef作为硬性限制，不允许无限扩展
    while (!candidates_heap.empty()) {
        Neighbor current = candidates_heap.top();
        candidates_heap.pop();
        
        // 提前终止：当前候选比W中最远的还远
        if (current.distance > W.top().distance) {
            break;
        }
        
        // 扩展邻居（使用新的数组结构）
        int* neighbors = nodes_[current.id].neighbors_layer0;
        int num_neighbors = nodes_[current.id].num_neighbors_layer0;
        
        for (int i = 0; i < num_neighbors; ++i) {
            int neighbor_id = neighbors[i];
            
            // 🔧 安全检查：跳过无效的邻居ID
            if (neighbor_id < 0 || neighbor_id >= static_cast<int>(num_elements_)) {
                continue;
            }
            
            if (neighbor_id >= node_id) continue;  // 只访问已插入的节点
            if (visited_bitmap_[neighbor_id] == current_version) continue;  // O(1)查重
            
            visited_bitmap_[neighbor_id] = current_version;
            float dist = distance(neighbor_id, query);
            
            if (W.size() < (size_t)ef || dist < W.top().distance) {
                candidates_heap.push(Neighbor(neighbor_id, dist));
                W.push(Neighbor(neighbor_id, dist));
                
                if (W.size() > (size_t)ef) {
                    W.pop();
                }
            }
        }
    }
    
    // 从W中提取候选并排序
    std::vector<Neighbor> candidates;
    candidates.reserve(W.size());
    while (!W.empty()) {
        candidates.push_back(W.top());
        W.pop();
    }
    std::sort(candidates.begin(), candidates.end());
    
    // 选择最近的M0个邻居并添加边
    size_t num_neighbors_to_add = std::min(M0_, candidates.size());
    
    for (size_t i = 0; i < num_neighbors_to_add; ++i) {
        int neighbor_id = candidates[i].id;
        
        // 添加正向边
        nodes_[node_id].neighbors_layer0[nodes_[node_id].num_neighbors_layer0++] = neighbor_id;
        
        // 添加反向边
        Node& neighbor_node = nodes_[neighbor_id];
        neighbor_node.neighbors_layer0[neighbor_node.num_neighbors_layer0++] = node_id;
        
        // ✅ 优化4：只在超过M0时才修剪
        if (neighbor_node.num_neighbors_layer0 > (int)M0_) {
            // 修剪：使用nth_element部分排序（O(n)）
            const float* neighbor_vec = data_ + neighbor_id * dimension_;
            std::vector<Neighbor> neighbor_candidates;
            neighbor_candidates.reserve(neighbor_node.num_neighbors_layer0);
            
            for (int j = 0; j < neighbor_node.num_neighbors_layer0; ++j) {
                int nn = neighbor_node.neighbors_layer0[j];
                float dist = distance(nn, neighbor_vec);
                neighbor_candidates.push_back(Neighbor(nn, dist));
            }
            
            // 部分排序：找到最近的M0个
            std::nth_element(neighbor_candidates.begin(), 
                            neighbor_candidates.begin() + M0_,
                            neighbor_candidates.end());
            
            // 只保留最近的M0个
            neighbor_node.num_neighbors_layer0 = M0_;
            for (size_t j = 0; j < M0_; ++j) {
                neighbor_node.neighbors_layer0[j] = neighbor_candidates[j].id;
            }
        }
    }
}

float HNSW::distance(int node_id, const float* query) const {
    // 🔧 安全检查：防止访问越界
    if (node_id < 0 || node_id >= static_cast<int>(num_elements_)) {
        return std::numeric_limits<float>::max();  // 返回最大距离
    }
    
    const float* node_data = data_ + node_id * dimension_;
    
    switch (distance_type_) {
        case DistanceType::IP:
            return ip_distance(node_data, query, dimension_);
        case DistanceType::L2:
        default:
            return l2_distance(node_data, query, dimension_);
    }
}

// ✅ 快速Layer1搜索（用于构建时，不等稳定）
std::vector<int> HNSW::search_layer1_fast(const float* query, size_t k, size_t ef_search) {
    if (layer1_nodes_.empty()) {
        return std::vector<int>();
    }
    
    // ✅ 优化：使用visited位图
    size_t current_version = ++visited_version_;
    
    std::priority_queue<Neighbor> W;
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates;
    
    // 选择多个随机入口点（提高搜索覆盖率）
    int num_entries = std::min(3, (int)layer1_nodes_.size());
    for (int i = 0; i < num_entries; ++i) {
        int entry = layer1_nodes_[rng_() % layer1_nodes_.size()];
        if (visited_bitmap_[entry] == current_version) continue;
        
        float entry_dist = distance(entry, query);
        candidates.push(Neighbor(entry, entry_dist));
        W.push(Neighbor(entry, entry_dist));
        visited_bitmap_[entry] = current_version;
        
        if (W.size() > ef_search) {
            W.pop();
        }
    }
    
    // 贪心搜索
    while (!candidates.empty()) {
        Neighbor current = candidates.top();
        candidates.pop();
        
        if (current.distance > W.top().distance) {
            break;
        }
        
        // 使用新的数组结构
        int* neighbors = nodes_[current.id].neighbors_layer1;
        int num_neighbors = nodes_[current.id].num_neighbors_layer1;
        
        for (int i = 0; i < num_neighbors; ++i) {
            int neighbor_id = neighbors[i];
            if (visited_bitmap_[neighbor_id] == current_version) {
                continue;
            }
            
            visited_bitmap_[neighbor_id] = current_version;
            float dist = distance(neighbor_id, query);
            
            if (W.size() < ef_search || dist < W.top().distance) {
                candidates.push(Neighbor(neighbor_id, dist));
                W.push(Neighbor(neighbor_id, dist));
                
                if (W.size() > ef_search) {
                    W.pop();
                }
            }
        }
    }
    
    // 提取top-k
    std::vector<Neighbor> results;
    results.reserve(k);
    while (!W.empty() && results.size() < k) {
        results.push_back(W.top());
        W.pop();
    }
    
    std::sort(results.begin(), results.end());
    std::vector<int> entry_points;
    entry_points.reserve(results.size());
    for (const auto& n : results) {
        entry_points.push_back(n.id);
    }
    
    return entry_points;
}

std::vector<int> HNSW::search_layer1_stable(const float* query, size_t k, size_t ef_search, size_t& visited_count) {
    visited_count = 0;
    
    // 如果第1层为空，返回空
    if (layer1_nodes_.empty()) {
        return std::vector<int>();
    }
    
    // ✅ 优化：使用visited位图
    size_t current_version = ++visited_version_;
    
    // 初始化搜索状态
    std::priority_queue<Neighbor> top_k;  // max heap
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates;  // min heap
    
    // 🔧 修复：Query-guided多入口点选择
    // 采样大量候选，选择距离最近的作为初始入口点
    size_t num_samples = std::min((size_t)100, layer1_nodes_.size());  // 采样100个
    size_t num_initial_entries = std::min((size_t)20, layer1_nodes_.size());  // 选择最近的20个
    
    std::vector<Neighbor> sampled_entries;
    sampled_entries.reserve(num_samples);
    std::unordered_set<int> sampled_set;
    
    // 采样并计算距离
    while (sampled_set.size() < num_samples) {
        int entry = layer1_nodes_[rng_() % layer1_nodes_.size()];
        if (sampled_set.insert(entry).second) {
            float entry_dist = distance(entry, query);
            sampled_entries.push_back(Neighbor(entry, entry_dist));
        }
    }
    
    // 选择距离最近的 num_initial_entries 个作为初始入口点
    std::sort(sampled_entries.begin(), sampled_entries.end());
    for (size_t i = 0; i < std::min(num_initial_entries, sampled_entries.size()); ++i) {
        int entry = sampled_entries[i].id;
        float entry_dist = sampled_entries[i].distance;
        
        candidates.push(Neighbor(entry, entry_dist));
        visited_bitmap_[entry] = current_version;
        top_k.push(Neighbor(entry, entry_dist));
        visited_count++;
        
        // 限制top_k大小
        if (top_k.size() > ef_search) {
            top_k.pop();
        }
    }
    
    // 🔧 修复：充分搜索直到 ef_search，不提前终止
    // 移除"稳定性检查"以提高 recall
    while (!candidates.empty()) {
        Neighbor current = candidates.top();
        candidates.pop();
        
        // 标准 HNSW 终止条件：当前候选距离大于 top-k 中最远的距离
        if (top_k.size() >= ef_search && current.distance > top_k.top().distance) {
            break;
        }
        
        // 检查邻居（使用新的数组结构）
        int* neighbors = nodes_[current.id].neighbors_layer1;
        int num_neighbors = nodes_[current.id].num_neighbors_layer1;
        
        for (int i = 0; i < num_neighbors; ++i) {
            int neighbor_id = neighbors[i];
            if (visited_bitmap_[neighbor_id] == current_version) {
                continue;
            }
            
            visited_bitmap_[neighbor_id] = current_version;
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
    
    // ✅ 优化：使用visited位图
    size_t current_version = ++visited_version_;
    
    // 初始化搜索状态
    // W: 候选集合（最多ef_search个最好的候选），使用max heap（距离大的在top）
    std::priority_queue<Neighbor> W;  // max heap
    // candidates: 待扩展的节点，使用min heap（距离小的优先扩展）
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates;
    
    // 将所有入口点加入候选队列和W
    #ifdef DEBUG_SEARCH
    std::cout << "[DEBUG] 搜索开始，current_version=" << current_version 
              << ", visited_bitmap_.size()=" << visited_bitmap_.size()
              << ", num_elements_=" << num_elements_ << std::endl;
    #endif
    
    for (int entry_id : entry_points) {
        #ifdef DEBUG_SEARCH
        std::cout << "[DEBUG] 处理入口点 " << entry_id 
                  << ", visited_bitmap[" << entry_id << "]=" << (int)visited_bitmap_[entry_id]
                  << ", current_version=" << current_version << std::endl;
        #endif
        
        if (visited_bitmap_[entry_id] == current_version) {
            continue;
        }
        
        float dist = distance(entry_id, query);
        candidates.push(Neighbor(entry_id, dist));
        W.push(Neighbor(entry_id, dist));
        visited_bitmap_[entry_id] = current_version;
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
        
        // 扩展当前节点的所有邻居（使用新的数组结构）
        int* neighbors = nodes_[current.id].neighbors_layer0;
        int num_neighbors = nodes_[current.id].num_neighbors_layer0;
        
        for (int i = 0; i < num_neighbors; ++i) {
            int neighbor_id = neighbors[i];
            
            #ifdef DEBUG_SEARCH
            if (visited_count < 10) {  // 只打印前几次
                std::cout << "[DEBUG]   检查邻居 " << neighbor_id 
                          << ", visited_bitmap[" << neighbor_id << "]=" << (int)visited_bitmap_[neighbor_id]
                          << ", current_version=" << current_version << std::endl;
            }
            #endif
            
            if (visited_bitmap_[neighbor_id] == current_version) {
                continue;
            }
            
            visited_bitmap_[neighbor_id] = current_version;
            visited_count++;
            float dist = distance(neighbor_id, query);
            
            // 如果W还没满，或者这个邻居比W中最远的还近，就加入W
            if (W.size() < ef_search || dist < W.top().distance) {
                candidates.push(Neighbor(neighbor_id, dist));
                W.push(Neighbor(neighbor_id, dist));
                
                #ifdef DEBUG_SEARCH
                if (visited_count < 10) {
                    std::cout << "[DEBUG]   添加邻居 " << neighbor_id << " 到W, 当前W大小=" << W.size() << std::endl;
                }
                #endif
                
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
    
    #ifdef DEBUG_SEARCH
    std::cout << "[DEBUG] W中有 " << results.size() << " 个候选，前10个: ";
    for (size_t i = 0; i < std::min((size_t)10, results.size()); ++i) {
        std::cout << results[i].id << " ";
    }
    std::cout << std::endl;
    #endif
    
    // 按距离排序（从近到远）
    std::sort(results.begin(), results.end());
    
    #ifdef DEBUG_SEARCH
    std::cout << "[DEBUG] 排序后前10个: ";
    for (size_t i = 0; i < std::min((size_t)10, results.size()); ++i) {
        std::cout << results[i].id << " ";
    }
    std::cout << std::endl;
    #endif
    
    // 只返回前k个
    std::vector<int> result_ids;
    for (size_t i = 0; i < std::min(k, results.size()); ++i) {
        result_ids.push_back(results[i].id);
    }
    
    #ifdef DEBUG_SEARCH
    std::cout << "[DEBUG] search_layer0_multi_entry返回 " << result_ids.size() << " 个邻居: ";
    for (size_t i = 0; i < std::min((size_t)10, result_ids.size()); ++i) {
        std::cout << result_ids[i] << " ";
    }
    std::cout << std::endl;
    #endif
    
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
        // 使用全局最优的策略：从第0层选择多个随机节点作为入口
        // ⚠️ 重要：使用unordered_set避免重复
        std::unordered_set<int> entry_set;
        while (entry_set.size() < std::min(num_entry_points, num_elements_)) {
            entry_set.insert(rng_() % num_elements_);
        }
        entry_points.assign(entry_set.begin(), entry_set.end());
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

std::vector<int> HNSW::get_layer0_neighbors(int node_id) const {
    if (node_id < 0 || (size_t)node_id >= num_elements_) {
        return std::vector<int>();
    }
    
    // ✅ 适配新的数组结构
    const Node& node = nodes_[node_id];
    std::vector<int> neighbors;
    neighbors.reserve(node.num_neighbors_layer0);
    for (int i = 0; i < node.num_neighbors_layer0; ++i) {
        neighbors.push_back(node.neighbors_layer0[i]);
    }
    return neighbors;
}

void HNSW::load_layer0(const float* data, size_t num_vectors, const std::vector<std::vector<int>>& neighbors_list) {
    std::cout << "从预加载数据加载第0层图结构：" << num_vectors << " 个向量" << std::endl;
    
    num_elements_ = num_vectors;
    
    // 🔧 关键修复：扩大 visited_bitmap_ 以匹配实际节点数
    if (num_vectors > visited_bitmap_.size()) {
        std::cout << "扩大 visited_bitmap_ 从 " << visited_bitmap_.size() 
                  << " 到 " << num_vectors << std::endl;
        visited_bitmap_.resize(num_vectors, 0);
    }
    
    // 复制数据
    if (data_ != nullptr) {
        delete[] data_;
    }
    data_ = new float[num_elements_ * dimension_];
    std::copy(data, data + num_elements_ * dimension_, data_);
    
    // 初始化节点
    nodes_.resize(num_elements_);
    layer1_nodes_.clear();
    
    // ✅ 预分配邻居数组
    size_t Mmax0 = 2 * M0_;
    size_t Mmax1 = 2 * M1_;
    
    // 第一步：确定哪些节点在第1层并分配内存
    std::cout << "确定第1层节点并分配内存..." << std::endl;
    for (size_t i = 0; i < num_elements_; ++i) {
        int layer = select_layer_randomly();
        
        // 为第0层分配邻居数组
        nodes_[i].neighbors_layer0 = new int[Mmax0];
        nodes_[i].num_neighbors_layer0 = 0;
        
        if (layer >= 1) {
            nodes_[i].in_layer1 = true;
            layer1_nodes_.push_back(i);
            // 为第1层分配邻居数组
            nodes_[i].neighbors_layer1 = new int[Mmax1];
            nodes_[i].num_neighbors_layer1 = 0;
        }
        
        if ((i + 1) % 10000 == 0) {
            std::cout << "  进度: " << (i + 1) << "/" << num_elements_ << std::endl;
        }
    }
    
    std::cout << "第1层节点数: " << layer1_nodes_.size() << " / " << num_elements_ 
              << " (" << (100.0 * layer1_nodes_.size() / num_elements_) << "%)" << std::endl;
    
    // 第二步：直接加载第0层连接（从文件读取的邻居列表）
    std::cout << "加载第0层连接..." << std::endl;
    for (size_t i = 0; i < num_elements_ && i < neighbors_list.size(); ++i) {
        const std::vector<int>& neighbors = neighbors_list[i];
        nodes_[i].num_neighbors_layer0 = std::min((size_t)neighbors.size(), Mmax0);
        for (int j = 0; j < nodes_[i].num_neighbors_layer0; ++j) {
            nodes_[i].neighbors_layer0[j] = neighbors[j];
        }
        
        if ((i + 1) % 10000 == 0) {
            std::cout << "  进度: " << (i + 1) << "/" << num_elements_ << std::endl;
        }
    }
    
    std::cout << "第0层图结构加载完成！（第1层待后续构建）" << std::endl;
}

void HNSW::build_layer1_only(const float* data) {
    std::cout << "开始构建第1层连接..." << std::endl;
    
    // 确保已经加载了向量数据
    if (data_ == nullptr || num_elements_ == 0) {
        std::cerr << "错误：必须先加载第0层数据才能构建第1层！" << std::endl;
        return;
    }
    
    // 如果第1层节点列表为空，说明还没有确定哪些节点在第1层
    // 这种情况下应该已经在 load_layer0 中完成了
    if (layer1_nodes_.empty()) {
        std::cout << "警告：没有第1层节点，跳过第1层构建" << std::endl;
        return;
    }
    
    std::cout << "第1层节点数: " << layer1_nodes_.size() << " / " << num_elements_ 
              << " (" << (100.0 * layer1_nodes_.size() / num_elements_) << "%)" << std::endl;
    
    // 构建第1层连接
    for (size_t i = 0; i < layer1_nodes_.size(); ++i) {
        int node_id = layer1_nodes_[i];
        
        // 安全检查：确保节点状态正常
        if (!nodes_[node_id].in_layer1 || nodes_[node_id].neighbors_layer1 == nullptr) {
            continue;
        }
        
        connect_neighbors_layer1(node_id, ef_construction_);
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "  进度: " << (i + 1) << "/" << layer1_nodes_.size() << std::endl;
        }
    }
    
    std::cout << "第1层构建完成！" << std::endl;
}

} // namespace hnsw

