#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnsw.h"
#include <vector>

namespace py = pybind11;

class HNSWWrapper {
public:
    HNSWWrapper(size_t dimension, size_t M0, size_t ef_construction, 
                        size_t max_elements, int seed = 42)
        : index_(dimension, M0, ef_construction, max_elements, seed) {}
    
    void build(py::array_t<float> vectors) {
        py::buffer_info buf = vectors.request();
        
        if (buf.ndim != 2) {
            throw std::runtime_error("向量必须是2维数组 (N x D)");
        }
        
        size_t num_vectors = buf.shape[0];
        size_t dimension = buf.shape[1];
        
        float* data = static_cast<float*>(buf.ptr);
        index_.build(data, num_vectors);
    }
    
    py::dict search(py::array_t<float> query, size_t k, 
                   size_t ef_search = 200, size_t num_entry_points = 10) {
        py::buffer_info buf = query.request();
        
        if (buf.ndim != 1) {
            throw std::runtime_error("查询向量必须是1维数组 (D,)");
        }
        
        float* query_ptr = static_cast<float*>(buf.ptr);
        hnsw::SearchResult result = index_.search(query_ptr, k, ef_search, num_entry_points);
        
        // 转换neighbors为numpy数组
        py::array_t<int> neighbors_array(result.neighbors.size());
        py::buffer_info neighbors_buf = neighbors_array.request();
        int* neighbors_ptr = static_cast<int*>(neighbors_buf.ptr);
        
        for (size_t i = 0; i < result.neighbors.size(); ++i) {
            neighbors_ptr[i] = result.neighbors[i];
        }
        
        // 返回字典
        py::dict result_dict;
        result_dict["neighbors"] = neighbors_array;
        result_dict["visited_count"] = result.visited_count;
        result_dict["latency_us"] = result.latency_us;
        result_dict["layer1_visited"] = result.layer1_visited;
        result_dict["layer0_visited"] = result.layer0_visited;
        
        return result_dict;
    }
    
    py::list batch_search(py::array_t<float> queries, size_t k,
                         size_t ef_search = 200, size_t num_entry_points = 10) {
        py::buffer_info buf = queries.request();
        
        if (buf.ndim != 2) {
            throw std::runtime_error("查询向量必须是2维数组 (N x D)");
        }
        
        size_t num_queries = buf.shape[0];
        float* queries_ptr = static_cast<float*>(buf.ptr);
        
        std::vector<hnsw::SearchResult> results = index_.batch_search(
            queries_ptr, num_queries, k, ef_search, num_entry_points
        );
        
        // 转换为Python list of dicts
        py::list result_list;
        for (const auto& result : results) {
            py::array_t<int> neighbors_array(result.neighbors.size());
            py::buffer_info neighbors_buf = neighbors_array.request();
            int* neighbors_ptr = static_cast<int*>(neighbors_buf.ptr);
            
            for (size_t i = 0; i < result.neighbors.size(); ++i) {
                neighbors_ptr[i] = result.neighbors[i];
            }
            
            py::dict result_dict;
            result_dict["neighbors"] = neighbors_array;
            result_dict["visited_count"] = result.visited_count;
            result_dict["latency_us"] = result.latency_us;
            result_dict["layer1_visited"] = result.layer1_visited;
            result_dict["layer0_visited"] = result.layer0_visited;
            
            result_list.append(result_dict);
        }
        
        return result_list;
    }
    
    static double compute_recall(py::array_t<int> results, py::array_t<int> ground_truth, size_t k) {
        py::buffer_info results_buf = results.request();
        py::buffer_info gt_buf = ground_truth.request();
        
        std::vector<int> results_vec(results_buf.shape[0]);
        std::vector<int> gt_vec(gt_buf.shape[0]);
        
        int* results_ptr = static_cast<int*>(results_buf.ptr);
        int* gt_ptr = static_cast<int*>(gt_buf.ptr);
        
        for (size_t i = 0; i < results_vec.size(); ++i) {
            results_vec[i] = results_ptr[i];
        }
        for (size_t i = 0; i < gt_vec.size(); ++i) {
            gt_vec[i] = gt_ptr[i];
        }
        
        return hnsw::HNSW::compute_recall(results_vec, gt_vec, k);
    }
    
    size_t get_num_nodes() const {
        return index_.get_num_nodes();
    }
    
    size_t get_num_layer1_nodes() const {
        return index_.get_num_layer1_nodes();
    }
    
    std::vector<int> get_neighbors_layer0(int node_id) const {
        return index_.get_neighbors_layer0(node_id);
    }
    
    std::vector<int> get_neighbors_layer1(int node_id) const {
        return index_.get_neighbors_layer1(node_id);
    }
    
    bool is_in_layer1(int node_id) const {
        return index_.is_in_layer1(node_id);
    }
    
private:
    hnsw::HNSW index_;
};

PYBIND11_MODULE(hnsw_core, m) {
    m.doc() = "2层HNSW索引的C++核心实现";
    
    py::class_<HNSWWrapper>(m, "HNSW")
        .def(py::init<size_t, size_t, size_t, size_t, int>(),
             py::arg("dimension"),
             py::arg("M0"),
             py::arg("ef_construction"),
             py::arg("max_elements"),
             py::arg("seed") = 42,
             "创建2层HNSW索引\n\n"
             "参数:\n"
             "  dimension: 向量维度\n"
             "  M0: 第0层出度\n"
             "  ef_construction: 构建时搜索宽度\n"
             "  max_elements: 最大元素数\n"
             "  seed: 随机种子")
        .def("build", &HNSWWrapper::build,
             py::arg("vectors"),
             "构建索引\n\n"
             "参数:\n"
             "  vectors: numpy数组 (N x D)")
        .def("search", &HNSWWrapper::search,
             py::arg("query"),
             py::arg("k"),
             py::arg("ef_search") = 200,
             py::arg("num_entry_points") = 10,
             "搜索最近邻\n\n"
             "参数:\n"
             "  query: 查询向量 (D,)\n"
             "  k: 返回的邻居数量\n"
             "  ef_search: 搜索宽度\n"
             "  num_entry_points: 入口点数量\n\n"
             "返回:\n"
             "  字典，包含:\n"
             "    - neighbors: numpy数组，k个最近邻的ID\n"
             "    - visited_count: 访问的节点总数\n"
             "    - latency_us: 搜索延迟（微秒）\n"
             "    - layer1_visited: 第1层访问的节点数\n"
             "    - layer0_visited: 第0层访问的节点数")
        .def("batch_search", &HNSWWrapper::batch_search,
             py::arg("queries"),
             py::arg("k"),
             py::arg("ef_search") = 200,
             py::arg("num_entry_points") = 10,
             "批量搜索最近邻\n\n"
             "参数:\n"
             "  queries: 查询向量 (N x D)\n"
             "  k: 返回的邻居数量\n"
             "  ef_search: 搜索宽度\n"
             "  num_entry_points: 入口点数量\n\n"
             "返回:\n"
             "  列表，每个元素是一个字典（同search返回）")
        .def_static("compute_recall", &HNSWWrapper::compute_recall,
             py::arg("results"),
             py::arg("ground_truth"),
             py::arg("k"),
             "计算recall@k\n\n"
             "参数:\n"
             "  results: 搜索结果 (numpy数组)\n"
             "  ground_truth: ground truth (numpy数组)\n"
             "  k: 计算recall的k值\n\n"
             "返回:\n"
             "  recall值 (0.0-1.0)")
        .def("get_num_nodes", &HNSWWrapper::get_num_nodes,
             "获取总节点数")
        .def("get_num_layer1_nodes", &HNSWWrapper::get_num_layer1_nodes,
             "获取第1层节点数")
        .def("get_neighbors_layer0", &HNSWWrapper::get_neighbors_layer0,
             py::arg("node_id"),
             "获取节点在第0层的邻居列表")
        .def("get_neighbors_layer1", &HNSWWrapper::get_neighbors_layer1,
             py::arg("node_id"),
             "获取节点在第1层的邻居列表")
        .def("is_in_layer1", &HNSWWrapper::is_in_layer1,
             py::arg("node_id"),
             "检查节点是否在第1层");
}

