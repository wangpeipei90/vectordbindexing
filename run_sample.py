from io_utils import read_fbin, read_ibin
import faiss
print(faiss.__version__)
import numpy as np
file_path = "/root/code/vectordbindexing/Text2Image/base.1M.fbin"
query_path = "/root/code/vectordbindexing/Text2Image/query.public.100K.fbin"
ground_truth_path = "/root/code/vectordbindexing/Text2Image/groundtruth.public.100K.ibin"


# read datasets
print("\n\nreading image vector: ---")
data_vector = read_fbin(file_path)
print(type(data_vector))
print(data_vector.ndim, data_vector.shape, data_vector.dtype, data_vector.size)
# print(data_vector[:1])  # Print first 1 elements to verify content

train_data_vector = data_vector[:500000]
insert_1_percent = data_vector[500000:505000]
insert_2_percent = data_vector[505000:510000]
insert_3_percent = data_vector[510000:515000]
insert_4_percent = data_vector[515000:520000]
insert_5_percent = data_vector[520000:525000]
insert_10_percent = data_vector[525000:550000]

# read querys
print("\n\nreading querys: ---")
query_vector = read_fbin(query_path)
print(type(query_vector))
print(query_vector.ndim, query_vector.shape, query_vector.dtype, query_vector.size)
# print(query_vector[0])  # Print first 3 elements to verify content

import time
import hnsw_cosine
import simple_sim_hash
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
importlib.reload(hnsw_cosine)

# M=64 比较合适，甚至更宽的宽度
# 这里是个经验值：会在增加宽度的同时，逐渐达到一个稳定值
index = hnsw_cosine.HNSWIndex(M=64, ef_construction=128, ef_search=64, random_seed=1)
simHash = simple_sim_hash.SimpleSimHash(dim=200)

IMAGE_IDX_SET = set()

# 形状 [N,200]（先用1M子集或更小切片做原型）
for img_id, vec in enumerate(train_data_vector):        # 可加 tqdm、批量 flush
    index.add_item_fast10k(vec, lsh=simHash, limit=500)
    IMAGE_IDX_SET.add(img_id)

# 读取faiss搜索结果，获取 query_vector 和 search 结果
import json
train_query_list = {}
test_query_list = {}

# ground_truth = read_ibin(ground_truth_path)
# print(type(ground_truth))
# print(ground_truth.ndim, ground_truth.shape, ground_truth.dtype, ground_truth.size)
# for query_idx in range(ground_truth.shape[0]):
#     if int(query_idx) % 6 != 0:
#         train_query_list[query_idx] = ground_truth[query_idx]
#     else:
#         test_query_list[query_idx] = ground_truth[query_idx]

with open("./TempResults/search_results_100K.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    for query_idx, vec_list in data.items():
        mList = []
        for x in vec_list:
            mList.append(x - int(query_idx))
        if int(query_idx) % 6 != 0:
            train_query_list[int(query_idx)] = mList
        else:
            test_query_list[int(query_idx)] = mList
print(f"num of train: {len(train_query_list)}")
print(f"num of test: {len(test_query_list)}")


# reset added links
for layer in range(0, 100):
    if layer not in index.neighbours:
        continue
    for idx in range(0, 500000):
        if idx not in index.neighbours[layer]:
            continue
        index.neighbours[layer][idx] = index.neighbours[layer][idx][:64]


# OOD search steps
NUM_STEPS = []
for qid, target_list in test_query_list.items():
    q = query_vector[qid]
    for target_id in target_list[:10]:
        if target_id not in IMAGE_IDX_SET:
            continue
        out = index.search_steps_to_target(q, target_id, k=10, ef=64)
        NUM_STEPS.append(len(out["trace"]))

arr_ori_bak = np.array(NUM_STEPS, dtype=np.float64)
arr_ori = arr_ori_bak.copy()
arr_ori.sort()

mean_steps = arr_ori.mean()
P50_steps = np.percentile(arr_ori, 50)
p99_steps = np.percentile(arr_ori, 99)
print(f"mean steps: {mean_steps}")
print(f"middle steps: {P50_steps}")
print(f"p99 steps: {p99_steps}")

# 使用新的 RoarGraph 风格的 cross distribution 边构建
print("\n=== 构建 RoarGraph 风格的 Cross Distribution 边 ===")
stats = index.build_cross_distribution_edges(
    test_query_list,
    layer=0,  # 只在第0层构建
    max_new_edges_per_node=4,
    occlude_alpha=1.0,  # 遮挡阈值
    use_metric=True,
    chain_extra=1,  # 额外的链式连接
)
print("Cross distribution 边构建统计:")
print(stats)

# 获取 cross distribution 边的统计信息
cross_stats = index.get_cross_distribution_stats()
print("\nCross distribution 边统计:")
print(f"总添加的 cross distribution 边: {cross_stats['total_cross_edges']}")
print(f"被删除的 cross distribution 边: {cross_stats['deleted_cross_edges']}")
print(f"活跃的 cross distribution 边: {cross_stats['active_cross_edges']}")

# OOD search steps - after add cross distribution links
NUM_STEPS = []
for qid, target_list in test_query_list.items():
    q = query_vector[qid]
    for target_id in target_list[:10]:
        if target_id not in IMAGE_IDX_SET:
            continue
        out = index.search_steps_to_target(q, target_id, k=10, ef=64)
        NUM_STEPS.append(len(out["trace"]))

arr_bak = np.array(NUM_STEPS, dtype=np.float64)
arr = arr_bak.copy()
arr.sort()

mean_steps = arr.mean()
P50_steps = np.percentile(arr, 50)
p99_steps = np.percentile(arr, 99)
print(f"\n添加 cross distribution 边后:")
print(f"mean steps: {mean_steps}")
print(f"middle steps: {P50_steps}")
print(f"p99 steps: {p99_steps}")

# 检查差距最大的那个结果
max_idx = 0
max_steps = 0
for idx in range(len(arr_ori_bak)):
    if arr_ori_bak[idx] - arr_bak[idx] > max_steps:
        max_idx = idx
        max_steps = arr_ori_bak[idx] - arr_bak[idx]
print(f"{max_idx} item got the biggest steps reduction: {max_steps}")

# 插入额外的数据；并继续search上面的测试集合，查看search所需steps
insert_data_vectors = {
    "insert_1%": insert_1_percent,
    "insert_2%": insert_2_percent,
    "insert_3%": insert_3_percent,
    "insert_4%": insert_4_percent,
    "insert_5%": insert_5_percent,
    "insert_10%": insert_10_percent,
}
img_id = 500000
for name, insert_vectors in insert_data_vectors.items():
    print(f"-------------{name}--------------")
    # insert 新节点
    for _, vec in enumerate(insert_vectors):        # 可加 tqdm、批量 flush
        index.add_item(vec, id=img_id)
        img_id += 1
        IMAGE_IDX_SET.add(img_id)
    
    # 获取增量插入后的 cross distribution 边统计
    cross_stats_after = index.get_cross_distribution_stats()
    print(f"增量插入后 cross distribution 边统计:")
    print(f"总添加的 cross distribution 边: {cross_stats_after['total_cross_edges']}")
    print(f"被删除的 cross distribution 边: {cross_stats_after['deleted_cross_edges']}")
    print(f"活跃的 cross distribution 边: {cross_stats_after['active_cross_edges']}")
    print(f"新增删除的 cross distribution 边: {cross_stats_after['deleted_cross_edges'] - cross_stats['deleted_cross_edges']}")
    
    NUM_STEPS = []
    for qid, target_list in test_query_list.items():
        q = query_vector[qid]
        for target_id in target_list[:10]:
            if target_id not in IMAGE_IDX_SET:
                continue
            out = index.search_steps_to_target(q, target_id, k=10, ef=64)
            NUM_STEPS.append(len(out["trace"]))

    arr = np.array(NUM_STEPS, dtype=np.float64)

    print(f"{max_idx} item got the biggest steps reduction: {arr[max_idx]}")

    arr.sort()

    mean_steps = arr.mean()
    P50_steps = np.percentile(arr, 50)
    p99_steps = np.percentile(arr, 99)
    print(f"mean steps: {mean_steps}")
    print(f"middle steps: {P50_steps}")
    print(f"p99 steps: {p99_steps}")
    
    # 更新统计信息用于下次比较
    cross_stats = cross_stats_after.copy()