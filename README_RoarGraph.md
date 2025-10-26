# RoarGraph

This is the manner for "How to use RoarGraph and It's SourceCode"

### How to use
- follow the guideline of [RoarGraph](https://github.com/matchyc/RoarGraph)
- Additional Notes
```shell
# 1, the third party package need update before cmake
git submodule update --init --recursive

# compute groundtruth
cp ./thirdparty/DiskANN/tests/utils/compute_groundtruth compute_groundtruth

./compute_groundtruth --data_type float --dist_fn mips --base_file ../data/t2i-10M/base.10M.fbin  --query_file ../data/t2i-10M/query.train.10M.fbin  --gt_file ../data/t2i-10M/train.gt.bin --K 100

# build the graph index
./tests/test_build_roargraph --data_type float --dist ip \
--base_data_path ../data/t2i-10M/base.10M.fbin  \
--sampled_query_data_path ../data/t2i-10M/query.train.10M.fbin \
--projection_index_save_path ../data/t2i-10M/t2i_10M_roar.index \
--learn_base_nn_path ../data/t2i-10M/train.gt.bin \
--M_sq 100 --M_pjbp 35 --L_pjpq 500 -T 64

# search
./tests/test_search_roargraph --data_type float \
--dist ip --base_data_path ../data/t2i-10M/base.10M.fbin \
--projection_index_save_path ../data/t2i-10M/t2i_10M_roar.index \
--gt_path ../data/t2i-10M/groundtruth-computed.10k.ibin  \
--query_path ../data/t2i-10M/query.10k.fbin \
--L_pq 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300 350 400 450 500 550 600 650 700 750 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
--k 10  -T 16 \
--evaluation_save_path ../data/t2i-10M/test_search_t2i_10M_top10_T16.csv
```