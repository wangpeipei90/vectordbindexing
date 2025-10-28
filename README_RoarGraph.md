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

### RoarGraph 在Text2Image-10M上的结果：
load meta from file: ../data/t2i-10M/base.10M.fbin points_num: 10000000 dim: 200
load meta from file: ../data/t2i-10M/query.10k.fbin points_num: 10000 dim: 200
load data from file: ../data/t2i-10M/query.10k.fbin
points_num: 10000 dim: 200
Finish load data from file: ../data/t2i-10M/query.10k.fbin points_num: 10000 dim: 200
new_dim: 200
load gt from file: ../data/t2i-10M/groundtruth-computed.10k.ibin points_num: 10000 dim: 500
Using inner product as distance metric
Inside using IP distance after normalization.
load meta from file: ../data/t2i-10M/base.10M.fbin points_num: 10000000 dim: 200
load data from file: ../data/t2i-10M/base.10M.fbin
points_num: 10000000 dim: 200
Finish load data from file: ../data/t2i-10M/base.10M.fbin points_num: 10000000 dim: 200
new_dim: 200
Load graph index: ../data/t2i-10M/t2i_10M_roar.index
Projection graph, ep: 2610664
Projection graph, avg_degree: 43.5648
k: 10
Using thread: 16
L_pq            QPS                     avg_visited     mean_latency    recall@10       avg_hops
10              59880.2         817.74          0.0167          0.69366         18.7729
15              56818.2         1006.87         0.0176          0.76229         23.9534
20              47393.4         1179.97         0.0211          0.80266         28.9484
25              40485.8         1345            0.0247          0.83043         33.9024
30              35211.3         1504.31         0.0284          0.84995         38.8243
35              33333.3         1661.17         0.03            0.86696         43.8067
40              30395.1         1816.37         0.0329          0.88068         48.8109
45              27248           1965.36         0.0367          0.89058         53.7468
50              27027           2113.63         0.037           0.90002         58.7355
55              25316.5         2257.87         0.0395          0.9074          63.6907
60              23310           2400.39         0.0429          0.91343         68.6506
65              22321.4         2543.44         0.0448          0.91923         73.664
70              20242.9         2684.05         0.0494          0.92508         78.6792
75              19342.4         2821.17         0.0517          0.92887         83.6221
80              18281.5         2955.71         0.0547          0.93234         88.5568
85              17543.9         3092.49         0.057           0.93623         93.5843
90              16949.2         3226.31         0.059           0.93973         98.5653
95              15128.6         3359.6          0.0661          0.94246         103.554
100             14792.9         3491.35         0.0676          0.94517         108.544
110             12484.4         3749.59         0.0801          0.94998         118.498
120             12285           4005.54         0.0814          0.95402         128.476
130             13642.6         4255.51         0.0733          0.95741         138.406
140             11481.1         4504.02         0.0871          0.96014         148.362
150             10559.7         4749.69         0.0947          0.96283         158.338
160             11312.2         4991.67         0.0884          0.96478         168.319
170             10787.5         5231.43         0.0927          0.96677         178.306
180             10582           5470.49         0.0945          0.96879         188.314
190             10152.3         5705.87         0.0985          0.97044         198.288
200             9250.69         5938.48         0.1081          0.9717          208.218
220             8896.8          6398.66         0.1124          0.97437         228.189
240             7385.52         6853.61         0.1354          0.97646         248.18
260             6997.9          7301.2          0.1429          0.97854         268.203
280             6618.13         7743.94         0.1511          0.98042         288.202
300             6868.13         8179.94         0.1456          0.98208         308.243
350             6172.84         9242.33         0.162           0.98462         358.116
400             5249.34         10278.9         0.1905          0.98662         407.948
450             5020.08         11295.1         0.1992          0.98842         457.898
500             4518.75         12293.4         0.2213          0.99022         507.937
550             4151.1          13277.9         0.2409          0.99169         558.081
600             3723.01         14234.1         0.2686          0.99237         607.882
650             3741.11         15179.8         0.2673          0.99317         657.786
700             3453.04         16109.5         0.2896          0.99366         707.601
750             3324.47         17034.9         0.3008          0.99436         757.664
800             3044.14         17946.6         0.3285          0.99471         807.633
900             2704.9          19735.6         0.3697          0.99547         907.391
1000            2510.67         21490.3         0.3983          0.99609         1007.28
1100            2339.73         23219.7         0.4274          0.99669         1107.3
1200            2168.73         24912.7         0.4611          0.99699         1207.09
1300            2036.25         26581.8         0.4911          0.99742         1306.88
1400            1892.15         28227.7         0.5285          0.99769         1406.78
1500            1770.85         29850.7         0.5647          0.9979          1506.56
1600            1694.92         31457.3         0.59            0.99802         1606.43
1700            1604.11         33044.6         0.6234          0.99812         1706.24
1800            1505.34         34618.1         0.6643          0.99833         1806.2
1900            1422.48         36176.2         0.703           0.99853         1906.13
2000            1343.18         37719.4         0.7445          0.99858         2006