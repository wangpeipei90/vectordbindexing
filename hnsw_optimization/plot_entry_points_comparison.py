"""
在notebook中运行此代码来生成对比图表
复制粘贴到新的cell中执行
"""

# 导入绘图函数
from generate_comparison_plots import generate_comparison_plots

# 生成所有对比图表
plot_results = generate_comparison_plots(
    results_by_entry=results_by_entry,
    entry_point_values=entry_point_values,
    output_dir='/root/code/vectordbindexing/hnsw_optimization'
)

print("\n" + "=" * 70)
print("图表已生成，文件列表：")
print("=" * 70)
print("1. recall_vs_entry_points.png - Recall准确率对比")
print("2. latency_vs_entry_points.png - 查询延迟对比")  
print("3. recall_latency_tradeoff.png - Recall-Latency权衡对比（双Y轴）")

