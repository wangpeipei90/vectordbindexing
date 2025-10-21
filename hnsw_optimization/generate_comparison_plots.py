#!/usr/bin/env python3
"""
生成不同并行入口点数量的性能对比图表
对比内容：
1. Recall准确率 vs 并行入口点数量
2. 查询延迟 vs 并行入口点数量
3. Recall-Latency权衡对比（双Y轴）
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_comparison_plots(results_by_entry, entry_point_values, output_dir='./'):
    """
    生成性能对比图表

    参数:
        results_by_entry: 字典，键为入口点数量，值为包含recall和latency的结果字典
        entry_point_values: 列表，不同的入口点数量
        output_dir: 输出目录
    """

    print("=" * 70)
    print("生成专业对比图表：不同并行入口点数量的性能分析")
    print("=" * 70)

    # 准备数据
    x_values = entry_point_values
    recall_10_values = [results_by_entry[n]['recall_10']
                        for n in entry_point_values]
    recall_100_values = [results_by_entry[n]['recall_100']
                         for n in entry_point_values]
    latency_values = [results_by_entry[n]['avg_time_ms']
                      for n in entry_point_values]

    # 设置绘图风格
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans',
                                       'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # ==================== 图1: Recall对比（不同并行入口点数量） ====================
    print("\n生成图1: Recall vs 入口点数量...")
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制双条形折线图
    line1 = ax1.plot(x_values, recall_10_values, marker='o', linewidth=3, markersize=10,
                     label='Recall@10', color='#2E86AB', linestyle='-')
    line2 = ax1.plot(x_values, recall_100_values, marker='s', linewidth=3, markersize=10,
                     label='Recall@100', color='#A23B72', linestyle='--')

    ax1.set_xlabel('Number of Parallel Entry Points',
                   fontsize=14, fontweight='bold')
    ax1.set_ylabel('Recall', fontsize=14, fontweight='bold')
    ax1.set_title('Recall Accuracy vs Number of Parallel Entry Points',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=13, loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_xticks(x_values)
    ax1.set_ylim([0, 1.05])

    # 在数据点上添加数值标签
    for i, (x, y10, y100) in enumerate(zip(x_values, recall_10_values, recall_100_values)):
        ax1.text(x, y10 + 0.03, f'{y10:.3f}', ha='center', fontsize=10,
                 fontweight='bold', color='#2E86AB')
        if i % 2 == 0:  # 交错显示避免重叠
            ax1.text(x, y100 - 0.06, f'{y100:.3f}', ha='center', fontsize=10,
                     fontweight='bold', color='#A23B72')
        else:
            ax1.text(x + 0.15, y100, f'{y100:.3f}', ha='left', fontsize=10,
                     fontweight='bold', color='#A23B72')

    plt.tight_layout()
    output_path1 = f'{output_dir}/recall_vs_entry_points.png'
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"✅ 图1已保存: {output_path1}")
    plt.show()

    # ==================== 图2: Latency对比（不同并行入口点数量） ====================
    print("\n生成图2: Latency vs 入口点数量...")
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # 绘制延迟折线图
    line3 = ax2.plot(x_values, latency_values, marker='D', linewidth=3, markersize=10,
                     label='Query Latency', color='#F18F01', linestyle='-')

    ax2.set_xlabel('Number of Parallel Entry Points',
                   fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Query Latency (ms)',
                   fontsize=14, fontweight='bold')
    ax2.set_title('Query Latency vs Number of Parallel Entry Points',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=13, loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_xticks(x_values)
    ax2.set_ylim([0, max(latency_values) * 1.15])

    # 添加数值标签和变化百分比
    for i, (x, y) in enumerate(zip(x_values, latency_values)):
        ax2.text(x, y + max(latency_values)*0.03, f'{y:.2f} ms', ha='center',
                 fontsize=10, fontweight='bold', color='#F18F01')

        # 显示相对于单入口的变化
        if i > 0:
            change_pct = ((y - latency_values[0]) / latency_values[0]) * 100
            ax2.text(x, y - max(latency_values)*0.08, f'(+{change_pct:.0f}%)',
                     ha='center', fontsize=9, color='#C73E1D', style='italic')

    # 添加基线参考线
    ax2.axhline(y=latency_values[0], color='gray', linestyle=':', linewidth=2,
                alpha=0.5, label=f'Baseline (1 entry): {latency_values[0]:.2f} ms')
    ax2.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    output_path2 = f'{output_dir}/latency_vs_entry_points.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ 图2已保存: {output_path2}")
    plt.show()

    # ==================== 图3: 综合对比图（双Y轴） ====================
    print("\n生成图3: Recall-Latency权衡对比...")
    fig3, ax3_1 = plt.subplots(figsize=(14, 7))

    # 左Y轴：Recall
    color1 = '#2E86AB'
    ax3_1.set_xlabel('Number of Parallel Entry Points',
                     fontsize=14, fontweight='bold')
    ax3_1.set_ylabel('Recall@10', fontsize=14, fontweight='bold', color=color1)
    line4 = ax3_1.plot(x_values, recall_10_values, marker='o', linewidth=3, markersize=12,
                       label='Recall@10', color=color1, linestyle='-')
    ax3_1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax3_1.set_xticks(x_values)
    ax3_1.grid(True, alpha=0.3, linestyle='--')
    ax3_1.set_ylim([0, 1.05])

    # 右Y轴：Latency
    ax3_2 = ax3_1.twinx()
    color2 = '#F18F01'
    ax3_2.set_ylabel('Average Query Latency (ms)', fontsize=14,
                     fontweight='bold', color=color2)
    line5 = ax3_2.plot(x_values, latency_values, marker='s', linewidth=3, markersize=12,
                       label='Latency', color=color2, linestyle='--')
    ax3_2.tick_params(axis='y', labelcolor=color2, labelsize=12)

    # 标题和图例
    ax3_1.set_title('Recall vs Latency Trade-off (Parallel Entry Points)',
                    fontsize=16, fontweight='bold', pad=20)

    # 合并图例
    lines = line4 + line5
    labels = [l.get_label() for l in lines]
    ax3_1.legend(lines, labels, fontsize=13,
                 loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    output_path3 = f'{output_dir}/recall_latency_tradeoff.png'
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"✅ 图3已保存: {output_path3}")
    plt.show()

    # ==================== 数据总结表格 ====================
    print("\n" + "=" * 80)
    print("详细性能数据总结")
    print("=" * 80)
    print(f"\n{'入口点':<8} {'Recall@10':<12} {'Recall@100':<12} {'延迟(ms)':<12} "
          f"{'延迟增加':<12} {'效率比':<12}")
    print("-" * 80)

    for i, num_entries in enumerate(entry_point_values):
        result = results_by_entry[num_entries]
        latency_increase = result['avg_time_ms'] - latency_values[0]
        efficiency = result['recall_10'] / result['avg_time_ms']

        print(f"{num_entries:<8} {result['recall_10']:<12.4f} {result['recall_100']:<12.4f} "
              f"{result['avg_time_ms']:<12.3f} {latency_increase:+12.3f} {efficiency:<12.6f}")

    # 找出最佳平衡点
    efficiency_scores = [(results_by_entry[n]['recall_10'] / results_by_entry[n]['avg_time_ms'])
                         for n in entry_point_values]
    best_efficiency_idx = np.argmax(efficiency_scores)
    best_efficiency_entry = entry_point_values[best_efficiency_idx]

    print("\n" + "=" * 80)
    print("关键结论")
    print("=" * 80)
    print(f"\n🏆 最佳效率配置: {best_efficiency_entry} 个入口点")
    print(
        f"   - Recall@10: {results_by_entry[best_efficiency_entry]['recall_10']:.4f}")
    print(
        f"   - Recall@100: {results_by_entry[best_efficiency_entry]['recall_100']:.4f}")
    print(
        f"   - 延迟: {results_by_entry[best_efficiency_entry]['avg_time_ms']:.3f} ms")
    print(
        f"   - 效率比: {efficiency_scores[best_efficiency_idx]:.6f} (Recall/ms)")

    print(f"\n📊 性能趋势:")
    recall_trend = "增加" if recall_10_values[-1] > recall_10_values[0] else "降低"
    latency_trend = "增加" if latency_values[-1] > latency_values[0] else "降低"
    print(f"   - Recall趋势: 入口点从1到8，Recall@10 {recall_trend}")
    print(f"   - 延迟趋势: 入口点从1到8，延迟 {latency_trend}")
    print(
        f"   - 单入口点性能: Recall={recall_10_values[0]:.4f}，延迟={latency_values[0]:.2f}ms")
    print(
        f"   - 多入口点性能: Recall={recall_10_values[-1]:.4f}，延迟={latency_values[-1]:.2f}ms")

    print("\n✅ 所有图表生成完成！")

    return {
        'recall_10_values': recall_10_values,
        'recall_100_values': recall_100_values,
        'latency_values': latency_values,
        'best_efficiency_entry': best_efficiency_entry,
        'efficiency_scores': efficiency_scores
    }


# 示例：在notebook中使用
# 假设你已经有了 results_by_entry 和 entry_point_values
if __name__ == "__main__":
    print("这是一个模块文件，请在notebook中导入使用：")
    print("from generate_comparison_plots import generate_comparison_plots")
    print("generate_comparison_plots(results_by_entry, entry_point_values, output_dir='/root/code/vectordbindexing/hnsw_optimization')")
