import time
import numpy as np
import matplotlib

# 强制无GUI后端，适配Streamlit
matplotlib.use('Agg')
# 二次全局字体设置，双重保险
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

import matplotlib.pyplot as plt


def run_performance_test(original_img, binary_img, test_times=10):
    """
    算法性能对比测试
    """
    from .hough_detector import hough_original_detect, hough_improved_detect

    t_original_list = []
    t_improved_list = []

    for _ in range(test_times):
        # 普通霍夫计时
        start = time.time()
        _, _, _ = hough_original_detect(original_img, binary_img)
        t_original = time.time() - start
        t_original_list.append(t_original)

        # 改良霍夫计时
        start = time.time()
        _, _, _ = hough_improved_detect(original_img, binary_img)
        t_improved = time.time() - start
        t_improved_list.append(t_improved)

    # 计算平均耗时和加速比
    t_original_avg = np.mean(t_original_list)
    t_improved_avg = np.mean(t_improved_list)
    speedup_ratio = t_original_avg / t_improved_avg if t_improved_avg != 0 else 1

    return t_original_avg, t_improved_avg, speedup_ratio, t_original_list, t_improved_list


def plot_performance_result(t_original_avg, t_improved_avg, speedup_ratio):
    """
    生成性能对比图，修复中文乱码
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1：平均运行时间对比
    algorithms = ['普通霍夫变换', '改良霍夫变换']
    times = [t_original_avg, t_improved_avg]
    bars = ax1.bar(algorithms, times, color=['#1f77b4', '#ff7f0e'], width=0.6)
    ax1.set_title('平均运行时间对比（秒）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('耗时 (s)', fontsize=12)
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}s',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 子图2：加速比
    ax2.bar(['改良版加速比'], [speedup_ratio], color='#2ca02c', width=0.4)
    ax2.set_title(f'改良版相对普通版加速比: {speedup_ratio:.2f}x', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(speedup_ratio * 1.2, 1.5))
    ax2.text(0, speedup_ratio + max(speedup_ratio * 0.1, 0.1),
             f'{speedup_ratio:.2f}x',
             ha='center', va='bottom', fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig