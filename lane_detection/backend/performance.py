import time
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 适配Streamlit无GUI环境
import matplotlib.pyplot as plt

# 设置中文字体（解决Matplotlib中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_performance_test(original_img, binary_img, test_times=10):
    """
    【课程知识点】算法性能对比测试
    原理：重复运行多次算法，统计平均耗时，计算加速比
    :param original_img: 原始测试图像
    :param binary_img: 二值化测试图像
    :param test_times: 重复测试次数（取平均值，减少误差）
    :return: 普通版平均耗时、改良版平均耗时、加速比、耗时列表
    """
    from .hough_detector import hough_original_detect, hough_improved_detect

    t_original_list = []  # 普通霍夫耗时列表
    t_improved_list = []  # 改良霍夫耗时列表

    # 重复测试多次
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
    speedup_ratio = t_original_avg / t_improved_avg if t_improved_avg != 0 else 1  # 避免除零

    return t_original_avg, t_improved_avg, speedup_ratio, t_original_list, t_improved_list


def plot_performance_result(t_original_avg, t_improved_avg, speedup_ratio):
    """
    生成性能对比图（Matplotlib可视化，课程知识点：数据可视化）
    :param t_original_avg: 普通版平均耗时
    :param t_improved_avg: 改良版平均耗时
    :param speedup_ratio: 加速比
    :return: Matplotlib figure对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1：平均运行时间对比
    algorithms = ['普通霍夫变换', '改良霍夫变换']
    times = [t_original_avg, t_improved_avg]
    ax1.bar(algorithms, times, color=['#1f77b4', '#ff7f0e'], width=0.6)
    ax1.set_title('平均运行时间对比（秒）', fontsize=14)
    ax1.set_ylabel('运行时间 (s)')
    # 在柱状图上标注数值
    for i, v in enumerate(times):
        ax1.text(i, v + max(times) * 0.02, f'{v:.3f}s', ha='center', fontsize=12)

    # 子图2：加速比
    ax2.bar(['改良版加速比'], [speedup_ratio], color='#2ca02c', width=0.4)
    ax2.set_title(f'改良版相对普通版加速比: {speedup_ratio:.2f}x', fontsize=14)
    ax2.set_ylim(0, max(speedup_ratio * 1.2, 1.5))  # 设置y轴范围
    ax2.text(0, speedup_ratio + max(speedup_ratio * 0.1, 0.1),
             f'{speedup_ratio:.2f}x', ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout()  # 自动调整布局
    return fig