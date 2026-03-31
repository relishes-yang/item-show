import time
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .hough_detector import hough_original_detect, hough_improved_detect

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_performance_test(original_img, binary_img, test_times=10):
    """
    性能对比测试
    """
    t_original_list = []
    t_improved_list = []

    for _ in range(test_times):
        start = time.time()
        _, _, _ = hough_original_detect(original_img, binary_img)
        t_original = time.time() - start
        t_original_list.append(t_original)

        start = time.time()
        _, _, _ = hough_improved_detect(original_img, binary_img)
        t_improved = time.time() - start
        t_improved_list.append(t_improved)

    t_original_avg = np.mean(t_original_list)
    t_improved_avg = np.mean(t_improved_list)
    speedup_ratio = t_original_avg / t_improved_avg if t_improved_avg != 0 else 1

    return t_original_avg, t_improved_avg, speedup_ratio, t_original_list, t_improved_list


def plot_performance_result(t_original_avg, t_improved_avg, speedup_ratio):
    """
    生成性能对比图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    algorithms = ['普通霍夫变换', '改良霍夫变换']
    times = [t_original_avg, t_improved_avg]
    ax1.bar(algorithms, times, color=['#1f77b4', '#ff7f0e'], width=0.6)
    ax1.set_title('平均运行时间对比（秒）', fontsize=14)
    ax1.set_ylabel('运行时间 (s)')
    for i, v in enumerate(times):
        ax1.text(i, v + max(times) * 0.02, f'{v:.3f}s', ha='center', fontsize=12)

    ax2.bar(['加速比'], [speedup_ratio], color='#2ca02c', width=0.4)
    ax2.set_title(f'改良版相对普通版加速比: {speedup_ratio:.2f}x', fontsize=14)
    ax2.set_ylim(0, max(speedup_ratio * 1.2, 1.5))
    ax2.text(0, speedup_ratio + max(speedup_ratio * 0.1, 0.1),
             f'{speedup_ratio:.2f}x', ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig


# 兼容旧版导入的别名
def performance_test(binary_img_path: str, original_img_path: str, n_runs: int = 10, save_fig_path: str = None):
    binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_img_path)
    t1_avg, t2_avg, speedup, _, _ = run_performance_test(original_img, binary_img, n_runs)
    if save_fig_path:
        fig = plot_performance_result(t1_avg, t2_avg, speedup)
        fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    return {
        "original_avg_time": t1_avg,
        "improved_avg_time": t2_avg,
        "speedup_ratio": speedup
    }