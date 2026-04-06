import numpy as np
import matplotlib.pyplot as plt

# 定义函数，计算迭代序列
def compute_sequence(b, k_max=20):
    r = (b - 1) / (b + 1)
    f0 = 0.5 * (b**2 + b)
    k = np.arange(k_max + 1)
    x = b * (r ** k)
    y = r ** k
    f = ((1 - b) / (1 + b)) ** k * f0
    return k, x, y, f

# 计算b=0.5和b=0.01的序列
k1, x1, y1, f1 = compute_sequence(0.5, k_max=20)
k2, x2, y2, f2 = compute_sequence(0.01, k_max=20)

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(k1, x1, label='b=0.5', marker='o')
plt.plot(k2, x2, label='b=0.01', marker='s')
plt.title('x_k')
plt.legend()
plt.grid(True)

plt.subplot(132)
plt.plot(k1, y1, label='b=0.5', marker='o')
plt.plot(k2, y2, label='b=0.01', marker='s')
plt.title('y_k')
plt.legend()
plt.grid(True)

plt.subplot(133)
plt.plot(k1, f1, label='b=0.5', marker='o')
plt.plot(k2, f2, label='b=0.01', marker='s')
plt.title('f_k')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()