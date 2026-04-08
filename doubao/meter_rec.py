# 工业指针式仪表识别 作业专用 新手零修改版
# ================== 唯一需要你改的地方 ==================
IMAGE_PATH = "./IMAGE_PATH/001.jpg"  # 改成你的指针表图片名字
MIN_SCALE = 0    # 你的仪表最小量程
MAX_SCALE = 1.6  # 你的仪表最大量程
# ========================================================

import cv2
import numpy as np
import math
import os

# 读取图片
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"错误：找不到图片，请检查路径是否正确！当前路径：{IMAGE_PATH}")
    exit()
img_original = img.copy()

# 图片预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 霍夫圆检测找表盘
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=100, param2=60, minRadius=50, maxRadius=500)

if circles is None:
    print("错误：未检测到圆形表盘，请确保图片清晰、表盘完整")
    exit()

# 提取表盘圆心和半径
circles = np.uint16(np.around(circles))
circle = circles[0][0]
cx, cy, radius = circle[0], circle[1], circle[2]
print(f"✅ 检测到表盘：圆心({cx},{cy})，半径{radius}像素")

# 画表盘和圆心
cv2.circle(img_original, (cx, cy), radius, (0, 255, 0), 2)
cv2.circle(img_original, (cx, cy), 2, (0, 0, 255), 3)

# 边缘检测+霍夫线检测找指针
edges = cv2.Canny(blur, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=radius*0.3, maxLineGap=10)

if lines is None:
    print("错误：未检测到指针，请确保图片清晰、指针无遮挡")
    exit()

# 筛选有效指针（经过圆心附近的线段）
pointer_line = None
min_dist = float('inf')
for line in lines:
    x1, y1, x2, y2 = line[0]
    # 计算线段到圆心的距离
    dist = abs((y2-y1)*cx - (x2-x1)*cy + x2*y1 - y2*x1) / math.sqrt((y2-y1)**2 + (x2-x1)**2)
    line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    # 筛选条件：距离圆心近、长度符合表盘范围
    if dist < radius*0.1 and line_length > radius*0.3 and line_length < radius*0.9:
        if dist < min_dist:
            min_dist = dist
            pointer_line = line[0]

if pointer_line is None:
    print("错误：未筛选到有效指针，请调整图片清晰度")
    exit()

x1, y1, x2, y2 = pointer_line
print(f"✅ 检测到指针：线段({x1},{y1})-({x2},{y2})")

# 画指针
cv2.line(img_original, (x1, y1), (x2, y2), (0, 0, 255), 3)

# 计算指针角度，换算仪表读数
# 确定指针端点（远离圆心的一端）
dist1 = math.sqrt((x1-cx)**2 + (y1-cy)**2)
dist2 = math.sqrt((x2-cx)**2 + (y2-cy)**2)
px, py = (x1, y1) if dist1 > dist2 else (x2, y2)

# 角度计算（适配工业仪表通用规则：0刻度在最左侧，顺时针增加）
angle = math.atan2(py - cy, px - cx) * 180 / math.pi
angle = angle + 360 if angle < 0 else angle

# 角度映射到量程
if 90 <= angle <= 270:
    scale_ratio = (270 - angle) / 180
else:
    scale_ratio = (270 - angle + 360) / 180
scale_ratio = max(0, min(1, scale_ratio))
reading = MIN_SCALE + scale_ratio * (MAX_SCALE - MIN_SCALE)

# 把读数写在图片上，交作业专用
cv2.putText(img_original, f"Reading: {reading:.2f}", (cx - radius, cy + radius + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 保存结果图片
save_path = "D:\\software\\python\\item-show\\doubao\\output\\"+os.path.basename(IMAGE_PATH)
cv2.imwrite(save_path, img_original)

# 最终结果输出
print("="*50)
print(f"📊 仪表量程：{MIN_SCALE} ~ {MAX_SCALE}")
print(f"🎯 最终识别读数：{reading:.2f}")
print(f"💾 作业用结果图已保存到：{save_path}")
print("="*50)