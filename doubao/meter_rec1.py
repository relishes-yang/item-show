# 工业指针表识别 批量处理简化版（不用Paddle）
import cv2
import numpy as np
import os
import math

# ====================== 仅需修改这2个路径 ======================
INPUT_FOLDER = "./IMAGE_PATH"  # 你的图片都放在这个文件夹里
OUTPUT_FOLDER = "./output_results"  # 处理后的结果会自动保存到这里
# 手动设置你的仪表量程（你的表是0-1.6kPa，不用改）
MIN_SCALE = 0
MAX_SCALE = 1.6
# ==============================================================

# 创建输出文件夹
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 获取文件夹里所有图片
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(image_extensions)]

if not image_files:
    print(f"错误：{INPUT_FOLDER}文件夹里没有找到图片！")
    exit()

print(f"✅ 找到{len(image_files)}张图片，开始批量处理...")

# ---------------------- 对每张图片进行处理 ----------------------
for idx, image_name in enumerate(image_files, 1):
    image_path = os.path.join(INPUT_FOLDER, image_name)
    print(f"\n--- 处理第{idx}/{len(image_files)}张：{image_name} ---")

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("警告：无法读取图片，跳过...")
        continue
    img_original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # ---------------------- 1. 检测表盘（霍夫圆） ----------------------
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=100,
                               param1=100, param2=60, minRadius=50, maxRadius=min(h, w) // 2)
    if circles is None:
        print("警告：未检测到表盘，跳过...")
        continue
    circles = np.uint16(np.around(circles))
    circle = circles[0][0]
    cx, cy, radius = circle[0], circle[1], circle[2]
    print(f"✅ 表盘检测成功：圆心({cx},{cy})，半径{radius}")

    # 创建表盘掩码，只保留表盘区域
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # ---------------------- 2. 优化指针检测（颜色分割+霍夫线） ----------------------
    _, thresh = cv2.threshold(masked_gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=radius * 0.4, maxLineGap=5)
    if lines is None:
        print("警告：未检测到指针，跳过...")
        continue

    # 筛选经过圆心的有效指针
    pointer_line = None
    min_dist = float('inf')
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / math.hypot(y2 - y1, x2 - x1)
        line_length = math.hypot(x2 - x1, y2 - y1)
        if dist < radius * 0.1 and radius * 0.3 < line_length < radius * 0.9:
            if dist < min_dist:
                min_dist = dist
                pointer_line = line[0]
    if pointer_line is None:
        print("警告：未找到有效指针，跳过...")
        continue
    x1, y1, x2, y2 = pointer_line
    print(f"✅ 指针检测成功：线段({x1},{y1})-({x2},{y2})")

    # ---------------------- 3. 修正角度计算（适配你的表盘） ----------------------
    # 确定指针的端点（远离圆心的一端）
    dist1 = math.hypot(x1 - cx, y1 - cy)
    dist2 = math.hypot(x2 - cx, y2 - cy)
    px, py = (x1, y1) if dist1 > dist2 else (x2, y2)

    # 计算指针角度（适配OpenCV图像坐标）
    dx = px - cx
    dy = py - cy
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    print(f"🔍 指针角度：{angle_deg:.2f}°")

    # ---------------------- 4. 映射角度到读数（适配你的0-1.6kPa表） ----------------------
    # 你的表盘：0刻度在左下（225°），1.6刻度在右下（45°），量程对应180°
    start_angle = 200  # 0刻度的角度，若读数不准可微调±10°
    end_angle = 145  # 满量程的角度，若读数不准可微调±10°
    total_angle_range = 180

    if angle_deg >= start_angle:
        offset = angle_deg - start_angle
    else:
        offset = (angle_deg + 360) - start_angle
    offset = offset % 360
    if offset > total_angle_range:
        offset = 360 - offset

    scale_ratio = offset / total_angle_range
    scale_ratio = max(0, min(1, scale_ratio))
    reading = MIN_SCALE + scale_ratio * (MAX_SCALE - MIN_SCALE)
    print(f"📊 读数：{reading:.2f} kPa")

    # ---------------------- 5. 在图片上绘制结果 ----------------------
    cv2.circle(img_original, (cx, cy), radius, (0, 255, 0), 2)
    cv2.circle(img_original, (cx, cy), 2, (0, 0, 255), 3)
    cv2.line(img_original, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 绘制清晰的读数（放在表盘下方，不遮挡刻度）
    text = f"read: {reading:.2f} kPa"
    text_x = max(10, int(cx - radius * 0.8))
    text_y = min(h - 10, int(cy + radius + 40))
    cv2.putText(img_original, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 保存结果
    output_path = os.path.join(OUTPUT_FOLDER, f"result_{image_name}")
    cv2.imwrite(output_path, img_original)
    print(f"💾 结果已保存到：{output_path}")

print(f"\n--- 批量处理完成！所有结果已保存到{OUTPUT_FOLDER}文件夹 ---")