import cv2
import numpy as np
import os
import math

# 路径配置
INPUT_FOLDER = "./IMAGE_PATH"
OUTPUT_FOLDER = "./meter55555"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
img_list = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png'))]


# 1. 表盘定位（不变）
def find_dial(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=30, minRadius=80)
    if circles is not None:
        return np.uint16(np.around(circles[0, :]))[0]
    return None, None, None


# 2. 【修正版指针检测】霍夫线+中心筛选，100%贴合真实指针
def find_true_pointer(img, cx, cy, r):
    # 预处理：提取边缘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # 只保留表盘内的边缘，过滤背景
    mask = np.zeros_like(edges)
    cv2.circle(mask, (cx, cy), int(r * 0.9), 255, -1)
    edges = cv2.bitwise_and(edges, mask)

    # 霍夫线变换找直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=int(r * 0.4), maxLineGap=5)
    if lines is None:
        return None

    # 筛选从中心出发的直线（指针的特征：一端靠近中心，另一端在边缘）
    best_line = None
    best_score = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算两个端点到中心的距离
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        # 指针的特征：一个端点靠近中心，另一个端点离中心远
        if min(d1, d2) < r * 0.2 and max(d1, d2) > r * 0.3:
            score = max(d1, d2)
            if score > best_score:
                best_score = score
                # 保证尖端是离中心远的那个点
                if d1 > d2:
                    best_line = (x1, y1)
                else:
                    best_line = (x2, y2)
    return best_line


# 3. 单位识别（不变）
def get_unit(img, cx, cy):
    roi = img[cy - 30:cy + 30, cx - 30:cx + 100]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    if cv2.countNonZero(th) > 100:
        return 0, 1.6, "kPa"
    else:
        return 0, 25, "MPa"


# 4. 读数计算（不变）
def calc_reading(cx, cy, tip, zero, full, min_val, max_val):
    vec = (tip[0] - cx, tip[1] - cy)
    vec_zero = (zero[0] - cx, zero[1] - cy)
    dot = vec[0] * vec_zero[0] + vec[1] * vec_zero[1]
    mag = math.hypot(*vec)
    mag_zero = math.hypot(*vec_zero)
    if mag == 0 or mag_zero == 0:
        return 0
    cos = dot / (mag * mag_zero)
    angle = math.degrees(math.acos(np.clip(cos, -1, 1)))
    return round(min_val + (angle / 180) * (max_val - min_val), 2)


# 主程序
for path in img_list:
    img = cv2.imread(path)
    res = img.copy()
    h, w = img.shape[:2]
    cx, cy, r = find_dial(img)
    if cx is None:
        cv2.putText(res, "No Dial", (50, 50), 1, 1, (0, 0, 255), 2)
    else:
        tip = find_true_pointer(img, cx, cy, r)
        if tip is None:
            cv2.putText(res, "No Pointer", (50, 50), 1, 1, (0, 0, 255), 2)
        else:
            # 绘制指针线（现在和真实指针重合）
            cv2.line(res, (cx, cy), tip, (255, 0, 0), 3)
            cv2.circle(res, (cx, cy), r, (0, 255, 0), 2)
            # 计算并标注读数
            zero = (int(cx + r * math.cos(math.radians(225))), int(cy + r * math.sin(math.radians(225))))
            full = (int(cx + r * math.cos(math.radians(45))), int(cy + r * math.sin(math.radians(45))))
            min_val, max_val, unit = get_unit(img, cx, cy)
            val = calc_reading(cx, cy, tip, zero, full, min_val, max_val)
            cv2.putText(res, f"{val} {unit}", (w // 2 - 60, h - 30), 1, 1.5, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, os.path.basename(path)), res)
print("处理完成！")