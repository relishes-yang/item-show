import cv2
import numpy as np
import os
import math
import easyocr
from pathlib import Path

# ======================== 可调参数 ========================
# 表盘检测参数
HOUGH_DP = 1.2
HOUGH_MIN_DIST_FACTOR = 0.2  # 圆心最小距离 = 对角线 * 因子
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
MIN_RADIUS = 80
MAX_RADIUS_FACTOR = 0.45  # 最大半径 = min(高,宽) * 因子

# 指针检测参数
CANNY_THRESH1 = 50
CANNY_THRESH2 = 150
HOUGH_LINE_THRESH = 30
MIN_LINE_LEN_FACTOR = 0.4
MAX_LINE_GAP = 8

# OCR 参数
OCR_LANGUAGES = ['en', 'ch_sim']  # 支持英文和简体中文
OCR_CONFIDENCE_THRESH = 0.5  # 识别置信度阈值
SCALE_RADIUS_RATIO = (0.7, 0.95)  # 刻度文字所在的半径范围（相对于表盘半径）

# 路径配置
INPUT_FOLDER = "./IMAGE_PATH"
OUTPUT_FOLDER = "./meter55555"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 支持的文件扩展名
img_list = [str(p) for p in Path(INPUT_FOLDER).glob("*.*") if p.suffix.lower() in ('.jpg', '.png', '.jpeg')]

# 初始化 EasyOCR reader（首次运行会下载模型）
reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)


# ======================== 表盘检测 ========================
def find_dial_robust(img):
    """检测圆形表盘，返回 (cx, cy, radius)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    diag = np.sqrt(h ** 2 + w ** 2)
    min_dist = int(diag * HOUGH_MIN_DIST_FACTOR)
    max_radius = int(min(h, w) * MAX_RADIUS_FACTOR)
    min_radius = max(MIN_RADIUS, int(max_radius * 0.3))

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=min_dist,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 选择半径最大的圆（表盘主体）
        largest = max(circles, key=lambda x: x[2])
        cx, cy, r = largest
        if 0 < cx < w and 0 < cy < h and r > 0:
            return cx, cy, r
    return None, None, None


# ======================== 指针角度检测 ========================
def detect_pointer_angle(img, cx, cy, r):
    """返回指针角度（0~360°，水平向右为0°，逆时针为正）"""
    # 提取表盘ROI并创建掩膜
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(img.shape[1], cx + r)
    y2 = min(img.shape[0], cy + r)
    roi = img[y1:y2, x1:x2]
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    roi_cx = cx - x1
    roi_cy = cy - y1
    cv2.circle(mask, (roi_cx, roi_cy), r, 255, -1)
    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

    # 边缘检测
    gray = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_THRESH1, CANNY_THRESH2)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    # 霍夫线检测
    min_line_len = int(r * MIN_LINE_LEN_FACTOR)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_LINE_THRESH,
                            minLineLength=min_line_len, maxLineGap=MAX_LINE_GAP)
    if lines is None:
        return None

    candidate_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        d1 = math.hypot(x1 - roi_cx, y1 - roi_cy)
        d2 = math.hypot(x2 - roi_cx, y2 - roi_cy)
        # 指针特征：一端靠近圆心（<0.2r），另一端远离（>0.35r）
        if (d1 < r * 0.2 and d2 > r * 0.35) or (d2 < r * 0.2 and d1 > r * 0.35):
            far_x, far_y = (x1, y1) if d1 > d2 else (x2, y2)
            angle = math.degrees(math.atan2(far_y - roi_cy, far_x - roi_cx)) % 360
            candidate_angles.append(angle)

    if not candidate_angles:
        return None

    # 去除异常值（取中位数）
    candidate_angles.sort()
    if len(candidate_angles) >= 3:
        candidate_angles = candidate_angles[1:-1]
    return np.median(candidate_angles)


# ======================== 自动识别量程和单位（OCR） ========================
def extract_scale_params(img, cx, cy, r):
    """
    从表盘上自动识别最小刻度值、最大刻度值、单位，以及它们对应的角度。
    返回: (min_value, max_value, unit, min_angle, max_angle)
    """
    # 1. 生成表盘区域的极坐标展开图（用于OCR定位）
    h, w = img.shape[:2]
    # 提取圆形ROI
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)
    dial_roi = img[y1:y2, x1:x2]
    roi_h, roi_w = dial_roi.shape[:2]
    roi_cx = cx - x1
    roi_cy = cy - y1

    # 创建掩膜，只保留表盘圆形区域
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.circle(mask, (roi_cx, roi_cy), r, 255, -1)
    dial_masked = cv2.bitwise_and(dial_roi, dial_roi, mask=mask)

    # 2. 使用 EasyOCR 识别文本
    # 将图像转为 RGB（EasyOCR 需要）
    rgb_image = cv2.cvtColor(dial_masked, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_image, paragraph=False)

    # 3. 解析每个识别结果
    scale_items = []  # 存储 (角度, 数值)
    unit = None

    for (bbox, text, conf) in results:
        if conf < OCR_CONFIDENCE_THRESH:
            continue
        # 计算文本边界框的中心点（在 dial_roi 坐标系中）
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        cx_text = np.mean(xs)
        cy_text = np.mean(ys)
        # 计算该点到圆心的距离和角度
        dx = cx_text - roi_cx
        dy = cy_text - roi_cy
        dist = math.hypot(dx, dy)
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360

        # 筛选属于刻度环的文本（距离在指定范围内）
        if SCALE_RADIUS_RATIO[0] * r <= dist <= SCALE_RADIUS_RATIO[1] * r:
            # 尝试将文本转换为数字
            try:
                # 移除可能的前后空格和单位字符
                clean_text = text.strip()
                # 简单处理：提取数字部分（包括小数点）
                import re
                num_match = re.search(r'[-+]?\d*\.?\d+', clean_text)
                if num_match:
                    value = float(num_match.group())
                    scale_items.append((angle, value))
            except:
                pass

            # 识别单位（常见压力单位）
            if unit is None:
                text_lower = text.lower()
                for unit_str in ['mpa', 'kpa', 'bar', 'psi', 'kg/cm²', 'kg/cm2']:
                    if unit_str in text_lower:
                        unit = unit_str.upper()
                        break
                if text_lower in ['mpa', 'kpa', 'bar', 'psi']:
                    unit = text.upper()

    if len(scale_items) < 2:
        # OCR 识别失败，返回默认值并打印警告
        print("警告：未能自动识别量程，使用默认值 0-16 MPa")
        return 0.0, 16.0, "MPa", 30.0, 330.0

    # 找出最小值和最大值对应的刻度
    min_item = min(scale_items, key=lambda x: x[1])
    max_item = max(scale_items, key=lambda x: x[1])
    min_value = min_item[1]
    max_value = max_item[1]
    min_angle = min_item[0]
    max_angle = max_item[0]

    # 确保角度顺序正确（如果 min_angle > max_angle，说明零位在右侧，需要加360）
    if min_angle > max_angle:
        # 将较小角度的值加360度处理（仅用于线性映射）
        # 注意：实际映射时需要处理角度跨越0度的情况
        # 这里简单处理，将 min_angle 视为起点，max_angle 视为终点，若 min_angle > max_angle，则 max_angle += 360
        max_angle += 360

    if unit is None:
        unit = "unknown"

    return min_value, max_value, unit, min_angle % 360, max_angle % 360


# ======================== 读数计算 ========================
def angle_to_reading(angle, min_angle, max_angle, min_value, max_value):
    """线性映射角度到读数，处理角度跨越0°的情况"""
    # 如果起始角度 > 终止角度（例如 330° -> 30° 跨越0°），需要调整
    if min_angle > max_angle:
        # 将指针角度也做偏移处理
        if angle < min_angle:
            angle += 360
        max_angle += 360
    # 限幅
    if angle < min_angle:
        angle = min_angle
    if angle > max_angle:
        angle = max_angle
    ratio = (angle - min_angle) / (max_angle - min_angle)
    return min_value + ratio * (max_value - min_value)


# ======================== 绘制结果 ========================
def draw_result(img, cx, cy, r, angle, reading, min_angle, max_angle, min_val, max_val, unit):
    result = img.copy()
    # 绘制表盘圆和圆心
    cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)
    # 绘制指针
    if angle is not None:
        end_x = int(cx + 0.85 * r * math.cos(math.radians(angle)))
        end_y = int(cy + 0.85 * r * math.sin(math.radians(angle)))
        cv2.line(result, (cx, cy), (end_x, end_y), (255, 0, 0), 3)
    # 显示读数
    text = f"{reading:.2f} {unit}"
    cv2.putText(result, text, (cx - r // 2, cy - r // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    # 显示自动识别的量程范围（调试信息）
    cv2.putText(result, f"Range: {min_val}~{max_val} {unit}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    return result


# ======================== 主程序 ========================
def main():
    for img_path in img_list:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        print(f"处理: {Path(img_path).name}")
        result_img = img.copy()

        # 1. 定位表盘
        cx, cy, r = find_dial_robust(img)
        if cx is None:
            cv2.putText(result_img, "Dial not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, Path(img_path).name), result_img)
            print("  表盘定位失败")
            continue

        # 2. 自动识别量程和单位
        min_val, max_val, unit, min_angle, max_angle = extract_scale_params(img, cx, cy, r)
        print(f"  识别结果: 量程 {min_val}~{max_val} {unit}, 角度范围 {min_angle:.1f}°~{max_angle:.1f}°")

        # 3. 检测指针角度
        angle = detect_pointer_angle(img, cx, cy, r)
        if angle is None:
            cv2.putText(result_img, "Pointer not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("  指针检测失败")
        else:
            # 4. 计算读数
            reading = angle_to_reading(angle, min_angle, max_angle, min_val, max_val)
            print(f"  指针角度: {angle:.1f}°, 读数: {reading:.2f} {unit}")
            result_img = draw_result(img, cx, cy, r, angle, reading, min_angle, max_angle, min_val, max_val, unit)

        # 保存结果
        out_path = os.path.join(OUTPUT_FOLDER, Path(img_path).name)
        cv2.imwrite(out_path, result_img)
        print(f"  已保存: {out_path}\n")

    print("批量处理完成！")


if __name__ == "__main__":
    main()