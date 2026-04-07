import cv2
import numpy as np
import pytesseract
from PIL import Image


# 若Tesseract不在系统PATH中，请取消注释并设置路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_meter_circle(image, param2=30):
    """
    检测仪表盘圆形区域
    返回 (center_x, center_y, radius) 或 None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=param2, minRadius=50, maxRadius=500)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 选择半径最大的圆（假设表盘最大）
        x, y, r = max(circles, key=lambda c: c[2])
        return (x, y, r)
    return None


def polar_unwrap(image, center, radius, start_angle=-150, end_angle=150):
    """
    极坐标展开：将圆形表盘扇形区域展开为矩形图像
    image: 原始图像（BGR）
    center: (cx, cy)
    radius: 半径
    start_angle, end_angle: 角度范围（度）
    返回展开后的矩形图像（高度=半径，宽度=角度范围对应的像素数）
    """
    cx, cy = center
    # 转换为极坐标映射
    theta_range = np.radians(end_angle - start_angle)
    width = int(radius * theta_range)  # 弧长 ≈ 半径 * 弧度
    height = radius

    # 创建映射矩阵
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for i in range(height):  # 径向距离 r
        r = i
        for j in range(width):  # 角度 theta
            theta = start_angle + (j / width) * (end_angle - start_angle)
            theta_rad = np.radians(theta)
            x = cx + r * np.cos(theta_rad)
            y = cy + r * np.sin(theta_rad)
            map_x[i, j] = x
            map_y[i, j] = y

    unwrapped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return unwrapped


def find_pointer_in_unwrapped(unwrapped_gray):
    """
    在展开的矩形图中找到指针位置（垂直投影法）
    unwrapped_gray: 单通道灰度图
    返回指针所在的列索引（x坐标）
    """
    # 二值化：指针通常较亮
    _, binary = cv2.threshold(unwrapped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 垂直投影：每一列的白点数量
    projection = np.sum(binary == 255, axis=0)
    # 平滑投影曲线
    projection = cv2.GaussianBlur(projection.astype(np.float32), (5, 1), 0)
    # 找到峰值（指针最粗的位置）
    max_col = np.argmax(projection)
    return max_col


def unwrapped_col_to_angle(col, width, start_angle, end_angle):
    """将展开图中的列坐标转换为实际角度"""
    angle = start_angle + (col / width) * (end_angle - start_angle)
    return angle


def angle_to_reading(angle, min_angle, max_angle, min_val, max_val):
    """角度转读数（线性映射）"""
    if angle is None:
        return None
    proportion = (angle - min_angle) / (max_angle - min_angle)
    proportion = np.clip(proportion, 0, 1)
    return min_val + proportion * (max_val - min_val)


def preprocess_pointer_meter(roi):
    """预处理指针表盘ROI（备用，主要用于调试）"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray


def read_pointer_meter_advanced(image, config):
    """
    改进的指针式仪表读数（极坐标展开法）
    返回: (reading, info_dict)
    """
    h, w = image.shape[:2]
    # 1. 检测表盘圆形
    circle = detect_meter_circle(image, param2=config.get('hough_circle_param2', 30))
    if circle is None:
        return None, {"error": "未检测到表盘圆形"}
    cx, cy, r = circle

    # 2. 确保圆心和半径在图像内
    if cx - r < 0 or cx + r > w or cy - r < 0 or cy + r > h:
        # 调整半径（缩小到安全范围）
        r = min(cx, cy, w - cx, h - cy, r)

    # 3. 极坐标展开
    start_angle = config['min_angle']
    end_angle = config['max_angle']
    unwrapped = polar_unwrap(image, (cx, cy), r, start_angle, end_angle)
    if unwrapped is None or unwrapped.size == 0:
        return None, {"error": "极坐标展开失败"}

    # 4. 在展开图上找指针
    unwrapped_gray = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2GRAY)
    pointer_col = find_pointer_in_unwrapped(unwrapped_gray)
    height, width = unwrapped_gray.shape
    # 5. 列坐标转角度
    angle = unwrapped_col_to_angle(pointer_col, width, start_angle, end_angle)

    # 6. 角度转读数
    reading = angle_to_reading(angle, start_angle, end_angle,
                               config['min_value'], config['max_value'])

    # 7. 构造可视化信息
    info_dict = {
        "circle": (cx, cy, r),
        "angle": angle,
        "reading": reading,
        "reading_text": f"{reading:.3f} {config.get('unit', '')}",
        "unwrapped": unwrapped,
        "pointer_col": pointer_col
    }
    return reading, info_dict


def read_digital_meter_advanced(image, config):
    """
    改进的数字仪表识别（基于ROI裁剪 + 增强OCR）
    """
    # 简单实现：假设数字区域在图像中央偏下，可配置相对位置
    h, w = image.shape[:2]
    roi_rel = config.get('roi_relative', [0.3, 0.5, 0.7, 0.9])
    x1 = int(w * roi_rel[0])
    y1 = int(h * roi_rel[1])
    x2 = int(w * roi_rel[2])
    y2 = int(h * roi_rel[3])
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None, {"error": "数字区域无效"}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 增强对比度
    gray = cv2.equalizeHist(gray)
    # 自适应二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # 形态学闭运算，连接数字断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # OCR
    whitelist = config.get('ocr_whitelist', '0123456789.')
    psm = config.get('psm', 7)
    custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
    pil_img = Image.fromarray(binary)
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    # 提取数字和小数点
    reading_str = ''.join([c for c in text.strip() if c.isdigit() or c == '.'])
    try:
        reading = float(reading_str) if reading_str else None
    except:
        reading = None

    info_dict = {
        "roi": (x1, y1, x2, y2),
        "binary": binary,
        "reading_text": str(reading) if reading is not None else "None"
    }
    return reading, info_dict