import cv2
import numpy as np
import math
import os
import sys
from typing import Tuple, List, Optional, Union
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============ 第一部分：配置与工具类 ============

class Config:
    """全局配置类 - 所有可调参数集中在这里"""

    # 图像预处理参数
    BLUR_KERNEL_SIZE = (5, 5)  # 高斯模糊核大小
    CANNY_LOW = 50  # Canny边缘检测低阈值
    CANNY_HIGH = 150  # Canny边缘检测高阈值
    DILATE_ITER = 2  # 膨胀迭代次数
    ERODE_ITER = 1  # 腐蚀迭代次数

    # 指针仪表参数
    POINTER_MIN_LENGTH = 50  # 指针最小长度（像素）
    POINTER_MAX_LENGTH = 300  # 指针最大长度（像素）
    POINTER_ANGLE_RANGE = 270  # 指针量程角度（默认270度）

    # 数字仪表参数
    DIGIT_MIN_WIDTH = 10  # 数字最小宽度
    DIGIT_MAX_WIDTH = 100  # 数字最大宽度
    DIGIT_MIN_HEIGHT = 20  # 数字最小高度
    DIGIT_MAX_HEIGHT = 150  # 数字最大高度
    DIGIT_ASPECT_RATIO = (0.2, 1.0)  # 数字宽高比范围

    # 颜色阈值（用于区分仪表类型）
    BLUE_LOWER = np.array([100, 50, 50])  # HSV蓝色下限
    BLUE_UPPER = np.array([130, 255, 255])  # HSV蓝色上限

    # 输出设置
    OUTPUT_DIR = "output"  # 输出目录
    SAVE_DEBUG_IMAGES = True  # 是否保存调试图片


class Utils:
    """工具类 - 提供各种静态方法"""

    @staticmethod
    def ensure_dir(directory: str):
        """确保目录存在，不存在则创建"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[创建目录] {directory}")

    @staticmethod
    def save_image(image: np.ndarray, filename: str, subdir: str = ""):
        """保存图片到输出目录"""
        if not Config.SAVE_DEBUG_IMAGES:
            return

        output_path = os.path.join(Config.OUTPUT_DIR, subdir)
        Utils.ensure_dir(output_path)

        filepath = os.path.join(output_path, filename)
        cv2.imwrite(filepath, image)
        print(f"[保存图片] {filepath}")

    @staticmethod
    def display_image(image: np.ndarray, title: str = "Image", wait: bool = True):
        """显示图片（调试用）"""
        # 调整显示尺寸如果图片太大
        h, w = image.shape[:2]
        max_size = 800
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            display_img = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            display_img = image

        cv2.imshow(title, display_img)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """计算两点间欧氏距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def calculate_angle(center: Tuple[int, int], point: Tuple[int, int]) -> float:
        """
        计算从中心点到某点的角度（0-360度，0度为3点钟方向，顺时针增加）
        仪表通常：0度在左边(9点)，顺时针增加到右边(3点)是270度
        """
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = math.degrees(math.atan2(dy, dx))

        # 转换为0-360度，0度在3点钟方向
        if angle < 0:
            angle += 360

        return angle


# ============ 第二部分：图像预处理模块 ============

class ImagePreprocessor:
    """
    图像预处理类
    功能：加载图片、降噪、增强对比度、边缘检测
    """

    def __init__(self):
        self.original_image = None
        self.gray_image = None
        self.blurred_image = None
        self.edges = None

    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图片文件

        参数:
            image_path: 图片文件路径

        返回:
            加载的图片（BGR格式）

        异常:
            FileNotFoundError: 文件不存在
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图片文件: {image_path}\n请检查路径是否正确！")

        # 检查文件扩展名
        valid_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in valid_ext:
            raise ValueError(f"不支持的图片格式: {ext}\n支持的格式: {valid_ext}")

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}\n文件可能已损坏或格式不支持")

        self.original_image = image.copy()
        print(f"[加载成功] 图片尺寸: {image.shape[1]}x{image.shape[0]}")
        return image

    def preprocess(self, image: np.ndarray) -> dict:
        """
        完整的预处理流程

        处理步骤:
        1. 转换为灰度图
        2. 高斯模糊降噪
        3. 自适应直方图均衡化增强对比度
        4. Canny边缘检测

        参数:
            image: 输入的BGR图片

        返回:
            包含各阶段处理结果的字典
        """
        results = {
            'original': image.copy(),
            'gray': None,
            'blurred': None,
            'enhanced': None,
            'edges': None,
            'binary': None
        }

        # 步骤1: 灰度转换
        # 原因：彩色信息对仪表读数识别无用，灰度图减少计算量
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray_image = gray
        results['gray'] = gray.copy()
        print("[预处理] 1/5 灰度转换完成")

        # 步骤2: 高斯模糊去噪
        # 原因：消除图片中的高频噪声，避免边缘检测时出现伪边缘
        blurred = cv2.GaussianBlur(gray, Config.BLUR_KERNEL_SIZE, 0)
        self.blurred_image = blurred
        results['blurred'] = blurred.copy()
        print("[预处理] 2/5 高斯模糊完成")

        # 步骤3: 自适应直方图均衡化(CLAHE)
        # 原因：增强对比度，使仪表刻度更清晰
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        results['enhanced'] = enhanced.copy()
        print("[预处理] 3/5 对比度增强完成")

        # 步骤4: Canny边缘检测
        # 原因：找出图像中的边缘，为后续轮廓检测做准备
        edges = cv2.Canny(enhanced, Config.CANY_LOW, Config.CANY_HIGH)
        self.edges = edges
        results['edges'] = edges.copy()
        print("[预处理] 4/5 边缘检测完成")

        # 步骤5: 二值化（备用）
        # 使用Otsu自动阈值二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        results['binary'] = binary

        # 形态学操作：先膨胀后腐蚀（闭运算），连接断开的边缘
        kernel = np.ones((3, 3), np.uint8)
        edges_morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,
                                       iterations=1)
        results['edges_morph'] = edges_morph
        print("[预处理] 5/5 形态学处理完成")

        return results

    def detect_circles(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """
        使用霍夫圆检测定位表盘

        参数:
            gray_image: 灰度图

        返回:
            检测到的圆 [x, y, radius]，如果没检测到返回None
        """
        # 霍夫圆检测参数
        # 参数说明：
        # cv2.HOUGH_GRADIENT: 检测方法
        # dp=1: 分辨率反比
        # minDist=100: 圆心之间的最小距离
        # param1=100: Canny边缘检测的高阈值
        # param2=30: 累加器阈值（越小检测到的圆越多）
        # minRadius/maxRadius: 半径范围

        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=50,
            maxRadius=min(gray_image.shape) // 2
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(f"[圆检测] 检测到 {len(circles[0])} 个圆")
            return circles[0]

        print("[圆检测] 未检测到圆形表盘")
        return None


# ============ 第三部分：仪表类型检测模块 ============

class MeterTypeDetector:
    """
    仪表类型检测器
    功能：自动判断是指针式仪表还是数字式仪表
    """

    # 类型常量
    TYPE_POINTER = "pointer"  # 指针式
    TYPE_DIGITAL = "digital"  # 数字式
    TYPE_UNKNOWN = "unknown"  # 未知

    def __init__(self):
        self.confidence = 0.0

    def detect(self, image: np.ndarray) -> Tuple[str, float]:
        """
        检测仪表类型

        判断逻辑（多特征综合）:
        1. 检测是否有圆形表盘（指针式通常有）
        2. 检测蓝色LCD区域（数字式通常有）
        3. 检测字符区域特征

        参数:
            image: 输入BGR图片

        返回:
            (类型, 置信度)
        """
        scores = {
            self.TYPE_POINTER: 0,
            self.TYPE_DIGITAL: 0
        }

        # 特征1: 检测圆形表盘（指针式特征）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
            param1=100, param2=40,
            minRadius=50, maxRadius=min(gray.shape) // 2
        )

        if circles is not None and len(circles[0]) > 0:
            scores[self.TYPE_POINTER] += 40  # 有圆形表盘，加分

        # 特征2: 检测蓝色LCD区域（数字式特征）
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, Config.BLUE_LOWER, Config.BLUE_UPPER)
        blue_ratio = np.sum(blue_mask > 0) / (blue_mask.shape[0] * blue_mask.shape[1])

        if blue_ratio > 0.05:  # 蓝色区域超过5%
            scores[self.TYPE_DIGITAL] += 50  # 强特征

        # 特征3: 检测数字轮廓特征
        digital_score = self._detect_digital_features(gray)
        scores[self.TYPE_DIGITAL] += digital_score

        # 特征4: 检测指针特征（长直线）
        pointer_score = self._detect_pointer_features(gray)
        scores[self.TYPE_POINTER] += pointer_score

        # 确定类型
        total = scores[self.TYPE_POINTER] + scores[self.TYPE_DIGITAL]
        if total == 0:
            return self.TYPE_UNKNOWN, 0.0

        if scores[self.TYPE_POINTER] > scores[self.TYPE_DIGITAL]:
            confidence = scores[self.TYPE_POINTER] / total
            return self.TYPE_POINTER, confidence
        else:
            confidence = scores[self.TYPE_DIGITAL] / total
            return self.TYPE_DIGITAL, confidence

    def _detect_digital_features(self, gray: np.ndarray) -> int:
        """
        检测数字特征：七段数码管样式或LCD数字的矩形特征

        返回: 分数 0-30
        """
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_like_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0

            # 数字通常有一定宽高比，且大小适中
            if (Config.DIGIT_MIN_WIDTH < w < Config.DIGIT_MAX_WIDTH and
                    Config.DIGIT_MIN_HEIGHT < h < Config.DIGIT_MAX_HEIGHT and
                    0.1 < aspect < 0.8):
                digit_like_count += 1

        # 如果有多个类似数字的区域，很可能是数字表
        return min(digit_like_count * 5, 30)

    def _detect_pointer_features(self, gray: np.ndarray) -> int:
        """
        检测指针特征：从中心发出的长直线

        返回: 分数 0-30
        """
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=50,
                                minLineLength=50,
                                maxLineGap=10)

        if lines is None:
            return 0

        # 统计长直线数量
        long_lines = 0
        height, width = gray.shape
        center = (width // 2, height // 2)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = Utils.calculate_distance((x1, y1), (x2, y2))

            # 检查是否经过中心区域
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            dist_to_center = Utils.calculate_distance(center, (mid_x, mid_y))

            # 长直线且经过中心附近，可能是指针
            if length > 80 and dist_to_center < min(height, width) // 3:
                long_lines += 1

        return min(long_lines * 10, 30)


# ============ 第四部分：指针式仪表识别模块 ============

class PointerMeterRecognizer:
    """
    指针式仪表识别器
    功能：定位表盘、识别指针、计算读数
    """

    def __init__(self):
        self.center = None
        self.radius = None
        self.pointer_angle = None
        self.zero_angle = None  # 0刻度角度
        self.full_angle = None  # 满量程角度

    def recognize(self, image: np.ndarray, preprocessed: dict) -> dict:
        """
        识别指针式仪表读数

        流程:
        1. 定位表盘中心和半径
        2. 检测指针位置
        3. 计算指针角度
        4. 转换为读数

        参数:
            image: 原始图片
            preprocessed: 预处理结果字典

        返回:
            包含读数、角度、可视化结果的字典
        """
        gray = preprocessed['gray']
        edges = preprocessed.get('edges_morph', preprocessed['edges'])

        # 步骤1: 定位表盘
        circle = self._locate_dial(gray)
        if circle is None:
            return {
                'success': False,
                'error': '无法定位表盘',
                'reading': None
            }

        self.center = (circle[0], circle[1])
        self.radius = circle[2]
        print(f"[指针表] 表盘中心: {self.center}, 半径: {self.radius}")

        # 步骤2: 检测刻度（用于确定0点和满量程位置）
        scale_info = self._detect_scales(gray, edges)

        # 步骤3: 检测指针
        pointer_result = self._detect_pointer(gray, edges)
        if pointer_result is None:
            return {
                'success': False,
                'error': '无法检测指针',
                'reading': None,
                'center': self.center,
                'radius': self.radius
            }

        pointer_angle, pointer_tip = pointer_result
        self.pointer_angle = pointer_angle

        # 步骤4: 计算读数
        # 默认假设：仪表是270度量程，0在左下方(225度)，满量程在右下方(315度)
        # 这是最常见的工业仪表布局
        reading = self._calculate_reading(pointer_angle, scale_info)

        # 生成可视化结果
        visualization = self._visualize(image, pointer_tip, reading)

        return {
            'success': True,
            'type': 'pointer',
            'reading': round(reading, 2),
            'unit': scale_info.get('unit', ''),
            'angle': round(pointer_angle, 2),
            'center': self.center,
            'radius': self.radius,
            'pointer_tip': pointer_tip,
            'range': scale_info.get('range', (0, 100)),
            'visualization': visualization
        }

    def _locate_dial(self, gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        定位表盘位置（圆心和半径）

        策略:
        1. 先尝试霍夫圆检测
        2. 如果失败，尝试轮廓检测找最大圆
        """
        # 方法1: 霍夫圆检测
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=100, param2=50,
            minRadius=50, maxRadius=min(gray.shape) // 2
        )

        if circles is not None:
            # 选择半径最大的圆（通常是表盘外圈）
            circles = np.uint16(np.around(circles[0]))
            best_circle = max(circles, key=lambda c: c[2])
            return tuple(best_circle)

        # 方法2: 轮廓检测（备用）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 找最大的近似圆形的轮廓
        best_circle = None
        best_score = 0

        for cnt in contours:
            if len(cnt) < 5:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            area = cv2.contourArea(cnt)
            circle_area = np.pi * radius * radius

            # 圆形度 = 轮廓面积 / 外接圆面积，越接近1越圆
            circularity = area / circle_area if circle_area > 0 else 0

            if circularity > 0.7 and radius > 50:  # 至少70%圆度
                score = area * circularity
                if score > best_score:
                    best_score = score
                    best_circle = (int(x), int(y), int(radius))

        return best_circle

    def _detect_scales(self, gray: np.ndarray, edges: np.ndarray) -> dict:
        """
        检测刻度线和数字，确定量程

        返回:
            包含range(最小值,最大值), unit(单位), zero_angle, full_angle的字典
        """
        # 创建掩码，只在表盘区域内搜索
        mask = np.zeros_like(gray)
        if self.center and self.radius:
            cv2.circle(mask, self.center, int(self.radius * 0.9), 255, -1)

        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

        # 查找轮廓，筛选出刻度线（短直线）
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        scale_angles = []
        for cnt in contours:
            length = cv2.arcLength(cnt, False)
            if 10 < length < self.radius * 0.4:  # 刻度线长度范围
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    angle = Utils.calculate_angle(self.center, (cx, cy))
                    scale_angles.append(angle)

        # 默认量程假设（270度仪表）
        # 如果没有检测到足够的刻度，使用默认值
        if len(scale_angles) < 3:
            return {
                'range': (0, 100),
                'unit': '',
                'zero_angle': 225,  # 左下
                'full_angle': 315  # 右下（顺时针270度）
            }

        # 根据刻度分布确定0点和满量程点
        scale_angles.sort()

        # 简单假设：最小角度是0点，最大角度是满量程（适用于270度表）
        # 实际应用中可能需要更复杂的逻辑识别数字

        return {
            'range': (0, 100),
            'unit': '',
            'zero_angle': min(scale_angles),
            'full_angle': max(scale_angles)
        }

    def _detect_pointer(self, gray: np.ndarray, edges: np.ndarray) -> Optional[Tuple[float, Tuple[int, int]]]:
        """
        检测指针位置和角度

        策略:
        1. 在表盘中心区域搜索直线
        2. 选择最长且经过中心的直线作为指针
        3. 计算指针角度

        返回:
            (角度, 指针尖端坐标)
        """
        # 创建ROI掩码，排除表盘外围和中心小圆
        mask = np.zeros_like(gray)
        if self.center and self.radius:
            # 外圈限制
            cv2.circle(mask, self.center, int(self.radius * 0.85), 255, -1)
            # 内圈挖空（去除中心轴）
            cv2.circle(mask, self.center, int(self.radius * 0.1), 0, -1)

        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

        # 霍夫直线检测
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=int(self.radius * 0.3),  # 指针至少占半径30%
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            # 备用方案：直接搜索最亮/最暗的线（根据指针颜色）
            return self._detect_pointer_by_color(gray)

        # 找到最佳指针候选
        best_line = None
        best_score = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算直线中点
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

            # 距离中心的距离（越近越好）
            dist_to_center = Utils.calculate_distance(self.center, (mid_x, mid_y))

            # 直线长度
            length = Utils.calculate_distance((x1, y1), (x2, y2))

            # 检查是否经过中心：计算中心到直线的距离
            # 直线方程: (y2-y1)x - (x2-x1)y + (x2y1-x1y2) = 0
            if length > 0:
                dist_line_to_center = abs(
                    (y2 - y1) * self.center[0] - (x2 - x1) * self.center[1] + x2 * y1 - y1 * x2) / length
            else:
                dist_line_to_center = float('inf')

            # 评分：长 + 经过中心 + 距离中心适中
            if dist_line_to_center < self.radius * 0.2:  # 直线经过中心
                score = length * (1 - dist_line_to_center / (self.radius + 1))
                if score > best_score:
                    best_score = score
                    best_line = (x1, y1, x2, y2)

        if best_line is None:
            return None

        # 计算指针角度（从中心指向远端）
        x1, y1, x2, y2 = best_line

        # 确定哪个端点离中心更远（指针尖端）
        dist1 = Utils.calculate_distance(self.center, (x1, y1))
        dist2 = Utils.calculate_distance(self.center, (x2, y2))

        if dist1 > dist2:
            tip = (x1, y1)
            tail = (x2, y2)
        else:
            tip = (x2, y2)
            tail = (x1, y1)

        # 计算角度
        angle = Utils.calculate_angle(self.center, tip)

        return angle, tip

    def _detect_pointer_by_color(self, gray: np.ndarray) -> Optional[Tuple[float, Tuple[int, int]]]:
        """
        备用方案：通过颜色/亮度检测指针（红色指针在灰度图中较暗）
        """
        # 在中心区域搜索最暗的径向线（假设指针是黑色或红色）
        height, width = gray.shape
        cx, cy = self.center

        best_angle = 0
        best_contrast = 0
        best_tip = (cx, cy)

        # 每隔5度搜索一次
        for angle in range(0, 360, 5):
            rad = math.radians(angle)

            # 沿该半径采样
            samples = []
            for r in range(int(self.radius * 0.2), int(self.radius * 0.8), 5):
                x = int(cx + r * math.cos(rad))
                y = int(cy + r * math.sin(rad))

                if 0 <= x < width and 0 <= y < height:
                    samples.append(gray[y, x])

            if len(samples) > 5:
                # 计算对比度（标准差）
                contrast = np.std(samples)
                mean_val = np.mean(samples)

                # 指针通常是暗的（低均值）且有对比度
                if mean_val < 100 and contrast > best_contrast:
                    best_contrast = contrast
                    best_angle = angle
                    best_tip = (int(cx + self.radius * 0.7 * math.cos(rad)),
                                int(cy + self.radius * 0.7 * math.sin(rad)))

        if best_contrast > 10:
            return float(best_angle), best_tip

        return None

    def _calculate_reading(self, pointer_angle: float, scale_info: dict) -> float:
        """
        将指针角度转换为读数

        参数:
            pointer_angle: 指针角度（0-360，0在3点钟方向）
            scale_info: 刻度信息

        返回:
            读数值
        """
        min_val, max_val = scale_info.get('range', (0, 100))
        zero_angle = scale_info.get('zero_angle', 225)  # 默认左下
        full_angle = scale_info.get('full_angle', 315)  # 默认右下

        # 处理角度环绕（例如从350度到10度）
        total_range = Config.POINTER_ANGLE_RANGE

        # 将角度转换为相对于0刻度的角度
        relative_angle = pointer_angle - zero_angle
        if relative_angle < 0:
            relative_angle += 360

        # 如果超过了满量程角度，限制在满量程
        if relative_angle > total_range:
            if relative_angle > 300:  # 接近0点
                relative_angle = 0
            else:
                relative_angle = total_range

        # 线性插值计算读数
        ratio = relative_angle / total_range
        reading = min_val + ratio * (max_val - min_val)

        return reading

    def _visualize(self, image: np.ndarray, pointer_tip: Tuple[int, int],
                   reading: float) -> np.ndarray:
        """生成可视化结果图"""
        vis = image.copy()

        # 画表盘圆
        if self.center and self.radius:
            cv2.circle(vis, self.center, self.radius, (0, 255, 0), 2)
            cv2.circle(vis, self.center, 5, (0, 0, 255), -1)  # 中心点

            # 画指针
            cv2.line(vis, self.center, pointer_tip, (255, 0, 0), 3)
            cv2.circle(vis, pointer_tip, 5, (255, 0, 0), -1)

        # 添加读数文字
        text = f"Reading: {reading:.2f}"
        cv2.putText(vis, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        return vis


# ============ 第五部分：数字式仪表识别模块 ============

class DigitalMeterRecognizer:
    """
    数字式仪表识别器（OCR）
    功能：定位数字区域、分割字符、识别数字
    """

    def __init__(self):
        self.digit_roi = None  # 数字区域
        self.digits = []  # 分割出的数字图片

    def recognize(self, image: np.ndarray, preprocessed: dict) -> dict:
        """
        识别数字式仪表读数

        流程:
        1. 定位数字显示区域（LCD屏幕）
        2. 分割单个数字
        3. 模板匹配或特征识别数字
        4. 组合读数

        参数:
            image: 原始图片
            preprocessed: 预处理结果

        返回:
            包含读数、置信度、可视化结果的字典
        """
        gray = preprocessed['gray']

        # 步骤1: 定位数字区域
        roi, roi_coords = self._locate_digital_area(image, gray)
        if roi is None:
            return {
                'success': False,
                'error': '无法定位数字显示区域',
                'reading': None
            }

        self.digit_roi = roi
        print(f"[数字表] 找到数字区域: {roi.shape[1]}x{roi.shape[0]}")

        # 步骤2: 分割数字
        digit_images = self._segment_digits(roi)
        if not digit_images:
            return {
                'success': False,
                'error': '无法分割数字',
                'reading': None
            }

        print(f"[数字表] 分割出 {len(digit_images)} 个字符")

        # 步骤3: 识别每个数字
        recognized_digits = []
        confidences = []

        for i, digit_img in enumerate(digit_images):
            digit, conf = self._recognize_digit(digit_img)
            recognized_digits.append(digit)
            confidences.append(conf)
            print(f"  字符{i + 1}: '{digit}' (置信度: {conf:.2f})")

        # 组合结果
        reading_str = ''.join(recognized_digits)

        # 处理小数点（简单启发式：如果识别结果中有特殊标记）
        reading = self._parse_reading(reading_str)

        # 生成可视化
        visualization = self._visualize(image, roi_coords, digit_images,
                                        recognized_digits, reading)

        avg_confidence = np.mean(confidences) if confidences else 0

        return {
            'success': True,
            'type': 'digital',
            'reading': reading,
            'raw_string': reading_str,
            'confidence': round(avg_confidence, 3),
            'digit_count': len(digit_images),
            'visualization': visualization
        }

    def _locate_digital_area(self, image: np.ndarray, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """
        定位数字显示区域（LCD屏幕）

        策略:
        1. 检测蓝色/绿色LCD背景
        2. 查找具有高对比度的矩形区域
        3. 查找具有多个等高矩形的区域（七段数码管特征）

        返回:
            (ROI图片, (x, y, w, h)) 或 (None, None)
        """
        height, width = image.shape[:2]

        # 方法1: 颜色检测（LCD通常是蓝色或绿色背景）
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 蓝色LCD
        blue_mask = cv2.inRange(hsv, Config.BLUE_LOWER, Config.BLUE_UPPER)
        # 绿色LCD
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        color_mask = cv2.bitwise_or(blue_mask, green_mask)

        # 方法2: 查找具有高对比度的暗色文字区域
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # 合并掩码
        combined_mask = cv2.bitwise_or(color_mask, dark_mask)

        # 形态学操作连接区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        best_roi = None
        best_score = 0
        best_coords = None

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # 筛选条件：宽度大于高度、面积适中、位置通常在图片中上部
            if (w > h * 2 and
                    area > 1000 and
                    h > Config.DIGIT_MIN_HEIGHT and
                    y < height * 0.7):  # 不在最底部

                # 检查区域内是否有多个垂直条（七段数码管特征）
                roi = gray[y:y + h, x:x + w]
                score = self._score_digital_roi(roi)

                if score > best_score:
                    best_score = score
                    best_roi = roi
                    best_coords = (x, y, w, h)

        # 如果没找到，尝试全局搜索
        if best_roi is None:
            # 查找具有最多轮廓的区域（通常是数字区域）
            best_roi, best_coords = self._find_digit_area_by_contours(gray)

        return best_roi, best_coords

    def _score_digital_roi(self, roi: np.ndarray) -> float:
        """
        评分ROI是否像数字区域（七段数码管特征）
        """
        if roi.size == 0:
            return 0

        # 二值化
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 统计类似数字段的矩形
        segment_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0

            # 七段数码管的段通常是细长的
            if (0.1 < aspect < 0.4 or 2.5 < aspect < 10) and w * h > 50:
                segment_count += 1

        # 数字通常有多个段（至少7段一个数字）
        return segment_count

    def _find_digit_area_by_contours(self, gray: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """
        备用方案：通过轮廓分布查找数字区域
        """
        height, width = gray.shape

        # 滑动窗口搜索
        best_count = 0
        best_box = (0, int(height * 0.2), width, int(height * 0.6))

        # 使用默认区域（图片中间）
        x, y, w, h = best_box
        roi = gray[y:y + h, x:x + w]

        return roi, best_box

    def _segment_digits(self, roi: np.ndarray) -> List[np.ndarray]:
        """
        分割ROI中的单个数字

        策略:
        1. 垂直投影分割
        2. 查找等宽的字符区域
        3. 过滤非数字区域
        """
        if roi is None or roi.size == 0:
            return []

        # 预处理ROI
        # 根据背景色决定二值化方式
        mean_val = np.mean(roi)
        if mean_val > 127:
            # 背景亮，文字暗
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # 背景暗，文字亮
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 垂直投影（统计每列的白色像素数）
        vertical_proj = np.sum(binary, axis=0)

        # 查找字符边界（投影值高的区域是字符）
        height, width = binary.shape
        in_char = False
        char_start = 0
        char_boundaries = []

        threshold = np.max(vertical_proj) * 0.1  # 10%最大值为阈值

        for i, proj in enumerate(vertical_proj):
            if not in_char and proj > threshold:
                in_char = True
                char_start = i
            elif in_char and proj < threshold:
                in_char = False
                if i - char_start > 5:  # 最小宽度
                    char_boundaries.append((char_start, i))

        # 如果边界太少，尝试固定宽度分割
        if len(char_boundaries) < 2:
            char_boundaries = self._fixed_width_segmentation(binary)

        # 提取每个字符
        digits = []
        for start, end in char_boundaries:
            # 扩展边界
            start = max(0, start - 2)
            end = min(width, end + 2)

            char_img = binary[:, start:end]

            # 清理并标准化
            char_img = self._normalize_digit(char_img)

            if char_img is not None:
                digits.append(char_img)

        return digits

    def _fixed_width_segmentation(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """
        固定宽度分割（备用方案）
        """
        height, width = binary.shape

        # 估计字符宽度（假设有4-6位数字）
        for num_digits in range(4, 7):
            char_width = width // num_digits
            boundaries = []
            for i in range(num_digits):
                start = i * char_width
                end = (i + 1) * char_width
                boundaries.append((start, end))
            return boundaries

        return [(0, width)]

    def _normalize_digit(self, char_img: np.ndarray) -> Optional[np.ndarray]:
        """
        标准化数字图片为统一尺寸（28x28）
        """
        if char_img.size == 0:
            return None

        # 去除边缘空白
        rows = np.any(char_img, axis=1)
        cols = np.any(char_img, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        char_crop = char_img[y1:y2 + 1, x1:x2 + 1]

        # 调整大小为28x28，保持宽高比
        h, w = char_crop.shape
        scale = 20 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        if new_h < 5 or new_w < 5:
            return None

        resized = cv2.resize(char_crop, (new_w, new_h))

        # 放入28x28画布中心
        normalized = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        normalized[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return normalized

    def _recognize_digit(self, digit_img: np.ndarray) -> Tuple[str, float]:
        """
        识别单个数字图片

        使用简单的模板匹配（七段数码管特征）
        返回: (字符, 置信度)
        """
        if digit_img is None:
            return '?', 0.0

        # 七段数码管识别逻辑
        # 将28x28图像分为7个区域，检测每个段是否点亮

        h, w = digit_img.shape
        segment_on = []

        # 定义7段的位置（归一化坐标）
        segments = [
            (0.5, 0.15),  # 上 (a)
            (0.85, 0.35),  # 右上 (b)
            (0.85, 0.65),  # 右下 (c)
            (0.5, 0.85),  # 下 (d)
            (0.15, 0.65),  # 左下 (e)
            (0.15, 0.35),  # 左上 (f)
            (0.5, 0.5),  # 中 (g)
        ]

        for sx, sy in segments:
            x = int(sx * w)
            y = int(sy * h)
            # 检查该区域的平均亮度
            region = digit_img[max(0, y - 2):min(h, y + 2), max(0, x - 2):min(w, x + 2)]
            is_on = np.mean(region) > 127
            segment_on.append(is_on)

        a, b, c, d, e, f, g = segment_on

        # 七段数码管解码表
        patterns = {
            (True, True, True, True, True, True, False): '0',
            (False, True, True, False, False, False, False): '1',
            (True, True, False, True, True, False, True): '2',
            (True, True, True, True, False, False, True): '3',
            (False, True, True, False, False, True, True): '4',
            (True, False, True, True, False, True, True): '5',
            (True, False, True, True, True, True, True): '6',
            (True, True, True, False, False, False, False): '7',
            (True, True, True, True, True, True, True): '8',
            (True, True, True, True, False, True, True): '9',
        }

        pattern = tuple(segment_on)
        if pattern in patterns:
            return patterns[pattern], 0.9
        else:
            # 模糊匹配：找最相似的模式
            best_match = '?'
            best_score = 0
            for pat, digit in patterns.items():
                score = sum(a == b for a, b in zip(pattern, pat)) / 7
                if score > best_score:
                    best_score = score
                    best_match = digit

            return best_match, best_score

    def _parse_reading(self, raw_string: str) -> Union[int, float]:
        """
        解析读数字符串，处理小数点
        """
        # 简单处理：如果长度>3，假设最后一位是小数位
        # 实际应用中需要更复杂的逻辑检测小数点
        try:
            # 尝试直接转换
            if '.' in raw_string:
                return float(raw_string)
            else:
                return int(raw_string)
        except ValueError:
            # 清理非数字字符后尝试
            cleaned = ''.join(c for c in raw_string if c.isdigit() or c == '.')
            if cleaned:
                return float(cleaned) if '.' in cleaned else int(cleaned)
            return 0

    def _visualize(self, image: np.ndarray, roi_coords: Tuple,
                   digit_images: List[np.ndarray],
                   recognized_digits: List[str],
                   reading: Union[int, float]) -> np.ndarray:
        """生成可视化结果"""
        vis = image.copy()

        if roi_coords:
            x, y, w, h = roi_coords
            # 画数字区域框
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 在区域上方显示读数
            text = f"Reading: {reading}"
            cv2.putText(vis, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # 添加整体读数
        cv2.putText(vis, f"Digital: {reading}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return vis


# ============ 第六部分：主控制系统 ============

class MeterReadingSystem:
    """
    电力仪表读数识别系统主类
    整合所有模块，提供统一的识别接口
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.type_detector = MeterTypeDetector()
        self.pointer_recognizer = PointerMeterRecognizer()
        self.digital_recognizer = DigitalMeterRecognizer()

        # 确保输出目录存在
        Utils.ensure_dir(Config.OUTPUT_DIR)

    def process(self, image_path: str, meter_type: str = "auto") -> dict:
        """
        处理单张图片

        参数:
            image_path: 图片文件路径
            meter_type: "auto"(自动检测), "pointer"(指针式), "digital"(数字式)

        返回:
            完整的识别结果字典
        """
        print(f"\n{'=' * 50}")
        print(f"开始处理: {image_path}")
        print(f"{'=' * 50}")

        result = {
            'input_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'meter_type': None,
            'type_confidence': 0,
            'reading': None,
            'visualization_path': None
        }

        try:
            # 步骤1: 加载图片
            image = self.preprocessor.load_image(image_path)
            Utils.save_image(image, "01_original.jpg")

            # 步骤2: 预处理
            print("\n[阶段] 图像预处理...")
            preprocessed = self.preprocessor.preprocess(image)

            # 保存预处理结果
            Utils.save_image(preprocessed['gray'], "02_gray.jpg")
            Utils.save_image(preprocessed['edges'], "03_edges.jpg")

            # 步骤3: 检测仪表类型
            if meter_type == "auto":
                print("\n[阶段] 自动检测仪表类型...")
                detected_type, confidence = self.type_detector.detect(image)
                result['meter_type'] = detected_type
                result['type_confidence'] = round(confidence, 3)
                print(f"[结果] 检测到: {detected_type} (置信度: {confidence:.2%})")
            else:
                result['meter_type'] = meter_type
                result['type_confidence'] = 1.0
                print(f"[设置] 强制指定类型: {meter_type}")

            # 步骤4: 根据类型选择识别器
            print(f"\n[阶段] 开始识别...")
            if result['meter_type'] == 'pointer':
                recognition_result = self.pointer_recognizer.recognize(image, preprocessed)
            elif result['meter_type'] == 'digital':
                recognition_result = self.digital_recognizer.recognize(image, preprocessed)
            else:
                # 未知类型，两种都尝试
                print("[警告] 未知类型，尝试两种识别方式...")
                ptr_result = self.pointer_recognizer.recognize(image, preprocessed)
                dig_result = self.digital_recognizer.recognize(image, preprocessed)

                if ptr_result['success'] and not dig_result['success']:
                    recognition_result = ptr_result
                    result['meter_type'] = 'pointer'
                elif dig_result['success'] and not ptr_result['success']:
                    recognition_result = dig_result
                    result['meter_type'] = 'digital'
                else:
                    # 两者都成功或都失败，选择置信度高的
                    ptr_conf = ptr_result.get('confidence', 0) if ptr_result['success'] else 0
                    dig_conf = dig_result.get('confidence', 0) if dig_result['success'] else 0

                    if ptr_conf > dig_conf:
                        recognition_result = ptr_result
                        result['meter_type'] = 'pointer'
                    else:
                        recognition_result = dig_result
                        result['meter_type'] = 'digital'

            # 步骤5: 整合结果
            if recognition_result['success']:
                result['success'] = True
                result['reading'] = recognition_result['reading']

                # 保存可视化结果
                vis_image = recognition_result['visualization']
                vis_filename = f"result_{os.path.basename(image_path)}"
                Utils.save_image(vis_image, vis_filename)
                result['visualization_path'] = os.path.join(Config.OUTPUT_DIR, vis_filename)

                # 显示结果摘要
                print(f"\n{'=' * 50}")
                print(f"[识别成功]")
                print(f"仪表类型: {result['meter_type']}")
                print(f"读数: {result['reading']}")
                if 'unit' in recognition_result:
                    print(f"单位: {recognition_result['unit']}")
                print(f"{'=' * 50}")
            else:
                result['error'] = recognition_result.get('error', '识别失败')
                print(f"\n[错误] {result['error']}")

        except Exception as e:
            result['error'] = str(e)
            print(f"\n[异常] {str(e)}")
            import traceback
            traceback.print_exc()

        return result

    def batch_process(self, image_dir: str, meter_type: str = "auto") -> List[dict]:
        """
        批量处理目录中的所有图片

        参数:
            image_dir: 图片目录路径
            meter_type: 仪表类型

        返回:
            结果列表
        """
        print(f"\n开始批量处理目录: {image_dir}")

        if not os.path.exists(image_dir):
            print(f"[错误] 目录不存在: {image_dir}")
            return []

        # 支持的图片格式
        valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']

        image_files = [f for f in os.listdir(image_dir)
                       if os.path.splitext(f)[1].lower() in valid_ext]

        if not image_files:
            print(f"[警告] 目录中没有图片文件")
            return []

        print(f"发现 {len(image_files)} 个图片文件")

        results = []
        for filename in image_files:
            filepath = os.path.join(image_dir, filename)
            result = self.process(filepath, meter_type)
            results.append(result)

        # 保存汇总报告
        self._save_report(results)

        return results

    def _save_report(self, results: List[dict]):
        """保存处理报告"""
        report_path = os.path.join(Config.OUTPUT_DIR, "report.json")

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n[报告已保存] {report_path}")

        # 打印统计
        total = len(results)
        success = sum(1 for r in results if r['success'])
        print(f"\n处理统计:")
        print(f"  总计: {total}")
        print(f"  成功: {success}")
        print(f"  失败: {total - success}")
        print(f"  成功率: {success / total * 100:.1f}%" if total > 0 else "N/A")


# ============ 第七部分：测试与演示 ============

def create_test_image_pointer(filename: str = "test_pointer.jpg"):
    """
    创建模拟指针式仪表测试图（用于演示）
    """
    # 创建黑色背景
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    center = (200, 200)
    radius = 150

    # 画表盘外圆（白色）
    cv2.circle(img, center, radius, (255, 255, 255), 3)
    cv2.circle(img, center, 10, (0, 0, 255), -1)  # 中心轴

    # 画刻度（270度量程，从225度到315度）
    for i in range(11):
        angle = 225 + i * 27  # 270度分10格
        rad = math.radians(angle)
        x1 = int(center[0] + (radius - 20) * math.cos(rad))
        y1 = int(center[1] + (radius - 20) * math.sin(rad))
        x2 = int(center[0] + radius * math.cos(rad))
        y2 = int(center[1] + radius * math.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # 标注数字
        label = str(i * 10)
        tx = int(center[0] + (radius - 40) * math.cos(rad))
        ty = int(center[1] + (radius - 40) * math.sin(rad))
        cv2.putText(img, label, (tx - 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 画指针（指向65，即225 + 6.5*27 = 400.5度）
    pointer_angle = 400.5
    rad = math.radians(pointer_angle)
    tip_x = int(center[0] + (radius - 30) * math.cos(rad))
    tip_y = int(center[1] + (radius - 30) * math.sin(rad))
    cv2.line(img, center, (tip_x, tip_y), (0, 0, 255), 4)

    # 添加标签
    cv2.putText(img, "VOLTAGE", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(filename, img)
    print(f"[创建测试图] {filename} (模拟指针式电压表，读数应为65)")
    return filename


def create_test_image_digital(filename: str = "test_digital.jpg"):
    """
    创建模拟数字式仪表测试图（用于演示）
    """
    # 创建深蓝色背景（模拟LCD）
    img = np.ones((200, 400, 3), dtype=np.uint8) * 50
    img[:, :] = (139, 0, 0)  # 深蓝色背景

    # LCD区域（稍亮的蓝色）
    cv2.rectangle(img, (50, 50), (350, 150), (200, 50, 50), -1)

    # 模拟七段数码管显示 "123.4"
    def draw_segment(img, x, y, w, h, color, thickness=3):
        # 画一个矩形段
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # 简化的数字显示（使用OpenCV文字模拟）
    cv2.putText(img, "123.4", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)
    cv2.putText(img, "A", (320, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imwrite(filename, img)
    print(f"[创建测试图] {filename} (模拟数字式电流表，读数应为123.4)")
    return filename


def main():
    """
    主函数 - 完整执行流程演示
    """
    print("=" * 60)
    print("电力仪表读数识别系统 - 演示程序")
    print("=" * 60)

    # 初始化系统
    system = MeterReadingSystem()

    # 创建测试图片（如果没有真实图片）
    test_dir = "test_images"
    Utils.ensure_dir(test_dir)

    # 创建测试图片
    pointer_test = create_test_image_pointer(os.path.join(test_dir, "pointer_meter.jpg"))
    digital_test = create_test_image_digital(os.path.join(test_dir, "digital_meter.jpg"))

    print("\n" + "=" * 60)
    print("测试1: 指针式仪表识别")
    print("=" * 60)

    # 处理指针式仪表
    result1 = system.process(pointer_test, meter_type="pointer")

    print("\n" + "=" * 60)
    print("测试2: 数字式仪表识别")
    print("=" * 60)

    # 处理数字式仪表
    result2 = system.process(digital_test, meter_type="digital")

    print("\n" + "=" * 60)
    print("测试3: 自动类型检测（指针式）")
    print("=" * 60)

    # 自动检测类型
    result3 = system.process(pointer_test, meter_type="auto")

    # 显示结果汇总
    print("\n" + "=" * 60)
    print("处理结果汇总")
    print("=" * 60)
    for i, r in enumerate([result1, result2, result3], 1):
        status = "✓ 成功" if r['success'] else "✗ 失败"
        print(f"\n测试{i}: {status}")
        print(f"  文件: {os.path.basename(r['input_path'])}")
        print(f"  类型: {r['meter_type']} (置信度: {r['type_confidence']:.2%})")
        if r['success']:
            print(f"  读数: {r['reading']}")
        else:
            print(f"  错误: {r['error']}")

    print("\n" + "=" * 60)
    print("演示完成！所有结果保存在 output/ 目录")
    print("=" * 60)

    # 提示用户如何使用
    print("\n【使用说明】")
    print("1. 将你的仪表照片放入 test_images/ 目录")
    print("2. 修改 main() 中的图片路径，或调用 system.process('你的图片路径')")
    print("3. 查看 output/ 目录中的识别结果和可视化图片")

    return system


if __name__ == "__main__":
    # 运行主程序
    system = main()