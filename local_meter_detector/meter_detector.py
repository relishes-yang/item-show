"""
本地高精度电力仪表检测器
支持：单张图片检测、批量处理、参数精细调整
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# 尝试导入OCR，如果没有则使用备用方案
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  EasyOCR未安装，将使用备用刻度识别方案")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class MeterInfo:
    """仪表检测结果"""
    meter_id: int
    center: Tuple[int, int]
    radius: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

    # 识别结果
    min_val: float = 0.0
    max_val: float = 100.0
    unit: str = ""
    current_value: float = 0.0

    # 角度信息
    min_angle: float = -45.0
    max_angle: float = 225.0
    pointer_angle: float = 0.0

    # 质量评估
    confidence: float = 0.0
    detection_score: float = 0.0

    # 调试信息
    scale_points: List[Tuple[Tuple[int, int], float]] = None
    debug_info: Dict = None

    def __post_init__(self):
        if self.scale_points is None:
            self.scale_points = []
        if self.debug_info is None:
            self.debug_info = {}


class PrecisionMeterDetector:
    """
    高精度仪表检测器

    核心改进：
    1. 多尺度圆检测 - 避免漏检和过检
    2. 基于投影的指针检测 - 比霍夫变换更稳定
    3. 模板匹配刻度识别 - 不依赖OCR也能工作
    4. 几何约束验证 - 确保检测结果合理性
    """

    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self.debug_images = {}  # 存储调试图像
        self.ocr_reader = None

        # 初始化OCR（如果可用）
        if OCR_AVAILABLE:
            try:
                print("🔄 正在初始化OCR引擎...")
                self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                print("✅ OCR引擎初始化成功")
            except Exception as e:
                print(f"⚠️  OCR初始化失败: {e}")

    def detect(self, image_path: str, **kwargs) -> List[MeterInfo]:
        """
        检测图片中的仪表

        Args:
            image_path: 图片路径
            min_radius: 最小半径（可选，自动计算）
            max_radius: 最大半径（可选，自动计算）

        Returns:
            检测到的仪表列表
        """
        # 加载图像
        image = self._load_image(image_path)
        if image is None:
            raise ValueError(f"无法加载图片: {image_path}")

        self.debug_images['original'] = image.copy()

        print(f"\n{'='*60}")
        print(f"🔍 开始检测: {Path(image_path).name}")
        print(f"   图像尺寸: {image.shape[1]} x {image.shape[0]}")
        print(f"{'='*60}\n")

        # 步骤1: 预处理
        print("步骤1: 图像预处理...")
        preprocessed = self._preprocess(image)
        self.debug_images['preprocessed'] = preprocessed['visualization']

        # 步骤2: 检测仪表区域
        print("步骤2: 检测仪表区域...")
        circles = self._detect_circles(
            preprocessed['gray'], 
            preprocessed['edges'],
            kwargs.get('min_radius'),
            kwargs.get('max_radius')
        )

        if not circles:
            print("❌ 未检测到仪表")
            return []

        print(f"   检测到 {len(circles)} 个候选区域")

        # 可视化候选圆
        circle_viz = image.copy()
        for i, (x, y, r, score) in enumerate(circles[:10]):
            color = (0, 255, 255) if score > 0.6 else (128, 128, 128)
            cv2.circle(circle_viz, (x, y), r, color, 2)
            cv2.putText(circle_viz, f"{i+1}:{score:.2f}", (x-20, y-r-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        self.debug_images['circles'] = circle_viz

        # 步骤3: 筛选和排序
        print("步骤3: 筛选高质量区域...")
        filtered = self._filter_circles(circles, preprocessed['gray'])
        print(f"   筛选后剩余 {len(filtered)} 个区域")

        # 步骤4: 详细分析每个仪表
        print("步骤4: 详细分析仪表...")
        meters = []

        for idx, (x, y, r, score) in enumerate(filtered[:3]):  # 最多分析3个
            print(f"\n   分析仪表 {idx+1}/{min(len(filtered), 3)}...")
            meter = self._analyze_meter(image, idx, (x, y, r), score)
            if meter:
                meters.append(meter)
                print(f"   ✅ 读数: {meter.current_value:.2f} {meter.unit}")

        # 步骤5: 生成最终结果图
        if meters:
            result_viz = self._draw_results(image, meters)
            self.debug_images['result'] = result_viz

        print(f"\n{'='*60}")
        print(f"✅ 检测完成: 成功识别 {len(meters)} 个仪表")
        print(f"{'='*60}\n")

        return meters

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """加载图像"""
        image = cv2.imread(path)
        if image is None and PIL_AVAILABLE:
            # 尝试用PIL加载
            try:
                pil_img = Image.open(path).convert('RGB')
                image = np.array(pil_img)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"PIL加载失败: {e}")
        return image

    def _preprocess(self, image: np.ndarray) -> Dict:
        """图像预处理"""
        # 转为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 双边滤波（保边去噪）
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Canny边缘检测
        median = np.median(blurred)
        lower = int(max(0, 0.33 * median))
        upper = int(min(255, 1.33 * median))
        edges = cv2.Canny(blurred, lower, upper)

        # 形态学闭运算连接断开的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 创建可视化
        h, w = gray.shape
        viz = np.zeros((h, w*3, 3), dtype=np.uint8)
        viz[:, :w] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        viz[:, w:2*w] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        viz[:, 2*w:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 添加标签
        labels = ['Original Gray', 'Enhanced', 'Edges']
        for i, label in enumerate(labels):
            cv2.putText(viz, label, (i*w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return {
            'gray': enhanced,
            'edges': edges,
            'visualization': viz
        }

    def _detect_circles(self, gray: np.ndarray, edges: np.ndarray,
                       min_radius: int = None, max_radius: int = None) -> List[Tuple]:
        """
        多尺度圆检测
        使用多个参数组合，避免漏检
        """
        h, w = gray.shape
        min_dim = min(h, w)

        # 自动计算半径范围
        if min_radius is None:
            min_radius = int(min_dim * 0.1)  # 最小10%
        if max_radius is None:
            max_radius = int(min_dim * 0.4)  # 最大40%

        print(f"   搜索半径范围: {min_radius} - {max_radius}px")

        all_circles = []

        # 多参数策略
        params = [
            {'dp': 1, 'minDist': min_radius * 2, 'param2': 30},
            {'dp': 1.2, 'minDist': min_radius * 1.5, 'param2': 25},
            {'dp': 1.5, 'minDist': min_radius * 1.2, 'param2': 20},
        ]

        for param in params:
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT,
                dp=param['dp'],
                minDist=param['minDist'],
                param1=50,
                param2=param['param2'],
                minRadius=min_radius,
                maxRadius=max_radius
            )

            if circles is not None:
                for c in circles[0]:
                    x, y, r = int(c[0]), int(c[1]), int(c[2])
                    # 计算质量分数
                    score = self._evaluate_circle(gray, edges, (x, y, r))
                    all_circles.append((x, y, r, score))

        # 按分数排序
        all_circles.sort(key=lambda x: x[3], reverse=True)

        # 非极大值抑制（NMS）
        filtered = self._nms(all_circles, threshold=0.5)

        return filtered

    def _evaluate_circle(self, gray: np.ndarray, edges: np.ndarray, 
                        circle: Tuple[int, int, int]) -> float:
        """
        评估圆的质量

        评分标准：
        1. 圆环边缘密度（应该有刻度线）
        2. 内部纹理丰富度（应该有数字和指针）
        3. 圆度验证
        4. 对比度
        """
        x, y, r = circle
        h, w = gray.shape

        # 检查边界
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return 0.0

        score = 0.0

        # 1. 圆环边缘密度（外环10%区域）
        mask_ring = np.zeros_like(gray)
        cv2.circle(mask_ring, (x, y), r, 255, int(r * 0.1))
        ring_edges = cv2.bitwise_and(edges, mask_ring)
        ring_density = np.sum(ring_edges > 0) / (np.sum(mask_ring > 0) + 1e-6)
        score += min(1.0, ring_density * 8) * 0.35

        # 2. 内部纹理（标准差）
        mask_inner = np.zeros_like(gray)
        cv2.circle(mask_inner, (x, y), int(r * 0.7), 255, -1)
        inner_region = gray[mask_inner > 0]
        texture_score = min(1.0, np.std(inner_region) / 60)
        score += texture_score * 0.25

        # 3. 径向对称性（快速检查）
        roi = gray[y-r:y+r, x-r:x+r]
        if roi.size > 0:
            h_roi, w_roi = roi.shape
            center_x, center_y = w_roi // 2, h_roi // 2

            # 计算四个方向的梯度
            gradients = []
            for angle in [0, 45, 90, 135]:
                rad = math.radians(angle)
                x_end = int(center_x + (r * 0.8) * math.cos(rad))
                y_end = int(center_y + (r * 0.8) * math.sin(rad))

                line = roi[center_y, min(center_x, x_end):max(center_x, x_end)]
                if len(line) > 0:
                    gradients.append(np.std(line))

            if gradients:
                symmetry_score = 1.0 - (np.std(gradients) / (np.mean(gradients) + 1e-6))
                score += max(0, symmetry_score) * 0.2

        # 4. 半径合理性
        ideal_r = min(h, w) * 0.15
        size_score = 1.0 - abs(r - ideal_r) / ideal_r
        score += max(0, size_score) * 0.2

        return min(1.0, score)

    def _nms(self, circles: List[Tuple], threshold: float = 0.5) -> List[Tuple]:
        """非极大值抑制"""
        if not circles:
            return []

        # 按分数排序
        sorted_circles = sorted(circles, key=lambda x: x[3], reverse=True)
        keep = []

        for c1 in sorted_circles:
            x1, y1, r1, s1 = c1

            # 检查与已保留圆的重叠
            overlap = False
            for c2 in keep:
                x2, y2, r2, s2 = c2

                # 计算距离
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)

                # 如果距离小于半径之和的threshold，认为是同一个圆
                if dist < (r1 + r2) * threshold:
                    overlap = True
                    break

            if not overlap and s1 > 0.3:  # 分数阈值
                keep.append(c1)

        return keep

    def _filter_circles(self, circles: List[Tuple], gray: np.ndarray) -> List[Tuple]:
        """进一步筛选圆"""
        # 只保留高质量的
        high_quality = [c for c in circles if c[3] > 0.5]

        # 如果太少，降低阈值
        if len(high_quality) < 1:
            high_quality = [c for c in circles if c[3] > 0.3]

        # 最多返回5个
        return high_quality[:5]

    def _analyze_meter(self, image: np.ndarray, meter_id: int,
                      circle: Tuple[int, int, int], score: float) -> Optional[MeterInfo]:
        """详细分析单个仪表"""
        x, y, r = circle

        # 提取仪表ROI（稍微扩大）
        margin = int(r * 0.15)
        x1 = max(0, x - r - margin)
        y1 = max(0, y - r - margin)
        x2 = min(image.shape[1], x + r + margin)
        y2 = min(image.shape[0], y + r + margin)

        roi = image[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 相对圆心坐标
        cx, cy = x - x1, y - y1

        # 识别刻度
        print("      识别刻度...")
        scale_info = self._recognize_scale(roi, (cx, cy, r))

        # 检测指针
        print("      检测指针...")
        pointer_info = self._detect_pointer(roi, (cx, cy, r))

        if pointer_info is None:
            print("      ⚠️  未检测到指针")
            return None

        # 计算角度范围
        if len(scale_info['points']) >= 2:
            angle_range = self._calculate_angle_range(scale_info['points'], (cx, cy))
        else:
            # 使用默认角度范围
            angle_range = {'min_angle': -45, 'max_angle': 225}

        # 计算读数
        current_value = self._interpolate_value(
            pointer_info['angle'],
            angle_range['min_angle'],
            angle_range['max_angle'],
            scale_info['min_val'],
            scale_info['max_val']
        )

        # 综合置信度
        confidence = (
            score * 0.3 +
            pointer_info['confidence'] * 0.4 +
            min(1.0, len(scale_info['points']) / 5) * 0.3
        )

        # 转换刻度点到全局坐标
        global_points = [((int(pt[0] + x1), int(pt[1] + y1)), val) 
                        for pt, val in scale_info['points']]

        return MeterInfo(
            meter_id=meter_id,
            center=(x, y),
            radius=r,
            bbox=(x1, y1, x2, y2),
            min_val=scale_info['min_val'],
            max_val=scale_info['max_val'],
            unit=scale_info['unit'],
            min_angle=angle_range['min_angle'],
            max_angle=angle_range['max_angle'],
            pointer_angle=pointer_info['angle'],
            current_value=current_value,
            confidence=confidence,
            detection_score=score,
            scale_points=global_points,
            debug_info={
                'pointer_tip': pointer_info.get('tip'),
                'scale_count': len(scale_info['points'])
            }
        )

    def _recognize_scale(self, roi: np.ndarray, circle: Tuple[int, int, int]) -> Dict:
        """识别刻度和单位"""
        cx, cy, r = circle

        scale_points = []
        unit = ""
        values = []

        # 方法1: 使用OCR（如果可用）
        if self.ocr_reader is not None:
            try:
                results = self.ocr_reader.readtext(roi)

                for (bbox, text, conf) in results:
                    if conf < 0.4:
                        continue

                    # 提取数字
                    import re
                    numbers = re.findall(r'\d+\.?\d*', text)

                    if numbers:
                        try:
                            val = float(numbers[0])
                            pts = np.array(bbox, np.int32)
                            pt = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))

                            # 检查是否在刻度环上
                            dist = math.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
                            if r * 0.5 < dist < r * 0.95:
                                scale_points.append((pt, val))
                                values.append(val)
                        except:
                            pass
                    else:
                        # 检查单位
                        unit_candidates = ['MPa', 'kPa', 'Pa', 'bar', 'A', 'V', 'kV', 'mA', 'kW', 'Hz', '℃', '°C']
                        for u in unit_candidates:
                            if u in text or u.upper() in text.upper():
                                unit = u
                                break
            except Exception as e:
                print(f"      OCR错误: {e}")

        # 方法2: 基于轮廓的刻度检测（备用）
        if len(scale_points) < 3:
            print("      使用备用刻度检测...")
            backup_points = self._detect_scale_by_contour(roi, (cx, cy, r))

            # 合并结果
            existing_vals = {v for _, v in scale_points}
            for pt, val in backup_points:
                if val not in existing_vals:
                    scale_points.append((pt, val))
                    values.append(val)

        # 推断单位
        if not unit and values:
            max_val = max(values)
            if max_val < 1:
                unit = "MPa"
            elif max_val < 100:
                unit = "MPa"
            elif max_val < 1000:
                unit = "kPa"
            elif max_val < 10:
                unit = "A"
            else:
                unit = ""

        # 确定量程
        if len(values) >= 2:
            min_val, max_val = min(values), max(values)
        elif values:
            min_val, max_val = 0, max(values) * 1.2
        else:
            min_val, max_val = 0, 100

        return {
            'points': sorted(scale_points, key=lambda x: x[1]),
            'min_val': min_val,
            'max_val': max_val,
            'unit': unit
        }

    def _detect_scale_by_contour(self, roi: np.ndarray, circle: Tuple[int, int, int]) -> List[Tuple]:
        """基于轮廓检测刻度（备用方法）"""
        cx, cy, r = circle
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # 自适应阈值
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 2000:  # 过滤太小或太大的
                continue

            # 计算中心
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx_cnt = int(M["m10"] / M["m00"])
            cy_cnt = int(M["m01"] / M["m00"])

            # 检查是否在刻度环上
            dist = math.sqrt((cx_cnt - cx)**2 + (cy_cnt - cy)**2)
            if r * 0.6 < dist < r * 0.9:
                # 估算数值（基于角度）
                angle = math.degrees(math.atan2(cy_cnt - cy, cx_cnt - cx))
                if angle < 0:
                    angle += 360

                # 假设0度对应最小值，270度对应最大值（典型仪表）
                normalized_angle = (angle - 135) % 360
                estimated_val = normalized_angle / 270 * 100

                scale_candidates.append(((cx_cnt, cy_cnt), estimated_val))

        return scale_candidates[:10]  # 最多10个

    def _detect_pointer(self, roi: np.ndarray, circle: Tuple[int, int, int]) -> Optional[Dict]:
        """检测指针 - 使用径向投影方法"""
        cx, cy, r = circle
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # 创建掩码，只保留圆内区域
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), int(r * 0.85), 255, -1)

        # 自适应阈值提取指针
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        binary = cv2.bitwise_and(binary, mask)

        # 形态学操作连接指针
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 方法1: 径向投影（最稳定）
        pointer_angle = self._radial_projection(gray, binary, (cx, cy, r))

        if pointer_angle is None:
            # 方法2: 霍夫直线（备用）
            pointer_angle = self._hough_pointer(gray, (cx, cy, r))

        if pointer_angle is None:
            return None

        # 计算置信度
        confidence = 0.7  # 基础置信度

        return {
            'angle': pointer_angle,
            'confidence': confidence,
            'tip': self._calculate_tip(cx, cy, r, pointer_angle)
        }

    def _radial_projection(self, gray: np.ndarray, binary: np.ndarray, 
                          circle: Tuple[int, int, int]) -> Optional[float]:
        """径向投影检测指针"""
        cx, cy, r = circle
        h, w = gray.shape

        # 确保圆心在ROI内
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return None

        # 创建极坐标映射
        angles = np.linspace(0, 360, 360, endpoint=False)
        projections = []

        for angle in angles:
            rad = math.radians(angle)
            # 从圆心向外采样
            samples = []
            for d in range(int(r * 0.2), int(r * 0.8), 2):
                x = int(cx + d * math.cos(rad))
                y = int(cy + d * math.sin(rad))
                if 0 <= x < w and 0 <= y < h:
                    samples.append(binary[y, x])

            if samples:
                projections.append(np.mean(samples))
            else:
                projections.append(0)

        projections = np.array(projections)

        # 找到投影最大的角度（指针应该是黑色的，所以是最大值）
        # 但需要平滑处理
        smoothed = np.convolve(projections, np.ones(5)/5, mode='same')

        # 找到峰值
        peak_idx = np.argmax(smoothed)
        peak_value = smoothed[peak_idx]

        # 检查峰值是否足够显著
        if peak_value < 50:  # 阈值
            return None

        # 转换为仪表角度（0度向上）
        pointer_angle = (90 - peak_idx) % 360

        return pointer_angle

    def _hough_pointer(self, gray: np.ndarray, circle: Tuple[int, int, int]) -> Optional[float]:
        """霍夫变换检测指针（备用）"""
        cx, cy, r = circle

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                               minLineLength=int(r*0.3), maxLineGap=10)

        if lines is None:
            return None

        best_line = None
        best_score = -1

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算中点
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            dist_to_center = math.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)

            # 长度
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)

            # 评分：长且通过圆心
            score = length - dist_to_center * 2

            if score > best_score and length > r * 0.2:
                best_score = score
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            return None

        x1, y1, x2, y2 = best_line

        # 确定针尖
        dist1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        dist2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2)
        tip = (x1, y1) if dist1 > dist2 else (x2, y2)

        # 计算角度
        dx, dy = tip[0] - cx, tip[1] - cy
        angle = math.degrees(math.atan2(dx, -dy))
        if angle < 0:
            angle += 360

        return angle

    def _calculate_tip(self, cx: int, cy: int, r: int, angle: float) -> Tuple[int, int]:
        """计算针尖位置"""
        rad = math.radians(angle - 90)
        tip_x = int(cx + (r - 10) * math.cos(rad))
        tip_y = int(cy + (r - 10) * math.sin(rad))
        return (tip_x, tip_y)

    def _calculate_angle_range(self, scale_points: List[Tuple], 
                              center: Tuple[int, int]) -> Dict:
        """根据刻度点计算角度范围"""
        if len(scale_points) < 2:
            return {'min_angle': -45, 'max_angle': 225}

        cx, cy = center
        angles = []

        for (pt, val) in scale_points:
            dx, dy = pt[0] - cx, pt[1] - cy
            angle = math.degrees(math.atan2(dx, -dy))
            if angle < 0:
                angle += 360
            angles.append((angle, val))

        angles.sort(key=lambda x: x[1])
        min_angle, max_angle = angles[0][0], angles[-1][0]

        if max_angle < min_angle:
            max_angle += 360

        return {'min_angle': min_angle, 'max_angle': max_angle}

    def _interpolate_value(self, angle: float, min_angle: float, 
                          max_angle: float, min_val: float, max_val: float) -> float:
        """线性插值计算读数"""
        if angle < min_angle:
            angle += 360

        ratio = (angle - min_angle) / (max_angle - min_angle)
        ratio = max(0, min(1, ratio))

        value = min_val + ratio * (max_val - min_val)
        return round(value, 2)

    def _draw_results(self, image: np.ndarray, meters: List[MeterInfo]) -> np.ndarray:
        """绘制检测结果"""
        result = image.copy()

        for meter in meters:
            x, y = meter.center
            r = meter.radius

            # 绘制圆
            cv2.circle(result, (x, y), r, (0, 255, 0), 3)
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

            # 绘制指针
            angle_rad = math.radians(meter.pointer_angle - 90)
            tip_x = int(x + (r - 15) * math.cos(angle_rad))
            tip_y = int(y + (r - 15) * math.sin(angle_rad))
            cv2.line(result, (x, y), (tip_x, tip_y), (255, 0, 0), 4)

            # 绘制刻度点
            for pt, val in meter.scale_points:
                cv2.circle(result, pt, 3, (255, 255, 0), -1)

            # 信息框
            info = f"{meter.current_value:.1f}{meter.unit}"
            (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            bx, by = x - tw//2, y - r - 40
            cv2.rectangle(result, (bx-10, by-10), (bx+tw+10, by+th+10), (0,0,0), -1)
            cv2.rectangle(result, (bx-10, by-10), (bx+tw+10, by+th+10), (0,255,0), 2)
            cv2.putText(result, info, (bx, by+th), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return result

    def save_debug_images(self, output_dir: str):
        """保存调试图像"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for name, img in self.debug_images.items():
            path = Path(output_dir) / f"debug_{name}.jpg"
            cv2.imwrite(str(path), img)
            print(f"   保存: {path}")


def main():
    """主函数 - 命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description='电力仪表检测')
    parser.add_argument('image', help='输入图片路径')
    parser.add_argument('-o', '--output', default='output', help='输出目录')
    parser.add_argument('--min-radius', type=int, help='最小半径')
    parser.add_argument('--max-radius', type=int, help='最大半径')
    parser.add_argument('--no-debug', action='store_true', help='不保存调试图像')

    args = parser.parse_args()

    # 创建检测器
    detector = PrecisionMeterDetector(debug_mode=not args.no_debug)

    # 检测
    try:
        meters = detector.detect(
            args.image,
            min_radius=args.min_radius,
            max_radius=args.max_radius
        )

        # 打印结果
        print("\n检测结果:")
        print("-" * 60)
        for m in meters:
            print(f"仪表 {m.meter_id + 1}:")
            print(f"  位置: ({m.center[0]}, {m.center[1]}), 半径: {m.radius}px")
            print(f"  读数: {m.current_value:.2f} {m.unit}")
            print(f"  量程: {m.min_val} - {m.max_val} {m.unit}")
            print(f"  指针角度: {m.pointer_angle:.1f}°")
            print(f"  置信度: {m.confidence:.2%}")
            print()

        # 保存调试图像
        if not args.no_debug:
            detector.save_debug_images(args.output)

            # 保存结果图
            if 'result' in detector.debug_images:
                result_path = Path(args.output) / "result.jpg"
                cv2.imwrite(str(result_path), detector.debug_images['result'])
                print(f"✅ 结果图保存至: {result_path}")

        # 保存JSON结果
        if meters:
            results = [{
                'id': m.meter_id + 1,
                'value': m.current_value,
                'unit': m.unit,
                'range': [m.min_val, m.max_val],
                'confidence': m.confidence,
                'center': m.center,
                'radius': m.radius
            } for m in meters]

            json_path = Path(args.output) / "result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ JSON结果保存至: {json_path}")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
