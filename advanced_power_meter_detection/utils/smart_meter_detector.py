
import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import easyocr
import re

@dataclass
class MeterInfo:
    """单个仪表的完整信息"""
    meter_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]
    radius: int
    min_val: float
    max_val: float
    unit: str
    min_angle: float
    max_angle: float
    pointer_angle: float
    current_value: float
    confidence: float
    scale_points: List[Tuple[Tuple[int, int], float]]  # [(point, value), ...]

@dataclass
class ProcessingStep:
    """处理步骤可视化"""
    step_name: str
    description: str
    image: np.ndarray

class AdvancedMeterDetector:
    """智能电力仪表检测器 - 自动识别量程、角度、多仪表"""

    def __init__(self):
        self.reader = None  # OCR读取器（延迟初始化）
        self.debug_steps = []  # 存储处理步骤用于可视化

    def init_ocr(self):
        """初始化OCR（延迟加载以节省内存）"""
        if self.reader is None:
            print("🔄 正在初始化OCR引擎...")
            self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        return self.reader

    def detect_meters(self, image: np.ndarray) -> List[MeterInfo]:
        """
        主检测函数：检测图像中所有仪表
        """
        self.debug_steps = []  # 重置步骤记录
        original = image.copy()

        # 步骤1: 预处理
        step1 = self._preprocess(image)
        self.debug_steps.append(ProcessingStep(
            "1. 图像预处理", 
            "灰度化、去噪、边缘增强",
            step1['visualization']
        ))

        # 步骤2: 检测所有可能的仪表区域（圆形检测）
        circles = self._detect_all_circles(step1['edges'], step1['gray'])
        if not circles:
            return []

        circle_viz = original.copy()
        for i, (x, y, r) in enumerate(circles):
            cv2.circle(circle_viz, (x, y), r, (0, 255, 0), 2)
            cv2.putText(circle_viz, f"M{i+1}", (x-10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        self.debug_steps.append(ProcessingStep(
            "2. 仪表区域检测",
            f"检测到 {len(circles)} 个圆形仪表区域",
            circle_viz
        ))

        # 步骤3: 对每个仪表进行详细分析
        meters = []
        for idx, (x, y, r) in enumerate(circles):
            meter = self._analyze_single_meter(original, idx, (x, y, r))
            if meter:
                meters.append(meter)

        # 步骤4: 最终可视化
        final_viz = self._draw_final_result(original, meters)
        self.debug_steps.append(ProcessingStep(
            "4. 最终检测结果",
            f"成功识别 {len(meters)} 个仪表",
            final_viz
        ))

        return meters

    def _preprocess(self, image: np.ndarray) -> Dict:
        """步骤1: 图像预处理"""
        # 转为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 自适应直方图均衡化（增强对比度）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Canny边缘检测
        edges = cv2.Canny(enhanced, 50, 150)

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # 创建可视化图像
        viz = np.zeros((image.shape[0], image.shape[1]*2, 3), dtype=np.uint8)
        viz[:, :image.shape[1]] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        viz[:, image.shape[1]:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cv2.putText(viz, "Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(viz, "Edges", (image.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return {'gray': enhanced, 'edges': edges, 'visualization': viz}

    def _detect_all_circles(self, edges: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int]]:
        """检测所有圆形仪表"""
        # 使用霍夫圆变换
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,  # 圆心之间的最小距离
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=min(gray.shape) // 3
        )

        if circles is None:
            return []

        circles = np.uint16(np.around(circles[0]))

        # 过滤：根据圆内的特征验证是否为仪表
        valid_circles = []
        for (x, y, r) in circles:
            # 提取圆内区域
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), int(r*0.8), 255, -1)
            roi = cv2.bitwise_and(gray, mask)

            # 检查圆内是否有足够的边缘（排除纯色圆）
            edge_count = cv2.countNonZero(cv2.Canny(roi, 50, 150))
            if edge_count > 100:  # 阈值可调
                valid_circles.append((int(x), int(y), int(r)))

        return valid_circles

    def _analyze_single_meter(self, image: np.ndarray, meter_id: int, 
                             circle: Tuple[int, int, int]) -> Optional[MeterInfo]:
        """分析单个仪表的所有信息"""
        x, y, r = circle

        # 提取仪表区域
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(image.shape[1], x+r), min(image.shape[0], y+r)
        meter_roi = image[y1:y2, x1:x2]

        # 步骤3a: OCR识别刻度和单位
        ocr_result = self._recognize_scale(meter_roi, (x, y, r))

        step_viz = meter_roi.copy()
        # 绘制检测到的数字位置
        for (pt, val) in ocr_result['scale_points']:
            abs_pt = (pt[0] + x1, pt[1] + y1)
            cv2.circle(step_viz, (pt[0], pt[1]), 5, (0, 255, 255), -1)
            cv2.putText(step_viz, str(val), (pt[0]-10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        self.debug_steps.append(ProcessingStep(
            f"3.{meter_id+1}. 仪表{meter_id+1}刻度识别",
            f"识别到 {len(ocr_result['scale_points'])} 个刻度，单位: {ocr_result['unit']}",
            step_viz
        ))

        # 步骤3b: 检测指针
        pointer_result = self._detect_pointer_precise(image, (x, y, r))

        if pointer_result is None:
            return None

        # 计算角度范围（基于刻度位置）
        angle_range = self._calculate_angle_range(ocr_result['scale_points'], (x, y))

        # 计算当前读数
        current_value = self._interpolate_value(
            pointer_result['angle'],
            angle_range['min_angle'],
            angle_range['max_angle'],
            ocr_result['min_val'],
            ocr_result['max_val']
        )

        return MeterInfo(
            meter_id=meter_id,
            bbox=(x1, y1, x2, y2),
            center=(x, y),
            radius=r,
            min_val=ocr_result['min_val'],
            max_val=ocr_result['max_val'],
            unit=ocr_result['unit'],
            min_angle=angle_range['min_angle'],
            max_angle=angle_range['max_angle'],
            pointer_angle=pointer_result['angle'],
            current_value=current_value,
            confidence=pointer_result['confidence'],
            scale_points=ocr_result['scale_points']
        )

    def _recognize_scale(self, roi: np.ndarray, circle: Tuple[int, int, int]) -> Dict:
        """使用OCR识别仪表刻度和单位"""
        reader = self.init_ocr()

        # OCR检测
        results = reader.readtext(roi)

        scale_points = []  # [(point, value), ...]
        unit = ""
        values = []

        cx, cy, r = circle

        for (bbox, text, conf) in results:
            # 提取数字
            numbers = re.findall(r'\d+\.?\d*', text)

            if numbers:
                try:
                    val = float(numbers[0])
                    # 计算数字中心点
                    pts = np.array(bbox, np.int32)
                    center_pt = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))

                    # 只保留在圆内的点
                    dist = math.sqrt((center_pt[0] - cx)**2 + (center_pt[1] - cy)**2)
                    if dist < r * 0.9 and dist > r * 0.4:  # 在刻度区域
                        scale_points.append((center_pt, val))
                        values.append(val)
                except:
                    pass
            else:
                # 识别单位（MPa, kPa, A, V等）
                unit_candidates = ['MPa', 'kPa', 'Pa', 'A', 'V', 'kV', 'mA', 'kW', 'Hz']
                for u in unit_candidates:
                    if u in text:
                        unit = u
                        break

        # 如果没有检测到单位，尝试根据数值范围推断
        if not unit and values:
            unit = self._infer_unit(values)

        # 确定量程
        if len(values) >= 2:
            min_val = min(values)
            max_val = max(values)
        else:
            # 默认值
            min_val = 0.0
            max_val = max(values) * 2 if values else 100.0

        return {
            'scale_points': sorted(scale_points, key=lambda x: x[1]),
            'min_val': min_val,
            'max_val': max_val,
            'unit': unit
        }

    def _infer_unit(self, values: List[float]) -> str:
        """根据数值范围推断单位"""
        max_val = max(values) if values else 0
        if max_val < 1:
            return "MPa"
        elif max_val < 100:
            return "MPa"
        elif max_val < 1000:
            return "kPa"
        elif max_val < 10:
            return "A"
        else:
            return ""

    def _detect_pointer_precise(self, image: np.ndarray, circle: Tuple[int, int, int]) -> Optional[Dict]:
        """精确检测指针"""
        x, y, r = circle

        # 提取ROI
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(image.shape[1], x+r), min(image.shape[0], y+r)
        roi = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 自适应阈值（更好地分离指针）
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # 形态学操作连接指针部分
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 霍夫直线检测
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                               minLineLength=int(r*0.3), maxLineGap=10)

        if lines is None:
            return None

        # 找到通过圆心附近的最长线段
        best_line = None
        best_score = -1

        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]

            # 计算中点
            mid_x = (x1_l + x2_l) / 2
            mid_y = (y1_l + y2_l) / 2

            # 到圆心的距离
            dist_to_center = math.sqrt((mid_x - r)**2 + (mid_y - r)**2)

            # 线段长度
            length = math.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)

            # 评分：长且通过圆心
            score = length - dist_to_center * 2

            if score > best_score and length > r * 0.2:
                best_score = score
                best_line = (x1_l, y1_l, x2_l, y2_l)

        if best_line is None:
            return None

        x1_l, y1_l, x2_l, y2_l = best_line

        # 确定针尖（离圆心远的一端）
        dist1 = math.sqrt((x1_l - r)**2 + (y1_l - r)**2)
        dist2 = math.sqrt((x2_l - r)**2 + (y2_l - r)**2)

        if dist1 > dist2:
            tip = (x1_l, y1_l)
        else:
            tip = (x2_l, y2_l)

        # 计算角度（相对于垂直向上）
        dx = tip[0] - r
        dy = tip[1] - r
        angle = math.degrees(math.atan2(dx, -dy))
        if angle < 0:
            angle += 360

        return {
            'angle': angle,
            'confidence': min(1.0, best_score / 100),
            'tip': (tip[0] + x1, tip[1] + y1)
        }

    def _calculate_angle_range(self, scale_points: List[Tuple[Tuple[int, int], float]], 
                              center: Tuple[int, int]) -> Dict:
        """根据刻度点计算角度范围"""
        if len(scale_points) < 2:
            return {'min_angle': -45, 'max_angle': 225}

        cx, cy = center
        angles = []

        for (pt, val) in scale_points:
            dx = pt[0] - cx
            dy = pt[1] - cy
            angle = math.degrees(math.atan2(dx, -dy))
            if angle < 0:
                angle += 360
            angles.append((angle, val))

        # 按值排序
        angles.sort(key=lambda x: x[1])

        min_angle = angles[0][0]
        max_angle = angles[-1][0]

        # 处理跨越0度的情况
        if max_angle < min_angle:
            max_angle += 360

        return {'min_angle': min_angle, 'max_angle': max_angle}

    def _interpolate_value(self, angle: float, min_angle: float, max_angle: float,
                          min_val: float, max_val: float) -> float:
        """线性插值计算读数"""
        # 标准化角度
        if angle < min_angle:
            angle += 360

        ratio = (angle - min_angle) / (max_angle - min_angle)
        ratio = max(0, min(1, ratio))  # 限制在0-1范围

        value = min_val + ratio * (max_val - min_val)
        return round(value, 2)

    def _draw_final_result(self, image: np.ndarray, meters: List[MeterInfo]) -> np.ndarray:
        """绘制最终结果"""
        result = image.copy()

        for meter in meters:
            x, y = meter.center
            r = meter.radius

            # 绘制表盘圆
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

            # 绘制刻度点
            for (pt, val) in meter.scale_points:
                cv2.circle(result, pt, 3, (255, 255, 0), -1)

            # 绘制指针
            angle_rad = math.radians(meter.pointer_angle - 90)
            tip_x = int(x + (r - 20) * math.cos(angle_rad))
            tip_y = int(y + (r - 20) * math.sin(angle_rad))
            cv2.line(result, (x, y), (tip_x, tip_y), (255, 0, 0), 3)

            # 绘制信息框
            info_text = f"ID:{meter.meter_id+1} {meter.current_value}{meter.unit}"
            (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # 信息框背景
            cv2.rectangle(result, (x-r, y-r-40), (x-r+text_w+20, y-r), (0, 0, 0), -1)
            cv2.rectangle(result, (x-r, y-r-40), (x-r+text_w+20, y-r), (0, 255, 0), 2)

            cv2.putText(result, info_text, (x-r+10, y-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result

    def get_processing_steps(self) -> List[ProcessingStep]:
        """获取所有处理步骤用于可视化"""
        return self.debug_steps
