import cv2
import numpy as np
import json
import os
from pathlib import Path

class AutoMeterReader:
    def __init__(self, min_value=0.0, max_value=1.6, unit="MPa"):
        """
        参数：
        min_value: 仪表最小值（如 0）
        max_value: 仪表最大值（如 1.6）
        unit: 单位（如 "MPa"）
        """
        self.min_value = min_value
        self.max_value = max_value
        self.unit = unit
        self.min_angle = None   # 自动检测
        self.max_angle = None   # 自动检测

    def load_image(self, path):
        self.img = cv2.imread(path)
        if self.img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.equalizeHist(self.gray)
        self.gray = cv2.bilateralFilter(self.gray, 9, 75, 75)

    def find_meter_ellipse(self):
        """检测表盘椭圆，返回圆心和半径（取长轴）"""
        edges = cv2.Canny(self.gray, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            self.center = (int(ellipse[0][0]), int(ellipse[0][1]))
            self.radius = max(int(ellipse[1][0]//2), int(ellipse[1][1]//2))
            return True
        return False

    def detect_scale_lines_angles(self):
        """
        自动检测所有刻度线的角度（基于径向线检测）
        返回角度列表（范围 -180 到 180）
        """
        cx, cy = self.center
        r = self.radius
        # 创建环形掩膜（只保留刻度环区域）
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r*0.9), 255, -1)
        cv2.circle(mask, (cx, cy), int(r*0.6), 0, -1)
        ring = cv2.bitwise_and(self.gray, mask)
        # 边缘检测
        edges = cv2.Canny(ring, 50, 150)
        # 霍夫线检测
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
        if lines is None:
            return []
        angles = []
        for rho, theta in lines[:, 0]:
            # 计算直线角度（度）
            angle = np.degrees(theta)
            # 转换为径向线方向（0~180）
            angle = angle % 180
            # 检查该直线是否通过圆心附近
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            dist = abs(a*cx + b*cy - rho)
            if dist < 15:   # 通过圆心
                # 将角度归一化到 -180~180（以水平为0，向上为正？我们需要统一）
                # 这里简单存储原始角度，后续再转换
                angles.append(angle)
        # 去重（相近角度合并）
        if len(angles) == 0:
            return []
        angles = np.unique(np.round(angles, 1))
        # 转换为以圆心为原点的射线角度（0°向右，逆时针为正）
        # 霍夫线返回的theta是直线法线角度，径向线的角度 = theta + 90°
        radial_angles = []
        for ang in angles:
            radial = (ang + 90) % 180
            # 再映射到 0~360
            radial_360 = radial
            radial_angles.append(radial_360)
        # 由于刻度线通常成对出现（180°对称），我们只取0~180范围
        radial_angles = [a for a in radial_angles if 0 <= a <= 180]
        # 转换为以向上为0°的极坐标角度（与指针检测一致）
        # 我们需要的角度：0°向上，顺时针为正。需要转换：
        # 原角度（向右0°逆时针） -> 向上0°顺时针 = 90 - 原角度
        final_angles = []
        for a in radial_angles:
            converted = (90 - a) % 360
            if converted > 180:
                converted -= 360
            final_angles.append(converted)
        final_angles = sorted(final_angles)
        return final_angles

    def estimate_range_angles(self, scale_angles):
        """
        根据检测到的刻度线角度，估算零刻度角度和满刻度角度
        假设刻度线均匀分布，取最小和最大角度
        """
        if len(scale_angles) < 2:
            # 默认角度范围
            return -150, 150
        # 找到最小和最大角度（注意角度可能跨越 -180/180 边界）
        # 简单起见，直接取最小和最大
        min_ang = min(scale_angles)
        max_ang = max(scale_angles)
        # 如果角度范围太小，使用默认
        if max_ang - min_ang < 90:
            return -150, 150
        return min_ang, max_ang

    def get_pointer_angle(self):
        """极坐标变换找指针角度（0°向上，顺时针为正，范围 -180~180）"""
        cx, cy = self.center
        r = self.radius
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        masked = cv2.bitwise_and(self.gray, mask)
        polar = cv2.warpPolar(masked, (r, 360), (cx, cy), r,
                              cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
        h, w = polar.shape
        roi = polar[int(h*0.3):int(h*0.8), :]
        proj = np.sum(roi, axis=0)
        proj = cv2.GaussianBlur(proj.astype(np.float32), (5,1), 0)
        min_col = np.argmin(proj)
        angle = (min_col / w) * 360.0
        angle = (angle - 90) % 360
        if angle > 180:
            angle -= 360
        return angle

    def angle_to_reading(self, angle):
        if self.min_angle is None or self.max_angle is None:
            return 0.0
        # 将角度限制在范围内
        if angle < self.min_angle:
            angle = self.min_angle
        if angle > self.max_angle:
            angle = self.max_angle
        proportion = (angle - self.min_angle) / (self.max_angle - self.min_angle)
        return self.min_value + proportion * (self.max_value - self.min_value)

    def draw_result(self, angle, reading):
        img = self.img.copy()
        cv2.circle(img, self.center, self.radius, (0,255,0), 2)
        rad = np.radians(angle)
        tip_x = int(self.center[0] + (self.radius-20) * np.cos(rad))
        tip_y = int(self.center[1] + (self.radius-20) * np.sin(rad))
        cv2.line(img, self.center, (tip_x, tip_y), (0,0,255), 3)
        text = f"{reading:.3f} {self.unit}"
        cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        return img

    def process(self, image_path):
        self.load_image(image_path)
        if not self.find_meter_ellipse():
            raise Exception("未检测到表盘")
        # 自动检测刻度线角度
        scale_angles = self.detect_scale_lines_angles()
        if len(scale_angles) < 2:
            # 如果检测失败，使用默认角度范围（常见压力表）
            self.min_angle, self.max_angle = -150, 150
            print("警告：未检测到足够刻度线，使用默认角度范围 -150° ~ 150°")
        else:
            self.min_angle, self.max_angle = self.estimate_range_angles(scale_angles)
            print(f"自动检测角度范围: {self.min_angle:.1f}° ~ {self.max_angle:.1f}°")
        pointer_angle = self.get_pointer_angle()
        reading = self.angle_to_reading(pointer_angle)
        annotated = self.draw_result(pointer_angle, reading)
        return reading, self.unit, annotated

def batch_process(input_dir='images', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    # ========== 请根据您的仪表修改下面两行 ==========
    MIN_VAL = 0.0      # 仪表最小值
    MAX_VAL = 1.6      # 仪表最大值
    UNIT = "MPa"       # 单位
    # =============================================
    reader = AutoMeterReader(min_value=MIN_VAL, max_value=MAX_VAL, unit=UNIT)
    results = []
    for img_path in Path(input_dir).glob('*.*'):
        if img_path.suffix.lower() not in ['.jpg','.jpeg','.png']:
            continue
        print(f"处理: {img_path.name}")
        try:
            reading, unit, anno = reader.process(str(img_path))
            out_path = os.path.join(output_dir, f"annotated_{img_path.name}")
            cv2.imwrite(out_path, anno)
            results.append({"image": img_path.name, "reading": reading, "unit": unit, "status": "success"})
            print(f"  读数: {reading:.3f} {unit}")
        except Exception as e:
            print(f"  失败: {e}")
            results.append({"image": img_path.name, "error": str(e), "status": "fail"})
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("处理完成，结果保存在", output_dir)

if __name__ == '__main__':
    batch_process()