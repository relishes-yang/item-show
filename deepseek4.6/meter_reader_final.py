import cv2
import numpy as np
import pytesseract
import json
import os
import re
from pathlib import Path
from scipy import ndimage
import pytesseract
# 将路径修改为你实际的Tesseract安装目录
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
class FullyAutoMeterReader:
    def __init__(self):
        self.img = None
        self.gray = None
        self.center = None
        self.r = None
        self.scale_angles = []
        self.scale_values = []
        self.unit = ""

    def load(self, path):
        self.img = cv2.imread(path)
        if self.img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.equalizeHist(self.gray)
        self.gray = cv2.bilateralFilter(self.gray, 9, 75, 75)

    def find_meter_ellipse(self):
        edges = cv2.Canny(self.gray, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return False
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            self.center = (int(ellipse[0][0]), int(ellipse[0][1]))
            self.r = max(int(ellipse[1][0]//2), int(ellipse[1][1]//2))
            return True
        return False

    def extract_scales(self):
        """通过OCR自动提取刻度数字及其角度，实现量程自动识别"""
        cx, cy = self.center
        r = self.r
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r*0.95), 255, -1)
        cv2.circle(mask, (cx, cy), int(r*0.6), 0, -1)
        ring = cv2.bitwise_and(self.img, self.img, mask=mask)
        gray_ring = cv2.cvtColor(ring, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_ring, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_map = {}
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 15 or h < 15 or w > 80 or h > 80: continue
            roi = self.img[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # OCR识别数字（含小数点）
            text = pytesseract.image_to_string(roi_bin, config='--psm 8 -c tessedit_char_whitelist=0123456789.')
            try:
                val = float(text.strip())
                # 计算数字中心的角度
                dx = (x + w/2) - cx
                dy = (y + h/2) - cy
                angle = np.degrees(np.arctan2(dy, dx)) % 360
                scale_map[angle] = val
            except: continue
        # 按角度排序，确保顺序正确
        if len(scale_map) >= 2:
            self.scale_angles = sorted(scale_map.keys())
            self.scale_values = [scale_map[a] for a in self.scale_angles]
            # 识别单位（在表盘下方区域搜索文字）
            h, w = self.img.shape[:2]
            roi_unit = self.img[int(h*0.7):h, int(w*0.5):w]
            unit_gray = cv2.cvtColor(roi_unit, cv2.COLOR_BGR2GRAY)
            _, unit_bin = cv2.threshold(unit_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            unit_text = pytesseract.image_to_string(unit_bin, config='--psm 7')
            if 'MPa' in unit_text: self.unit = 'MPa'
            elif 'kPa' in unit_text: self.unit = 'kPa'
            return True
        return False

    def get_pointer_angle(self):
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.circle(mask, self.center, self.r, 255, -1)
        masked = cv2.bitwise_and(self.gray, mask)
        polar = cv2.warpPolar(masked, (self.r, 360), self.center, self.r, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
        h, w = polar.shape
        roi = polar[int(h*0.3):int(h*0.8), :]
        proj = np.sum(roi, axis=0)
        proj = cv2.GaussianBlur(proj.astype(np.float32), (5,1), 0)
        min_col = np.argmin(proj)
        angle = (min_col / w) * 360.0
        angle = (angle - 90) % 360
        if angle > 180: angle -= 360
        return angle

    def angle_to_reading(self, angle):
        if len(self.scale_angles) < 2:
            return 0.0
        angle_norm = angle % 360
        if angle_norm <= self.scale_angles[0]: return self.scale_values[0]
        if angle_norm >= self.scale_angles[-1]: return self.scale_values[-1]
        for i in range(len(self.scale_angles)-1):
            if self.scale_angles[i] <= angle_norm <= self.scale_angles[i+1]:
                t = (angle_norm - self.scale_angles[i]) / (self.scale_angles[i+1] - self.scale_angles[i])
                return self.scale_values[i] + t * (self.scale_values[i+1] - self.scale_values[i])
        return 0.0

    def draw_result(self, angle, reading):
        img = self.img.copy()
        cv2.circle(img, self.center, self.r, (0,255,0), 2)
        rad = np.radians(angle)
        tip_x = int(self.center[0] + (self.r-20) * np.cos(rad))
        tip_y = int(self.center[1] + (self.r-20) * np.sin(rad))
        cv2.line(img, self.center, (tip_x, tip_y), (0,0,255), 3)
        text = f"{reading:.3f} {self.unit}"
        cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        return img

    def process(self, image_path):
        self.load(image_path)
        if not self.find_meter_ellipse(): raise Exception("未检测到表盘")
        if not self.extract_scales(): raise Exception("自动识别量程失败，请检查图片清晰度")
        angle = self.get_pointer_angle()
        reading = self.angle_to_reading(angle)
        annotated = self.draw_result(angle, reading)
        return reading, self.unit, annotated

def batch_process(input_dir='images', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    reader = FullyAutoMeterReader()
    results = []
    for img_path in Path(input_dir).glob('*.*'):
        if img_path.suffix.lower() not in ['.jpg','.jpeg','.png']: continue
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