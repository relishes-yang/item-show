import os
import json
import cv2
import numpy as np
from pathlib import Path
from utils import (
    read_pointer_meter_advanced,
    read_digital_meter_advanced,
    detect_meter_circle
)


class MeterReader:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.meter_type = self.config.get('meter_type', 'pointer')
        self.pointer_cfg = self.config.get('pointer_meter', {})
        self.digital_cfg = self.config.get('digital_meter', {})
        # 预处理参数
        self.preprocess_cfg = self.config.get('preprocess', {})

    def draw_pointer_result(self, img, info_dict):
        """在原始图像上绘制指针仪表检测结果"""
        img_copy = img.copy()
        # 绘制圆形表盘
        if 'circle' in info_dict:
            cx, cy, r = info_dict['circle']
            cv2.circle(img_copy, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(img_copy, (cx, cy), 2, (0, 0, 255), -1)
        # 绘制指针（根据角度画线）
        if 'angle' in info_dict and 'circle' in info_dict:
            angle_rad = np.radians(info_dict['angle'])
            cx, cy, r = info_dict['circle']
            # 指针长度取半径的0.8倍
            tip_x = int(cx + 0.8 * r * np.cos(angle_rad))
            tip_y = int(cy + 0.8 * r * np.sin(angle_rad))
            cv2.line(img_copy, (cx, cy), (tip_x, tip_y), (255, 0, 0), 3)
        # 添加读数文本
        text = info_dict.get('reading_text', 'No reading')
        cv2.putText(img_copy, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)
        return img_copy

    def draw_digital_result(self, img, info_dict):
        """在原始图像上绘制数字仪表检测结果"""
        img_copy = img.copy()
        if 'roi' in info_dict:
            x1, y1, x2, y2 = info_dict['roi']
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = info_dict.get('reading_text', 'No reading')
        cv2.putText(img_copy, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)
        return img_copy

    def read_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"无法读取图片 {image_path}", "image": os.path.basename(image_path)}

        if self.meter_type == 'pointer':
            reading, info_dict = read_pointer_meter_advanced(img, self.pointer_cfg)
            annotated = self.draw_pointer_result(img, info_dict)
        elif self.meter_type == 'digital':
            reading, info_dict = read_digital_meter_advanced(img, self.digital_cfg)
            annotated = self.draw_digital_result(img, info_dict)
        else:
            return {"error": "未知仪表类型", "image": os.path.basename(image_path)}

        status = "success" if reading is not None else "failed"
        return {
            "image": os.path.basename(image_path),
            "meter_type": self.meter_type,
            "reading": reading,
            "info": info_dict.get('reading_text', '识别失败'),
            "status": status,
            "annotated_image": annotated
        }


def batch_process(image_folder, config_path='config.json', output_json='output/results.json',
                  output_image_folder='output'):
    reader = MeterReader(config_path)
    results = []
    os.makedirs(output_image_folder, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for img_path in Path(image_folder).glob('*'):
        if img_path.suffix.lower() not in image_extensions:
            continue
        print(f"处理中: {img_path.name}")
        res = reader.read_image(str(img_path))

        # 保存标注图片
        if 'annotated_image' in res:
            out_img_path = os.path.join(output_image_folder, f"annotated_{img_path.name}")
            cv2.imwrite(out_img_path, res['annotated_image'])
            res['annotated_image_path'] = out_img_path
            del res['annotated_image']  # 避免JSON序列化错误

        results.append(res)
        print(f"  结果: {res.get('info', '识别失败')}")

    # 保存JSON结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n批量处理完成，结果已保存至 {output_json}，标注图片保存在 {output_image_folder}")
    return results


if __name__ == '__main__':
    # 运行批量处理
    batch_process('images', 'config.json', 'output/results.json', 'output')