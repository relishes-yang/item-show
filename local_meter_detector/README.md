# 本地高精度仪表检测器

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 快速测试
```bash
python test.py your_image.jpg
```

### 2. 命令行工具
```bash
python meter_detector.py your_image.jpg -o output
```

### 3. 在代码中使用
```python
from meter_detector import PrecisionMeterDetector

detector = PrecisionMeterDetector()
meters = detector.detect("image.jpg")

for meter in meters:
    print(f"读数: {meter.current_value} {meter.unit}")
```

## 输出文件

- `debug_original.jpg` - 原始图像
- `debug_preprocessed.jpg` - 预处理结果
- `debug_circles.jpg` - 候选圆检测
- `debug_result.jpg` - 最终结果
- `result.json` - JSON格式结果

## 核心改进

1. **多尺度圆检测** - 避免漏检
2. **径向投影指针检测** - 比霍夫变换更稳定
3. **OCR + 轮廓双模式刻度识别** - 提高鲁棒性
4. **NMS过滤** - 去除重复检测
