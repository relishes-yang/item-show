#!/usr/bin/env python3
"""
本地仪表检测测试脚本
用法: python test.py <图片路径>
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meter_detector import PrecisionMeterDetector

def test_single_image(image_path):
    """测试单张图片"""
    print(f"\n{'='*60}")
    print(f"测试图片: {image_path}")
    print(f"{'='*60}\n")

    # 创建检测器
    detector = PrecisionMeterDetector(debug_mode=True)

    # 检测
    try:
        meters = detector.detect(image_path)

        if not meters:
            print("❌ 未检测到仪表")
            return

        print(f"\n{'='*60}")
        print(f"检测成功: 共 {len(meters)} 个仪表")
        print(f"{'='*60}\n")

        for m in meters:
            print(f"【仪表 {m.meter_id + 1}】")
            print(f"  位置: ({m.center[0]}, {m.center[1]})")
            print(f"  半径: {m.radius}px")
            print(f"  读数: {m.current_value:.2f} {m.unit}")
            print(f"  量程: {m.min_val} - {m.max_val} {m.unit}")
            print(f"  指针角度: {m.pointer_angle:.1f}°")
            print(f"  置信度: {m.confidence:.2%}")
            print()

        # 保存调试信息
        output_dir = "debug_output"
        os.makedirs(output_dir, exist_ok=True)
        detector.save_debug_images(output_dir)
        print(f"✅ 调试图像已保存至: {output_dir}/")

    except Exception as e:
        print(f"❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test.py <图片路径>")
        print("示例: python test.py meter.jpg")
        sys.exit(1)

    test_single_image(sys.argv[1])
