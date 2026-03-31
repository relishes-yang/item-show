import cv2
import numpy as np
from PIL import Image


def draw_static_lane_scan(original_img, lane_lines, scan_line_count=5, scan_color=(0, 0, 255), line_width=3):
    """
    【课程知识点】静态车道扫描标注（对应你提供的效果图）
    原理：在检测到的车道线区域，绘制从下到上的红色扫描线，标注车道范围
    :param original_img: 原始BGR图像
    :param lane_lines: 检测到的车道线列表（[[x1,y1,x2,y2], ...]，来自霍夫变换的valid_lines）
    :param scan_line_count: 扫描线数量（默认5条，对应图中的5条红线）
    :param scan_color: 扫描线颜色（默认红色BGR: (0,0,255)）
    :param line_width: 扫描线宽度
    :return: 带扫描标注的图像
    """
    result_img = original_img.copy()
    h, w = result_img.shape[:2]

    # 1. 提取车道线的y坐标范围，确定扫描上下边界
    all_ys = []
    for line in lane_lines:
        x1, y1, x2, y2 = line
        all_ys.extend([y1, y2])
    if not all_ys:
        return result_img  # 无车道线，直接返回原图

    min_y = int(min(all_ys))  # 车道线最上端
    max_y = int(max(all_ys))  # 车道线最下端（图像底部）

    # 2. 生成从下到上均匀分布的扫描线y坐标
    scan_ys = np.linspace(max_y, min_y, scan_line_count, dtype=int)

    # 3. 提取车道线的x坐标范围，确定扫描左右边界
    all_xs = []
    for line in lane_lines:
        x1, y1, x2, y2 = line
        all_xs.extend([x1, x2])
    if not all_xs:
        return result_img
    min_x = int(min(all_xs))
    max_x = int(max(all_xs))

    # 4. 绘制横向扫描线
    for y in scan_ys:
        cv2.line(result_img, (min_x, y), (max_x, y), scan_color, line_width)

    # 5. 绘制左右斜向引导线（对应你图中的两侧红线）
    cv2.line(result_img, (min_x, max_y), (min_x, min_y), scan_color, line_width)
    cv2.line(result_img, (max_x, max_y), (max_x, min_y), scan_color, line_width)

    # 6. 添加标注文字（对应图中的算法说明）
    cv2.putText(result_img, f"普通霍夫变换（检测到{len(lane_lines)}条有效车道线）",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, scan_color, 3)

    return result_img


def generate_dynamic_lane_scan_gif(original_img, lane_lines, output_path="output/lane_scan.gif",
                                   frame_count=30, scan_color=(0, 0, 255), line_width=3, duration=50):
    """
    【课程知识点】动态车道扫描动画（可视化检测过程）
    原理：生成逐帧扫描动画，从下到上逐步扫描，同时显示检测到的车道线，最终合成GIF
    :param original_img: 原始BGR图像
    :param lane_lines: 检测到的车道线列表
    :param output_path: GIF保存路径
    :param frame_count: 动画总帧数（越多越流畅）
    :param scan_color: 扫描线颜色
    :param line_width: 扫描线宽度
    :param duration: 每帧持续时间（毫秒，默认50ms=20fps）
    :return: 无，保存GIF到output_path
    """
    h, w = original_img.shape[:2]
    frames = []

    # 1. 确定车道区域范围
    all_ys = []
    all_xs = []
    for line in lane_lines:
        x1, y1, x2, y2 = line
        all_ys.extend([y1, y2])
        all_xs.extend([x1, x2])
    if not all_ys or not all_xs:
        return
    min_y = int(min(all_ys))
    max_y = int(max(all_ys))
    min_x = int(min(all_xs))
    max_x = int(max(all_xs))

    # 2. 逐帧生成动画：扫描线从下到上移动，逐步显示车道线
    for i in range(frame_count):
        frame = original_img.copy()
        # 当前扫描线的y坐标（从底部逐步向上移动）
        current_y = int(max_y - (max_y - min_y) * (i / frame_count))

        # 绘制当前扫描线
        cv2.line(frame, (min_x, current_y), (max_x, current_y), scan_color, line_width)

        # 逐步显示已扫描到的车道线（仅展示扫描线下方的部分）
        for line in lane_lines:
            x1, y1, x2, y2 = line
            if y1 >= current_y and y2 >= current_y:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), line_width)
            elif y1 >= current_y or y2 >= current_y:
                # 截断车道线，仅显示已扫描区域
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                t = (current_y - y1) / (y2 - y1) if (y2 - y1) != 0 else 0
                intersect_x = x1 + t * (x2 - x1)
                cv2.line(frame, (int(intersect_x), current_y), (x2, y2), (0, 255, 0), line_width)

        # 转换为PIL Image，用于合成GIF
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        frames.append(pil_frame)

    # 3. 合成无限循环GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )


def video_lane_scan(frame, lane_lines, scan_line_count=5, scan_color=(0, 0, 255), line_width=3):
    """
    视频帧车道扫描标注（适配视频检测模块，逐帧添加扫描效果）
    :param frame: 输入视频帧
    :param lane_lines: 检测到的车道线
    :param scan_line_count: 扫描线数量
    :param scan_color: 扫描线颜色
    :param line_width: 扫描线宽度
    :return: 带扫描标注的视频帧
    """
    return draw_static_lane_scan(frame, lane_lines, scan_line_count, scan_color, line_width)