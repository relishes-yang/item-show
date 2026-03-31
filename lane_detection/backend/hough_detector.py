import cv2
import numpy as np


def hough_original_detect(original_img, binary_img, roi_ratio=0.5, hough_threshold=80,
                          min_line_length=50, max_line_gap=30, slope_min=0.4, slope_max=0.9):
    """
    【课程知识点】普通霍夫变换车道线检测
    原理：将图像空间的直线转换为参数空间的点，通过累加器投票检测直线
    :param original_img: 原始BGR图像
    :param binary_img: 二值化结果图
    :param roi_ratio: ROI区域占比（图像下半部分高度比例，排除天空等无关背景）
    :param hough_threshold: 霍夫变换累加阈值（投票数超过该值才判定为直线）
    :param min_line_length: 最小线长（过滤短直线）
    :param max_line_gap: 最大线间隙（连接断裂的直线）
    :param slope_min: 斜率筛选最小值（排除水平/垂直干扰线）
    :param slope_max: 斜率筛选最大值
    :return: 检测结果图、ROI区域图、检测到的线条
    """
    result_img = original_img.copy()
    h, w = binary_img.shape

    # 【课程知识点】ROI（感兴趣区域）：只保留图像下半部分（车道核心区域）
    roi_y_start = int(h * (1 - roi_ratio))
    roi = binary_img[roi_y_start:h, 0:w]

    # 【课程知识点】霍夫直线变换（概率霍夫变换HoughLinesP，效率更高）
    lines = cv2.HoughLinesP(
        roi,
        rho=1,  # 距离分辨率（像素）
        theta=np.pi / 180,  # 角度分辨率（弧度）
        threshold=hough_threshold,  # 累加阈值
        minLineLength=min_line_length,  # 最小线长
        maxLineGap=max_line_gap  # 最大线间隙
    )

    # 绘制符合斜率要求的车道线（排除水平/垂直干扰线）
    valid_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 恢复y坐标到原图（因为我们只处理了ROI区域）
            y1 += roi_y_start
            y2 += roi_y_start
            # 计算斜率，筛选符合车道线特征的直线
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # 加1e-6避免除零
            if (slope < -slope_min and slope > -slope_max) or (slope > slope_min and slope < slope_max):
                # 绘制车道线（红色，线宽3）
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                valid_lines.append([x1, y1, x2, y2])

    # 生成ROI可视化图（方便理解ROI区域）
    roi_visual = original_img.copy()
    cv2.rectangle(roi_visual, (0, roi_y_start), (w, h), (0, 255, 0), 2)
    cv2.putText(roi_visual, "ROI车道区域", (10, roi_y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return result_img, roi_visual, valid_lines


def hough_improved_detect(original_img, binary_img, roi_ratio=0.5, grid_rows=9, grid_cols=16,
                          transition_thresh=15, hough_threshold=50, min_line_length=30,
                          max_line_gap=20, slope_min=0.4, slope_max=0.9):
    """
    【课程知识点】改良版霍夫变换车道线检测（分块跳变检测优化）
    优化思路：先通过分块跳变检测筛选有效车道区域，生成掩码，再在掩码上做霍夫变换
    优势：减少背景干扰，提升检测精度和速度
    :param original_img: 原始BGR图像
    :param binary_img: 二值化结果图
    :param roi_ratio: ROI区域占比
    :param grid_rows: 网格行数（分块策略，可配置）
    :param grid_cols: 网格列数
    :param transition_thresh: 像素跳变阈值（超过该值判定为有效车道块）
    :param hough_threshold: 霍夫变换累加阈值
    :param min_line_length: 最小线长
    :param max_line_gap: 最大线间隙
    :param slope_min: 斜率筛选最小值
    :param slope_max: 斜率筛选最大值
    :return: 检测结果图、掩码区域图、检测到的线条
    """
    result_img = original_img.copy()
    h, w = binary_img.shape

    # 第一步：只保留下半部分ROI区域
    bottom_h = int(h * roi_ratio)
    roi_y_start = h - bottom_h
    bottom_img = binary_img[roi_y_start:h, :]

    # 第二步：分块（可配置的行列数，课程知识点：分块策略优化）
    cell_h = bottom_h // grid_rows
    cell_w = w // grid_cols
    mask = np.zeros_like(bottom_img, dtype=np.uint8)

    # 第三步：从左到右扫描每个小方块，检测0→255→0跳变（车道线特征）
    for i in range(grid_rows):
        for j in range(grid_cols):
            # 计算当前块的坐标
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = bottom_img[y1:y2, x1:x2]

            # 统计0→255和255→0的跳变次数（车道线边缘有明显跳变）
            transitions = 0
            for row in cell:
                for k in range(1, len(row)):
                    if row[k - 1] != row[k]:
                        transitions += 1

            # 跳变次数超过阈值，判定为有效车道块，保留到掩码中
            if transitions > transition_thresh:
                mask[y1:y2, x1:x2] = cell

    # 第四步：只在有效掩码上做霍夫变换（核心优化：减少背景干扰，提升速度）
    lines = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    # 绘制符合斜率要求的车道线
    valid_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 恢复y坐标到原图
            y1 += roi_y_start
            y2 += roi_y_start
            # 斜率筛选
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if (slope < -slope_min and slope > -slope_max) or (slope > slope_min and slope < slope_max):
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                valid_lines.append([x1, y1, x2, y2])

    # 生成掩码可视化图（红色通道显示掩码，方便理解优化效果）
    mask_visual = np.zeros_like(original_img)
    mask_visual[roi_y_start:h, :, 2] = mask  # 红色通道显示掩码
    mask_visual = cv2.addWeighted(original_img, 0.7, mask_visual, 0.3, 0)
    cv2.putText(mask_visual, "有效车道掩码区域", (10, roi_y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return result_img, mask_visual, valid_lines