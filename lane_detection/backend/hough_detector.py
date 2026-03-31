import cv2
import numpy as np


def hough_original_detect(original_img, binary_img, roi_ratio=0.5, hough_threshold=80,
                          min_line_length=50, max_line_gap=30, slope_min=0.4, slope_max=0.9):
    """
    普通霍夫变换车道检测
    """
    result_img = original_img.copy()
    h, w = binary_img.shape
    roi_y_start = int(h * (1 - roi_ratio))
    roi = binary_img[roi_y_start:h, 0:w]

    lines = cv2.HoughLinesP(
        roi, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    valid_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y1 += roi_y_start
            y2 += roi_y_start
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if (slope < -slope_min and slope > -slope_max) or (slope > slope_min and slope < slope_max):
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                valid_lines.append([x1, y1, x2, y2])

    roi_visual = original_img.copy()
    cv2.rectangle(roi_visual, (0, roi_y_start), (w, h), (0, 255, 0), 2)
    return result_img, roi_visual, valid_lines


def hough_improved_detect(original_img, binary_img, roi_ratio=0.5, grid_rows=9, grid_cols=16,
                          transition_thresh=15, hough_threshold=50, min_line_length=30,
                          max_line_gap=20, slope_min=0.4, slope_max=0.9):
    """
    改良霍夫变换车道检测
    """
    result_img = original_img.copy()
    h, w = binary_img.shape
    bottom_h = int(h * roi_ratio)
    roi_y_start = h - bottom_h
    bottom_img = binary_img[roi_y_start:h, :]

    cell_h = bottom_h // grid_rows
    cell_w = w // grid_cols
    mask = np.zeros_like(bottom_img, dtype=np.uint8)

    for i in range(grid_rows):
        for j in range(grid_cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = bottom_img[y1:y2, x1:x2]
            transitions = 0
            for row in cell:
                for k in range(1, len(row)):
                    if row[k - 1] != row[k]:
                        transitions += 1
            if transitions > transition_thresh:
                mask[y1:y2, x1:x2] = cell

    lines = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    valid_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y1 += roi_y_start
            y2 += roi_y_start
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if (slope < -slope_min and slope > -slope_max) or (slope > slope_min and slope < slope_max):
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                valid_lines.append([x1, y1, x2, y2])

    mask_visual = np.zeros_like(original_img)
    mask_visual[roi_y_start:h, :, 2] = mask
    mask_visual = cv2.addWeighted(original_img, 0.7, mask_visual, 0.3, 0)
    return result_img, mask_visual, valid_lines


# 兼容旧版导入的别名
def hough_original(binary_img_path: str, original_img_path: str, save_path: str = None):
    binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_img_path)
    result, _, _ = hough_original_detect(original_img, binary_img)
    if save_path:
        cv2.imwrite(save_path, result)
    return result


def hough_improved(binary_img_path: str, original_img_path: str, transition_thresh: int = 10, save_path: str = None):
    binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_img_path)
    result, mask = hough_improved_detect(original_img, binary_img, transition_thresh=transition_thresh)
    if save_path:
        cv2.imwrite(save_path, result)
    return result, mask