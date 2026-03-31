# 统一导出后端接口，文件名、函数名必须完全对应
from .binary_processor import (
    image_preprocess,
    binary_threshold,
    canny_edge_detect
)
from .hough_detector import (
    hough_original_detect,
    hough_improved_detect
)
from .performance import (
    run_performance_test,
    plot_performance_result
)
# 新增：车道扫描功能导出
from .lane_scanner import (
    draw_static_lane_scan,
    generate_dynamic_lane_scan_gif,
    video_lane_scan
)