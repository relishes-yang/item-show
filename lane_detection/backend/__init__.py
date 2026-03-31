# 统一导出所有后端接口，确保和app.py、binary_processor.py完全一致
from .binary_processor import (
    image_preprocess,
    binary_threshold,
    canny_edge_detect,
    image_binary_process,
    video_frame_binary_process
)
from .hough_detector import (
    hough_original_detect,
    hough_improved_detect
)
from .performance_evaluator import (
    run_performance_test,
    plot_performance_result
)