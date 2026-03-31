import cv2
import numpy as np

def image_preprocess(image, blur_kernel=5):
    """
    图像预处理：高斯模糊去噪（前端导入的核心函数，必须存在）
    :param image: 输入原始BGR图像
    :param blur_kernel: 高斯模糊核大小（奇数，越大去噪越强）
    :return: 预处理后的图像
    """
    blur_img = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    return blur_img

def binary_threshold(image, mode="Otsu", manual_thresh_low=50, manual_thresh_high=255):
    """
    二值化核心函数
    :param image: 输入预处理后的BGR图像
    :param mode: 二值化模式，可选 Otsu/Manual（手动阈值）
    :param manual_thresh_low: 手动模式下的低阈值
    :param manual_thresh_high: 手动模式下的高阈值
    :return: 灰度图、二值化结果图
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "Otsu":
        _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_img = cv2.threshold(gray, manual_thresh_low, manual_thresh_high, cv2.THRESH_BINARY)
    return gray, binary_img

def canny_edge_detect(image, threshold1=50, threshold2=150):
    """
    Canny边缘检测（适配课程第5章「边缘检测」知识点）
    :param image: 输入预处理后的BGR图像
    :param threshold1: 低阈值
    :param threshold2: 高阈值
    :return: 边缘检测结果图
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray, threshold1, threshold2)
    return edge_img

# 兼容旧版函数名（防止app.py旧代码报错）
def image_binary_process(image_path: str, save_path: str = None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到图片文件：{image_path}")
    preprocessed = image_preprocess(img)
    gray, binary = binary_threshold(preprocessed, mode="Otsu")
    if save_path:
        cv2.imwrite(save_path, binary)
    return img, gray, binary

def video_frame_binary_process(frame: np.ndarray) -> np.ndarray:
    preprocessed = image_preprocess(frame)
    _, binary = binary_threshold(preprocessed, mode="Otsu")
    return binary