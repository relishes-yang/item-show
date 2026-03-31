import cv2
import numpy as np


def image_preprocess(image, blur_kernel=5):
    """
    【课程知识点】图像预处理：高斯模糊去噪
    原理：通过高斯核平滑图像，减少噪声对后续二值化/边缘检测的干扰
    :param image: 输入原始BGR图像
    :param blur_kernel: 高斯模糊核大小（奇数，越大去噪越强但图像越模糊）
    :return: 预处理后的图像
    """
    # 高斯模糊：对图像进行平滑处理，抑制高频噪声
    blur_img = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    return blur_img


def binary_threshold(image, mode="Otsu", manual_thresh_low=50, manual_thresh_high=255):
    """
    【课程知识点】图像二值化
    原理：将灰度图像转换为黑白二值图像，突出目标区域（车道线）
    :param image: 输入预处理后的BGR图像
    :param mode: 二值化模式，可选 "Otsu"（大津法自动最优阈值）或 "Manual"（手动阈值）
    :param manual_thresh_low: 手动模式下的低阈值
    :param manual_thresh_high: 手动模式下的高阈值
    :return: 灰度图、二值化结果图
    """
    # 第一步：转为灰度图（二值化必须基于单通道灰度图）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 第二步：二值化处理
    if mode == "Otsu":
        # 【课程知识点】大津法（Otsu）：自动计算最优阈值，无需手动调节
        # 原理：基于类间方差最大化，自动找到最佳分割阈值
        _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # 手动阈值二值化：根据场景手动调节阈值
        _, binary_img = cv2.threshold(gray, manual_thresh_low, manual_thresh_high, cv2.THRESH_BINARY)

    return gray, binary_img


def canny_edge_detect(image, threshold1=50, threshold2=150):
    """
    【课程知识点】Canny边缘检测
    原理：多阶段算法，包括高斯模糊、梯度计算、非极大值抑制、双阈值检测
    :param image: 输入预处理后的BGR图像
    :param threshold1: 低阈值（用于弱边缘连接）
    :param threshold2: 高阈值（用于强边缘保留）
    :return: 边缘检测结果图
    """
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Canny边缘检测：经典边缘检测算法，定位准确、抗噪性强
    edge_img = cv2.Canny(gray, threshold1, threshold2)
    return edge_img