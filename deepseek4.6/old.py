import cv2
import numpy as np

def detect_circle_meter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯滤波去噪
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # 霍夫圆检测
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=50, param1=50, param2=30,
                               minRadius=50, maxRadius=500)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 取半径最大的圆作为表盘（假设图片中仪表是主要物体）
        x, y, r = max(circles, key=lambda c: c[2])
        return (x, y, r)
    return None