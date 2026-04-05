import cv2
import numpy as np
import pytesseract

# ========== 1.配置（自己改图片路径就行）==========
IMG_PATH = "meter.jpg"  # 把仪表照片放同文件夹
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 你的安装路径


# ========== 2.图片预处理（去噪+矫正）==========
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 降噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 二值化
    _, binary = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
    # 腐蚀膨胀强化轮廓
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(binary, kernel, iterations=2)
    return gray, binary, dilate


# ========== 3.数字液晶屏识别 ==========
def ocr_read(gray_img):
    # 裁剪屏幕区域可自己微调
    roi = gray_img[100:300, 200:500]
    text = pytesseract.image_to_string(roi, config='--psm 6 digits')
    return text.strip()


# ========== 4.主运行流程 ==========
if __name__ == "__main__":
    # 读图片
    img = cv2.imread(IMG_PATH)
    if img is None:
        print("错误：找不到图片！检查路径和文件名")
        exit()

    gray, binary, dilate = preprocess(img)
    # 识别数字读数
    result = ocr_read(gray)

    print("=====电力仪表识别结果=====")
    print(f"识别读数：{result}")

    # 展示效果图
    cv2.imshow("原图", img)
    cv2.imshow("预处理图", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()