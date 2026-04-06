import cv2
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np  # 需要导入numpy用于锐化操作

"""
图片名称不支持中文名称，运行失败记得改一下文件名称
"""
class ImageProcessor:
    def __init__(self):
        # 获取桌面路径
        self.desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
        self.photos_folder = os.path.join(self.desktop_path, 'photos')

        # 确保输出文件夹存在
        if not os.path.exists(self.photos_folder):
            os.makedirs(self.photos_folder)

        # 初始化Tkinter（用于隐藏主窗口，只显示对话框）
        self.root = tk.Tk()
        self.root.withdraw()

    def load_image(self):
        """弹出文件选择框，让用户选择图片"""
        file_path = filedialog.askopenfilename(
            title="请选择一张图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        # 【修正处】这里之前写成了 filepath，现在改为 file_path
        if not file_path:
            print("未选择图片，程序退出。")
            return None

        # 读取图片
        img = cv2.imread(file_path)
        if img is None:
            print("无法读取图片，请检查文件格式。")
            return None

        print(f"成功加载图片: {file_path}")
        return img

    def apply_filter(self, img, choice):
        """根据用户选择应用不同的美颜功能"""
        if choice == 1:  # 磨皮 (基于高斯模糊)
            # 4.2.3 高斯滤波：平滑皮肤纹理
            result = cv2.GaussianBlur(img, (15, 15), 0)
            description = "SmoothSkin"

        elif choice == 2:  # 滤镜 (暖色调)
            # 调整颜色空间，增加红色和黄色
            result = img.copy()
            # 注意：OpenCV默认是BGR格式，索引2是红色通道
            result[:, :, 2] = cv2.add(result[:, :, 2], 30)
            description = "WarmFilter"

        elif choice == 3:  # 去噪 (中值滤波)
            # 4.3.1 中值滤波：有效去除椒盐噪声
            result = cv2.medianBlur(img, 5)
            description = "Denoise"

        elif choice == 4:  # 锐化 (增强细节)
            # 定义锐化核
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            result = cv2.filter2D(img, -1, kernel)
            description = "Sharpen"

        elif choice == 5:  # 双边滤波 (高级磨皮)
            # 4.3.3 双边滤波：既能模糊又能保留边缘
            result = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            description = "BilateralBlur"

        else:
            print("无效的选择，使用原图。")
            return img, "Original"

        return result, description

    def save_image(self, img, description):
        """保存图片到桌面的 photos 文件夹"""
        # 生成文件名
        filename = f"result_{description}.jpg"
        save_path = os.path.join(self.photos_folder, filename)

        # 保存图片
        cv2.imwrite(save_path, img)
        print(f"图片已保存至: {save_path}")
        return save_path

    def run(self):
        """主运行流程"""
        # 1. 加载图片
        image = self.load_image()
        if image is None:
            return

        # 2. 弹出选择框，让用户选择功能
        choice_str = simpledialog.askstring(
            "功能选择",
            "请输入功能编号:\n1. 磨皮 (高斯滤波)\n2. 滤镜 (暖色调)\n3. 去噪 (中值滤波)\n4. 锐化\n5. 双边滤波 (高级磨皮)"
        )

        if not choice_str:
            print("未选择功能。")
            return

        try:
            choice = int(choice_str)
            if choice < 1 or choice > 5:
                raise ValueError
        except ValueError:
            print("请输入有效的数字 1-5")
            return

        # 3. 处理图片
        processed_img, desc = self.apply_filter(image, choice)

        # 4. 保存图片
        save_path = self.save_image(processed_img, desc)

        # 5. 提示用户
        messagebox.showinfo("完成", f"处理完成！\n文件已保存到:\n{save_path}")


if __name__ == "__main__":
    app = ImageProcessor()
    app.run()