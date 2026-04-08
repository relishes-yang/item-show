from setuptools import setup, find_packages

# 自动发现所有子目录的包，并统一声明依赖
setup(
    name="item-show",
    version="1.0.0",
    packages=find_packages(),  # 递归扫描所有子文件夹
    install_requires=[
        "streamlit>=1.30.0",
        "numpy>=1.26.0",
        "pillow>=10.0.0",
        "opencv-python-headless>=4.8.0",
    ],
)