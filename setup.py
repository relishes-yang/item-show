from setuptools import setup, find_packages

setup(
    name="item_show",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "numpy",
        "pillow",
        "opencv-python-headless",
    ],
)