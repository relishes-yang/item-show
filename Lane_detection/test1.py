import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import os

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="车道线检测", layout="wide")
st.title("📷 车道线检测系统 | 稳定无报错版")
st.caption("计算机视觉课程作业 | 视频处理已修复")


# -------------------------- 核心函数：全参数可调的车道检测 --------------------------
def lane_detection_full(image,
                        canny_low, canny_high,
                        hough_thresh, min_line_len, max_line_gap,
                        roi_top_ratio, roi_left_ratio, roi_right_ratio,
                        white_low, white_high,
                        angle_min, angle_max):
    """
    全参数可调车道检测，适配各种路况
    """
    height, width = image.shape[:2]

    # 1. 颜色分割：提取白色车道线
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, white_low])
    upper_white = np.array([180, 30, white_high])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 2. 边缘检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, canny_low, canny_high)
    combined = cv2.bitwise_or(canny, white_mask)

    # 3. ROI区域（排除护栏干扰）
    roi_vertices = np.array([
        [
            (width * roi_left_ratio, height),
            (width * 0.40, height * roi_top_ratio),
            (width * 0.60, height * roi_top_ratio),
            (width * roi_right_ratio, height)
        ]
    ], dtype=np.int32)
    mask = np.zeros_like(combined)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_edges = cv2.bitwise_and(combined, mask)

    # 4. 霍夫变换：检测线段
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )

    # 5. 线段过滤 + 车道线拟合
    lane_image = image.copy()
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 过滤角度不合理的线
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if not (angle_min < abs(angle) < angle_max):
                continue
            # 按斜率分类
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.3:
                left_lines.append(line)
            elif slope > 0.3:
                right_lines.append(line)

    # 绘制左车道线
    if left_lines:
        left_x = []
        left_y = []
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            left_x.extend([x1, x2])
            left_y.extend([y1, y2])
        left_fit = np.polyfit(left_y, left_x, 1)
        left_line = np.poly1d(left_fit)
        y_min = int(height * roi_top_ratio)
        y_max = height
        left_x1 = int(left_line(y_max))
        left_x2 = int(left_line(y_min))
        cv2.line(lane_image, (left_x1, y_max), (left_x2, y_min), (0, 0, 255), 5)

    # 绘制右车道线
    if right_lines:
        right_x = []
        right_y = []
        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            right_x.extend([x1, x2])
            right_y.extend([y1, y2])
        right_fit = np.polyfit(right_y, right_x, 1)
        right_line = np.poly1d(right_fit)
        y_min = int(height * roi_top_ratio)
        y_max = height
        right_x1 = int(right_line(y_max))
        right_x2 = int(right_line(y_min))
        cv2.line(lane_image, (right_x1, y_max), (right_x2, y_min), (0, 0, 255), 5)

    # 填充车道区域
    if left_lines and right_lines:
        pts_left = np.array([[left_x1, y_max], [left_x2, y_min]])
        pts_right = np.array([[right_x1, y_max], [right_x2, y_min]])
        pts = np.vstack((pts_left, pts_right[::-1]))
        cv2.fillPoly(lane_image, [pts], (0, 255, 0, 50))

    return roi_edges, lane_image


# -------------------------- 侧边栏：全参数可调 --------------------------
st.sidebar.header("⚙️ 参数调节")

# 1. 边缘检测
st.sidebar.subheader("1. 边缘检测")
canny_low = st.sidebar.slider("Canny低阈值", 0, 255, 50)
canny_high = st.sidebar.slider("Canny高阈值", 0, 255, 150)

# 2. 颜色分割
st.sidebar.subheader("2. 颜色分割")
white_low = st.sidebar.slider("白色亮度下限", 0, 255, 200)
white_high = st.sidebar.slider("白色亮度上限", 0, 255, 255)

# 3. ROI区域（关键！排除护栏）
st.sidebar.subheader("3. ROI区域")
roi_top_ratio = st.sidebar.slider("ROI上边界高度", 0.3, 0.8, 0.6)
roi_left_ratio = st.sidebar.slider("ROI左边界", 0.0, 0.3, 0.15)
roi_right_ratio = st.sidebar.slider("ROI右边界", 0.7, 1.0, 0.95)

# 4. 霍夫变换
st.sidebar.subheader("4. 霍夫变换")
hough_thresh = st.sidebar.slider("霍夫阈值", 10, 200, 30)
min_line_len = st.sidebar.slider("最小线段长度", 10, 200, 50)
max_line_gap = st.sidebar.slider("最大线段间隙", 10, 200, 150)

# 5. 线段过滤
st.sidebar.subheader("5. 线段过滤")
angle_min = st.sidebar.slider("最小角度", 20, 45, 30)
angle_max = st.sidebar.slider("最大角度", 60, 85, 80)

# -------------------------- 选项卡：图片/视频 --------------------------
tab1, tab2 = st.tabs(["🖼️ 图片处理", "🎥 视频处理"])

# ========== 图片处理 ==========
with tab1:
    uploaded_img = st.file_uploader("上传车道图片", type=["jpg", "png", "jpeg"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        start = time.time()
        binary_img, result_img = lane_detection_full(
            img, canny_low, canny_high,
            hough_thresh, min_line_len, max_line_gap,
            roi_top_ratio, roi_left_ratio, roi_right_ratio,
            white_low, white_high,
            angle_min, angle_max
        )
        end = time.time()

        st.subheader("检测结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "原始图像", use_container_width=True)
        with col2:
            st.image(binary_img, "二值化边缘图", use_container_width=True)
        with col3:
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), "车道检测结果", use_container_width=True)

        st.success(f"处理完成！耗时：{end - start:.3f}s")

# ========== 视频处理（已修复） ==========
with tab2:
    uploaded_vid = st.file_uploader("上传车道视频", type=["mp4", "avi", "mov"])
    if uploaded_vid is not None:
        # 临时文件处理（先关闭句柄再使用）
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_vid.read())
            temp_input_path = tfile.name

        cap = cv2.VideoCapture(temp_input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 输出视频临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
            out_path = temp_output.name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        progress = st.progress(0)
        frame_show = st.empty()
        start = time.time()

        try:
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                # 逐帧检测
                _, res_frame = lane_detection_full(
                    frame, canny_low, canny_high,
                    hough_thresh, min_line_len, max_line_gap,
                    roi_top_ratio, roi_left_ratio, roi_right_ratio,
                    white_low, white_high,
                    angle_min, angle_max
                )
                out.write(res_frame)
                # 实时预览
                frame_show.image(cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB),
                                 f"处理帧：{i + 1}/{total_frames}", use_container_width=True)
                progress.progress((i + 1) / total_frames)
        finally:
            # 确保释放资源
            cap.release()
            out.release()

        st.success(f"视频处理完成！耗时：{time.time() - start:.3f}s")

        # 提供下载
        with open(out_path, 'rb') as f:
            st.download_button("📥 下载检测后视频", f, "lane_detected_video.mp4", "video/mp4")

        # 安全删除临时文件（不影响程序运行）
        try:
            os.unlink(temp_input_path)
            os.unlink(out_path)
        except:
            pass