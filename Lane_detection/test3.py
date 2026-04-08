import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import os
import io

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="二值化车道检测", layout="wide")
st.title("📷 基于颜色分割+霍夫变换的车道线检测系统")
st.caption("计算机视觉课程作业 | 图片/视频均可下载")

# -------------------------- 核心函数 --------------------------
def lane_detection_optimized(image, canny_low, canny_high, hough_thresh, roi_height_ratio):
    height, width = image.shape[:2]

    # 颜色分割
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([10, 70, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Canny
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, canny_low, canny_high)
    combined = cv2.bitwise_or(canny, color_mask)

    # ROI
    roi_vertices = np.array([
        [
            (width * 0.05, height),
            (width * 0.40, height * roi_height_ratio),
            (width * 0.60, height * roi_height_ratio),
            (width * 0.95, height)
        ]
    ], dtype=np.int32)
    mask = np.zeros_like(combined)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_edges = cv2.bitwise_and(combined, mask)

    # 霍夫变换
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=30,
        maxLineGap=100
    )

    lane_image = image.copy()
    if lines is not None:
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.3:
                left_lines.append(line)
            elif slope > 0.3:
                right_lines.append(line)

        # 左车道
        if left_lines:
            left_x, left_y = [], []
            for line in left_lines:
                x1, y1, x2, y2 = line[0]
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            left_fit = np.polyfit(left_y, left_x, 1)
            left_line = np.poly1d(left_fit)
            y_min = int(height * roi_height_ratio)
            y_max = height
            left_x1 = int(left_line(y_max))
            left_x2 = int(left_line(y_min))
            cv2.line(lane_image, (left_x1, y_max), (left_x2, y_min), (0, 0, 255), 5)

        # 右车道
        if right_lines:
            right_x, right_y = [], []
            for line in right_lines:
                x1, y1, x2, y2 = line[0]
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])
            right_fit = np.polyfit(right_y, right_x, 1)
            right_line = np.poly1d(right_fit)
            y_min = int(height * roi_height_ratio)
            y_max = height
            right_x1 = int(right_line(y_max))
            right_x2 = int(right_line(y_min))
            cv2.line(lane_image, (right_x1, y_max), (right_x2, y_min), (0, 0, 255), 5)

        # 填充
        if left_lines and right_lines:
            pts_left = np.array([[left_x1, y_max], [left_x2, y_min]])
            pts_right = np.array([[right_x1, y_max], [right_x2, y_min]])
            pts = np.vstack((pts_left, pts_right[::-1]))
            cv2.fillPoly(lane_image, [pts], (0, 255, 0, 50))

    return roi_edges, lane_image

# -------------------------- 侧边栏参数（带详细注释） --------------------------
st.sidebar.header("⚙️ 参数调节面板")
st.sidebar.caption("鼠标悬停可查看参数说明")

canny_low = st.sidebar.slider(
    "Canny低阈值",
    0, 255, 50,
    help="边缘检测低阈值：值越小，检测到的边缘越多，噪声也会增加"
)
canny_high = st.sidebar.slider(
    "Canny高阈值",
    0, 255, 150,
    help="边缘检测高阈值：值越大，过滤的弱边缘越多，检测更精准"
)
hough_thresh = st.sidebar.slider(
    "霍夫变换阈值",
    10, 200, 30,
    help="线段检测阈值：值越小，检测到的线段越多；值越大，只保留明显线段"
)
roi_height_ratio = st.sidebar.slider(
    "ROI上边界高度比例",
    0.3, 0.8, 0.6,
    help="车道检测区域上边界：值越大，检测区域越靠下；值越小，检测范围越大"
)

# -------------------------- 选项卡 --------------------------
tab1, tab2 = st.tabs(["🖼️ 图片处理", "🎥 视频处理"])

# ========== 图片处理 + 下载 ==========
with tab1:
    uploaded_img = st.file_uploader("上传车道图片", type=["jpg", "png", "jpeg"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        start = time.time()
        binary_img, result_img = lane_detection_optimized(img, canny_low, canny_high, hough_thresh, roi_height_ratio)
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

        # 图片下载
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(result_rgb)
        buf = io.BytesIO()
        im_pil.save(buf, format='PNG')
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 下载检测结果图片",
            data=byte_im,
            file_name="lane_result.png",
            mime="image/png"
        )

# ========== 视频处理 + 下载 =================
with tab2:
    uploaded_vid = st.file_uploader("上传车道视频", type=["mp4", "avi", "mov"])
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        tfile.close()
        temp_input_path = tfile.name

        cap = cv2.VideoCapture(temp_input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output.close()
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
                _, res_frame = lane_detection_optimized(frame, canny_low, canny_high, hough_thresh, roi_height_ratio)
                out.write(res_frame)
                frame_show.image(cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB),
                                 f"处理帧：{i + 1}/{total_frames}", use_container_width=True)
                progress.progress((i + 1) / total_frames)
        finally:
            cap.release()
            out.release()

        st.success(f"视频处理完成！耗时：{time.time() - start:.3f}s")

        # 视频下载
        with open(out_path, 'rb') as f:
            st.download_button("📥 下载检测后视频", f, "lane_detected_video.mp4", "video/mp4")

        # 清理临时文件
        try:
            os.unlink(temp_input_path)
            os.unlink(out_path)
        except:
            pass