import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO
import tempfile

# -------------------------- 初始化会话状态（解决标签页变量访问问题） --------------------------
if 'input_img' not in st.session_state:
    st.session_state.input_img = None
if 'binary_img' not in st.session_state:
    st.session_state.binary_img = None
if 'preprocess_img' not in st.session_state:
    st.session_state.preprocess_img = None
if 'gray_img' not in st.session_state:
    st.session_state.gray_img = None
if 'edge_img' not in st.session_state:
    st.session_state.edge_img = None

# 导入后端核心函数（完全对齐__init__.py导出）
from backend import (
    image_preprocess, binary_threshold, canny_edge_detect,
    hough_original_detect, hough_improved_detect,
    run_performance_test, plot_performance_result
)

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="车道线检测可视化系统", layout="wide", page_icon="📷")
st.title("📷 基于二值化与霍夫变换的车道线检测可视化系统")
st.caption("《计算机视觉检测》课程作业 | Streamlit可视化")

# 创建输出目录
os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -------------------------- 侧边栏参数 --------------------------
st.sidebar.header("⚙️ 算法参数配置")
with st.sidebar.expander("🔧 预处理与二值化参数", expanded=True):
    blur_kernel = st.slider("高斯模糊核大小（去噪）", 1, 15, 5, step=2)
    binary_mode = st.radio("二值化模式", ["Otsu自动阈值", "手动阈值"], index=0, horizontal=True)
    manual_thresh_low = st.slider("手动二值化低阈值", 0, 255, 50) if binary_mode == "手动阈值" else 0
    manual_thresh_high = st.slider("手动二值化高阈值", 0, 255, 255) if binary_mode == "手动阈值" else 255
    canny_thresh1 = st.slider("Canny边缘检测低阈值", 0, 255, 50)
    canny_thresh2 = st.slider("Canny边缘检测高阈值", 0, 255, 150)

with st.sidebar.expander("📏 霍夫变换通用参数", expanded=True):
    roi_ratio = st.slider("ROI车道区域占比（图像下半部分）", 0.1, 0.9, 0.5, step=0.05)
    slope_min = st.slider("车道线斜率筛选最小值", 0.1, 1.0, 0.4, step=0.05)
    slope_max = st.slider("车道线斜率筛选最大值", 0.5, 2.0, 0.9, step=0.05)

with st.sidebar.expander("📐 普通霍夫变换专属参数", expanded=False):
    hough_original_thresh = st.slider("霍夫累加阈值", 10, 200, 80)
    original_min_line = st.slider("最小线长", 10, 200, 50)
    original_max_gap = st.slider("最大线间隙", 5, 100, 30)

with st.sidebar.expander("📐 改良霍夫变换专属参数", expanded=False):
    grid_rows = st.slider("网格行数", 3, 20, 9)
    grid_cols = st.slider("网格列数", 5, 30, 16)
    transition_thresh = st.slider("像素跳变筛选阈值", 5, 50, 15)
    hough_improved_thresh = st.slider("霍夫累加阈值", 10, 200, 50)
    improved_min_line = st.slider("最小线长", 10, 200, 30)
    improved_max_gap = st.slider("最大线间隙", 5, 100, 20)

with st.sidebar.expander("📊 性能测试配置", expanded=False):
    test_times = st.slider("性能测试重复次数", 1, 20, 10)

# -------------------------- 主页面 --------------------------
tab1, tab2, tab3 = st.tabs(["📸 图片车道检测", "🎥 视频车道检测", "📊 算法性能对比"])

# ================================== 图片检测模块 ==================================
with tab1:
    st.subheader("1. 上传车道图片")
    uploaded_file = st.file_uploader("上传车道图片", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    # 读取输入图片
    input_img = None
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file)
        input_img = np.array(img_pil)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    else:
        if os.path.exists("data/test.jpg"):
            input_img = cv2.imread("data/test.jpg")

    if input_img is not None:
        # 存储到会话状态，供其他标签页使用
        st.session_state.input_img = input_img

        st.subheader("2. 检测全流程可视化")
        # 1. 预处理
        preprocess_img = image_preprocess(input_img, blur_kernel=blur_kernel)
        st.session_state.preprocess_img = preprocess_img
        # 2. 二值化
        gray_img, binary_img = binary_threshold(
            preprocess_img, mode="Otsu" if binary_mode == "Otsu自动阈值" else "Manual",
            manual_thresh_low=manual_thresh_low, manual_thresh_high=manual_thresh_high
        )
        st.session_state.gray_img = gray_img
        st.session_state.binary_img = binary_img
        # 3. 边缘检测
        edge_img = canny_edge_detect(preprocess_img, threshold1=canny_thresh1, threshold2=canny_thresh2)
        st.session_state.edge_img = edge_img
        # 4. 两种霍夫检测
        original_result, roi_visual, original_lines = hough_original_detect(
            input_img, binary_img, roi_ratio=roi_ratio,
            hough_threshold=hough_original_thresh, min_line_length=original_min_line,
            max_line_gap=original_max_gap, slope_min=slope_min, slope_max=slope_max
        )
        improved_result, mask_visual, improved_lines = hough_improved_detect(
            input_img, binary_img, roi_ratio=roi_ratio, grid_rows=grid_rows, grid_cols=grid_cols,
            transition_thresh=transition_thresh, hough_threshold=hough_improved_thresh,
            min_line_length=improved_min_line, max_line_gap=improved_max_gap,
            slope_min=slope_min, slope_max=slope_max
        )

        # -------------------------- 流程可视化（修复参数+优化布局） --------------------------
        # 第一行：预处理流程
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        with col1:
            st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB),
                     caption="原始图像", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(preprocess_img, cv2.COLOR_BGR2RGB),
                     caption="高斯模糊去噪", use_container_width=True)
        with col3:
            st.image(gray_img, caption="灰度化", use_container_width=True)
        with col4:
            st.image(binary_img, caption=f"{binary_mode}二值化", use_container_width=True)

        # 第二行：边缘检测&ROI
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.image(edge_img, caption="Canny边缘检测", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(roi_visual, cv2.COLOR_BGR2RGB),
                     caption="ROI车道区域", use_container_width=True)
        with col3:
            st.image(cv2.cvtColor(mask_visual, cv2.COLOR_BGR2RGB),
                     caption="改良版有效掩码", use_container_width=True)

        # -------------------------- 最终结果对比 --------------------------
        st.subheader("3. 算法检测结果对比")
        res_col1, res_col2 = st.columns(2, gap="medium")
        with res_col1:
            st.image(cv2.cvtColor(original_result, cv2.COLOR_BGR2RGB),
                     caption=f"普通霍夫变换（{len(original_lines)}条有效线）",
                     use_container_width=True)
        with res_col2:
            st.image(cv2.cvtColor(improved_result, cv2.COLOR_BGR2RGB),
                     caption=f"改良霍夫变换（{len(improved_lines)}条有效线）",
                     use_container_width=True)

        # -------------------------- 结果下载 --------------------------
        st.subheader("4. 结果下载")


        def img_to_bytes(img, is_bgr=True):
            if is_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            buf = BytesIO()
            pil_img.save(buf, format="PNG", quality=95)
            return buf.getvalue()


        d1, d2, d3 = st.columns(3, gap="medium")
        with d1:
            st.download_button(
                label="📥 下载二值化图",
                data=img_to_bytes(binary_img, is_bgr=False),
                file_name="binary_result.png",
                mime="image/png",
                use_container_width=True
            )
        with d2:
            st.download_button(
                label="📥 下载普通霍夫结果",
                data=img_to_bytes(original_result),
                file_name="hough_original_result.png",
                mime="image/png",
                use_container_width=True
            )
        with d3:
            st.download_button(
                label="📥 下载改良霍夫结果",
                data=img_to_bytes(improved_result),
                file_name="hough_improved_result.png",
                mime="image/png",
                use_container_width=True
            )
    else:
        st.info("请上传车道图片或使用默认测试图片，开始检测")

# ================================== 视频检测模块 ==================================
with tab2:
    st.subheader("视频车道检测")
    st.info("支持上传mp4格式视频，逐帧进行车道检测，可预览并下载处理后的视频")
    uploaded_video = st.file_uploader("上传车道视频", type=["mp4"], label_visibility="collapsed")

    if uploaded_video is not None:
        # 保存临时视频文件
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        # 打开视频
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.write(f"✅ 视频信息：{width}×{height} | {fps:.1f}fps | 总帧数：{total_frames}")

        # 检测算法选择
        detect_algorithm = st.radio("选择检测算法", ["改良霍夫变换", "普通霍夫变换"], horizontal=True)
        process_btn = st.button("开始处理视频", type="primary", use_container_width=True)

        if process_btn:
            # 视频写入配置
            output_path = "output/video_detection_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 逐帧处理
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 核心检测逻辑
                preprocess_frame = image_preprocess(frame, blur_kernel=blur_kernel)
                _, binary_frame = binary_threshold(
                    preprocess_frame, mode="Otsu" if binary_mode == "Otsu自动阈值" else "Manual",
                    manual_thresh_low=manual_thresh_low, manual_thresh_high=manual_thresh_high
                )

                if detect_algorithm == "改良霍夫变换":
                    frame_result, _, _ = hough_improved_detect(
                        frame, binary_frame, roi_ratio=roi_ratio, grid_rows=grid_rows, grid_cols=grid_cols,
                        transition_thresh=transition_thresh, hough_threshold=hough_improved_thresh,
                        min_line_length=improved_min_line, max_line_gap=improved_max_gap,
                        slope_min=slope_min, slope_max=slope_max
                    )
                else:
                    frame_result, _, _ = hough_original_detect(
                        frame, binary_frame, roi_ratio=roi_ratio,
                        hough_threshold=hough_original_thresh, min_line_length=original_min_line,
                        max_line_gap=original_max_gap, slope_min=slope_min, slope_max=slope_max
                    )

                # 写入视频
                out.write(frame_result)

                # 更新进度
                frame_count += 1
                progress = int(frame_count / total_frames * 100)
                progress_bar.progress(progress)
                status_text.text(f"处理进度：{frame_count}/{total_frames} 帧")

            # 释放资源
            cap.release()
            out.release()
            os.unlink(tfile.name)
            progress_bar.empty()
            status_text.text("✅ 视频处理完成！")

            # 视频下载
            with open(output_path, "rb") as f:
                st.download_button(
                    label="📥 下载处理后的视频",
                    data=f.read(),
                    file_name="lane_detection_video_result.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

# ================================== 性能对比模块 ==================================
with tab3:
    st.subheader("算法性能对比测试")
    st.info("基于当前上传的图片，重复运行多次算法，统计平均耗时与加速比")

    # 从会话状态获取图片，解决标签页间变量访问问题
    input_img = st.session_state.input_img
    binary_img = st.session_state.binary_img

    if input_img is not None and binary_img is not None:
        run_test_btn = st.button("开始性能对比测试", type="primary", use_container_width=True)
        if run_test_btn:
            with st.spinner("正在执行性能测试，请稍候..."):
                # 运行性能测试
                t_original_avg, t_improved_avg, speedup_ratio, _, _ = run_performance_test(
                    input_img, binary_img, test_times=test_times
                )
                # 生成对比图
                fig = plot_performance_result(t_original_avg, t_improved_avg, speedup_ratio)

                # 展示结果
                st.subheader("📊 性能测试结果")
                result_col1, result_col2, result_col3 = st.columns(3, gap="medium")
                with result_col1:
                    st.metric("普通霍夫平均耗时", f"{t_original_avg:.3f} s")
                with result_col2:
                    st.metric("改良霍夫平均耗时", f"{t_improved_avg:.3f} s")
                with result_col3:
                    st.metric("改良版加速比", f"{speedup_ratio:.2f} x")

                # 展示性能对比图
                st.subheader("性能对比可视化")
                st.pyplot(fig, use_container_width=True)

                # 保存并下载报告
                fig.savefig("output/performance_result.png", dpi=300, bbox_inches='tight')
                with open("output/performance_result.png", "rb") as f:
                    st.download_button(
                        label="📥 下载性能对比图",
                        data=f.read(),
                        file_name="lane_detection_performance.png",
                        mime="image/png",
                        use_container_width=True
                    )
    else:
        st.warning("⚠️ 请先在【图片车道检测】标签页上传图片，再进行性能测试")

# -------------------------- 页脚 --------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 20px;'>"
    "📌 基于《计算机视觉检测》课程知识点开发 | 车道线检测 - 二值化+霍夫变换<br>"
    "🌈前端：杨金伟 |  🌈后端：马骏玮<br>"
    "2026.3.31"
    "</div>",
    unsafe_allow_html=True
)