import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt
# 适配Streamlit Cloud的中文字体（优先用服务器自带的DejaVu Sans）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 全局工具函数：图片转字节流（只定义这一次！）
def img_to_bytes(img, is_bgr=True):
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG", quality=95)
    return buf.getvalue()

# 导入后端核心函数
from backend import (
    image_preprocess, binary_threshold, canny_edge_detect,
    hough_original_detect, hough_improved_detect,
    run_performance_test, plot_performance_result,
    # 新增：车道扫描函数
    draw_static_lane_scan,
    generate_dynamic_lane_scan_gif,
    video_lane_scan
)

# -------------------------- 页面基础配置 --------------------------
st.set_page_config(
    page_title="车道线检测可视化系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 创建输出目录
os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -------------------------- 侧边栏：算法参数配置（含鼠标悬停解释） --------------------------
st.sidebar.title("⚙️ 算法参数配置")

with st.sidebar.expander("🔧 预处理与二值化参数", expanded=True):
    blur_kernel = st.slider(
        "高斯模糊核大小",
        min_value=1, max_value=15, value=5, step=2,
        help="💡 鼠标悬停解释：高斯模糊核大小，必须为奇数。越大去噪越强，但图像越模糊，建议3-7"
    )

    binary_mode = st.radio(
        "二值化模式",
        ["Otsu自动阈值", "手动阈值"],
        index=0, horizontal=True,
        help="💡 鼠标悬停解释：Otsu自动计算最优阈值，适合大多数场景；手动阈值可根据光照/阴影调节"
    )

    manual_thresh_low = st.slider(
        "手动二值化低阈值",
        min_value=0, max_value=255, value=50,
        help="💡 鼠标悬停解释：手动模式下，低于该值的像素设为0（黑色）"
    ) if binary_mode == "手动阈值" else 0

    manual_thresh_high = st.slider(
        "手动二值化高阈值",
        min_value=0, max_value=255, value=255,
        help="💡 鼠标悬停解释：手动模式下，高于该值的像素设为255（白色）"
    ) if binary_mode == "手动阈值" else 255

    canny_thresh1 = st.slider(
        "Canny边缘检测低阈值",
        min_value=0, max_value=255, value=50,
        help="💡 鼠标悬停解释：Canny边缘检测的低阈值，用于弱边缘连接"
    )

    canny_thresh2 = st.slider(
        "Canny边缘检测高阈值",
        min_value=0, max_value=255, value=150,
        help="💡 鼠标悬停解释：Canny边缘检测的高阈值，用于强边缘保留"
    )

with st.sidebar.expander("📏 霍夫变换通用参数", expanded=True):
    roi_ratio = st.slider(
        "ROI车道区域占比",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="💡 鼠标悬停解释：只保留图像下半部分的比例，排除天空等无关背景，建议0.5-0.7"
    )

    slope_min = st.slider(
        "车道线斜率筛选最小值",
        min_value=0.1, max_value=1.0, value=0.4, step=0.05,
        help="💡 鼠标悬停解释：筛选车道线的最小斜率，排除水平干扰线"
    )

    slope_max = st.slider(
        "车道线斜率筛选最大值",
        min_value=0.5, max_value=2.0, value=0.9, step=0.05,
        help="💡 鼠标悬停解释：筛选车道线的最大斜率，排除垂直干扰线"
    )

with st.sidebar.expander("📐 普通霍夫变换专属参数", expanded=False):
    hough_original_thresh = st.slider(
        "霍夫累加阈值",
        min_value=10, max_value=200, value=80,
        help="💡 鼠标悬停解释：霍夫变换的累加器投票阈值，超过该值才判定为直线"
    )

    original_min_line = st.slider(
        "最小线长",
        min_value=10, max_value=200, value=50,
        help="💡 鼠标悬停解释：检测到的直线的最小长度，过滤短直线"
    )

    original_max_gap = st.slider(
        "最大线间隙",
        min_value=5, max_value=100, value=30,
        help="💡 鼠标悬停解释：连接断裂直线的最大间隙"
    )

with st.sidebar.expander("📐 改良霍夫变换专属参数", expanded=False):
    grid_rows = st.slider(
        "网格行数",
        min_value=3, max_value=20, value=9,
        help="💡 鼠标悬停解释：改良霍夫变换的分块策略，网格行数"
    )

    grid_cols = st.slider(
        "网格列数",
        min_value=5, max_value=30, value=16,
        help="💡 鼠标悬停解释：改良霍夫变换的分块策略，网格列数"
    )

    transition_thresh = st.slider(
        "像素跳变筛选阈值",
        min_value=5, max_value=50, value=15,
        help="💡 鼠标悬停解释：分块内像素跳变次数超过该值，判定为有效车道块"
    )

    hough_improved_thresh = st.slider(
        "霍夫累加阈值",
        min_value=10, max_value=200, value=50,
        help="💡 鼠标悬停解释：改良霍夫变换的累加器投票阈值"
    )

    improved_min_line = st.slider(
        "最小线长",
        min_value=10, max_value=200, value=30,
        help="💡 鼠标悬停解释：改良霍夫变换检测到的直线的最小长度"
    )

    improved_max_gap = st.slider(
        "最大线间隙",
        min_value=5, max_value=100, value=20,
        help="💡 鼠标悬停解释：改良霍夫变换连接断裂直线的最大间隙"
    )

with st.sidebar.expander("📊 性能测试配置", expanded=False):
    test_times = st.slider(
        "性能测试重复次数",
        min_value=1, max_value=20, value=10,
        help="💡 鼠标悬停解释：重复运行算法的次数，取平均值减少误差"
    )

# -------------------------- 主页面：标签页布局 --------------------------
st.title("🚗 基于二值化与霍夫变换的车道线检测可视化系统")
st.caption("《计算机视觉检测》课程作业 | 涵盖知识点：预处理、二值化、边缘检测、霍夫变换、性能分析")

tab1, tab2, tab3 = st.tabs(["📸 图片车道检测", "🎥 视频车道检测", "📊 算法性能对比"])

# ================================== 标签1：图片车道检测 ==================================
with tab1:
    st.subheader("1. 上传车道图片")
    uploaded_file = st.file_uploader(
        "上传车道图片（支持jpg/png/jpeg）",
        type=["jpg", "png", "jpeg"],
        help="💡 鼠标悬停解释：上传包含车道线的图片，或使用默认测试图片"
    )

    # 读取输入图片
    input_img = None
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file)
        input_img = np.array(img_pil)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    else:
        if os.path.exists("data/test.jpg"):
            input_img = cv2.imread("data/test.jpg")
            st.info("💡 使用默认测试图片，请将你的测试图片命名为test.jpg并放入data目录")

    if input_img is not None:
        st.subheader("2. 检测全流程可视化")

        # 1. 预处理
        preprocess_img = image_preprocess(input_img, blur_kernel=blur_kernel)
        # 2. 二值化
        gray_img, binary_img = binary_threshold(
            preprocess_img, mode="Otsu" if binary_mode == "Otsu自动阈值" else "Manual",
            manual_thresh_low=manual_thresh_low, manual_thresh_high=manual_thresh_high
        )
        # 3. 边缘检测
        edge_img = canny_edge_detect(preprocess_img, threshold1=canny_thresh1, threshold2=canny_thresh2)
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

        # -------------------------- 分步骤可视化展示 --------------------------
        st.markdown("### 🔍 处理流程（课程知识点全覆盖）")
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        with col1:
            st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB),
                     caption="步骤1：原始图像", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(preprocess_img, cv2.COLOR_BGR2RGB),
                     caption="步骤2：高斯模糊去噪", use_container_width=True)
        with col3:
            st.image(gray_img, caption="步骤3：灰度化", use_container_width=True)
        with col4:
            st.image(binary_img, caption=f"步骤4：{binary_mode}二值化", use_container_width=True)

        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.image(edge_img, caption="步骤5：Canny边缘检测", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(roi_visual, cv2.COLOR_BGR2RGB),
                     caption="步骤6：ROI车道区域", use_container_width=True)
        with col3:
            st.image(cv2.cvtColor(mask_visual, cv2.COLOR_BGR2RGB),
                     caption="步骤7：改良版有效掩码", use_container_width=True)

        # -------------------------- 最终结果对比 --------------------------
        st.markdown("### 🎯 算法检测结果对比")
        res_col1, res_col2 = st.columns(2, gap="medium")
        with res_col1:
            st.image(cv2.cvtColor(original_result, cv2.COLOR_BGR2RGB),
                     caption=f"普通霍夫变换（检测到{len(original_lines)}条有效车道线）",
                     use_container_width=True)
        with res_col2:
            st.image(cv2.cvtColor(improved_result, cv2.COLOR_BGR2RGB),
                     caption=f"改良霍夫变换（检测到{len(improved_lines)}条有效车道线）",
                     use_container_width=True)

        # -------------------------- 修复版：车道扫描效果展示 --------------------------
        st.markdown("### 🚗 车道扫描效果（静态标注/动态动画）")
        # 扫描参数配置
        scan_col1, scan_col2, scan_col3 = st.columns(3, gap="medium")
        with scan_col1:
            scan_line_count = st.slider(
                "扫描线数量", min_value=1, max_value=10, value=5, step=1,
                help="💡 静态扫描的横线数量，越多越密集"
            )
        with scan_col2:
            scan_algorithm = st.radio(
                "选择算法", ["普通霍夫变换", "改良霍夫变换"], index=0, horizontal=True,
                help="💡 选择基于哪种算法的车道线生成扫描效果"
            )
        with scan_col3:
            scan_mode = st.radio(
                "扫描模式", ["静态标注", "动态动画"], index=0, horizontal=True,
                help="💡 静态生成图片，动态生成GIF动画"
            )

        # 生成扫描效果
        if scan_algorithm == "普通霍夫变换":
            lane_lines = original_lines
            algorithm_name = "普通霍夫变换"
        else:
            lane_lines = improved_lines
            algorithm_name = "改良霍夫变换"

        if scan_mode == "静态标注":
            # 生成静态扫描标注图
            scan_result = draw_static_lane_scan(
                input_img, lane_lines, scan_line_count=scan_line_count, scan_color=(0, 0, 255), line_width=3
            )
            # 展示扫描结果
            st.image(
                cv2.cvtColor(scan_result, cv2.COLOR_BGR2RGB),
                caption=f"{algorithm_name} 车道扫描静态标注（{len(lane_lines)}条有效车道线）",
                use_container_width=True
            )
            # 下载按钮
            st.download_button(
                label="📥 下载静态扫描结果图",
                data=img_to_bytes(scan_result),
                file_name=f"lane_scan_{algorithm_name}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            # 生成动态扫描动画GIF
            with st.spinner("正在生成扫描动画，请稍候..."):
                gif_path = "output/lane_scan_dynamic.gif"
                generate_dynamic_lane_scan_gif(
                    input_img, lane_lines, output_path=gif_path, frame_count=30, duration=50
                )
                st.image(gif_path, caption=f"{algorithm_name} 车道扫描动态动画", use_container_width=True)
                # 下载GIF
                with open(gif_path, "rb") as f:
                    st.download_button(
                        label="📥 下载动态扫描GIF",
                        data=f.read(),
                        file_name=f"lane_scan_{algorithm_name}.gif",
                        mime="image/gif",
                        use_container_width=True
                    )

        # -------------------------- 结果下载 --------------------------
        st.markdown("### 📥 结果下载")

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
                label="📥 下载二值化结果图",
                data=img_to_bytes(binary_img, is_bgr=False),
                file_name="binary_result.png",
                mime="image/png",
                use_container_width=True
            )
        with d2:
            st.download_button(
                label="📥 下载普通霍夫检测结果",
                data=img_to_bytes(original_result),
                file_name="hough_original_result.png",
                mime="image/png",
                use_container_width=True
            )
        with d3:
            st.download_button(
                label="📥 下载改良霍夫检测结果",
                data=img_to_bytes(improved_result),
                file_name="hough_improved_result.png",
                mime="image/png",
                use_container_width=True
            )
    else:
        st.warning("⚠️ 请上传车道图片或使用默认测试图片")

# ================================== 标签2：视频车道检测 ==================================
with tab2:
    st.subheader("🎬 视频车道检测")
    st.info("💡 支持上传mp4格式视频，逐帧进行车道检测，可预览并下载处理后的视频")
    uploaded_video = st.file_uploader(
        "上传车道视频（支持mp4）",
        type=["mp4"],
        help="💡 鼠标悬停解释：上传包含车道线的视频，逐帧处理"
    )

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
        detect_algorithm = st.radio(
            "选择检测算法",
            ["改良霍夫变换", "普通霍夫变换"],
            horizontal=True,
            help="💡 鼠标悬停解释：改良霍夫变换精度更高、速度更快"
        )

        # 新增视频扫描线数量配置
        scan_line_count = st.slider(
            "视频扫描线数量",
            min_value=1, max_value=10, value=5, step=1,
            help="💡 鼠标悬停解释：视频中每帧的扫描线数量"
        )


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

                # 算法处理
                if detect_algorithm == "改良霍夫变换":
                    frame_result, _, frame_lines = hough_improved_detect(
                        frame, binary_frame, roi_ratio=roi_ratio, grid_rows=grid_rows, grid_cols=grid_cols,
                        transition_thresh=transition_thresh, hough_threshold=hough_improved_thresh,
                        min_line_length=improved_min_line, max_line_gap=improved_max_gap,
                        slope_min=slope_min, slope_max=slope_max
                    )
                elif detect_algorithm == "普通霍夫变换":
                    frame_result, _, frame_lines = hough_original_detect(
                        frame, binary_frame, roi_ratio=roi_ratio,
                        hough_threshold=hough_original_thresh, min_line_length=original_min_line,
                        max_line_gap=original_max_gap, slope_min=slope_min, slope_max=slope_max
                    )
                elif detect_algorithm == "带扫描标注的改良霍夫":
                    frame_result, _, frame_lines = hough_improved_detect(
                        frame, binary_frame, roi_ratio=roi_ratio, grid_rows=grid_rows, grid_cols=grid_cols,
                        transition_thresh=transition_thresh, hough_threshold=hough_improved_thresh,
                        min_line_length=improved_min_line, max_line_gap=improved_max_gap,
                        slope_min=slope_min, slope_max=slope_max
                    )
                    # 新增：添加车道扫描标注
                    frame_result = video_lane_scan(frame_result, frame_lines, scan_line_count=scan_line_count)
                else:  # 带扫描标注的普通霍夫
                    frame_result, _, frame_lines = hough_original_detect(
                        frame, binary_frame, roi_ratio=roi_ratio,
                        hough_threshold=hough_original_thresh, min_line_length=original_min_line,
                        max_line_gap=original_max_gap, slope_min=slope_min, slope_max=slope_max
                    )
                    # 新增：添加车道扫描标注
                    frame_result = video_lane_scan(frame_result, frame_lines, scan_line_count=scan_line_count)



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

# ================================== 标签3：算法性能对比 ==================================
with tab3:
    st.subheader("📊 算法性能对比测试")
    st.info("💡 基于当前上传的图片，重复运行多次算法，统计平均耗时与加速比，适合课程作业性能分析")

    # 从图片检测模块获取图片
    if 'input_img' in locals() and input_img is not None:
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
    "📌 基于《计算机视觉检测》课程知识点开发 | 车道线检测 - 二值化+霍夫变换 <br>"
    "支持 12 种图像处理功能 | 可多选 | 批量下载 | 智能布局<br>"
    "💡前端：杨金伟 | 💡后端：马骏玮<br>"
    "2026.3.31</div>",
    unsafe_allow_html=True
)