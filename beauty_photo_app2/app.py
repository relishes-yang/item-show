import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import zipfile
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="AI 美颜照片处理",
    page_icon="📸",
    layout="wide"
)

"""
# AI 美颜照片处理工具
"""

# ========== 初始化 session_state ==========
if "filter_states" not in st.session_state:
    st.session_state.filter_states = {}
if "processed" not in st.session_state:
    st.session_state.processed = False
if "results" not in st.session_state:
    st.session_state.results = []
if "original_img" not in st.session_state:
    st.session_state.original_img = None
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

# 增加运行计数，强制刷新
st.session_state.run_count += 1


def set_all_filters(value):
    """全选/取消全选"""
    for key in FILTER_OPTIONS.keys():
        st.session_state.filter_states[key] = value


def update_filter_state(key):
    """更新单个滤镜状态 - 通过 checkbox 的 on_change 调用"""
    # 从 widget 获取当前值
    widget_key = f"chk_{key}"
    if widget_key in st.session_state:
        st.session_state.filter_states[key] = st.session_state[widget_key]


# 自定义 CSS 样式
st.markdown("""
<style>
    .stTooltipContent {
        max-width: 400px;
        font-size: 14px;
    }
    .filter-card {
        padding: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 5px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .filter-card:hover {
        background-color: #e8eaed;
    }
    .result-title {
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 16px;
    }
    .stImage {
        margin: 0 !important;
    }
    .stImage img {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📸 AI 美颜照片处理工具")
st.markdown("---")

# 侧边栏说明
with st.sidebar:
    st.header("💡 使用说明")
    st.markdown("""
    1. **上传**一张人像照片
    2. **多选**美颜功能（可全选）
    3. **查看**对比效果
    4. **下载**处理后的图片
    """)

    st.header("⚙️ 参数设置")
    ksize = st.slider("滤波核大小 (奇数)", min_value=3, max_value=15, value=5, step=2)
    sigma = st.slider("Sigma 强度值", min_value=10, max_value=100, value=50)

# ========== 滤镜功能定义 ==========
FILTER_OPTIONS = {
    "mean": {"name": "🌟 均值滤波", "desc": "通过计算邻域像素平均值来平滑图像",
             "func": lambda img, k, s: cv2.blur(img, (k, k))},
    "box": {"name": "📦 方框滤波", "desc": "类似均值滤波，但可控制是否归一化",
            "func": lambda img, k, s: cv2.boxFilter(img, -1, (k, k), normalize=True)},
    "gaussian": {"name": "🌸 高斯滤波", "desc": "使用高斯分布加权平均，更好保留细节",
                 "func": lambda img, k, s: cv2.GaussianBlur(img, (k, k), sigmaX=s / 10)},
    "median": {"name": "🧹 中值滤波", "desc": "取邻域像素的中位数，去除椒盐噪声",
               "func": lambda img, k, s: cv2.medianBlur(img, k)},
    "bilateral": {"name": "✨ 双边滤波", "desc": "去噪同时保留边缘，适合磨皮",
                  "func": lambda img, k, s: cv2.bilateralFilter(img, d=k, sigmaColor=s, sigmaSpace=s)},
    "laplacian_sharpen": {"name": "🔍 拉普拉斯锐化", "desc": "增强图像边缘和细节",
                          "func": lambda img, k, s: np.clip(img.astype(np.float32) - cv2.Laplacian(img, cv2.CV_64F), 0,
                                                            255).astype(np.uint8)},
    "sobel": {"name": "📐 Sobel 边缘检测", "desc": "检测图像中的边缘轮廓",
              "func": lambda img, k, s: cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=k))},
    "canny": {"name": "🎯 Canny 边缘检测", "desc": "更精确的边缘检测算法",
              "func": lambda img, k, s: cv2.Canny(img, 100, 200)},
    "sharpen": {"name": "🔪 锐化增强", "desc": "通过卷积核增强图像细节",
                "func": lambda img, k, s: cv2.filter2D(img, -1,np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],dtype=np.float32))},
    "warm": {"name": "🌞 暖色滤镜", "desc": "增加红色通道强度，温暖色调", "func": lambda img, k, s: img.copy()},
    "denoise": {"name": "🧼 快速去噪", "desc": "非局部均值去噪算法",
                "func": lambda img, k, s: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)},
    "histogram": {"name": "📊 直方图均衡化", "desc": "增强图像对比度", "func": lambda img, k, s: cv2.cvtColor(
        cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]), cv2.COLOR_YUV2BGR)}
}

# ========== 初始化所有滤镜状态 ==========
for key in FILTER_OPTIONS.keys():
    if key not in st.session_state.filter_states:
        st.session_state.filter_states[key] = False


def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def apply_filters(img, selected_filters, ksize, sigma):
    results = []
    for filter_key in selected_filters:
        filter_info = FILTER_OPTIONS[filter_key]
        try:
            if filter_key == "warm":
                result = img.copy().astype(np.int16)
                result[:, :, 2] = np.clip(result[:, :, 2] + 30, 0, 255)
                result = result.astype(np.uint8)
            elif filter_key == "histogram":
                yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                result = filter_info["func"](img, ksize, sigma)

            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            results.append(
                {"name": filter_info["name"], "key": filter_key, "image": result, "desc": filter_info["desc"]})
        except Exception as e:
            st.error(f"❌ {filter_info['name']} 处理失败：{str(e)}")
    return results


def bgr_to_rgb(cv2_img):
    if len(cv2_img.shape) == 2:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
    else:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return cv2_img


def cv2_to_pil(cv2_img):
    rgb_img = bgr_to_rgb(cv2_img)
    return Image.fromarray(rgb_img)


def create_download_bytes(cv2_img, format='jpg'):
    _, buffer = cv2.imencode(f'.{format}', cv2_img)
    return buffer.tobytes()


def calculate_layout(num_images):
    if num_images <= 4:
        return [num_images]
    elif num_images == 5:
        return [3, 2]
    elif num_images == 6:
        return [3, 3]
    elif num_images == 7:
        return [4, 3]
    elif num_images == 8:
        return [4, 4]
    elif num_images == 9:
        return [3, 3, 3]
    elif num_images == 10:
        return [4, 3, 3]
    elif num_images == 11:
        return [4, 4, 3]
    elif num_images == 12:
        return [4, 4, 4]
    else:
        rows = []
        remaining = num_images
        while remaining > 0:
            rows.append(min(4, remaining))
            remaining -= 4
        return rows


# ========== 主界面 ==========
st.header("📤 步骤 1: 上传图片")
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    original_img = load_image(uploaded_file)
    st.session_state.original_img = original_img

    st.subheader("📷 原图")
    col_orig, col_info = st.columns([2, 1])
    with col_orig:
        st.image(cv2_to_pil(original_img), width=500, use_container_width=False)
    with col_info:
        st.metric("宽度", f"{original_img.shape[1]} px")
        st.metric("高度", f"{original_img.shape[0]} px")
        st.metric("通道", f"{original_img.shape[2] if len(original_img.shape) > 2 else 1}")

    st.markdown("---")

    # ========== 功能选择区域 ==========
    st.header("🎨 步骤 2: 选择美颜功能")

    # 全选/取消全选按钮
    filter_keys = list(FILTER_OPTIONS.keys())
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.button("✅ 全选", on_click=set_all_filters, args=(True,), use_container_width=True, key="btn_select_all")
    with col_btn2:
        st.button("❌ 取消全选", on_click=set_all_filters, args=(False,), use_container_width=True,
                  key="btn_deselect_all")

    # 获取当前选中的滤镜
    selected_filters = [key for key in filter_keys if st.session_state.filter_states.get(key, False)]
    st.markdown(f"✅ 已选择 **{len(selected_filters)}** 个功能")

    # ========== 关键修复：复选框直接绑定 session_state ==========
    cols_per_row = 4
    num_filters = len(filter_keys)

    for i in range(0, num_filters, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < num_filters:
                key = filter_keys[i + j]
                with col:
                    with st.container():
                        st.markdown(f"<div class='filter-card'>", unsafe_allow_html=True)

                        # ========== 关键：复选框直接读写 session_state ==========
                        st.checkbox(
                            FILTER_OPTIONS[key]["name"],
                            key=f"chk_{key}",
                            help=FILTER_OPTIONS[key]["desc"],
                            value=st.session_state.filter_states[key],  # 直接绑定
                            on_change=update_filter_state,
                            args=(key,)
                        )

                        st.markdown("</div>", unsafe_allow_html=True)

    # 重新获取选中的滤镜（确保最新）
    selected_filters = [key for key in filter_keys if st.session_state.filter_states.get(key, False)]

    # 处理按钮
    process_btn = st.button("🚀 开始处理", type="primary", disabled=len(selected_filters) == 0, key="btn_process")

    if process_btn:
        with st.spinner("正在处理图片，请稍候..."):
            results = apply_filters(original_img, selected_filters, ksize, sigma)
            if results:
                st.session_state.results = results
                st.session_state.processed = True

    # ========== 显示结果 ==========
    if st.session_state.processed and st.session_state.results:
        results = st.session_state.results
        original_img = st.session_state.original_img

        st.markdown("---")
        st.header("📊 步骤 3: 查看处理结果")

        num_results = len(results)
        layout = calculate_layout(num_results)
        st.info(f"📐 当前布局：{num_results} 张图片，共 {len(layout)} 行，每行数量：{layout}")

        result_idx = 0
        for row_num, cols_in_row in enumerate(layout):
            cols = st.columns(cols_in_row)
            for col_idx, col in enumerate(cols):
                if result_idx < num_results:
                    with col:
                        result = results[result_idx]
                        st.markdown(f"<div class='result-title'>{result['name']}</div>", unsafe_allow_html=True)
                        st.image(cv2_to_pil(result["image"]), use_container_width=True)
                        with st.expander("📖 功能说明"):
                            st.write(result["desc"])
                        result_idx += 1

        st.markdown("---")
        st.header("📥 步骤 4: 下载结果")

        st.subheader("单独下载")
        download_cols = st.columns(min(4, len(results)))
        for idx, result in enumerate(results):
            with download_cols[idx % len(download_cols)]:
                download_bytes = create_download_bytes(result["image"])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label=f"📸 下载 {result['name']}",
                    data=download_bytes,
                    file_name=f"{result['name']}_{timestamp}.jpg",
                    mime="image/jpeg",
                    key=f"dl_{result['key']}_{idx}",
                    use_container_width=True
                )

        st.subheader("📦 批量下载（ZIP 压缩包）")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for result in results:
                img_bytes = create_download_bytes(result["image"])
                zip_file.writestr(f"{result['name']}_{timestamp}.jpg", img_bytes)
            orig_bytes = create_download_bytes(original_img)
            zip_file.writestr(f"原图_{timestamp}.jpg", orig_bytes)

        zip_buffer.seek(0)
        st.download_button(
            label="📦 下载全部图片 (ZIP)",
            data=zip_buffer,
            file_name=f"美颜处理结果_{timestamp}.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )

        st.success(f"✅ 处理完成！共生成 {len(results)} 张图片。")

else:
    st.info("👆 请先上传一张图片")

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 20px;'>"
    "Powered by OpenCV + Streamlit | 📸 AI 美颜照片处理工具<br>"
    "支持 12 种图像处理功能 | 可多选 | 批量下载 | 智能布局<br>"
    "杨金伟 | 2026.3.20"
    "</div>",
    unsafe_allow_html=True
)