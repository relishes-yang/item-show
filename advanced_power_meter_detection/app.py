
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os
import sys


# app.py 开头添加以下代码
import sys
import importlib.util

# 先检查OpenCV是否已安装，未安装则自动安装headless版本
if not importlib.util.find_spec("cv2"):
    import os
    os.system(f"{sys.executable} -m pip install opencv-python-headless==4.8.0.74")
# 后续正常导入
import cv2

# 页面配置
st.set_page_config(
    page_title="🔌 智能电力仪表检测系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .step-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .meter-result {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #00cc00;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00cc00;
    }
</style>
""", unsafe_allow_html=True)

# 导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.smart_meter_detector import AdvancedMeterDetector, MeterInfo

# 初始化检测器
@st.cache_resource
def get_detector():
    return AdvancedMeterDetector()

def create_gauge(value, min_val, max_val, unit, meter_id):
    """创建仪表盘图表"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"仪表 {meter_id+1} ({unit})", 'font': {'size': 16}},
        number={'suffix': f" {unit}", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, min_val + (max_val-min_val)*0.3], 'color': 'lightgreen'},
                {'range': [min_val + (max_val-min_val)*0.3, min_val + (max_val-min_val)*0.7], 'color': 'yellow'},
                {'range': [min_val + (max_val-min_val)*0.7, max_val], 'color': 'salmon'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    st.markdown('<p class="main-header">🔌 智能电力仪表检测系统</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#666;'>自动识别量程、单位、多仪表检测 | 可视化处理流程</p>", unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ 系统配置")

        st.markdown("### 🔧 检测参数")
        min_radius = st.slider("最小仪表半径", 30, 100, 50)
        max_radius_ratio = st.slider("最大半径比例", 2, 5, 3)

        st.markdown("### 📊 显示选项")
        show_steps = st.checkbox("显示处理步骤", value=True)
        show_details = st.checkbox("显示详细信息", value=True)

        st.markdown("---")
        st.markdown("### 💡 使用说明")
        st.info("""
        1. 上传包含仪表的图片
        2. 系统自动检测所有仪表
        3. 自动识别量程和单位
        4. 查看处理步骤和结果
        """)

    # 主界面
    uploaded_file = st.file_uploader("📸 上传仪表图片", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file is not None:
        # 读取图片
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📷 原始图片")
            st.image(image, use_container_width=True)

            if st.button("🔍 开始智能检测", type="primary", use_container_width=True):
                with st.spinner("🔄 正在分析..."):
                    detector = get_detector()

                    # 执行检测
                    meters = detector.detect_meters(image_cv)

                    if not meters:
                        st.error("❌ 未检测到仪表，请检查图片质量")
                        return

                    st.session_state['meters'] = meters
                    st.session_state['steps'] = detector.get_processing_steps()
                    st.success(f"✅ 检测到 {len(meters)} 个仪表")

        with col2:
            if 'meters' in st.session_state:
                meters = st.session_state['meters']
                steps = st.session_state['steps']

                # 显示处理步骤
                if show_steps and steps:
                    st.markdown("### 🔬 处理流程可视化")

                    # 使用标签页展示步骤
                    step_tabs = st.tabs([step.step_name for step in steps])

                    for tab, step in zip(step_tabs, steps):
                        with tab:
                            st.markdown(f"**{step.description}**")
                            step_img = cv2.cvtColor(step.image, cv2.COLOR_BGR2RGB)
                            st.image(step_img, use_container_width=True)

                # 显示检测结果
                st.markdown("### 📊 检测结果")

                # 最终结果大图
                if steps:
                    final_img = cv2.cvtColor(steps[-1].image, cv2.COLOR_BGR2RGB)
                    st.image(final_img, caption="检测标注结果", use_container_width=True)

                # 每个仪表的详细信息
                if show_details:
                    st.markdown("### 🎯 仪表详细信息")

                    meter_cols = st.columns(min(len(meters), 3))

                    for idx, (col, meter) in enumerate(zip(meter_cols, meters)):
                        with col:
                            # 仪表盘可视化
                            fig = create_gauge(
                                meter.current_value,
                                meter.min_val,
                                meter.max_val,
                                meter.unit,
                                meter.meter_id
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{idx}")

                            # 详细信息
                            with st.expander(f"查看仪表 {meter.meter_id+1} 详情"):
                                st.markdown(f"""
                                - **当前读数**: `{meter.current_value} {meter.unit}`
                                - **量程范围**: `{meter.min_val} - {meter.max_val} {meter.unit}`
                                - **指针角度**: `{meter.pointer_angle:.1f}°`
                                - **角度范围**: `{meter.min_angle:.1f}° - {meter.max_angle:.1f}°`
                                - **检测置信度**: `{meter.confidence:.2%}`
                                - **圆心位置**: `({meter.center[0]}, {meter.center[1]})`
                                - **仪表半径**: `{meter.radius}px`
                                """)

                                # 显示刻度点
                                if meter.scale_points:
                                    st.markdown("**识别到的刻度:**")
                                    for pt, val in meter.scale_points[:5]:
                                        st.write(f"- 值 {val}: 位置 {pt}")

                # 数据导出
                st.markdown("### 📥 数据导出")
                import pandas as pd

                data = []
                for m in meters:
                    data.append({
                        '仪表ID': m.meter_id + 1,
                        '当前读数': m.current_value,
                        '单位': m.unit,
                        '量程最小值': m.min_val,
                        '量程最大值': m.max_val,
                        '指针角度': m.pointer_angle,
                        '置信度': m.confidence,
                        '圆心X': m.center[0],
                        '圆心Y': m.center[1],
                        '半径': m.radius
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载检测数据 (CSV)",
                    data=csv,
                    file_name="meter_detection_result.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
