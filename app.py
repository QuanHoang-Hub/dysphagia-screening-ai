import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google import genai

st.set_page_config(page_title="AI Dysphagia Screening & Analysis Platform", layout="wide")

st.title("Ứng dụng Web AI Phân tích Tín hiệu Tầm soát Chứng Nuốt Khó")
st.markdown("Hệ thống thông minh kết hợp thiết bị đeo ESP32 NodeMCU và mô hình trí tuệ nhân tạo Gemini AI để phân tích sinh lý nuốt.")

# Khởi tạo Gemini AI Client
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

if api_key:
    client = genai.Client(api_key=api_key)
else:
    st.warning("Vui lòng cung cấp Gemini API Key để kích hoạt tính năng phân tích chuyên sâu bằng AI.")

uploaded_file = st.file_uploader("Tải lên tệp dữ liệu cảm biến CSV từ thiết bị ESP32", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Kiểm tra cột thời gian để vẽ biểu đồ chuẩn xác
    x_col = 'Thoi_gian_ms' if 'Thoi_gian_ms' in df.columns else df.index
    
    st.subheader("Trực quan hóa Tín hiệu Dao động Cơ học Vùng Cổ")
    fig = px.line(
        df, 
        x=x_col, 
        y='Tien_trinh_nuot', 
        labels={'Thoi_gian_ms': 'Thời gian (ms)', 'Tien_trinh_nuot': 'Biên độ ADC (0 - 4095)'},
        title="Đồ thị thời gian thực biên độ dao động pha hầu"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Trích xuất đặc trưng thống kê sinh lý
    max_amp = int(df['Tien_trinh_nuot'].max())
    mean_amp = float(df['Tien_trinh_nuot'].mean())
    std_amp = float(df['Tien_trinh_nuot'].std())
    threshold = mean_amp + 1.5 * std_amp
    
    # Đếm số cụm sự kiện vượt ngưỡng (chống đếm trùng mẫu liên tiếp)
    above_thresh = df['Tien_trinh_nuot'] > threshold
    peak_events = int((above_thresh & ~above_thresh.shift(1, fill_value=False)).sum())
    duration_sec = (df['Thoi_gian_ms'].max() - df['Thoi_gian_ms'].min()) / 1000.0 if 'Thoi_gian_ms' in df.columns else len(df) * 0.02
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Biên độ đỉnh cực đại", f"{max_amp}")
    col2.metric("Biên độ trung bình nền", f"{mean_amp:.2f}")
    col3.metric("Số nhịp nuốt phát hiện", f"{peak_events}")
    col4.metric("Thời gian ghi nhận (giây)", f"{duration_sec:.1f} s")
    
    if st.button("Phân tích Chuyên sâu bằng Gemini AI") and api_key:
        with st.spinner("AI đang phân tích dữ liệu sinh lý và đánh giá nguy cơ nuốt khó..."):
            prompt = f"""
            Bạn là một chuyên gia y sinh và bác sĩ thần kinh học hàng đầu. Hãy phân tích dữ liệu đo đạc từ thiết bị đeo tầm soát chứng nuốt khó (Dysphagia) sử dụng cảm biến áp điện và vi điều khiển ESP32 NodeMCU (tần số lấy mẫu 50Hz, bộ lọc trung bình động):
            - Thời gian ghi nhận tổng cộng: {duration_sec:.1f} giây
            - Biên độ đỉnh cực đại (Peak Amplitude): {max_amp} / 4095
            - Biên độ trung bình nền (Baseline): {mean_amp:.2f}
            - Độ lệch chuẩn tín hiệu: {std_amp:.2f}
            - Số nhịp nuốt (Swallow events) phát hiện được: {peak_events}
            
            Hãy đưa ra báo cáo phân tích lâm sàng bao gồm:
            1. Đánh giá đặc trưng pha hầu (Pharyngeal phase) dựa trên biên độ và tần suất nhịp nuốt.
            2. Nhận định nguy cơ rối loạn nuốt (Dysphagia Risk Assessment).
            3. Khuyến nghị hành động tiếp theo cho bệnh nhân hoặc bác sĩ lâm sàng.
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            
            st.subheader("Báo cáo Phân tích Lâm sàng từ Gemini AI")
            st.markdown(response.text)
else:
    st.info("Vui lòng tải lên tệp dữ liệu CSV thu nhận từ phần cứng ESP32 NodeMCU để bắt đầu.")
