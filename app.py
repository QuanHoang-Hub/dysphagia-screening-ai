import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google import genai

# --- CẤU HÌNH GIAO DIỆN TRANG WEB ---
st.set_page_config(
    page_title="Hệ thống Tầm soát Chứng Nuốt Khó AI", 
    page_icon="🩺", 
    layout="wide"
)

st.title("🩺 Nền tảng Web AI Phân tích Tín hiệu Sinh lý Vùng Cổ")
st.markdown("""
**Đề tài Nghiên cứu Khoa học Kỹ thuật Cấp Thành Phố**  
Hệ thống kết hợp thiết bị đeo phần cứng **ESP32 NodeMCU** (cảm biến áp điện Piezo V2, lọc trung bình động) 
và mô hình trí tuệ nhân tạo **Gemini AI** nhằm tự động hóa quá trình đánh giá sinh lý nuốt pha hầu.
""")

# --- THANH BÊN (SIDEBAR): CẤU HÌNH VÀ TẢI DỮ LIỆU ---
st.sidebar.header("⚙️ Cấu hình Hệ thống")
api_key = st.sidebar.text_input("🔑 Nhập Google Gemini API Key:", type="password")

if api_key:
    try:
        client = genai.Client(api_key=api_key)
        st.sidebar.success("Đã kết nối thành công với Gemini API!")
    except Exception as e:
        st.sidebar.error(f"Lỗi khởi tạo Gemini Client: {e}")
else:
    st.sidebar.warning("⚠️ Vui lòng nhập khóa API Key từ Google AI Studio để kích hoạt trợ lý AI.")

st.sidebar.markdown("---")
st.sidebar.header("📂 Dữ liệu Thực nghiệm Phần cứng")
uploaded_file = st.sidebar.file_uploader(
    "Tải lên tệp CSV thu nhận từ ESP32 (Định dạng: Thoi_gian_ms,Tien_trinh_nuot)", 
    type=["csv"]
)

# --- XỬ LÝ DỮ LIỆU VÀ HIỂN THỊ GIAO DIỆN CHÍNH ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Kiểm tra tính hợp lệ của cột dữ liệu
        if 'Tien_trinh_nuot' in df.columns:
            x_col = 'Thoi_gian_ms' if 'Thoi_gian_ms' in df.columns else df.index
            
            # Hiển thị bảng dữ liệu thô
            with st.expander("🔍 Xem trước bảng dữ liệu thô từ cảm biến áp điện"):
                st.dataframe(df.head(15), use_container_width=True)
            
            # Vẽ biểu đồ tín hiệu trực quan bằng Plotly
            st.subheader("📊 Biểu đồ Dao động Cơ học Vùng Cổ theo Thời gian thực")
            fig = px.line(
                df, 
                x=x_col, 
                y='Tien_trinh_nuot', 
                labels={
                    'Thoi_gian_ms': 'Thời gian ghi nhận (ms)', 
                    'Tien_trinh_nuot': 'Biên độ tín hiệu ADC (0 - 4095)'
                },
                title="Đồ thị tín hiệu điện sinh học rung động cơ cổ khi nuốt"
            )
            fig.update_traces(line_color='#0066cc', line_width=2.5)
            fig.update_layout(
                plot_bgcolor='white', 
                paper_bgcolor='white',
                hovermode='x unified',
                font=dict(family="Times New Roman", size=14)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trích xuất các thông số đặc trưng tín hiệu
            max_amp = int(df['Tien_trinh_nuot'].max())
            mean_amp = float(df['Tien_trinh_nuot'].mean())
            std_amp = float(df['Tien_trinh_nuot'].std())
            threshold = mean_amp + 1.5 * std_amp
            
            above_thresh = df['Tien_trinh_nuot'] > threshold
            peak_events = int((above_thresh & ~above_thresh.shift(1, fill_value=False)).sum())
            
            if 'Thoi_gian_ms' in df.columns:
                duration_sec = (df['Thoi_gian_ms'].max() - df['Thoi_gian_ms'].min()) / 1000.0
            else:
                duration_sec = len(df) * 0.02
            
            # Khu vực hiển thị chỉ số thống kê
            st.subheader("📈 Các Chỉ số Đặc trưng Tín hiệu Sinh lý")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Biên độ đỉnh cực đại", f"{max_amp} / 4095")
            col2.metric("Biên độ trung bình nền", f"{mean_amp:.2f}")
            col3.metric("Số nhịp nuốt phát hiện", f"{peak_events} nhịp")
            col4.metric("Thời gian ghi nhận", f"{duration_sec:.1f} giây")
            
            st.markdown("---")
            
            # Khu vực phân tích bằng Gemini AI (Báo cáo tổng quan)
            st.subheader("🤖 Trợ lý Y sinh Thông minh (Đánh giá Lâm sàng Tự động)")
            
            if st.button("🚀 Thực hiện Phân tích Lâm sàng Tổng quan bằng Gemini AI", type="primary"):
                if not api_key:
                    st.error("Vui lòng nhập Google Gemini API Key ở thanh bên trái trước khi phân tích!")
                else:
                    with st.spinner("🤖 Gemini AI đang phân tích hình thái sóng và tổng hợp báo cáo lâm sàng..."):
                        prompt = f"""
                        Bạn là một chuyên gia y sinh và bác sĩ thần kinh học hàng đầu chuyên nghiên cứu về rối loạn nuốt (Dysphagia). 
                        Hãy lập một báo cáo phân tích lâm sàng khoa học dựa trên dữ liệu thực nghiệm thực tế thu nhận từ thiết bị đeo phần cứng:
                        
                        - Tổng thời gian khảo sát: {duration_sec:.1f} giây
                        - Biên độ đỉnh cực đại (Peak Amplitude): {max_amp} (trên thang đo ADC 12-bit từ 0 đến 4095)
                        - Biên độ trung bình nền (Baseline Mean): {mean_amp:.2f}
                        - Độ lệch chuẩn tín hiệu (Std Dev): {std_amp:.2f}
                        - Số nhịp nuốt (Swallow Events) nhận diện được: {peak_events} nhịp
                        
                        Nội dung báo cáo yêu cầu bám sát cấu trúc chuyên môn:
                        1. Đánh giá đặc trưng pha hầu (Pharyngeal Phase Analysis).
                        2. Nhận định nguy cơ rối loạn nuốt và bất thường biên độ sóng.
                        3. Đề xuất khuyến nghị lâm sàng hoặc hướng can thiệp phù hợp cho nhân viên y tế.
                        """
                        try:
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt,
                            )
                            st.success("Quá trình phân tích hoàn tất thành công!")
                            st.markdown("### 📋 Báo cáo Đánh giá Lâm sàng Chi tiết")
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Đã xảy ra lỗi khi giao tiếp với hệ thống Gemini API: {e}")

            # --- KHU VỰC TRÒ CHUYỆN TRỰC TIẾP VỚI TRỢ LÝ Y SINH ---
            st.markdown("---")
            st.subheader("💬 Trò chuyện trực tiếp cùng Trợ lý Ảo Y sinh")
            st.markdown("Hỏi đáp chuyên sâu dựa trên dữ liệu đo đạc thực tế của tệp CSV vừa tải lên.")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_prompt := st.chat_input("Nhập câu hỏi (ví dụ: 'Với mức biên độ đỉnh này thì bệnh nhân có nguy cơ hít sặc ở mức độ nào?')..."):
                if not api_key:
                    st.error("Vui lòng nhập API Key ở thanh bên trái trước khi sử dụng khung chat!")
                else:
                    st.session_state.messages.append({"role": "user", "content": user_prompt})
                    with st.chat_message("user"):
                        st.markdown(user_prompt)

                    system_context = f"""
                    Bạn là Trợ lý ảo y sinh và chuyên gia thần kinh học chuyên nghiên cứu về chứng nuốt khó.
                    Dữ liệu cảm biến phần cứng ESP32 hiện tại của bệnh nhân:
                    - Tổng thời gian: {duration_sec:.1f} giây
                    - Biên độ đỉnh cực đại: {max_amp} / 4095
                    - Biên độ trung bình nền: {mean_amp:.2f}
                    - Số nhịp nuốt phát hiện: {peak_events} nhịp
                    
                    Hãy trả lời câu hỏi của người dùng một cách khoa học, chính xác và bám sát các chỉ số sinh lý trên.
                    """

                    with st.chat_message("assistant"):
                        with st.spinner("Trợ lý ảo đang phân tích và phản hồi..."):
                            try:
                                full_prompt = f"{system_context}\n\nCâu hỏi từ người dùng: {user_prompt}"
                                response = client.models.generate_content(
                                    model='gemini-2.5-flash',
                                    contents=full_prompt,
                                )
                                assistant_reply = response.text
                                st.markdown(assistant_reply)
                                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                            except Exception as e:
                                st.error(f"Đã xảy ra lỗi kết nối với Gemini AI: {e}")
        else:
            st.error("❌ Tệp CSV tải lên không hợp lệ! Tệp bắt buộc phải chứa cột dữ liệu tên là 'Tien_trinh_nuot'.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi đọc tệp dữ liệu: {e}")
else:
    st.info("👈 Vui lòng tải lên tệp dữ liệu cảm biến định dạng `.csv` từ thanh bên trái để khởi động hệ thống phân tích.")