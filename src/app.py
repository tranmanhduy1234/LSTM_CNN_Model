import streamlit as st
import cv2
import torch
import numpy as np
import collections
import os
import pandas as pd
import time
import tempfile
import matplotlib.pyplot as plt

from models.driver_state_model import DriverStateCNNLSTM
from utils.face_geometry import get_geometric_features, crop_region
import mediapipe as mp

# Configure page metadata and aesthetics
st.set_page_config(
    page_title="Driver Drowsiness Analytics - CNN-LSTM",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stAlert {
        border-radius: 10px;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    .metric-container {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Face Mesh
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = load_face_mesh()

# Initialize Model Loading
@st.cache_resource
def load_model(model_path, device):
    model = DriverStateCNNLSTM(cnn_feature_dim=64, geom_dim=10, lstm_hidden_dim=128, lstm_layers=2)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # Create a default dummy model file for testing
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    model.to(device)
    model.eval()
    return model

# App Header
st.title("🚗 Hệ Thống Nhận Diện Trạng Thái Tài Xế (CNN-LSTM)")
st.subheader("Phân tích không gian - thời gian kết hợp trạng thái Mắt, Miệng và Hướng đầu tài xế")

# Sidebar settings
st.sidebar.header("⚙️ Cấu hình Hệ thống")

device_opt = "cuda" if torch.cuda.is_available() else "cpu"
device = st.sidebar.selectbox("Thiết bị tính toán (Device)", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
model_path = st.sidebar.text_input("Đường dẫn model (.pth)", "checkpoints/best_model.pth")
threshold = st.sidebar.slider("Ngưỡng cảnh báo buồn ngủ (Threshold)", 0.1, 1.0, 0.5, 0.05)
seq_len = st.sidebar.number_input("Chiều dài chuỗi thời gian (Sequence Length)", min_value=10, max_value=100, value=30)
input_mode = st.sidebar.radio("Nguồn video đầu vào", ["Video tải lên (.mp4, .avi)", "Webcam Live Demo"])

# Load model
device_torch = torch.device(device)
try:
    model = load_model(model_path, device_torch)
    st.sidebar.success("✅ Đã tải mô hình thành công!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi tải mô hình: {e}")

# Layout columns
col1, col2 = st.columns([2, 1.2])

with col1:
    st.write("### 🎥 Camera giám sát trực tiếp")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

with col2:
    st.write("### 📊 Thông số thời gian thực")
    
    # Metric cards layout
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        ear_metric = st.empty()
        pitch_metric = st.empty()
    with m_col2:
        mar_metric = st.empty()
        drowsy_metric = st.empty()
        
    # Crop visualization column
    st.write("### 👁️ Vùng trích xuất đặc trưng (CNN Crops)")
    crops_placeholder = st.empty()
    
    st.write("### 📈 Biểu đồ lịch sử chỉ số")
    chart_placeholder = st.empty()

# Sequence buffer
eye_l_buffer = collections.deque(maxlen=seq_len)
eye_r_buffer = collections.deque(maxlen=seq_len)
mouth_buffer = collections.deque(maxlen=seq_len)
geom_buffer = collections.deque(maxlen=seq_len)

# History lists for graphing
history_ear = []
history_mar = []
history_pitch = []
history_prob = []
history_time = []

# Main running logic
run_processing = False
video_source = None

if input_mode == "Video tải lên (.mp4, .avi)":
    uploaded_file = st.file_uploader("Tải lên video driver (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_source = tfile.name
        run_processing = st.button("▶️ Chạy phân tích video")
else:
    run_processing = st.checkbox("🟢 Bắt đầu Web Camera Live")
    video_source = 0 # default webcam

if run_processing and video_source is not None:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Không thể mở nguồn video. Vui lòng kiểm tra lại.")
    else:
        st.toast("Đang khởi chạy camera/video...", icon="🚀")
        start_time = time.time()
        frame_idx = 0
        last_valid = None
        drowsiness_prob = 0.0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            img_h, img_w, _ = frame.shape
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face Mesh Processing
            results = face_mesh.process(display_frame)
            
            ear_avg, mar, pitch, yaw, roll = 0.3, 0.15, 0.0, 0.0, 0.0
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                geom_vec, pose_landmarks_2d = get_geometric_features(landmarks, img_w, img_h)
                
                # Crops
                el_crop = crop_region(display_frame, landmarks, [33, 160, 158, 133, 153, 144], pad_ratio=0.3, target_size=(64, 64))
                er_crop = crop_region(display_frame, landmarks, [362, 385, 387, 263, 373, 380], pad_ratio=0.3, target_size=(64, 64))
                m_crop = crop_region(display_frame, landmarks, [13, 14, 78, 308], pad_ratio=0.3, target_size=(64, 64))
                
                ear_avg = geom_vec[2]
                mar = geom_vec[3]
                pitch = geom_vec[4] * 90.0
                yaw = geom_vec[5] * 90.0
                roll = geom_vec[6] * 90.0
                
                last_valid = (el_crop, er_crop, m_crop, geom_vec)
                
                # Draw facial mesh landmarks to frame
                for pt in pose_landmarks_2d:
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
            else:
                if last_valid is not None:
                    el_crop, er_crop, m_crop, geom_vec = last_valid
                    ear_avg = geom_vec[2]
                    mar = geom_vec[3]
                    pitch = geom_vec[4] * 90.0
                    yaw = geom_vec[5] * 90.0
                    roll = geom_vec[6] * 90.0
                else:
                    el_crop = np.zeros((64, 64, 3), dtype=np.uint8)
                    er_crop = np.zeros((64, 64, 3), dtype=np.uint8)
                    m_crop = np.zeros((64, 64, 3), dtype=np.uint8)
                    geom_vec = [0.3, 0.3, 0.3, 0.15, 0.0, 0.0, 0.0, 0.45, 0.30, 0.25]
            
            # Save history
            current_elapsed = time.time() - start_time
            history_ear.append(ear_avg)
            history_mar.append(mar)
            history_pitch.append(pitch)
            history_prob.append(drowsiness_prob)
            history_time.append(current_elapsed)
            
            # Keep history lists trimmed
            if len(history_time) > 100:
                history_ear.pop(0)
                history_mar.pop(0)
                history_pitch.pop(0)
                history_prob.pop(0)
                history_time.pop(0)
                
            # Push into sequence buffer
            eye_l_buffer.append(el_crop)
            eye_r_buffer.append(er_crop)
            mouth_buffer.append(m_crop)
            geom_buffer.append(geom_vec)
            
            # Evaluate sequence model
            if len(geom_buffer) == seq_len:
                el_t = np.array(eye_l_buffer) / 255.0
                er_t = np.array(eye_r_buffer) / 255.0
                m_t = np.array(mouth_buffer) / 255.0
                
                el_t = torch.from_numpy(el_t).permute(0, 3, 1, 2).unsqueeze(0).float().to(device_torch)
                er_t = torch.from_numpy(er_t).permute(0, 3, 1, 2).unsqueeze(0).float().to(device_torch)
                m_t = torch.from_numpy(m_t).permute(0, 3, 1, 2).unsqueeze(0).float().to(device_torch)
                geom_t = torch.from_numpy(np.array(geom_buffer)).unsqueeze(0).float().to(device_torch)
                
                with torch.no_grad():
                    drowsiness_prob = model(el_t, er_t, m_t, geom_t).item()
                    
            # 1. Update frame view
            video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            
            # 2. Update alerts
            if drowsiness_prob >= threshold:
                alert_placeholder.error(f"🚨 **CẢNH BÁO BUỒN NGỦ NGUY HIỂM!** (Xác suất: {drowsiness_prob*100:.1f}%)", icon="🚨")
            else:
                alert_placeholder.success(f"🟢 **Tài xế tỉnh táo** (Xác suất buồn ngủ: {drowsiness_prob*100:.1f}%)", icon="🟢")
                
            # 3. Update Metric Cards
            ear_metric.metric("Chỉ số Mắt (EAR)", f"{ear_avg:.2f}", delta="Nhắm mắt" if ear_avg < 0.2 else "Bình thường")
            mar_metric.metric("Chỉ số Miệng (MAR)", f"{mar:.2f}", delta="Ngáp" if mar > 0.45 else "Bình thường", delta_color="inverse")
            pitch_metric.metric("Độ gật đầu (Pitch)", f"{pitch:.1f}°", delta="Cúi đầu" if pitch > 15 else "Bình thường", delta_color="inverse")
            drowsy_metric.metric("Nguy cơ buồn ngủ", f"{drowsiness_prob*100:.1f}%", delta="Nguy hiểm" if drowsiness_prob >= threshold else "An toàn", delta_color="inverse")
            
            # 4. Render extracted crops side-by-side
            # Combine left eye, right eye and mouth images horizontally
            combined_crops = np.hstack([el_crop, er_crop, m_crop])
            crops_placeholder.image(combined_crops, caption="Left Eye | Right Eye | Mouth Crops (Feed to CNN)", use_container_width=True)
            
            # 5. Render charts
            chart_data = pd.DataFrame({
                "Mắt (EAR)": history_ear,
                "Miệng (MAR)": history_mar,
                "Nguy cơ buồn ngủ": history_prob
            }, index=history_time)
            chart_placeholder.line_chart(chart_data)
            
            # Add a small delay for smoother playback
            if input_mode != "Webcam Live Demo":
                time.sleep(0.02) # approx 30 fps
                
        cap.release()
        st.toast("Hoàn tất phân tích video!", icon="✅")
