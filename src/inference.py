import os
import cv2
import torch
import numpy as np
import collections
import mediapipe as mp
import argparse

from models.driver_state_model import DriverStateCNNLSTM
from utils.face_geometry import get_geometric_features, crop_region, MODEL_POINTS

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def draw_pose_axis(image, pitch, yaw, roll, nose_point_2d, length=50.0):
    """
    Draw head direction vectors on the screen showing the driver's head pose.
    """
    # Convert pitch, yaw, roll from degrees to radians
    p = np.radians(pitch)
    y = np.radians(yaw)
    r = np.radians(roll)
    
    # Rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(p), -np.sin(p)],
                    [0, np.sin(p), np.cos(p)]])
    
    R_y = np.array([[np.cos(y), 0, np.sin(y)],
                    [0, 1, 0],
                    [-np.sin(y), 0, np.cos(y)]])
    
    R_z = np.array([[np.cos(r), -np.sin(r), 0],
                    [np.sin(r), np.cos(r), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    # 3D points representing axis direction (X-red, Y-green, Z-blue pointing out)
    axis_3d = np.array([
        [length, 0, 0],  # X
        [0, length, 0],  # Y
        [0, 0, length]   # Z (looking forward)
    ], dtype=np.float32)
    
    # Transform 3D axis points relative to the nose point
    axis_2d = np.zeros((3, 2), dtype=np.int32)
    for i in range(3):
        # Rotate and project
        pt_transformed = np.dot(R, axis_3d[i])
        axis_2d[i] = [
            int(nose_point_2d[0] + pt_transformed[0]),
            int(nose_point_2d[1] - pt_transformed[1]) # Y-axis inverted in image coordinates
        ]
        
    # Draw axes
    cv2.line(image, tuple(nose_point_2d.astype(int)), tuple(axis_2d[0]), (0, 0, 255), 2) # X (Red)
    cv2.line(image, tuple(nose_point_2d.astype(int)), tuple(axis_2d[1]), (0, 255, 0), 2) # Y (Green)
    cv2.line(image, tuple(nose_point_2d.astype(int)), tuple(axis_2d[2]), (255, 0, 0), 2) # Z (Blue - Nose direction)

def run_realtime_inference(model_path, source=0, seq_len=30, img_size=(64, 64), threshold=0.5):
    """
    Capture webcam stream or load a video, preprocess frames, maintain a sequence buffer,
    predict drowsiness using the CNN-LSTM model, and display real-time results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the model
    print(f"Loading CNN-LSTM model from {model_path}...")
    model = DriverStateCNNLSTM(cnn_feature_dim=64, geom_dim=10, lstm_hidden_dim=128, lstm_layers=2)
    
    # Load state dict (check if weights are wrapped in a full checkpoint)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    # 2. Setup rolling sequence buffers
    # We store the last `seq_len` crops and geometric vectors
    eye_l_buffer = collections.deque(maxlen=seq_len)
    eye_r_buffer = collections.deque(maxlen=seq_len)
    mouth_buffer = collections.deque(maxlen=seq_len)
    geom_buffer = collections.deque(maxlen=seq_len)
    
    # 3. Setup video capture
    # Source can be integer index (webcam) or filepath
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
        
    print("Inference loop running. Press 'q' to quit.")
    
    # Store temporary values for smoothing detection failures
    last_valid = None
    drowsiness_prob = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        img_h, img_w, _ = frame.shape
        # Create a display copy
        display_frame = frame.copy()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        ear_avg, mar, pitch, yaw, roll = 0.3, 0.15, 0.0, 0.0, 0.0
        nose_pt = None
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Extract features
            geom_vec, pose_landmarks_2d = get_geometric_features(landmarks, img_w, img_h)
            
            # Crops (resized and padded)
            el_crop = crop_region(rgb_frame, landmarks, [33, 160, 158, 133, 153, 144], pad_ratio=0.3, target_size=img_size)
            er_crop = crop_region(rgb_frame, landmarks, [362, 385, 387, 263, 373, 380], pad_ratio=0.3, target_size=img_size)
            m_crop = crop_region(rgb_frame, landmarks, [13, 14, 78, 308], pad_ratio=0.3, target_size=img_size)
            
            ear_avg = geom_vec[2]
            mar = geom_vec[3]
            pitch = geom_vec[4] * 90.0
            yaw = geom_vec[5] * 90.0
            roll = geom_vec[6] * 90.0
            
            # Nose point coordinate (landmark 1, index 0 in pose_landmarks_2d)
            nose_pt = pose_landmarks_2d[0]
            
            last_valid = (el_crop, er_crop, m_crop, geom_vec)
            
            # Draw facial landmarks on face for UI richness
            for pt in pose_landmarks_2d:
                cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
        else:
            # Revert to last valid frame details
            if last_valid is not None:
                el_crop, er_crop, m_crop, geom_vec = last_valid
                ear_avg = geom_vec[2]
                mar = geom_vec[3]
                pitch = geom_vec[4] * 90.0
                yaw = geom_vec[5] * 90.0
                roll = geom_vec[6] * 90.0
            else:
                # Fill defaults
                el_crop = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                er_crop = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                m_crop = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                geom_vec = [0.3, 0.3, 0.3, 0.15, 0.0, 0.0, 0.0, 0.45, 0.30, 0.25]
                
        # Push to buffer
        eye_l_buffer.append(el_crop)
        eye_r_buffer.append(er_crop)
        mouth_buffer.append(m_crop)
        geom_buffer.append(geom_vec)
        
        # If buffer is full, trigger CNN-LSTM model evaluation
        if len(geom_buffer) == seq_len:
            # Prep tensors
            # Transpose HWC -> CHW, and scale
            el_t = np.array(eye_l_buffer) / 255.0
            er_t = np.array(eye_r_buffer) / 255.0
            m_t = np.array(mouth_buffer) / 255.0
            
            # Transpose images to (seq_len, C, H, W)
            el_t = torch.from_numpy(el_t).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
            er_t = torch.from_numpy(er_t).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
            m_t = torch.from_numpy(m_t).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
            
            geom_t = torch.from_numpy(np.array(geom_buffer)).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                drowsiness_prob = model(el_t, er_t, m_t, geom_t).item()
                
        # 4. Visualization & HUD overlay
        # Draw head pose vector if nose point is available
        if nose_pt is not None:
            draw_pose_axis(display_frame, pitch, yaw, roll, nose_pt)
            
        # UI Box for stats
        cv2.rectangle(display_frame, (10, 10), (320, 190), (0, 0, 0), -1)
        # Add subtle transparent overlay
        alpha = 0.6
        cv2.addWeighted(display_frame, alpha, frame, 1 - alpha, 0, display_frame)
        
        # Status indicators
        cv2.putText(display_frame, f"EAR (Eyes): {ear_avg:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"MAR (Mouth): {mar:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Pitch (Nod): {pitch:.1f} deg", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Yaw (Turn): {yaw:.1f} deg", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Buffer: {len(geom_buffer)}/{seq_len}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Probability & Alert
        # Drowsiness bar
        bar_x1, bar_y1 = 340, 30
        bar_x2, bar_y2 = 620, 50
        cv2.rectangle(display_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
        fill_w = int((bar_x2 - bar_x1) * drowsiness_prob)
        bar_color = (0, 255, 0) if drowsiness_prob < 0.4 else ((0, 165, 255) if drowsiness_prob < 0.7 else (0, 0, 255))
        cv2.rectangle(display_frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), bar_color, -1)
        cv2.putText(display_frame, f"Drowsiness Risk: {drowsiness_prob*100:.1f}%", (340, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Big warning alert
        if drowsiness_prob >= threshold:
            # Blinking red alert box
            pulse = int((cv2.getTickCount() / cv2.getTickFrequency() * 3) % 2)
            alert_color = (0, 0, 255) if pulse == 0 else (0, 0, 150)
            cv2.rectangle(display_frame, (340, 70), (620, 180), alert_color, -1)
            cv2.putText(display_frame, "WARNING!", (420, 110), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(display_frame, "DROWSY DETECTED", (360, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.rectangle(display_frame, (340, 70), (620, 180), (0, 100, 0), -1)
            cv2.putText(display_frame, "ACTIVE DRIVING", (390, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Display the visual crops inside screen corners to make it look advanced
        if last_valid is not None:
            # Place left eye crop at bottom left
            h_crop, w_crop = el_crop.shape[:2]
            display_frame[img_h-h_crop-10:img_h-10, 10:w_crop+10] = cv2.cvtColor(el_crop, cv2.COLOR_RGB2BGR)
            cv2.putText(display_frame, "L Eye", (15, img_h-h_crop-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Place right eye crop next to left eye
            display_frame[img_h-h_crop-10:img_h-10, w_crop+20:2*w_crop+20] = cv2.cvtColor(er_crop, cv2.COLOR_RGB2BGR)
            cv2.putText(display_frame, "R Eye", (w_crop+25, img_h-h_crop-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Place mouth crop next to right eye
            display_frame[img_h-h_crop-10:img_h-10, 2*w_crop+30:3*w_crop+30] = cv2.cvtColor(m_crop, cv2.COLOR_RGB2BGR)
            cv2.putText(display_frame, "Mouth", (2*w_crop+35, img_h-h_crop-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        cv2.imshow("Driver State Monitor - CNN-LSTM Model", display_frame)
        
        # Stop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time driver drowsiness detection using CNN-LSTM model.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to trained PyTorch model (.pth)")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or path to input video file")
    parser.add_argument("--threshold", type=type(0.5), default=0.5, help="Drowsiness classification threshold")
    
    args = parser.parse_args()
    
    # Check if string source is integer (for webcam)
    video_source = int(args.source) if args.source.isdigit() else args.source
    
    # If the model does not exist, remind user to train it
    if not os.path.exists(args.model_path):
        print(f"Model file not found at {args.model_path}. You might need to run train.py first to train a model.")
        print("Using standard initial model state for demo purposes...")
        # Save a default initialized model for testing
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model = DriverStateCNNLSTM(cnn_feature_dim=64, geom_dim=10, lstm_hidden_dim=128, lstm_layers=2)
        torch.save(model.state_dict(), args.model_path)
        
    run_realtime_inference(args.model_path, source=video_source, threshold=args.threshold)
