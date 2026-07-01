import os
import cv2
import numpy as np
import mediapipe as mp
import argparse
from tqdm import tqdm

from utils.face_geometry import get_geometric_features, crop_region

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_video(video_path, seq_len=30, step_size=15, img_size=(64, 64)):
    """
    Extract facial features frame-by-frame from a single video file.
    Uses a sliding window to generate multiple sequences of length `seq_len`
    with stride `step_size`.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return [], [], [], []
        
    frames_eye_l = []
    frames_eye_r = []
    frames_mouth = []
    frames_geom = []
    
    # Store the last valid crop/geometry in case face detection fails on some frames
    last_valid_el = None
    last_valid_er = None
    last_valid_m = None
    last_valid_geom = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        img_h, img_w, _ = frame.shape
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. Compute geometry vector
            geom_vec, _ = get_geometric_features(landmarks, img_w, img_h)
            
            # 2. Crop left eye, right eye, and mouth
            # Left eye indices: [33, 160, 158, 133, 153, 144]
            el_crop = crop_region(rgb_frame, landmarks, [33, 160, 158, 133, 153, 144], pad_ratio=0.3, target_size=img_size)
            # Right eye indices: [362, 385, 387, 263, 373, 380]
            er_crop = crop_region(rgb_frame, landmarks, [362, 385, 387, 263, 373, 380], pad_ratio=0.3, target_size=img_size)
            # Mouth indices: inner lips & corners
            m_crop = crop_region(rgb_frame, landmarks, [13, 14, 78, 308], pad_ratio=0.3, target_size=img_size)
            
            # Save as last valid
            last_valid_el = el_crop
            last_valid_er = er_crop
            last_valid_m = m_crop
            last_valid_geom = geom_vec
            
        else:
            # If face detection fails, use temporal smoothing (repeat last valid features)
            if last_valid_geom is not None:
                el_crop = last_valid_el
                er_crop = last_valid_er
                m_crop = last_valid_m
                geom_vec = last_valid_geom
            else:
                # No face detected yet, skip or use black image with zero geometry
                el_crop = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                er_crop = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                m_crop = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                geom_vec = [0.3, 0.3, 0.3, 0.15, 0.0, 0.0, 0.0, 0.45, 0.30, 0.25] # standard default values
                
        frames_eye_l.append(el_crop)
        frames_eye_r.append(er_crop)
        frames_mouth.append(m_crop)
        frames_geom.append(geom_vec)
        
    cap.release()
    
    num_frames = len(frames_geom)
    if num_frames < seq_len:
        print(f"Warning: Video {video_path} has {num_frames} frames, shorter than sequence length {seq_len}. Skipping.")
        return [], [], [], []
        
    # Apply sliding window slicing
    video_eye_l = []
    video_eye_r = []
    video_mouth = []
    video_geom = []
    
    for start in range(0, num_frames - seq_len + 1, step_size):
        end = start + seq_len
        video_eye_l.append(np.array(frames_eye_l[start:end], dtype=np.uint8))
        video_eye_r.append(np.array(frames_eye_r[start:end], dtype=np.uint8))
        video_mouth.append(np.array(frames_mouth[start:end], dtype=np.uint8))
        video_geom.append(np.array(frames_geom[start:end], dtype=np.float32))
        
    return video_eye_l, video_eye_r, video_mouth, video_geom

def preprocess_dataset(data_dir, output_path, seq_len=30, step_size=15):
    """
    Look for raw folders 'awake' (label 0) and 'drowsy' (label 1) in data_dir,
    process all videos inside, and compile into a single NPZ file.
    """
    eye_left_dataset = []
    eye_right_dataset = []
    mouth_dataset = []
    geom_dataset = []
    labels_dataset = []
    
    classes = {'awake': 0.0, 'drowsy': 1.0}
    
    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist, skipping class {class_name}")
            continue
            
        print(f"Processing class: {class_name}...")
        video_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in tqdm(video_files):
            video_path = os.path.join(class_dir, video_file)
            
            # Process video and get sequences
            v_el, v_er, v_m, v_g = process_video(video_path, seq_len=seq_len, step_size=step_size)
            
            # Add to main dataset lists
            if len(v_g) > 0:
                eye_left_dataset.extend(v_el)
                eye_right_dataset.extend(v_er)
                mouth_dataset.extend(v_m)
                geom_dataset.extend(v_g)
                labels_dataset.extend([label] * len(v_g))
                
    if len(geom_dataset) == 0:
        print("No video data was successfully preprocessed. Check your data directories.")
        return
        
    # Convert and save
    eye_left_dataset = np.array(eye_left_dataset, dtype=np.uint8)
    eye_right_dataset = np.array(eye_right_dataset, dtype=np.uint8)
    mouth_dataset = np.array(mouth_dataset, dtype=np.uint8)
    geom_dataset = np.array(geom_dataset, dtype=np.float32)
    labels_dataset = np.array(labels_dataset, dtype=np.float32)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        eye_left=eye_left_dataset,
        eye_right=eye_right_dataset,
        mouth=mouth_dataset,
        geom=geom_dataset,
        labels=labels_dataset
    )
    print(f"\nPreprocessing finished! Saved dataset to {output_path}")
    print(f"Total samples: {len(labels_dataset)}")
    print(f"  - Awake samples: {np.sum(labels_dataset == 0.0)}")
    print(f"  - Drowsy samples: {np.sum(labels_dataset == 1.0)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess driver videos to extract eyes, mouth and geometric features.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw dataset folder containing 'awake/' and 'drowsy/' subfolders")
    parser.add_argument("--output_path", type=str, default="data/processed_dataset.npz", help="Path to save processed NPZ file")
    parser.add_argument("--seq_len", type=int, default=30, help="Temporal sequence length (frames per sample)")
    parser.add_argument("--step_size", type=int, default=15, help="Sliding window step size (stride) in frames")
    
    args = parser.parse_args()
    
    print("Preprocessing raw videos. Create structure: data/raw/awake/ and data/raw/drowsy/ to run on real videos.")
    preprocess_dataset(args.data_dir, args.output_path, seq_len=args.seq_len, step_size=args.step_size)
