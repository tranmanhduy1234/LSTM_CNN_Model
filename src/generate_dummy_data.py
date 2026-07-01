import os
import numpy as np
import cv2

def generate_dummy_dataset(output_path, num_samples=100, seq_len=30, img_size=(64, 64)):
    """
    Generate synthetic dataset for testing the CNN-LSTM drowsiness detection model.
    Saves an NPZ file containing left eye, right eye, mouth images and geometric features.
    
    Realistic patterns are simulated:
      - Class 0 (Awake): High EAR (eyes open), low MAR (mouth closed), stable head pose.
      - Class 1 (Drowsy): Low EAR (eyes closed/drooping), high MAR (yawning), head tilting/nodding.
    """
    print(f"Generating dummy dataset with {num_samples} samples, sequence length {seq_len}...")
    
    eye_left_all = []
    eye_right_all = []
    mouth_all = []
    geom_all = []
    labels_all = []
    
    for i in range(num_samples):
        # Determine class: 0 = Awake, 1 = Drowsy
        label = 1.0 if i >= num_samples // 2 else 0.0
        
        # Initialize sequence arrays
        # Shape: (seq_len, H, W, C)
        el_seq = np.zeros((seq_len, img_size[1], img_size[0], 3), dtype=np.uint8)
        er_seq = np.zeros((seq_len, img_size[1], img_size[0], 3), dtype=np.uint8)
        m_seq = np.zeros((seq_len, img_size[1], img_size[0], 3), dtype=np.uint8)
        geom_seq = np.zeros((seq_len, 10), dtype=np.float32)
        
        # Simulate temporal variation
        for t in range(seq_len):
            if label == 0.0:
                # AWAKE PATTERNS
                # Eyes open (EAR ~0.3) with occasional short blinks (lasts 3 frames)
                is_blinking = (t % 15 in [7, 8, 9])
                ear = 0.12 if is_blinking else (0.28 + 0.04 * np.sin(t / 2.0))
                
                # Mouth closed (MAR ~0.15)
                mar = 0.15 + 0.03 * np.random.randn()
                
                # Head stable (pitch, yaw, roll close to 0)
                pitch = 0.02 * np.sin(t / 5.0)
                yaw = 0.01 * np.cos(t / 5.0)
                roll = 0.01 * np.sin(t / 3.0)
                
                # Visual appearance simulator (brightness/texture)
                eye_val = 180 if not is_blinking else 60 # Darker if closed
                mouth_val = 100 # Normal closed mouth
            else:
                # DROWSY PATTERNS
                # Long eye closures (EAR remains low for many frames) or drooping eyes (EAR ~0.15)
                ear = 0.12 + 0.03 * np.sin(t / 4.0) # Extended closure / squinting
                
                # Yawning simulation (MAR spikes up to 0.7 for 10 frames in the middle)
                is_yawning = (t >= 10 and t <= 20)
                mar = 0.75 + 0.1 * np.sin((t - 10) * np.pi / 10) if is_yawning else (0.15 + 0.03 * np.random.randn())
                
                # Head nodding (pitch goes down - face bows down)
                pitch = 0.35 + 0.15 * np.sin(t / 4.0) # positive pitch = head bowing down
                yaw = 0.05 * np.cos(t / 4.0)
                roll = 0.15 * np.sin(t / 4.0)
                
                # Visual appearance simulator
                eye_val = 70 # Darker because eyes are closed/squinting
                mouth_val = 200 if is_yawning else 100 # Brighter/redder inside mouth if open
                
            # Create simulated visual crops (simple shapes/gradients representing eyes & mouth)
            # Left Eye
            el_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 80
            # Draw eye sclera/iris simulation based on eye openness
            cv2.circle(el_img, (32, 32), int(16 * (ear / 0.35)), (200, 200, 200), -1) # sclera
            cv2.circle(el_img, (32, 32), int(8 * (ear / 0.35)), (50, 30, 20), -1)     # pupil
            
            # Right Eye (similar to left eye)
            er_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 80
            cv2.circle(er_img, (32, 32), int(16 * (ear / 0.35)), (200, 200, 200), -1)
            cv2.circle(er_img, (32, 32), int(8 * (ear / 0.35)), (50, 30, 20), -1)
            
            # Mouth
            m_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 90
            # Draw mouth opening simulation
            # Yawns are wider and taller vertical ellipses
            cv2.ellipse(m_img, (32, 32), (24, int(20 * (mar / 0.8))), 0, 0, 360, (20, 10, 10), -1)
            if mar > 0.4:
                # Add red throat/tongue representation inside yawn
                cv2.ellipse(m_img, (32, 40), (12, int(8 * (mar / 0.8))), 0, 0, 360, (150, 40, 40), -1)
                
            # Fill sequence arrays
            el_seq[t] = el_img
            er_seq[t] = er_img
            m_seq[t] = m_img
            
            # Fill geometry vector
            # Indices:
            # 0: EAR_L, 1: EAR_R, 2: EAR_avg, 3: MAR, 4: Pitch, 5: Yaw, 6: Roll
            # 7: nose-to-mouth (shortens slightly when mouth open or head bows)
            # 8: mouth-to-chin (lengthens when mouth open)
            # 9: eyebrow-to-eye (narrower when frowning/squinting)
            norm_nose_to_mouth = 0.45 - 0.05 * (mar / 0.8)
            norm_mouth_to_chin = 0.30 + 0.15 * (mar / 0.8)
            avg_brow_eye = 0.25 if label == 0.0 else 0.18 # Frowning/drooping when tired
            
            geom_seq[t] = [
                ear, ear, ear, mar,
                pitch, yaw, roll,
                norm_nose_to_mouth, norm_mouth_to_chin, avg_brow_eye
            ]
            
        eye_left_all.append(el_seq)
        eye_right_all.append(er_seq)
        mouth_all.append(m_seq)
        geom_all.append(geom_seq)
        labels_all.append(label)
        
    # Convert lists to NumPy arrays
    eye_left_all = np.array(eye_left_all, dtype=np.uint8)
    eye_right_all = np.array(eye_right_all, dtype=np.uint8)
    mouth_all = np.array(mouth_all, dtype=np.uint8)
    geom_all = np.array(geom_all, dtype=np.float32)
    labels_all = np.array(labels_all, dtype=np.float32)
    
    # Save to NPZ file
    np.savez_compressed(
        output_path,
        eye_left=eye_left_all,
        eye_right=eye_right_all,
        mouth=mouth_all,
        geom=geom_all,
        labels=labels_all
    )
    print(f"Dummy dataset successfully saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_dummy_dataset("data/train_dummy.npz", num_samples=160)
    generate_dummy_dataset("data/val_dummy.npz", num_samples=40)
