import cv2
import numpy as np

# MediaPipe Face Mesh landmark indices
# Left eye indices
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# Right eye indices
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
# Mouth indices (inner vertical opening and corners)
MOUTH_VERTICAL = [13, 14]
MOUTH_HORIZONTAL = [78, 308]

# 3D model points for Head Pose Estimation (standard 3D face model points)
# Coordinate system: X points right, Y points down, Z points forward
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135.0),     # Left eye corner (index 33)
    (225.0, 170.0, -135.0),      # Right eye corner (index 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
], dtype=np.float32)

def calculate_ear(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR).
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    # Vertical distances
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_points):
    """
    Calculate the Mouth Aspect Ratio (MAR).
    MAR = |p_top - p_bottom| / |p_left - p_right|
    """
    # Vertical distance (inner lips)
    v = np.linalg.norm(mouth_points[0] - mouth_points[1])
    # Horizontal distance (corners of mouth)
    h = np.linalg.norm(mouth_points[2] - mouth_points[3])
    
    if h < 1e-6:
        return 0.0
    return v / h

def estimate_head_pose(landmarks_2d, img_w, img_h):
    """
    Estimate head pose (pitch, yaw, roll) in degrees using OpenCV solvePnP.
    landmarks_2d should be a list/array of 6 points corresponding to MODEL_POINTS:
    [Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner]
    """
    # Camera matrix estimation
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, 
        landmarks_2d.astype(np.float32), 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0
        
    # Get rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles (pitch, yaw, roll)
    # Using Tait-Bryan angles (Z-Y-X rotation sequence)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)[6]
    
    pitch = float(euler_angles[0].item())
    yaw = float(euler_angles[1].item())
    roll = float(euler_angles[2].item())
    
    # Adjust angles to represent standard rotation relative to camera
    # Pitch (look down/up): positive look down, negative look up
    # Yaw (look left/right): positive look right, negative look left
    # Roll (tilt left/right)
    return pitch, yaw, roll

def crop_region(image, landmarks, indices, pad_ratio=0.3, target_size=(64, 64)):
    """
    Crop a square region around specified landmarks with optional padding.
    """
    img_h, img_w, _ = image.shape
    
    # Extract points and scale to image size
    pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in indices])
    
    # Compute bounding box
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    
    w = max_x - min_x
    h = max_y - min_y
    
    # Make it square based on the maximum dimension
    side = max(w, h)
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    
    # Apply padding
    side = side * (1.0 + pad_ratio)
    
    # Calculate coordinates
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(img_w, cx + side / 2))
    y2 = int(min(img_h, cy + side / 2))
    
    if x2 <= x1 or y2 <= y1:
        # Fallback to empty patch
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
    crop = image[y1:y2, x1:x2]
    # Resize to target size
    crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    return crop_resized

def get_geometric_features(landmarks, img_w, img_h):
    """
    Compute eye status, mouth status, head angles, and relation between features.
    Returns:
        geometric_vector: A list of floats representing normalized geometric features.
        landmarks_for_pose: The 6 landmark coords used for pose estimation.
    """
    # Helper to convert mp landmark to numpy array
    def get_pt(idx):
        return np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h])
        
    # Get eye points for EAR
    left_eye_pts = np.array([get_pt(i) for i in LEFT_EYE_LANDMARKS])
    right_eye_pts = np.array([get_pt(i) for i in RIGHT_EYE_LANDMARKS])
    
    ear_l = calculate_ear(left_eye_pts)
    ear_r = calculate_ear(right_eye_pts)
    ear_avg = (ear_l + ear_r) / 2.0
    
    # Get mouth points for MAR
    # MOUTH_VERTICAL (13, 14) and MOUTH_HORIZONTAL (78, 308)
    mouth_pts = np.array([
        get_pt(13),
        get_pt(14),
        get_pt(78),
        get_pt(308)
    ])
    mar = calculate_mar(mouth_pts)
    
    # Get 6 landmarks for head pose
    # nose_tip (1), chin (152), left_eye_corner (33), right_eye_corner (263), left_mouth_corner (61), right_mouth_corner (291)
    pose_indices = [1, 152, 33, 263, 61, 291]
    landmarks_for_pose = np.array([get_pt(i) for i in pose_indices])
    
    pitch, yaw, roll = estimate_head_pose(landmarks_for_pose, img_w, img_h)
    
    # Spatial relationships / distances
    # Distance between left and right eyes
    eye_distance = np.linalg.norm(get_pt(33) - get_pt(263))
    # Distance from nose to chin (indicator of face size / scale)
    nose_to_chin = np.linalg.norm(get_pt(1) - get_pt(152))
    
    # Normalized eye-to-mouth distance (to capture nodding/head tilt changes relative to face size)
    nose_to_mouth = np.linalg.norm(get_pt(1) - (get_pt(61) + get_pt(291)) / 2.0)
    norm_nose_to_mouth = nose_to_mouth / max(1e-6, nose_to_chin)
    
    # Mouth to chin distance normalized
    mouth_to_chin = np.linalg.norm((get_pt(61) + get_pt(291)) / 2.0 - get_pt(152))
    norm_mouth_to_chin = mouth_to_chin / max(1e-6, nose_to_chin)

    # Eyebrow to eye distances (shows forehead wrinkling or squinting)
    # Left brow (70) to left eye (159)
    left_brow_eye = np.linalg.norm(get_pt(70) - get_pt(159)) / max(1e-6, nose_to_chin)
    # Right brow (300) to right eye (386)
    right_brow_eye = np.linalg.norm(get_pt(300) - get_pt(386)) / max(1e-6, nose_to_chin)
    avg_brow_eye = (left_brow_eye + right_brow_eye) / 2.0
    
    # Pack features into vector: 10 dimensions
    # 1. Left EAR
    # 2. Right EAR
    # 3. Average EAR
    # 4. MAR (Mouth opening)
    # 5. Pitch (Head vertical angle)
    # 6. Yaw (Head horizontal angle)
    # 7. Roll (Head tilt angle)
    # 8. Normalized nose-to-mouth distance
    # 9. Normalized mouth-to-chin distance
    # 10. Normalized average eyebrow-to-eye distance
    geometric_vector = [
        float(ear_l),
        float(ear_r),
        float(ear_avg),
        float(mar),
        float(pitch) / 90.0,   # Normalize to ~[-1.0, 1.0] range
        float(yaw) / 90.0,     # Normalize to ~[-1.0, 1.0] range
        float(roll) / 90.0,    # Normalize to ~[-1.0, 1.0] range
        float(norm_nose_to_mouth),
        float(norm_mouth_to_chin),
        float(avg_brow_eye)
    ]
    
    return geometric_vector, landmarks_for_pose
