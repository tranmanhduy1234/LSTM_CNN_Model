# Hệ Thống Nhận Diện Trạng Thái Tài Xế Dựa Trên Kiến Trúc CNN-LSTM

Dự án này nghiên cứu và triển khai một giải pháp học sâu (Deep Learning) kết hợp mạng tích chập không gian và mạng hồi quy thời gian (**CNN-LSTM**) theo chuỗi thời gian để nhận diện trạng thái tài xế (tỉnh táo, buồn ngủ, gật đầu, ngáp) trong thời gian thực từ luồng camera. 

Hệ thống tận dụng giải pháp **MediaPipe Face Mesh** hiệu năng cao của Google để xác định các mốc khuôn mặt, kết hợp với các thuật toán hình học máy tính cổ điển và mạng học sâu tùy biến để phân tích biểu cảm và hành vi động lực học của tài xế.

---

## 📐 Kiến Trúc Mô Hình Kết Hợp CNN-LSTM

Mô hình được xây dựng theo kiến trúc phân tầng, phân tách nhiệm vụ trích xuất đặc trưng không gian (Spatial Features) và học đặc trưng động học thời gian (Temporal Features):

```mermaid
graph TD
    Input[Khung hình Video/Webcam] --> MP[MediaPipe Face Mesh]
    
    %% Nhánh trích xuất ảnh vùng
    MP --> Crop[Cắt vùng ảnh]
    Crop --> |Mắt trái| Crop_EL[Left Eye Crop: 64x64x3]
    Crop --> |Mắt phải| Crop_ER[Right Eye Crop: 64x64x3]
    Crop --> |Miệng| Crop_M[Mouth Crop: 64x64x3]
    
    Crop_EL --> SiameseCNN[Siamese Eye CNN<br/>Chia sẻ trọng số]
    Crop_ER --> SiameseCNN
    Crop_M --> MouthCNN[Mouth CNN<br/>Độc lập]
    
    SiameseCNN --> |Vectơ đặc trưng| Feat_EL[Left Eye: 64D]
    SiameseCNN --> |Vectơ đặc trưng| Feat_ER[Right Eye: 64D]
    MouthCNN --> |Vectơ đặc trưng| Feat_M[Mouth: 64D]
    
    %% Nhánh trích xuất hình học
    MP --> Geom[Tính toán hình học]
    Geom --> GeomVec[Vectơ đặc trưng hình học: 10D<br/>EAR, MAR, Pitch, Yaw, Roll, Tỷ lệ khoảng cách]
    
    %% Kết hợp và dự báo
    Feat_EL & Feat_ER & Feat_M & GeomVec --> Concat[Gộp đặc trưng: 202-dim]
    Concat --> Projection[Linear Projection: 128-dim]
    Projection --> |Chuỗi 30 khung hình| LSTM[2-Layer LSTM<br/>Hidden size: 128]
    LSTM --> Head[Phân lớp: Linear -> Dropout -> Sigmoid]
    Head --> Output[Xác suất buồn ngủ: [0.0 - 1.0]]
```

### 1. Trích xuất đặc trưng không gian (Spatial Features)
*   **Siamese CNN cho mắt**: Do cấu trúc hình học của mắt trái và mắt phải tương đồng nhau, hệ thống sử dụng mạng Siamese (chia sẻ chung bộ trọng số weights giữa 2 nhánh CNN) giúp giảm 50% số lượng tham số cần huấn luyện ở nhánh này, tăng tốc độ hội tụ và ngăn chặn overfitting.
*   **Mouth CNN độc lập**: Được thiết kế riêng để học các cấu trúc đặc thù của miệng khi đóng, mở rộng, nói chuyện hoặc ngáp dài.
*   **Vectơ hình học bổ trợ (10 chiều)**: Cung cấp trực tiếp các thông số định lượng giúp mạng LSTM có các gợi ý vật lý rõ ràng về trạng thái của khuôn mặt.

### 2. Học đặc trưng động lực học thời gian (Temporal Features)
*   **Chuỗi thời gian (Sliding Window)**: Mặc định chọn cửa sổ trượt dài $T = 30$ khung hình (~1 giây video ở tốc độ 30 FPS).
*   **Mạng LSTM (Long Short-Term Memory)**: Mạng LSTM 2 lớp với 128 đơn vị ẩn chịu trách nhiệm ghi nhớ mối quan hệ liên tiếp giữa các khung hình (ví dụ: chuỗi 5 khung hình nhắm mắt chỉ là chớp mắt tự nhiên, nhưng chuỗi 25 khung hình nhắm mắt liên tục là dấu hiệu buồn ngủ).

---

## 🧮 Cơ Sở Toán Học & Hình Học

Hệ thống tính toán các giá trị hình học trực tiếp từ các mốc tọa độ (landmarks) của MediaPipe Face Mesh làm đặc trưng bổ trợ:

### 1. Chỉ số mở mắt - Eye Aspect Ratio (EAR)
EAR đo lường độ mở của mắt dựa trên tỷ lệ giữa khoảng cách chiều dọc và chiều ngang của mắt.

```text
       p2     p3
     .    . .    .
  p1               p4
     .    . .    .
       p6     p5
```

Công thức tính cho mỗi mắt:
$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

*   **Ý nghĩa**: Khi mắt mở to, EAR dao động từ $0.28$ đến $0.35$. Khi mí mắt nhắm lại, EAR giảm xuống gần bằng $0.1$.
*   **Các mốc MediaPipe sử dụng**:
    *   **Mắt trái**: $p_1=33, p_2=160, p_3=158, p_4=133, p_5=153, p_6=144$
    *   **Mắt phải**: $p_1=362, p_2=385, p_3=387, p_4=263, p_5=373, p_6=380$

### 2. Chỉ số mở miệng - Mouth Aspect Ratio (MAR)
MAR phản ánh độ mở rộng của khoang miệng để phát hiện hành động ngáp:
$$MAR = \frac{||p_{13} - p_{14}||}{||p_{78} - p_{308}||}$$

*   **Ý nghĩa**: Điểm $13$ và $14$ nằm trên đường viền trong của môi trên và môi dưới. Điểm $78$ và $308$ là hai khóe miệng. Khi ngáp lớn, khoảng cách dọc $||p_{13} - p_{14}||$ tăng vọt khiến chỉ số MAR vượt ngưỡng $0.5$.

### 3. Ước lượng tư thế đầu - Head Pose (3D Perspective-n-Point)
Để phát hiện hành vi tài xế cúi đầu gật gù hoặc quay mặt đi hướng khác, hệ thống giải bài toán PnP (Perspective-n-Point) bằng cách chiếu các tọa độ 2D thực tế trên ảnh sang mô hình 3D chuẩn của khuôn mặt (Anthropometric 3D Face Model):

$$\mathbf{p}_{2D} = \mathbf{K} \left[ \mathbf{R} \mid \mathbf{t} \right] \mathbf{P}_{3D}$$

*   **Ma trận camera dự phóng $\mathbf{K}$**: Khởi tạo tự động dựa trên độ rộng ảnh ($f_x = f_y = W$, điểm trung tâm $c_x = W/2, c_y = H/2$).
*   **Giải thuật PnP**: Hàm `cv2.solvePnP` tìm ra ma trận xoay $\mathbf{R}$ (Rotation Vector) và ma trận tịnh tiến $\mathbf{t}$ (Translation Vector).
*   **Phân tích góc Euler**: Áp dụng hàm phân rã ma trận chiếu `cv2.decomposeProjectionMatrix` để trích xuất 3 góc Euler của đầu:
    *   **Pitch (Độ gật/cúi đầu)**: Dương khi cúi xuống, âm khi ngước lên.
    *   **Yaw (Độ quay đầu)**: Dương khi quay sang phải, âm khi quay sang trái.
    *   **Roll (Độ nghiêng đầu)**: Nghiêng sang trái hoặc phải.

---

## 📂 Danh Mục Các Tệp Nguồn

Dưới đây là liên kết và chi tiết chức năng của các tệp tin trong không gian làm việc:

*   [requirements.txt](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/requirements.txt): Liệt kê các gói thư viện cần cài đặt với các phiên bản tương thích (ghim cứng `mediapipe==0.10.14` để đảm bảo có module `solutions` và tương thích tốt với NumPy 2.0+).
*   `src/`: Thư mục chứa toàn bộ mã nguồn.
    *   [src/utils/face_geometry.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/utils/face_geometry.py): Thư viện tính toán EAR, MAR, Head Pose Euler angles và hàm cắt (crop) các vùng mắt/miệng để chuẩn hóa về kích thước 64x64.
    *   [src/models/driver_state_model.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py): Thiết kế kiến trúc mạng neural PyTorch gồm hai mạng tích chập [FeatureExtractorCNN](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py#L4) và lớp mạng tích hợp chuỗi [DriverStateCNNLSTM](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py#L44) kết thúc bằng hàm kích hoạt Sigmoid.
    *   [src/dataset.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/dataset.py): Triển khai lớp nạp dữ liệu [DriverStateDataset](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/dataset.py#L5) kế thừa từ `torch.utils.data.Dataset`, tự động chuyển đổi định dạng ảnh `uint8` sang Tensor `float32` và thực hiện chuẩn hóa pixel về đoạn $[0.0, 1.0]$.
    *   [src/generate_dummy_data.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/generate_dummy_data.py): Kịch bản mô phỏng tín hiệu chuỗi thời gian sinh động (mắt nhắm, mở, nhấp nháy, miệng mở ngáp, đầu gật) để tạo tập huấn luyện giả lập.
    *   [src/train.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/train.py): Quy trình huấn luyện mô hình PyTorch, tự động lưu mô hình tốt nhất dựa trên Val Loss vào thư mục `checkpoints/best_model.pth` và vẽ đồ thị học tập lưu vào `reports/`.
    *   [src/preprocess.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/preprocess.py): Tiền xử lý các tập tin video thô thực tế để chuyển đổi hàng loạt thành tệp dữ liệu huấn luyện nén `.npz`.
    *   [src/inference.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/inference.py): Kịch bản nhận diện thời gian thực qua webcam, vẽ trực quan ma trận trục tọa độ hướng đầu 3D (3D Head Pose Axis) và bảng điều khiển HUD sinh động.
    *   [src/app.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/app.py): Ứng dụng dashboard web được xây dựng bằng Streamlit hiển thị trực tiếp video, các vùng ảnh cắt (crops), thẻ thông số thời gian thực và biểu đồ động.

---

## 🛠️ Hướng Dẫn Cài Đặt Chi Tiết

### 1. Chuẩn bị môi trường
Khuyên dùng môi trường ảo Python 3.12 (ví dụ: Conda):
```bash
# Tạo môi trường ảo với Python 3.12
conda create -n tttn python=3.12 -y
# Kích hoạt môi trường ảo
conda activate tttn
```

### 2. Cài đặt các thư viện cần thiết
Cài đặt tất cả các gói phụ thuộc được định nghĩa trong [requirements.txt](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/requirements.txt):
```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Chạy Pipeline Dự Án

Dự án cung cấp một pipeline huấn luyện khép kín, cho phép chạy thử nghiệm ngay cả khi chưa có dữ liệu video thật thông qua dữ liệu giả lập.

### Bước 1: Tạo dữ liệu giả lập để thử nghiệm hệ thống
Để kiểm tra xem mô hình và quy trình có hoạt động chính xác không, hãy chạy tệp mô phỏng dữ liệu:
```bash
python src/generate_dummy_data.py
```
*Kết quả:* Tạo ra hai tệp [data/train_dummy.npz](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/data/train_dummy.npz) và [data/val_dummy.npz](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/data/val_dummy.npz).

### Bước 2: Huấn luyện mô hình từ đầu
Huấn luyện mô hình CNN-LSTM trên tập dữ liệu đã chuẩn bị:
```bash
python src/train.py
```
*Kết quả:*
*   Mô hình tốt nhất được lưu tại [checkpoints/best_model.pth](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/checkpoints/best_model.pth).
*   Đồ thị theo dõi Loss/Accuracy được lưu tại `reports/training_curves.png`.

### Bước 3: Chạy chương trình nhận diện thời gian thực (Webcam HUD)
Sau khi đã có tệp mô hình đã huấn luyện, chạy nhận diện trực tiếp qua camera của bạn:
```bash
python src/inference.py --model_path checkpoints/best_model.pth --source 0
```
*(Ấn phím **'q'** khi đang chọn cửa sổ hiển thị OpenCV để tắt chương trình).*

### Bước 4: Khởi động Dashboard phân tích thông minh (Streamlit Web App)
Khởi chạy giao diện web để tải lên video hoặc xem đồ thị động:
```bash
streamlit run src/app.py
```
Sau đó mở trình duyệt web truy cập địa chỉ được in trên màn hình Terminal (thường là `http://localhost:8501`).

---

## 📹 Xử Lý Dữ Liệu Video Tự Quay Hoặc Bộ Dữ Liệu Lớn

Để huấn luyện mô hình nhận diện hành vi buồn ngủ thực tế từ các bộ dữ liệu video (như YawDD, NTHU Drowsy, v.v.):

1.  **Tổ chức thư mục dữ liệu**:
    ```text
    data/raw_dataset/
    ├── awake/
    │   ├── driver_normal_1.mp4
    │   └── driver_normal_2.avi
    └── drowsy/
        ├── driver_yawn_1.mp4
        └── driver_sleepy_2.mp4
    ```

2.  **Tiền xử lý chuyển đổi video sang dạng mảng NumPy**:
    Chạy tiền xử lý để cắt ảnh các vùng quan tâm, tính toán chỉ số hình học và gộp thành các chuỗi 30 khung hình:
    ```bash
    python src/preprocess.py --data_dir data/raw_dataset --output_path data/real_processed.npz --seq_len 30 --step_size 15
    ```

3.  **Huấn luyện lại mô hình**:
    Chỉnh sửa đường dẫn tệp dữ liệu huấn luyện trong [src/train.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/train.py) thành:
    ```python
    train_data = "data/real_processed.npz"
    # Sau đó chạy huấn luyện: python src/train.py
    ```
