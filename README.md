# HƯỚNG DẪN CHI TIẾT: CƠ CHẾ HOẠT ĐỘNG CỦA HỆ THỐNG PHÁT HIỆN BUỒN NGỦ (CNN-LSTM-ATTENTION)

Tài liệu này giải thích chi tiết, từng bước một về cơ chế hoạt động của toàn bộ hệ thống từ lúc Camera thu hình cho đến khi còi hú cảnh báo, đồng thời làm rõ cơ sở khoa học tại sao mô hình Deep Learning của bạn là trái tim không thể thay thế của dự án.

---

## 🧭 BẢN ĐỒ LUỒNG DỮ LIỆU (DATA FLOW)

Khi bạn bật camera giám sát, dữ liệu sẽ chạy qua một chuỗi xử lý khép kín gồm 4 bước lớn:

```text
 [Bước 1: Thu hình & Trích xuất] -> [Bước 2: Phân tích Không gian (CNN)] -> [Bước 3: Ghi nhớ Thời gian (LSTM & Attention)] -> [Bước 4: Cảnh báo (Alerts)]
```

Dưới đây là chi tiết kỹ thuật của từng bước:

---

## BƯỚC 1: THU HÌNH & TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC (MediaPipe & OpenCV)

Mỗi giây, Camera sẽ chụp khoảng 30 khung hình (frames). Với mỗi khung hình độc lập, hệ thống thực hiện trích xuất thông tin:

### 1. Xác định tọa độ khuôn mặt (MediaPipe Face Mesh)
Hệ thống sử dụng mô hình AI của Google để quét khuôn mặt và trả về **468 mốc tọa độ 3D** (X, Y, Z). Các mốc này được ghim trực tiếp lên mí mắt, lông mày, mũi, môi và cằm.

### 2. Cắt ảnh các vùng quan trọng (Facial Crops)
Dựa vào các tọa độ mốc, hàm `crop_region` trong tệp [src/utils/face_geometry.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/utils/face_geometry.py#L93) sẽ tự động cắt ra 3 vùng ảnh nhỏ từ khuôn mặt, sau đó co giãn (resize) về kích thước chuẩn **64x64 pixel**:
*   **Vùng mắt trái** (chứa con ngươi và mí mắt trái).
*   **Vùng mắt phải** (chứa con ngươi và mí mắt phải).
*   **Vùng miệng** (chứa toàn bộ đôi môi).

### 3. Tính toán các chỉ số toán học hình học (Geometric Vector)
Hệ thống sử dụng toán học Euclid để tính ra một vectơ đặc trưng hình học gồm 10 chỉ số từ các mốc tọa độ:
*   **Chỉ số mở mắt - EAR (Eye Aspect Ratio):** Tính bằng cách lấy khoảng cách dọc mí mắt chia cho khoảng cách ngang của mắt.
    *   *Mắt mở:* EAR $\approx$ 0.3
    *   *Mắt nhắm:* EAR $\approx$ 0.1
*   **Chỉ số mở miệng - MAR (Mouth Aspect Ratio):** Tính bằng tỷ lệ khoảng cách dọc của lòng môi trong chia cho bề ngang khóe miệng.
    *   *Miệng đóng:* MAR $\approx$ 0.1
    *   *Miệng ngáp to:* MAR $\ge$ 0.5
*   **Tư thế đầu - Head Pose (Pitch, Yaw, Roll):** Thuật toán hình học **solvePnP** so khớp các điểm mốc 2D trên ảnh với mô hình đầu 3D chuẩn để tính ra góc xoay đầu:
    *   *Pitch:* Độ gật đầu (cúi xuống/ngửa lên).
    *   *Yaw:* Độ quay đầu (quay trái/quay phải).
    *   *Roll:* Độ nghiêng đầu (nghiêng sang vai trái/phải).
*   **Các tỷ lệ khoảng cách phụ:** Tỷ lệ khoảng cách từ mũi đến miệng, miệng đến cằm, lông mày đến mắt để bổ trợ thông tin khi tài xế méo miệng hoặc nhăn trán gục đầu.

---

## BƯỚC 2: PHÂN TÍCH KHÔNG GIAN (Mạng Tích Chập CNN)

Ba vùng ảnh cắt và vectơ hình học ở Bước 1 sẽ được nạp vào phần đầu của mô hình Deep Learning trong tệp [src/models/driver_state_model.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py):

### 1. Nhánh Siamese CNN cho mắt
Ảnh mắt trái và mắt phải (kích thước 64x64x3) được đưa qua chung một mạng CNN ([FeatureExtractorCNN](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py#L4)). Mạng này gồm các tầng Convolution (tích chập) liên tiếp để trích xuất các đường nét mí mắt, đồng tử và chuyển đổi mỗi bức ảnh mắt thành một vectơ đặc trưng nén **64 chiều (64D)** biểu thị trạng thái không gian của mắt.

### 2. Nhánh CNN cho miệng
Ảnh miệng được đưa qua một mạng CNN độc lập khác để trích xuất các nếp nhăn môi, bóng tối trong khoang miệng khi mở lớn và chuyển đổi thành một vectơ nén **64 chiều (64D)**.

### 3. Nhánh 1D CNN cho chuỗi hình học
Vectơ hình học 10 chiều (EAR, MAR, góc đầu) qua các khung hình liên tiếp được đưa qua một mạng tích chập 1 chiều ([GeometricFeatureExtractor1DCNN](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py#L45)) để làm mịn và trích chọn các biến động cục bộ của chỉ số, chuyển đổi thành vectơ đặc trưng **64 chiều (64D)**.

### 4. Gộp đặc trưng (Fusion)
Tại mỗi khung hình (tại thời điểm $t$), ta gộp tất cả đặc trưng không gian lại:
$$\text{Đặc trưng gộp} = [\text{Mắt trái (64D)} + \text{Mắt phải (64D)} + \text{Miệng (64D)} + \text{Hình học (64D)}] = 256 \text{ chiều (256D)}$$
Sau đó, một tầng Tuyến tính (Linear) sẽ nén vectơ này từ 256D xuống **128 chiều (128D)**.

---

## BƯỚC 3: GHI NHỚ THỜI GIAN (Mạng LSTM & Cơ Chế Attention)

Vì buồn ngủ là một quá trình diễn ra theo thời gian (tài xế nhắm mắt lâu hoặc cúi đầu gục xuống vài giây), nên hệ thống không thể chỉ nhìn vào 1 khung hình đơn lẻ để kết luận. 

Hệ thống sẽ gom **30 khung hình liên tiếp** (tương đương khoảng 1 giây lái xe thực tế) thành một chuỗi dữ liệu đầu vào kích thước `(30, 128)` và xử lý như sau:

### 1. Mạng bộ nhớ dài-ngắn hạn (LSTM)
Chuỗi 30 khung hình được nạp vào mạng LSTM 2 lớp ([self.lstm](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py#L141)). Mạng LSTM sẽ quét qua từng khung hình từ $t_1$ đến $t_{30}$ và học cách ghi nhớ các thay đổi:
*   Nếu tài xế chớp mắt nhanh (chỉ nhắm mắt ở khung hình $t_{14}, t_{15}$ rồi mở ra ngay ở $t_{16}$), LSTM sẽ nhận biết đây là chớp mắt bình thường.
*   Nếu tài xế nhắm mắt liên tục (từ khung hình $t_5$ đến $t_{30}$ vẫn chưa mở), LSTM sẽ tích lũy trạng thái mệt mỏi tăng dần.

### 2. Cơ chế chú ý (Temporal Attention)
*   *Vấn đề của LSTM truyền thống:* Nó thường chỉ lấy thông tin ở khung hình cuối cùng ($t_{30}$), dẫn đến việc dễ bỏ sót những khoảnh khắc tài xế gục đầu nhanh ở giữa chuỗi ($t_{15}$) rồi ngẩng lên ngay.
*   *Giải pháp Attention:* Lớp [TemporalAttention](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py#L75) sẽ tính toán mức độ quan trọng (trọng số $\alpha$) cho từng khung hình trong số 30 khung hình. Khung hình nào có hiện tượng nhắm mắt sâu hoặc gục đầu sẽ được gán trọng số $\alpha$ rất cao.
*   Sau đó, hệ thống cộng chập có trọng số để tạo ra một **Vectơ Ngữ Cảnh (Context Vector)** đại diện tối ưu cho toàn bộ chuỗi.

### 3. Phân lớp dự đoán (Classifier)
Vectơ ngữ cảnh này được đưa qua các tầng tuyến tính kết nối hoàn toàn (Dense layers) và kết thúc bằng hàm kích hoạt **Sigmoid** để đưa ra kết quả cuối cùng là **Xác suất buồn ngủ (Drowsiness Probability)** nằm trong khoảng từ `0.0` (tỉnh táo hoàn toàn) đến `1.0` (ngủ gật hoàn toàn).

---

## BƯỚC 4: HỆ THỐNG CẢNH BÁO ĐA PHƯƠNG THỨC (Alerts)

Xác suất buồn ngủ đầu ra của mô hình (ví dụ: `prob = 0.85`) sẽ được đưa vào bộ xử lý cảnh báo trong tệp [src/utils/alerts.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/utils/alerts.py#L32):

1.  **Cảnh báo trên màn hình HUD:** Cửa sổ OpenCV / Streamlit hiển thị dòng chữ cảnh báo màu đỏ chói nhấp nháy: `🚨 WARNING! DROWSY DETECTED 🚨` và thanh trạng thái chuyển sang màu đỏ cảnh báo.
2.  **Cảnh báo Âm thanh:** Thư viện `sounddevice` tự động tạo ra sóng hình sin tần số cao (1200Hz - tiếng còi rít chói tai) và phát trực tiếp ra loa máy tính của bạn theo các nhịp dồn dập nhằm đánh thức tài xế ngay lập tức.
3.  **Mô phỏng tín hiệu phần cứng:** In ra màn hình console dòng lệnh kích hoạt thiết bị ngoại vi: `[HARDWARE TRIGGER] LED = FLASHING RED, BUZZER = ON`. (Tín hiệu này có thể kết nối trực tiếp với cổng COM/Serial hoặc chân GPIO của vi điều khiển để bật còi hú/đèn LED vật lý trên xe ô tô).

---

## 🧠 VAI TRÒ CỐT LÕI CỦA MÔ HÌNH CNN-LSTM TỰ THIẾT KẾ (TẠI SAO KHÔNG DÙNG THUẬT TOÁN RẼ NHÁNH IF-ELSE?)

Một câu hỏi quan trọng trong nghiên cứu khoa học: *Nếu ta đã tính được EAR, MAR và góc đầu bằng hình học, tại sao không dùng các câu lệnh rẽ nhánh điều kiện `if-else` đơn giản để báo động (ví dụ: `if EAR < 0.2: báo_động()`)?*

Thực tế, nếu chỉ sử dụng các ngưỡng cố định thông thường, hệ thống sẽ **thất bại hoàn toàn và không thể đưa vào cabin ô tô thực tế** vì các lý do sau:

### 1. Sự bất khả thi của ngưỡng EAR cố định do sinh trắc học cá nhân
*   **Cấu trúc mắt khác nhau:** Người mắt một mí hoặc nhỏ tự nhiên có chỉ số EAR lúc thức chỉ khoảng `0.18 - 0.22`, trong khi người mắt to có EAR lên tới `0.35`.
*   **Hậu quả:** Nếu đặt một ngưỡng cứng `if EAR < 0.22`, hệ thống sẽ **hú còi báo động liên tục đối với người mắt nhỏ** dù họ đang thức, trong khi lại **bỏ sót hoàn toàn người mắt to** khi họ đã lờ đờ nhắm hờ mắt buồn ngủ.
*   **Giải pháp của CNN-LSTM:** Mạng neural học cách tự thích ứng với biên độ dao động EAR tương đối của từng tài xế cụ thể qua chuỗi thời gian, tự động định hình thế nào là mắt "nhắm" so với trạng thái bình thường của chính người đó.

### 2. Sự khác biệt về thời gian (Temporal Dynamics) - Sự cần thiết của LSTM
*   **Chớp mắt vs Micro-sleep (Ngủ gật ngắn):**
    *   Tài xế tỉnh táo chớp mắt rất nhiều, thời gian nhắm mắt rất nhanh (chỉ từ `0.1 - 0.3 giây`, tương đương 3 - 9 khung hình).
    *   Tài xế ngủ gật có hành vi nhắm mắt kéo dài (từ `1.0 - 3.0 giây` trở lên, tương đương 30 - 90 khung hình).
    *   Cả hai hành vi này đều đưa giá trị EAR về sát `0.1`. Thuật toán `if-else` tĩnh không thể phân biệt được thời lượng nhắm mắt này nếu có nhiễu bắt điểm, dẫn đến còi báo động kêu vô tội vạ mỗi lần tài xế chớp mắt. Mạng **LSTM** ghi nhớ toàn bộ chuỗi và chỉ phát tín hiệu khi trạng thái nhắm mắt tích lũy qua nhiều khung hình liên tiếp.
*   **Nói chuyện/Cười vs Ngáp dài:**
    *   Tài xế nói chuyện làm môi chuyển động mở ra liên tục (MAR tăng giảm nhanh theo nhịp).
    *   Tài xế ngáp sẽ mở to miệng và giữ nguyên khẩu hình ngáp trong `3 - 5 giây`.
    *   Mạng **LSTM** phân tích nhịp điệu chuyển động để lọc bỏ hoàn toàn các trường hợp nói chuyện hoặc cười, tránh báo động sai.

### 3. Sự giới hạn của chỉ số hình học EAR/MAR - Sức mạnh của Mạng Tích Chập 2D CNN
Chỉ số EAR và MAR chỉ là các chỉ số khoảng cách thô được lập trình thủ công (hand-crafted features), bỏ qua lượng thông tin khổng lồ từ các pixel ảnh thực tế:
*   **Độ trĩu nặng của mí mắt (Heavy Eyelids):** Khi bắt đầu buồn ngủ, mí mắt tài xế trĩu nặng sụp xuống từ từ, cơ mặt giãn ra, và hướng nhìn con ngươi bắt đầu đờ đẫn hướng xuống dưới. Chỉ số hình học EAR chỉ đo khoảng cách mí mắt nên không thể phân biệt được mắt đang đờ đẫn hay chỉ đang nhìn xuống mặt đường. Nhánh **Eye CNN** sẽ quét toàn bộ ma trận pixel của vùng mắt để nhận biết cấu trúc da mí mắt sụp và vị trí tròng mắt.
*   **Ngáp che tay (Facial Obstruction):** Khi tài xế ngáp và đưa tay lên che miệng theo lịch sự, MediaPipe Face Mesh sẽ không thể định vị chính xác môi và tính sai MAR. Tuy nhiên, nhánh **Mouth CNN** khi được huấn luyện trên dữ liệu thực tế sẽ học được đặc trưng ảnh của một bàn tay chắn trước miệng kết hợp với cơ mặt xung quanh chuyển động để vẫn nhận diện ra hành vi ngáp.

---

## 🛠️ TỔNG HỢP CÁC TỆP TIN DỰ ÁN & VAI TRÒ

| Tên tệp tin | Vai trò trong hệ thống |
| :--- | :--- |
| [requirements.txt](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/requirements.txt) | Cài đặt các thư viện cần thiết. |
| [src/utils/face_geometry.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/utils/face_geometry.py) | **(Bước 1)** Tính EAR, MAR, góc nghiêng đầu và cắt ảnh mắt/miệng. |
| [src/models/driver_state_model.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/models/driver_state_model.py) | **(Bước 2 & 3)** Định nghĩa mạng Deep Learning CNN-LSTM-Attention để dự đoán. |
| [src/utils/alerts.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/utils/alerts.py) | **(Bước 4)** Kích hoạt còi hú qua loa và xuất lệnh điều khiển đèn LED cảnh báo. |
| [src/dataset.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/dataset.py) | Định nghĩa bộ nạp dữ liệu tuần tự phục vụ cho việc huấn luyện. |
| [src/generate_dummy_data.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/generate_dummy_data.py) | Tạo dữ liệu giả lập (chuỗi nhắm mắt, ngáp, gật gù) để test thử nghiệm. |
| [src/train.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/train.py) | Chạy vòng lặp tối ưu hóa trọng số mô hình và vẽ đồ thị kết quả học tập. |
| [src/preprocess.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/preprocess.py) | Đọc và xử lý hàng loạt video thực tế để tạo tập dữ liệu huấn luyện. |
| [src/inference.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/inference.py) | Giao diện camera OpenCV thời gian thực vẽ các trục tọa độ đầu 3D. |
| [src/app.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/tttnC38/src/app.py) | Giao diện dashboard giám sát trên nền tảng Web Streamlit sinh động. |
