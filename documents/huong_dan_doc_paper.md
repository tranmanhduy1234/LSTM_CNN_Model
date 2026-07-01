# Hướng Dẫn Đọc Papers — Hệ Thống CNN-LSTM Nhận Diện Trạng Thái Tài Xế

> **Mục tiêu**: Đọc theo trình tự từ nền tảng lý thuyết → kiến trúc kết hợp → ứng dụng cụ thể.  
> Mỗi paper được ghi rõ: cần nắm gì, bỏ qua gì, dùng để trích dẫn cho phần nào trong báo cáo.

---

## 🗺️ Trình Tự Đọc

```
Tầng 1 — Lý thuyết nền tảng (đọc trước để hiểu WHY)
    └── [1] LSTM.pdf
    └── [2] CNN.pdf (ImageNet)

Tầng 2 — Kiến trúc kết hợp CNN + LSTM (đọc để hiểu HOW)
    └── [3] Long-term Recurrent Convolutional Networks (LRCN)
    └── [4] Beyond Short Snippets

Tầng 3 — Domain cụ thể: Drowsiness Detection (đọc để biết WHAT người khác đã làm)
    └── [5] Detecting Driver Drowsiness Based on Sensors: A Review
    └── [6] Eye Blink Detection Using Facial Landmarks
    └── [7] Real-time detection of driver fatigue based on CNN-LSTM
    └── [8] A CNN-LSTM-Based Deep Learning Approach for Driver Drowsiness Prediction
    └── [9] A CNN-LSTM Approach for Drowsiness AND Distraction Detection

Tầng 4 — SOTA tham khảo hướng mở rộng
    └── [10] SAFE-DRIVE-AI: CNN–LSTM–Attention Framework
```

---

## Tầng 1 — Lý Thuyết Nền Tảng

---

### [1] LSTM.pdf
**Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.**

#### Tại sao phải đọc
Đây là paper gốc định nghĩa LSTM — bất kỳ báo cáo nào dùng LSTM đều phải trích dẫn paper này. Hơn 80.000 citation.

#### Cần nắm
- **Vấn đề LSTM giải quyết**: Vanishing Gradient Problem trong RNN thông thường — khi chuỗi dài, gradient về gần zero, mạng không học được dependency xa.
- **Cơ chế gating**: 3 cổng điều khiển luồng thông tin:
  - **Forget gate**: quyết định thông tin cũ nào cần quên.
  - **Input gate**: quyết định thông tin mới nào cần nhớ.
  - **Output gate**: quyết định thông tin nào đưa ra ngoài.
- **Memory cell**: bộ nhớ dài hạn chạy xuyên suốt chuỗi, gradient truyền ổn định hơn RNN thường.
- **Ý nghĩa với bài toán của bạn**: tài xế ngủ gật không thể phát hiện từ 1 frame đơn lẻ mà cần theo dõi xu hướng qua nhiều frame liên tiếp — đây chính xác là thứ LSTM giỏi.

#### Bỏ qua
Phần chứng minh toán học chi tiết (Section 4–6) và các thí nghiệm về embedded Reber grammar — không liên quan đến bài toán của bạn.

#### Trích dẫn cho phần
> **Mục 2.2 — Cơ sở lý thuyết LSTM**: Giải thích tại sao chọn LSTM thay vì RNN thông thường để mô hình hóa đặc trưng thời gian.

---

### [2] CNN.pdf (ImageNet)
**Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS.**

#### Tại sao phải đọc
Paper đặt nền móng cho toàn bộ computer vision hiện đại. Chứng minh CNN (AlexNet) vượt trội trong trích xuất spatial features từ ảnh.

#### Cần nắm
- **Tại sao CNN phù hợp cho ảnh**: tính chất locality (đặc trưng cục bộ như cạnh, góc, texture) và translation invariance (đặc trưng không phụ thuộc vị trí trong ảnh).
- **Cấu trúc CNN cơ bản**: Convolutional layer → Pooling layer → Fully Connected layer.
- **ReLU activation**: thay sigmoid/tanh, giải quyết vanishing gradient trong CNN sâu, tăng tốc training.
- **Dropout**: regularization để tránh overfitting — quan trọng cho dataset drowsiness thường nhỏ.
- **Ý nghĩa với bài toán**: CNN trích xuất đặc trưng hình học (hình dạng mắt nhắm/mở, vùng miệng ngáp) từ từng frame độc lập trước khi đưa vào LSTM.

#### Bỏ qua
Phần mô tả chi tiết phần cứng GPU, các kỹ thuật data augmentation cụ thể cho ImageNet — không ảnh hưởng lý luận của bạn.

#### Trích dẫn cho phần
> **Mục 2.1 — Cơ sở lý thuyết CNN**: Lý luận tại sao CNN là lựa chọn tối ưu để trích xuất spatial features từ từng frame khuôn mặt tài xế.

---

## Tầng 2 — Kiến Trúc Kết Hợp CNN-LSTM

---

### [3] Long-term Recurrent Convolutional Networks (LRCN)
**Donahue, J., et al. (2015). Long-term Recurrent Convolutional Networks for Visual Recognition and Description. CVPR.**

#### Tại sao phải đọc
Paper kinh điển nhất justify việc ghép CNN + LSTM thành pipeline tuần tự — đây chính là kiến trúc bạn đang xây dựng.

#### Cần nắm
- **Pipeline CNN → LSTM**: CNN trích xuất feature vector từ từng frame → LSTM nhận chuỗi feature vectors → học temporal dynamics. Đây là mô hình tư duy cốt lõi cho thiết kế của bạn.
- **Lý do tách biệt hai giai đoạn**: CNN học "nhìn thấy gì" (spatial), LSTM học "diễn biến như thế nào" (temporal) — phân tách trách nhiệm rõ ràng.
- **Kết quả thực nghiệm**: cải thiện đáng kể so với CNN đơn thuần trên các bài toán nhận dạng hoạt động — dùng để lập luận "tại sao cần LSTM sau CNN".
- **Fixed-length feature vector**: sau CNN, flatten và đưa vào LSTM dưới dạng sequence — đây là cách nối hai kiến trúc mà bạn cần implement.

#### Bỏ qua
Phần về Visual Description Generation (image captioning) — không liên quan đến bài toán classification của bạn.

#### Trích dẫn cho phần
> **Mục 2.3 — Kiến trúc kết hợp CNN-LSTM**: Cơ sở lý luận cho việc pipeline CNN trích xuất spatial features rồi chuyển sang LSTM mô hình hóa temporal dynamics.

---

### [4] Beyond Short Snippets: Deep Networks for Video Classification
**Yue-Hei Ng, J., et al. (2015). Beyond Short Snippets: Deep Networks for Video Classification. CVPR.**

#### Tại sao phải đọc
Mở rộng LRCN cho video dài, so sánh trực tiếp nhiều cách tích hợp CNN và LSTM — giúp bạn hiểu tại sao chọn cách nối tuần tự (frame-by-frame) thay vì các cách khác.

#### Cần nắm
- **Temporal pooling vs LSTM**: paper so sánh thực nghiệm hai cách xử lý chuỗi frame — pooling đơn giản (mất thông tin thứ tự) và LSTM (giữ thứ tự) — LSTM tốt hơn khi thứ tự quan trọng.
- **Ứng dụng cho drowsiness**: trạng thái buồn ngủ có tính thứ tự (mắt chớp → mắt nặng → nhắm lâu → gật đầu) nên LSTM phù hợp hơn pooling.
- **Optical flow vs RGB**: paper so sánh dùng ảnh màu (RGB) và optical flow — với bài toán khuôn mặt tĩnh (không di chuyển nhiều), RGB là đủ.

#### Bỏ qua
Phần thực nghiệm trên dataset Sports-1M (quá lớn và không liên quan), phần về multi-resolution optical flow.

#### Trích dẫn cho phần
> **Mục 2.3 — Kiến trúc kết hợp CNN-LSTM**: Justify lý do chọn LSTM thay vì temporal pooling để xử lý chuỗi frame khuôn mặt theo thời gian.

---

## Tầng 3 — Domain Cụ Thể: Drowsiness Detection

---

### [5] Detecting Driver Drowsiness Based on Sensors: A Review
**Sahayadhas, A., Sundaraj, K., & Murugappan, M. (2012). Sensors, 12(12), 16937–16953.**

#### Tại sao phải đọc
Paper survey tổng quan nhất về drowsiness detection — dùng để mở đầu phần Related Work, định vị hướng tiếp cận của bạn trong bức tranh nghiên cứu rộng hơn.

#### Cần nắm
- **3 hướng tiếp cận chính**:
  1. **Vehicle-based**: phân tích hành vi lái xe (độ lệch làn đường, áp lực vô-lăng) — không cần camera nhưng phản ứng chậm.
  2. **Physiological-based**: đo EEG, ECG, nhịp tim — chính xác nhất nhưng cần thiết bị đặc biệt, khó triển khai thực tế.
  3. **Behavioral-based** (hướng của bạn): phân tích khuôn mặt qua camera — không xâm lấn, triển khai dễ, đây là lý do bạn chọn hướng này.
- **Các dấu hiệu buồn ngủ hành vi**: tần suất chớp mắt, thời gian mắt nhắm (PERCLOS), tần suất ngáp, góc nghiêng đầu — đây là features CNN của bạn cần học.
- **PERCLOS** (Percentage of Eye Closure): chỉ số chuẩn nhất cho drowsiness — mắt nhắm >80% thời gian trong 1 phút là buồn ngủ.

#### Bỏ qua
Phần chi tiết về cảm biến sinh lý học (EEG electrode placement, ECG signal processing) — không phải hướng của bạn.

#### Trích dẫn cho phần
> **Mục 1 — Giới thiệu** và **Mục 3 — Related Work**: Định vị hướng behavioral-based trong ba hướng nghiên cứu, lý giải lý do chọn camera thay vì sensor.

---

### [6] Eye Blink Detection Using Facial Landmarks
**Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. VISAPP.**

#### Tại sao phải đọc
Định nghĩa chỉ số **EAR (Eye Aspect Ratio)** — công cụ toán học chuẩn để đo trạng thái mắt, nếu bạn dùng landmark-based detection thì paper này là bắt buộc.

#### Cần nắm
- **Công thức EAR**:
  ```
  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  ```
  Trong đó p1–p6 là 6 điểm landmark quanh mắt.
- **Ngưỡng phân loại**: EAR < 0.25 trong liên tiếp 3 frame → mắt đang nhắm.
- **Ưu điểm**: đơn giản, real-time, không cần GPU mạnh — phù hợp cho hệ thống nhúng trong xe.
- **Giới hạn**: nhạy cảm với góc camera và điều kiện ánh sáng — đây là điểm yếu mà CNN-LSTM của bạn giải quyết tốt hơn (học được robustness tự động).

#### Bỏ qua
Phần chi tiết về thuật toán detect facial landmark (dlib) — bạn chỉ cần hiểu EAR như một khái niệm, không cần implement lại từ đầu.

#### Trích dẫn cho phần
> **Mục 2.4 — Đặc trưng nhận diện**: Định nghĩa chỉ số EAR làm cơ sở toán học cho việc phát hiện trạng thái mắt, kể cả khi bạn dùng CNN thay vì tính EAR thủ công.

---

### [7] Real-time Detection of Driver Fatigue Based on CNN-LSTM
**Liu, M.Z., Xu, X., Hu, J., & Jiang, Q.N. (2021). IET Image Processing, Vol. 16, No. 4.**

#### Tại sao phải đọc
Paper gần nhất với thiết kế của bạn — dùng đúng CNN-LSTM cho bài toán fatigue detection real-time. Đây là **baseline chính** để so sánh kết quả.

#### Cần nắm
- **Pipeline cụ thể**: Face detection → ROI extraction (vùng mắt, miệng) → CNN feature extraction → LSTM temporal modeling → Classification.
- **Thiết kế CNN**: bao nhiêu conv layer, filter size, pooling strategy — so sánh với lựa chọn của bạn.
- **Thiết kế LSTM**: số hidden units, số layer, có dùng bidirectional không.
- **Dataset**: dùng dataset nào, điều kiện ánh sáng, góc camera — so sánh với dataset của bạn.
- **Kết quả**: accuracy, latency, FPS thực tế — đây là con số để bạn so sánh trong phần Evaluation.

#### Bỏ qua
Phần implementation detail về hardware (FPGA, embedded system optimization) nếu bạn không triển khai trên phần cứng chuyên dụng.

#### Trích dẫn cho phần
> **Mục 3 — Related Work**: So sánh trực tiếp với kiến trúc của bạn về pipeline, dataset, và kết quả.

---

### [8] A CNN-LSTM-Based Deep Learning Approach for Driver Drowsiness Prediction
**Gomaa, M.W., Mahmoud, R.O., & Sarhan, A.M. (2022). Journal of Engineering Research, Vol. 6, No. 3.**

#### Tại sao phải đọc
Thêm một baseline CNN-LSTM cho bài toán drowsiness, focus vào prediction (dự đoán trước) thay vì chỉ detection (phát hiện tức thời) — hướng tiếp cận có chiều sâu hơn.

#### Cần nắm
- **Sự khác biệt detection vs prediction**: detection = phát hiện khi đã buồn ngủ; prediction = dự đoán sắp buồn ngủ dựa trên xu hướng — LSTM đặc biệt phù hợp cho prediction vì nhớ được lịch sử.
- **Feature engineering**: ngoài ảnh frame, paper còn dùng thêm các derived features như EAR, MAR (Mouth Aspect Ratio) — gợi ý bạn có thể thêm tầng này để tăng accuracy.
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, và đặc biệt **latency** (độ trễ phát hiện) — quan trọng với bài toán safety-critical.

#### Bỏ qua
Phần so sánh với các ML truyền thống (SVM, Random Forest) — không cần thiết nếu bạn chỉ focus vào deep learning approach.

#### Trích dẫn cho phần
> **Mục 3 — Related Work**: So sánh về độ trễ phát hiện và metrics đánh giá. Mục **5 — Thảo luận**: Khả năng mở rộng sang drowsiness prediction.

---

### [9] A CNN-LSTM Approach for Accurate Drowsiness AND Distraction Detection
**ICIC Express Letters, Vol. 18, No. 9, 2024, pp. 907–917.**

#### Tại sao phải đọc
Paper mới nhất (2024) trong đám, dùng FaceMesh kết hợp CNN-LSTM, phát hiện đồng thời buồn ngủ VÀ mất tập trung — scope rộng hơn nhưng kiến trúc tương tự.

#### Cần nắm
- **FaceMesh**: thay vì detect từng vùng riêng lẻ, dùng 468 facial landmark toàn khuôn mặt làm input — cho phép học nhiều dấu hiệu hơn (hướng nhìn, gật đầu, biểu cảm).
- **IOU cho yawning detection**: dùng Intersection over Union để phát hiện tay che miệng (dấu hiệu ngáp) — kỹ thuật sáng tạo không phổ biến.
- **Head pose estimation**: dùng góc đầu làm feature bổ sung — thêm chiều thông tin ngoài mắt/miệng.
- **Multi-task learning**: phát hiện drowsiness và distraction cùng lúc trong 1 model — trái ngược với thiết kế single-task của bạn, dùng để so sánh trade-off.

#### Bỏ qua
Phần về distraction detection (tay cầm điện thoại, ăn uống) nếu scope dự án của bạn chỉ là drowsiness.

#### Trích dẫn cho phần
> **Mục 3 — Related Work**: So sánh về phương pháp trích xuất đặc trưng (FaceMesh vs ROI-based), và về scope bài toán (single-task vs multi-task).

---

## Tầng 4 — SOTA Tham Khảo Hướng Mở Rộng

---

### [10] SAFE-DRIVE-AI: A CNN–LSTM–Attention Framework for Drowsiness Detection
**Nasir, O. (2025). Engineering, Technology & Applied Science Research, Vol. 15, No. 5.**

#### Tại sao phải đọc
Paper mới nhất (2025), thêm Attention Mechanism vào CNN-LSTM — đây là hướng mở rộng tự nhiên cho thiết kế của bạn, dùng để viết phần **Future Work**.

#### Cần nắm
- **Vấn đề của CNN-LSTM thuần**: LSTM xử lý chuỗi theo thứ tự, không biết frame nào quan trọng hơn — Attention giải quyết bằng cách học trọng số cho từng frame.
- **Self-attention trong temporal modeling**: thay vì LSTM "nhớ đều" tất cả, attention cho phép model tập trung vào các khoảnh khắc mắt nhắm, đầu gật mà bỏ qua các frame "bình thường".
- **Cải thiện so với CNN-LSTM thuần**: accuracy tăng bao nhiêu % — con số cụ thể để bạn đặt mục tiêu.
- **Tăng interpretability**: attention weights có thể visualize để giải thích "tại sao model kết luận tài xế đang buồn ngủ" — quan trọng cho safety-critical system.

#### Bỏ qua
Chi tiết implementation của attention layer nếu bạn chưa implement — chỉ cần hiểu concept để viết Future Work.

#### Trích dẫn cho phần
> **Mục 6 — Hướng Phát Triển (Future Work)**: Đề xuất mở rộng kiến trúc bằng attention mechanism như hướng nghiên cứu tiếp theo.

---

## 📋 Bảng Tóm Tắt Trích Dẫn Theo Phần Báo Cáo

| Phần báo cáo | Papers cần trích dẫn |
|---|---|
| **1. Giới thiệu** — Bối cảnh và động lực | [5] Survey Sensors |
| **2.1 CNN** — Spatial feature extraction | [2] ImageNet/AlexNet |
| **2.2 LSTM** — Temporal modeling | [1] Hochreiter 1997 |
| **2.3 Kiến trúc CNN-LSTM** — Lý do ghép | [3] LRCN, [4] Beyond Short Snippets |
| **2.4 Đặc trưng nhận diện** — Mắt, miệng | [6] EAR Eye Blink |
| **3. Related Work** | [7] Liu 2021, [8] Gomaa 2022, [9] CNN-LSTM 2024 |
| **5. Thảo luận** — So sánh | [7], [8], [9] |
| **6. Future Work** | [10] SAFE-DRIVE-AI 2025 |

---

## ⏱️ Ước Tính Thời Gian Đọc

| Paper | Độ ưu tiên | Thời gian đọc | Ghi chú |
|---|---|---|---|
| [1] LSTM | ⭐⭐⭐ Bắt buộc | 45 phút | Đọc Section 1-3 là đủ |
| [2] CNN/ImageNet | ⭐⭐⭐ Bắt buộc | 30 phút | Đọc Abstract + Section 3-4 |
| [3] LRCN | ⭐⭐⭐ Bắt buộc | 45 phút | Đọc Section 1, 3, 5 |
| [4] Beyond Short Snippets | ⭐⭐ Quan trọng | 30 phút | Đọc Section 1, 3, 4 |
| [5] Survey Sensors | ⭐⭐⭐ Bắt buộc | 30 phút | Đọc Section 1-3 |
| [6] Eye Blink EAR | ⭐⭐ Quan trọng | 20 phút | Đọc toàn bộ (ngắn) |
| [7] Liu CNN-LSTM 2021 | ⭐⭐⭐ Bắt buộc | 45 phút | Đọc toàn bộ |
| [8] Gomaa 2022 | ⭐⭐ Quan trọng | 40 phút | Đọc Section 2-4 |
| [9] CNN-LSTM 2024 | ⭐⭐ Quan trọng | 40 phút | Đọc Section 2-4 |
| [10] SAFE-DRIVE-AI | ⭐ Tham khảo | 30 phút | Đọc Abstract + Conclusion |
| **Tổng** | | **~6 giờ** | Có thể chia 2-3 buổi |
