Tầng 1 — Nền tảng kiến trúc (bắt buộc trích dẫn)
Đây là các paper gốc chứng minh tại sao bạn chọn CNN và LSTM thay vì kiến trúc khác.
[1] LSTM gốc

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.

Đây là paper giải quyết vanishing gradient problem trong RNN thông thường, lý do cốt lõi tại sao bạn dùng LSTM thay vì RNN thuần để mô hình hóa chuỗi hành vi theo thời gian. Đây là paper hơn 80.000 citation, không thể thiếu trong bất kỳ báo cáo nào dùng LSTM.
[2] CNN cho spatial feature (ImageNet paper)

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS.

Cơ sở lý luận cho việc dùng CNN trích xuất spatial features từ từng frame ảnh — paper đặt nền móng cho toàn bộ computer vision hiện đại.

Tầng 2 — Cơ sở cho kiến trúc kết hợp CNN-LSTM
Chứng minh rằng việc kết hợp CNN (spatial) + LSTM (temporal) là lựa chọn có nền tảng học thuật vững chắc, không phải bạn tự nghĩ ra.
[3] LRCN — paper kinh điển nhất cho CNN-LSTM

Donahue, J., et al. (2015). Long-term Recurrent Convolutional Networks for Visual Recognition and Description. CVPR.

Paper này kết hợp spatial feature extraction của CNN với temporal dynamics modeling của LSTM, chứng minh cải thiện đáng kể trên các bài toán nhận diện hành động — đây là nền tảng trực tiếp cho pipeline CNN→LSTM của bạn.
[4] CNN-LSTM cho video classification

Yue-Hei Ng, J., et al. (2015). Beyond Short Snippets: Deep Networks for Video Classification. CVPR.

Paper này mở rộng ứng dụng CNN-LSTM để xử lý chuỗi video dài, nhấn mạnh tầm quan trọng của việc nắm bắt temporal dependency mở rộng — trực tiếp justify lý do bạn cần LSTM sau CNN thay vì dừng ở CNN đơn thuần.

Tầng 3 — Paper trực tiếp về driver drowsiness detection
Đây là các paper cùng domain, dùng để so sánh và định vị công trình của bạn trong bức tranh nghiên cứu hiện tại.
[5] CNN-LSTM cho drowsiness — paper gần nhất với thiết kế của bạn

Liu, M.Z., Xu, X., Hu, J., & Jiang, Q.N. (2021). Real-time detection of driver fatigue based on CNN-LSTM. IET Image Processing, Vol. 16, No. 4.

Đây là paper dùng đúng kiến trúc CNN-LSTM cho bài toán phát hiện mệt mỏi tài xế theo thời gian thực — có thể dùng làm baseline so sánh trực tiếp.
[6] CNN-LSTM drowsiness prediction

Gomaa, M.W., Mahmoud, R.O., & Sarhan, A.M. (2022). A CNN-LSTM-Based Deep Learning Approach for Driver Drowsiness Prediction. Journal of Engineering Research (ERJ), Vol. 6, No. 3, pp. 59–71.

Paper này dùng CNN-LSTM để dự đoán buồn ngủ, cùng hướng tiếp cận với bạn — dùng để so sánh kết quả và phương pháp.
[7] CNN-LSTM + attention framework (SOTA gần nhất)

Nasir, O. (2025). SAFE-DRIVE-AI: A CNN–LSTM–Attention Framework for Drowsiness Detection. Engineering, Technology & Applied Science Research, Vol. 15, No. 5.

Paper này đề xuất framework CNN-LSTM-Attention cho drowsiness detection — bạn có thể trích dẫn để chỉ ra hướng mở rộng tiếp theo (thêm attention) mà thiết kế hiện tại của bạn chưa có, thể hiện bạn nắm rõ bức tranh tổng thể.
[8] CNN-LSTM drowsiness + distraction (2024)

A CNN-LSTM Approach for Accurate Drowsiness and Distraction Detection in Drivers. ICIC Express Letters, Vol. 18, No. 9, 2024, pp. 907–917.

Paper này dùng FaceMesh kết hợp CNN-LSTM để phát hiện đồng thời buồn ngủ và mất tập trung, tập trung vào vùng mắt và miệng — trùng khớp với mô tả thiết kế "nhận diện trạng thái mắt và vùng miệng" của bạn.

Tầng 4 — Paper justify lựa chọn đặc trưng (mắt, miệng)
[9] Eye Aspect Ratio (EAR) — paper gốc của chỉ số đóng/mở mắt

Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. VISAPP.

Paper định nghĩa chỉ số EAR kinh điển nhất để phát hiện mắt nhắm — nếu bạn dùng landmark mắt trong thiết kế thì paper này là bắt buộc.
[10] Survey về drowsiness detection methods

Sahayadhas, A., Sundaraj, K., & Murugappan, M. (2012). Detecting Driver Drowsiness Based on Sensors: A Review. Sensors, 12(12), 16937–16953.

Paper survey tổng quan, dùng để mở đầu phần Related Work, chứng minh bài toán đã được nghiên cứu rộng rãi và phương pháp behavioral (camera nhìn mặt) là một trong ba hướng chính.