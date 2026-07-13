# TASK.md

# Dự án benchmark và tối ưu mô hình phân loại ảnh

## 1. Mục tiêu

Xây dựng **một notebook duy nhất** để:

1. Phân loại 500 ảnh bằng mô hình pretrained trên ImageNet.
2. Đo baseline bằng TensorFlow/Keras.
3. Chuyển mô hình sang ONNX và benchmark bằng ONNX Runtime.
4. Chuyển mô hình sang TFLite/LiteRT và benchmark.
5. Áp dụng thêm **Post-Training Static INT8 Quantization**.
6. So sánh độ chính xác, tốc độ, tài nguyên và kích thước mô hình giữa các phiên bản.
7. Xuất bảng kết quả, biểu đồ và dữ liệu dự đoán phục vụ báo cáo.

---

## 2. Phạm vi đã chốt

### 2.1 Mô hình

- Kiến trúc: **EfficientNetB0**
- Trọng số: pretrained ImageNet
- Input: `224 x 224 x 3`
- Output: 1.000 lớp ImageNet
- Không huấn luyện lại baseline.
- Không fine-tune trong phạm vi chính.

### 2.2 Framework và runtime

- Baseline: TensorFlow/Keras
- ONNX conversion: `tf2onnx`
- ONNX inference: ONNX Runtime
- TFLite/LiteRT conversion và inference: TensorFlow Lite Interpreter
- Tối ưu bổ sung: static INT8 post-training quantization
- Giám sát tài nguyên:
  - CPU/RAM: `psutil`
  - NVIDIA GPU/VRAM nếu có: NVML hoặc `nvidia-smi`
  - GPU không hỗ trợ hoặc không được runtime sử dụng: ghi `N/A` hoặc `0%`, không suy đoán

### 2.3 Môi trường mục tiêu

- Laptop Acer Nitro AN515-58
- CPU: Intel Core i5-12500H
- RAM: 24 GB
- GPU rời: 4 GB, chưa xác định chính xác model
- Hệ điều hành: Windows 64-bit
- Benchmark chính: **CPU**, để TensorFlow, ONNX Runtime và TFLite có cùng thiết bị so sánh
- Benchmark GPU: tùy chọn, chỉ chạy sau khi xác minh GPU và execution provider

---

## 3. Yêu cầu chức năng

### FR-01 — Kiểm tra môi trường

Notebook phải ghi lại:

- Hệ điều hành
- CPU
- Tổng RAM
- GPU và VRAM nếu phát hiện được
- Phiên bản Python
- Phiên bản các thư viện chính
- Execution providers của ONNX Runtime
- Thiết bị TensorFlow nhận diện được
- Số core/thread CPU
- Thread configuration dùng khi benchmark

### FR-02 — Chuẩn bị dữ liệu

- Có đúng 500 ảnh benchmark.
- Mỗi ảnh có:
  - đường dẫn hoặc ID
  - nhãn ground truth nếu đánh giá accuracy
  - mapping nhãn sang ImageNet class index
- Tạo manifest cố định, ví dụ `sample_500.csv`.
- Sampling có seed để tái lập.
- Kiểm tra ảnh hỏng, ảnh trùng và định dạng không hợp lệ.
- Khuyến nghị:
  - 500 ảnh ImageNet validation để benchmark
  - thêm 100 ảnh đại diện để calibration INT8
- Phương án dự phòng:
  - dùng 100 ảnh trong tập 500 làm calibration và ghi rõ giới hạn phương pháp

### FR-03 — Tiền xử lý thống nhất

Tất cả runtime phải dùng cùng một pipeline:

1. Đọc ảnh.
2. Chuyển RGB.
3. Resize/crop theo input chuẩn của EfficientNetB0.
4. Chuyển `float32`.
5. Đảm bảo cùng layout tensor phù hợp từng runtime.
6. Xác minh cùng đầu vào logic trước khi so sánh output.

Không được để mỗi runtime dùng một cách resize hoặc normalization khác nhau.

### FR-04 — Baseline TensorFlow/Keras

- Load EfficientNetB0 pretrained ImageNet.
- Chạy dự đoán cho đủ 500 ảnh.
- Lưu top-1 và top-5.
- Ghi:
  - model size
  - total inference time
  - average latency
  - median latency
  - p95 latency
  - FPS
  - peak RAM
  - average/peak CPU
  - average/peak GPU utilization nếu có
  - peak VRAM nếu có
  - top-1/top-5 accuracy nếu có nhãn

### FR-05 — ONNX

- Export baseline sang ONNX.
- Kiểm tra model bằng ONNX checker.
- So sánh output ONNX với baseline trên một tập kiểm thử nhỏ trước benchmark.
- Benchmark đủ 500 ảnh bằng ONNX Runtime.
- Ghi cùng bộ chỉ số với baseline.
- Lưu prediction riêng cho ONNX.

### FR-06 — TFLite/LiteRT FP32

- Convert baseline sang TFLite float.
- Load bằng TFLite Interpreter.
- Kiểm tra shape, dtype, input/output details.
- So sánh output với baseline.
- Benchmark đủ 500 ảnh.
- Ghi cùng bộ chỉ số.
- Lưu prediction riêng cho TFLite.

### FR-07 — Tối ưu INT8

Áp dụng **post-training static/full-integer quantization**:

- Dùng calibration dataset đại diện.
- Tạo ít nhất:
  - ONNX INT8
  - TFLite INT8
- Kiểm tra model load và inference thành công.
- Đo accuracy drop hoặc top-1 agreement so với baseline.
- Benchmark đủ 500 ảnh.
- Lưu toàn bộ metric và prediction.

### FR-08 — Báo cáo so sánh

Tạo một bảng tổng hợp với tối thiểu các dòng:

- TensorFlow FP32 baseline
- ONNX FP32
- TFLite FP32
- ONNX INT8
- TFLite INT8

Các cột tối thiểu:

- Runtime
- Precision
- Device/provider
- Model size (MB)
- Top-1 accuracy
- Top-5 accuracy
- Top-1 agreement với baseline
- Mean latency (ms)
- Median latency (ms)
- P95 latency (ms)
- FPS
- Peak RAM (MB)
- Average CPU (%)
- Peak CPU (%)
- Average GPU (%)
- Peak GPU (%)
- Peak VRAM (MB)
- Size reduction (%)
- Speedup so với baseline
- Accuracy delta

### FR-09 — Xuất kết quả

Notebook tạo:

- `results/benchmark_results.csv`
- `results/predictions_tensorflow.csv`
- `results/predictions_onnx.csv`
- `results/predictions_tflite.csv`
- `results/predictions_onnx_int8.csv`
- `results/predictions_tflite_int8.csv`
- `reports/comparison_summary.md`
- biểu đồ size, latency, FPS, RAM và accuracy
- file thông tin môi trường

---

## 4. Yêu cầu phi chức năng

### NFR-01 — Tái lập

- Fix random seed.
- Lưu manifest 500 ảnh.
- Lưu phiên bản package.
- Lưu cấu hình benchmark.
- Notebook chạy tuần tự từ trên xuống dưới.

### NFR-02 — Công bằng khi benchmark

- Cùng 500 ảnh.
- Cùng preprocessing logic.
- Cùng batch size chính: `1`.
- Cùng số CPU threads hoặc ghi rõ khác biệt.
- Có warm-up trước khi đo.
- Không tính thời gian tải model vào latency inference.
- Phân biệt:
  - model-only inference
  - end-to-end inference gồm đọc và preprocess ảnh

### NFR-03 — Độ tin cậy phép đo

- Chạy warm-up tối thiểu 20 lần.
- Benchmark toàn bộ 500 ảnh.
- Ghi median và p95, không chỉ mean.
- Resource monitor lấy mẫu liên tục trong khoảng benchmark.
- Tắt hoặc ghi nhận các ứng dụng nền có thể gây nhiễu.

### NFR-04 — Đúng đắn

- Float ONNX/TFLite phải được kiểm tra sai số output.
- Mục tiêu top-1 agreement của bản float với baseline: `>= 99%`.
- Mục tiêu INT8:
  - accuracy giảm không quá 1–2 percentage points, hoặc
  - top-1 agreement với baseline `>= 97%` khi không có ground truth.
- Các ngưỡng là mục tiêu nghiệm thu, không phải cam kết trước khi đo.

### NFR-05 — Tương thích cấu hình máy

- Không yêu cầu train mô hình lớn.
- Không phụ thuộc GPU để hoàn thành bài.
- Peak RAM phải nằm an toàn dưới 24 GB.
- Peak VRAM phải nằm dưới 4 GB nếu chạy GPU.
- Có fallback CPU khi CUDA/GPU provider không hoạt động.

### NFR-06 — Khả năng phục hồi

- Các model và kết quả trung gian được lưu ra disk.
- Nếu notebook dừng giữa chừng, có thể bỏ qua bước đã hoàn thành.
- Mỗi stage phải kiểm tra file output trước khi tiếp tục.

### NFR-07 — Dễ chấm và dễ giải thích

- Notebook có markdown giải thích trước mỗi stage.
- Mỗi bảng metric ghi rõ đơn vị và cách đo.
- Không dùng số liệu ước lượng thay cho số đo thực tế.
- Các trường không đo được phải ghi `N/A` kèm lý do.

---

## 5. Tiêu chí nghiệm thu

- [ ] Notebook duy nhất chạy từ đầu đến cuối.
- [ ] Đủ 500 ảnh được classify ở từng phiên bản.
- [ ] Có baseline, ONNX FP32, TFLite FP32 và INT8.
- [ ] Các bản float được kiểm tra tương đương output.
- [ ] Có bảng metric đầy đủ.
- [ ] Có size reduction và speedup.
- [ ] Có accuracy/top-1 agreement để kiểm tra suy giảm.
- [ ] Có file prediction cho từng runtime.
- [ ] Có biểu đồ so sánh.
- [ ] Có ghi chú rõ CPU/GPU provider.
- [ ] Có kết luận model/runtime tối ưu nhất trên chính máy thử nghiệm.

---

## 6. Ngoài phạm vi

- Train EfficientNetB0 từ đầu.
- Fine-tune dài hạn trên ImageNet.
- Benchmark trên điện thoại thật.
- TensorRT deployment.
- Viết ứng dụng GUI/web.
- Distributed inference.
- So sánh nhiều kiến trúc khác nhau.
