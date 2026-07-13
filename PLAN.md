# PLAN.md

# Kế hoạch thực hiện dự án

## 1. Kiến trúc đề xuất

### 1.1 Luồng tổng thể

```text
Dataset/Manifest
      |
      v
Unified Preprocessing
      |
      +--------------------+
      |                    |
      v                    v
TensorFlow FP32       Export SavedModel
      |                    |
      |               +----+------+
      |               |           |
      v               v           v
Baseline metrics   ONNX FP32   TFLite FP32
                      |           |
                      v           v
                  ONNX INT8   TFLite INT8
                      \           /
                       \         /
                        v       v
                    Unified Benchmark
                          |
                          v
              CSV + Predictions + Charts
                          |
                          v
                Final Comparison Report
```

### 1.2 Lý do chọn EfficientNetB0

EfficientNetB0 phù hợp vì:

- Đủ mạnh để bài có ý nghĩa học thuật.
- Nhẹ hơn đáng kể so với ResNet50 và InceptionV3.
- Input 224 x 224, giảm tải tính toán.
- Phù hợp RAM 24 GB và GPU 4 GB.
- Có pretrained ImageNet trong Keras.
- TensorFlow/Keras chuyển trực tiếp sang TFLite.
- Có thể xuất ONNX bằng tf2onnx.
- Có kiến trúc CNN phù hợp với static INT8 quantization.

### 1.3 Lý do chọn TensorFlow/Keras làm nguồn

- TFLite/LiteRT là đường chuyển đổi native.
- ONNX vẫn có thể sinh qua tf2onnx.
- Tránh chuỗi chuyển PyTorch → ONNX → TensorFlow → TFLite vốn có nhiều điểm lỗi hơn.
- Một model nguồn duy nhất giúp giảm sai khác preprocessing và weights.

### 1.4 Thiết bị benchmark

#### Benchmark chính

- CPU Intel Core i5-12500H.
- Batch size = 1.
- Cùng số threads giữa các runtime.
- TFLite, ONNX và TensorFlow đều đo trên CPU.

#### Benchmark GPU tùy chọn

Chỉ kích hoạt khi:

- GPU được xác minh là NVIDIA.
- `nvidia-smi` hoạt động.
- Runtime có execution provider tương ứng.
- Có thể duy trì cùng điều kiện đo.

Không dùng kết quả GPU để so trực tiếp với TFLite CPU nếu không ghi rõ đây là hai thiết bị khác nhau.

---

## 2. Công nghệ dự kiến

### 2.1 Môi trường

- Python 3.10 hoặc 3.11
- Jupyter Notebook / VS Code Notebook
- Virtual environment riêng
- Windows native CPU là đường mặc định
- WSL2 chỉ dùng khi thật sự cần TensorFlow GPU

### 2.2 Thư viện

- TensorFlow/Keras
- NumPy
- Pandas
- Pillow
- Matplotlib
- ONNX
- ONNX Runtime
- tf2onnx
- psutil
- pynvml hoặc wrapper `nvidia-smi`, nếu NVIDIA
- tqdm
- scikit-learn, chỉ khi cần metric hỗ trợ

### 2.3 Chính sách phiên bản

Không cài “latest” không kiểm soát trong notebook cuối.

Quy trình:

1. Tạo môi trường thử nghiệm.
2. Chọn bộ phiên bản TensorFlow/tf2onnx/ONNX Runtime tương thích.
3. Chuyển đổi model thành công.
4. Freeze package versions.
5. Ghi vào notebook và `environment.json`.

Điểm cần lưu ý:

- tf2onnx có ma trận hỗ trợ TensorFlow và ONNX opset riêng.
- TensorFlow GPU native Windows không còn là đường mặc định ở các phiên bản mới.
- Cần ưu tiên bộ phiên bản ổn định hơn bộ phiên bản mới nhất.

---

## 3. Giao thức benchmark

### 3.1 Dữ liệu

#### Phương án khuyến nghị

- 500 ảnh ImageNet validation có nhãn làm benchmark.
- 100 ảnh đại diện khác làm calibration INT8.
- Sampling cố định bằng seed.
- Lưu danh sách vào CSV.

#### Phương án dự phòng

- Dùng 500 ảnh benchmark.
- Lấy một subset cố định 100 ảnh trong đó để calibration.
- Ghi rõ calibration và evaluation không hoàn toàn độc lập.

### 3.2 Các chế độ đo

#### A. Model-only benchmark

Không tính:

- đọc file
- decode JPEG
- resize
- preprocessing

Dùng tensor đã chuẩn bị sẵn để đo tốc độ runtime thuần.

#### B. End-to-end benchmark

Tính:

- đọc ảnh
- decode
- resize
- preprocessing
- inference
- postprocess top-k

Báo cáo cả hai để tránh diễn giải sai FPS.

### 3.3 Cấu hình chuẩn

- Batch size: 1
- Warm-up: tối thiểu 20 lượt
- Measured samples: 500
- Thread count: thiết lập cố định và ghi vào report
- Power mode: cắm sạc
- Windows power plan: Best performance hoặc tương đương
- Đóng game, trình duyệt nhiều tab và tác vụ nền nặng
- Chạy mỗi runtime theo cùng thứ tự hoặc lặp lại thứ tự đảo để giảm bias nhiệt

### 3.4 Công thức chính

- `FPS = số ảnh / tổng thời gian inference`
- `Mean latency = trung bình thời gian mỗi ảnh`
- `P95 latency = percentile 95 của latency`
- `Speedup = baseline latency / optimized latency`
- `Size reduction = 1 - optimized_size / baseline_size`
- `Accuracy delta = optimized_accuracy - baseline_accuracy`
- `Top-1 agreement = tỷ lệ optimized top-1 giống baseline top-1`

### 3.5 Đo tài nguyên

#### CPU

Ghi hai loại:

- process CPU %
- system CPU %

Cần chuẩn hóa cách hiểu trên CPU nhiều core.

#### RAM

- process RSS trung bình
- process peak RSS
- system used RAM trung bình/peak nếu cần

#### GPU

Nếu NVIDIA:

- GPU utilization %
- VRAM used MB
- peak VRAM MB

Nếu runtime chạy CPU:

- vẫn có thể ghi system GPU utilization
- trường runtime GPU usage phải ghi `N/A` hoặc `0`, tùy cách đo
- không được kết luận runtime dùng GPU chỉ vì máy có GPU

---

## 4. Phương pháp tối ưu bổ sung

## Static INT8 Post-Training Quantization

### 4.1 Lý do chọn

- Phù hợp CNN.
- Không cần train lại toàn bộ.
- Có thể giảm kích thước model đáng kể.
- Có khả năng tăng tốc CPU.
- Có thể áp dụng cho cả ONNX và TFLite.
- Dễ giải thích và có số liệu trước/sau rõ ràng.

### 4.2 Quy trình

1. Chuẩn bị calibration dataset đại diện.
2. Chạy calibration.
3. Quantize weights và activations sang INT8.
4. Kiểm tra operator compatibility.
5. So sánh prediction với baseline.
6. Đo accuracy delta.
7. Benchmark lại cùng protocol.
8. Chọn bản INT8 có trade-off tốt nhất.

### 4.3 Fallback

Nếu full INT8 làm accuracy giảm quá nhiều:

1. Thử calibration method khác.
2. Tăng số ảnh calibration.
3. Loại trừ layer nhạy cảm khỏi quantization.
4. Dùng dynamic range quantization.
5. QAT chỉ là phương án mở rộng, không phải phạm vi mặc định.

### 4.4 Tối ưu runtime tùy chọn

Vì máy dùng Intel CPU, có thể benchmark thêm ONNX Runtime OpenVINO Execution Provider như một bonus. Đây là tối ưu runtime, không thay thế yêu cầu INT8 và không nên làm phức tạp phạm vi chính trước khi pipeline chuẩn hoàn tất.

---

## 5. Milestone

## Milestone 0 — Chốt đầu vào

### Công việc

- Xác định GPU chính xác.
- Chốt nguồn 500 ảnh.
- Chốt có nhãn ground truth hay không.
- Chốt CPU-only hay có thêm GPU appendix.
- Chốt package versions.

### Deliverable

- Environment decision record.
- Dataset decision record.
- Manifest format.

### Exit criteria

- Biết chính xác dataset path.
- Biết chính xác GPU.
- Tạo được Python environment.
- Import các thư viện chính thành công.

---

## Milestone 1 — Data và benchmark harness

### Công việc

- Tạo sample manifest.
- Validate 500 ảnh.
- Xây preprocessing thống nhất.
- Xây resource monitor.
- Xây timer và metric collector.
- Xây schema bảng kết quả.

### Deliverable

- Dataset manifest.
- Environment report.
- Benchmark protocol cells trong notebook.

### Exit criteria

- 500 ảnh hợp lệ.
- Có thể tạo batch/tensor cho model.
- Resource monitor hoạt động.

---

## Milestone 2 — Baseline EfficientNetB0

### Công việc

- Load pretrained model.
- Kiểm tra output trên một vài ảnh.
- Chạy 500 ảnh.
- Lưu prediction.
- Benchmark model-only và end-to-end.

### Deliverable

- Baseline model artifact.
- Baseline prediction CSV.
- Baseline metrics row.

### Exit criteria

- Không OOM.
- Đủ 500 prediction.
- Metric đầy đủ và hợp lý.

---

## Milestone 3 — ONNX FP32

### Công việc

- Export SavedModel.
- Convert sang ONNX.
- Check ONNX graph.
- Verify numerical equivalence.
- Benchmark 500 ảnh.

### Deliverable

- `efficientnetb0_fp32.onnx`
- ONNX prediction CSV.
- ONNX metrics row.

### Exit criteria

- ONNX Runtime load thành công.
- Float top-1 agreement mục tiêu >= 99%.
- Có số liệu speedup/size.

---

## Milestone 4 — TFLite FP32

### Công việc

- Convert TFLite float.
- Verify input/output dtype và shape.
- So sánh output.
- Benchmark 500 ảnh.

### Deliverable

- `efficientnetb0_fp32.tflite`
- TFLite prediction CSV.
- TFLite metrics row.

### Exit criteria

- Interpreter chạy đủ 500 ảnh.
- Float top-1 agreement mục tiêu >= 99%.
- Có số liệu speedup/size.

---

## Milestone 5 — INT8

### Công việc

- Chuẩn bị calibration set.
- Quantize ONNX INT8.
- Quantize TFLite INT8.
- Validate output.
- Benchmark lại.
- Phân tích accuracy drop.

### Deliverable

- `efficientnetb0_int8.onnx`
- `efficientnetb0_int8.tflite`
- Hai prediction CSV.
- Hai metrics row.

### Exit criteria

- Cả hai model load được.
- Đủ 500 prediction.
- Accuracy delta được báo cáo.
- Xác định trade-off tốt nhất.

---

## Milestone 6 — Báo cáo cuối

### Công việc

- Gom metrics.
- Tạo biểu đồ.
- Viết nhận xét.
- Xác định model tốt nhất theo:
  - tốc độ
  - size
  - RAM
  - accuracy
  - trade-off tổng thể
- Chạy notebook sạch từ đầu.

### Deliverable

- Notebook hoàn chỉnh.
- `benchmark_results.csv`
- `comparison_summary.md`
- Figures.
- Package version lock.

### Exit criteria

- Notebook chạy top-to-bottom.
- Không có cell lỗi.
- Kết quả có thể tái lập.
- Báo cáo nêu rõ hạn chế.

---

## 6. Cấu trúc notebook dự kiến

Notebook duy nhất: `notebooks/efficientnet_onnx_tflite_benchmark.ipynb`

### Section 0 — Project configuration

- Paths
- Seed
- Batch size
- Thread count
- Warm-up count
- Runtime flags

### Section 1 — Environment audit

- CPU/RAM/GPU
- Package versions
- Providers

### Section 2 — Dataset preparation

- Load/create manifest
- Validate images
- Label mapping
- Calibration subset

### Section 3 — Shared preprocessing

- Decode
- Resize/crop
- Tensor preparation
- Preloaded tensor cache

### Section 4 — Monitoring and benchmark utilities

- Timing
- CPU/RAM monitor
- GPU monitor
- Metric aggregation
- Result schema

### Section 5 — TensorFlow baseline

- Model load
- Sanity check
- 500 predictions
- Benchmark

### Section 6 — ONNX FP32

- Export
- Validate
- Inference
- Benchmark

### Section 7 — TFLite FP32

- Convert
- Validate
- Inference
- Benchmark

### Section 8 — ONNX INT8

- Calibration
- Quantization
- Validate
- Benchmark

### Section 9 — TFLite INT8

- Representative dataset
- Quantization
- Validate
- Benchmark

### Section 10 — Comparison

- Accuracy/agreement
- Size
- Latency/FPS
- RAM/CPU/GPU
- Speedup

### Section 11 — Export report

- CSV
- Markdown summary
- Figures
- Final conclusion

---

## 7. Kế hoạch file/folder

```text
efficientnet-optimization/
├── TASK.md
├── PLAN.md
├── CONTEXT.md
├── notebooks/
│   └── efficientnet_onnx_tflite_benchmark.ipynb
├── data/
│   ├── raw/
│   │   ├── benchmark_500/
│   │   └── calibration_100/
│   ├── manifests/
│   │   ├── sample_500.csv
│   │   └── calibration_100.csv
│   └── labels/
│       └── imagenet_class_index.json
├── models/
│   ├── tensorflow/
│   │   └── efficientnetb0_savedmodel/
│   ├── onnx/
│   │   ├── efficientnetb0_fp32.onnx
│   │   └── efficientnetb0_int8.onnx
│   └── tflite/
│       ├── efficientnetb0_fp32.tflite
│       └── efficientnetb0_int8.tflite
├── results/
│   ├── benchmark_results.csv
│   ├── predictions_tensorflow.csv
│   ├── predictions_onnx.csv
│   ├── predictions_tflite.csv
│   ├── predictions_onnx_int8.csv
│   └── predictions_tflite_int8.csv
├── reports/
│   ├── comparison_summary.md
│   └── environment.json
├── figures/
│   ├── fps_comparison.png
│   ├── latency_comparison.png
│   ├── model_size_comparison.png
│   ├── memory_comparison.png
│   └── accuracy_comparison.png
└── logs/
    └── benchmark.log
```

### Quy tắc

- Chỉ có một file code chính là notebook.
- Các thư mục còn lại chỉ chứa input, model artifacts và output.
- Không tách helper `.py` trong phiên bản nộp bài.
- Có thể dùng notebook checkpoint trong quá trình làm, nhưng không đưa vào bản nộp.

---

## 8. Rủi ro và xử lý

| Rủi ro | Mức độ | Xử lý |
|---|---:|---|
| Chưa biết model GPU | Cao | Kiểm tra bằng `nvidia-smi` và Device Manager trước khi chọn CUDA |
| TensorFlow GPU native Windows không tương thích | Cao | CPU-first; WSL2 là tùy chọn |
| ImageNet cần quyền truy cập hoặc token | Cao | Chuẩn bị dataset trước; có manifest cố định |
| Nhãn folder không khớp 1.000 class index | Cao | Validate mapping bằng vài mẫu thủ công và script kiểm tra |
| tf2onnx không tương thích phiên bản TensorFlow | Cao | Pin phiên bản, test conversion sớm ở Milestone 0 |
| TFLite INT8 có operator không hỗ trợ | Trung bình | Thử full INT8, fallback dynamic range |
| Sai khác preprocessing | Cao | Một hàm preprocessing logic, test tensor/output |
| FPS bị nhiễu do I/O | Cao | Báo cả model-only và end-to-end |
| CPU throttling/nhiệt độ | Trung bình | Cắm sạc, warm-up, chạy lặp, ghi median/p95 |
| GPU usage không đo được | Trung bình | Ghi N/A, không bịa số |
| 500 ảnh quá ít để kết luận accuracy tổng quát | Trung bình | Nêu rõ sample benchmark, không đại diện toàn ImageNet |
| INT8 giảm accuracy | Trung bình | Calibration tốt hơn, debug layer nhạy cảm, fallback |
| Notebook quá dài | Thấp | Chia section rõ ràng, collapse output dài |

---

## 9. Quyết định cần xác nhận trước khi code

1. GPU chính xác là model nào?
2. Dataset 500 ảnh lấy từ đâu?
3. Có nhãn ground truth theo ImageNet class index không?
4. Có cho phép thêm 100 ảnh calibration không?
5. Bài yêu cầu CPU benchmark, GPU benchmark hay cả hai?
6. Model size tính file nào làm baseline:
   - SavedModel directory
   - `.keras`
   - frozen graph
7. Giáo viên có yêu cầu chính xác package/framework nào không?
8. Có bắt buộc báo cáo power consumption hay không?
