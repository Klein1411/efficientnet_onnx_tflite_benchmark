# Báo cáo cuối: Benchmark EfficientNetB0 trên Windows CPU

## 1. Mục tiêu

Đánh giá một pipeline duy nhất cho EfficientNetB0 pretrained ImageNet trên 500 ảnh benchmark, so sánh TensorFlow/Keras FP32, ONNX FP32, TFLite FP32, ONNX INT8 và TFLite INT8 trên CPU Windows native.

## 2. Môi trường thử nghiệm

- OS: Windows 11 Home Single Language 64-bit
- CPU: Intel Core i5-12500H
- Physical cores: 12
- Logical processors: 16
- RAM: 24 GB
- GPU chính: NVIDIA GeForce RTX 3050 Laptop GPU, nhưng benchmark chính chạy CPU
- Python: 3.11.9
- TensorFlow: 2.15.1
- ONNX Runtime: 1.18.1
- tf2onnx: 1.16.1
- Batch size: 1

## 3. Dataset

- Benchmark: 500 ảnh từ `data/manifests/sample_500.csv`
- Calibration: 100 ảnh từ `data/manifests/calibration_100.csv`
- Ground-truth: ImageNet index thật, suy ra từ synset folder và `imagenet_class_index.json`

## 4. Benchmark protocol

- Chạy CPU-only trên Windows native
- Warm-up trước khi đo
- Đo riêng model-only và end-to-end
- Không thay đổi kết quả FP32 hiện có
- INT8 dùng calibration dataset riêng
- Không dùng validation subset 10 ảnh thay cho benchmark 500 ảnh

## 5. Bảng kết quả

| Mô hình | Runtime | Precision | Kích thước (MB) | Top-1 | Top-5 | Mean latency (ms) | Median (ms) | P95 (ms) | FPS model-only | FPS end-to-end | RAM đỉnh (MB) | CPU process đỉnh (%) | Speedup vs TF | Giảm size (%) | Delta accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TensorFlow FP32 | TensorFlow/Keras | FP32 | 21.04 | 0.836 | 0.974 | 141.957355 | 136.66370 | 173.828145 | 7.044369 | 7.119030 | 955.15 | 78.1 | 1.000000 | 0.000000 | 0.000 |
| ONNX FP32 | ONNX Runtime | FP32 | 20.17 | 0.836 | 0.974 | 9.694867 | 9.50165 | 11.589970 | 103.147362 | 77.471986 | 1238.05 | 1562.5 | 14.642527 | 4.134981 | 0.000 |
| TFLite FP32 | TFLite Interpreter | FP32 | 20.18 | 0.836 | 0.974 | 53.554561 | 53.10985 | 57.301935 | 18.672546 | 17.773604 | 2113.09 | 100.8 | 2.650705 | 4.087452 | 0.000 |
| ONNX INT8 | ONNX Runtime | INT8 | 5.80 | 0.702 | 0.920 | 11.329606 | 11.35305 | 13.120610 | 88.264324 | 66.012372 | 2180.51 | 1354.2 | 12.529770 | 72.434021 | -0.134 |
| TFLite INT8 | TFLite Interpreter | INT8 | 5.90 | 0.714 | 0.920 | 134.371102 | 132.17990 | 146.385315 | 7.442076 | 7.304824 | 2643.02 | 88.7 | 1.041899 | 71.958175 | -0.122 |

## 6. Phân tích từng runtime

- TensorFlow FP32 là baseline và là bản chậm nhất trong run hiện tại.
- ONNX FP32 giữ nguyên accuracy, numerical equivalence gần tuyệt đối và là runtime nhanh nhất trong bộ đo này.
- TFLite FP32 giữ nguyên accuracy, nhanh hơn TensorFlow nhưng chậm hơn ONNX trên máy Windows CPU hiện tại.
- ONNX INT8 giảm mạnh model size nhưng không nhanh hơn ONNX FP32 trong run hiện tại; accuracy giảm từ 0.836 xuống 0.702.
- TFLite INT8 giảm mạnh model size nhưng chậm hơn TFLite FP32 trong run hiện tại; accuracy giảm xuống 0.714.

## 7. Phân tích INT8 accuracy degradation

INT8 không tự động tốt hơn FP32. Hiệu quả quantization phụ thuộc runtime, operator support, kernel CPU, calibration dataset và thread configuration. Với cấu hình hiện tại:

- ONNX INT8 giữ top-1 agreement 1.0 trên subset 10 ảnh nhưng accuracy full benchmark giảm xuống 0.702.
- TFLite INT8 cũng giữ top-1 agreement 1.0 trên subset 10 ảnh nhưng accuracy full benchmark giảm xuống 0.714.
- Điều này cho thấy subset validation không thay thế được benchmark đầy đủ.

## 8. Điểm bất thường cần ghi chú

- TensorFlow model-only latency lớn hơn end-to-end latency trong kết quả hiện tại. Hai benchmark được chạy độc lập nên có thể chịu ảnh hưởng warm-up, scheduling và caching; không cộng/trừ trực tiếp hai số này.
- `process CPU` có thể vượt 100% vì `psutil` tính theo nhiều logical processors.
- GPU utilization chỉ là tải nền; các benchmark chính chạy CPU.

## 9. Kết luận

- Chính xác nhất: ba bản FP32 bằng nhau về top-1/top-5 accuracy.
- Nhanh nhất: ONNX FP32.
- Nhỏ nhất: ONNX INT8.
- Trade-off tốt nhất: ONNX FP32.
- INT8 phù hợp khi ưu tiên size hơn accuracy, không nên mặc định kỳ vọng nhanh hơn.

## 10. Hạn chế

- Benchmark hiện tại chỉ phản ánh một máy Windows cụ thể.
- Kết quả INT8 rất nhạy với calibration set và runtime kernel.
- FPS và latency có thể thay đổi nếu thread config hoặc tải nền khác đi.
- GPU chỉ được dùng làm monitor nền, không được coi là phần benchmark chính.

## 11. Hướng cải tiến

- Mở rộng calibration set đa dạng hơn
- Thử entropy hoặc percentile calibration
- Thử per-channel quantization sâu hơn
- Loại trừ layer nhạy cảm khỏi quantization
- Cân nhắc Quantization-Aware Training nếu cần giữ accuracy
- Lặp benchmark nhiều lần để giảm nhiễu
- Đồng nhất cấu hình thread chặt hơn
- Benchmark trên thiết bị mobile thật nếu mục tiêu cuối là triển khai edge

## 12. Kết luận ngắn

Dựa trên dữ liệu hiện tại, lựa chọn tổng thể tốt nhất là ONNX FP32. INT8 chỉ phù hợp khi mục tiêu chính là giảm kích thước model và chấp nhận suy giảm accuracy.
