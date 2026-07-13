# CONTEXT.md

# Bối cảnh dự án

## 1. Tóm tắt đề bài

Chọn một mô hình Deep Learning và thực hiện phân loại trên 500 ảnh sample, khuyến khích dùng ảnh ImageNet. Sau đó:

- tối ưu/chuyển sang ONNX
- tối ưu/chuyển sang TFLite
- đề xuất thêm một phương pháp tối ưu
- đo FPS, RAM, CPU, GPU, model size và các chỉ số liên quan ở mọi bước, kể cả baseline

Yêu cầu triển khai của người dùng:

- Phân tích trước, chưa viết code.
- Chốt kiến trúc.
- Toàn bộ code sau này nằm trong một notebook.
- Tạo `TASK.md`, `PLAN.md`, `CONTEXT.md`.

---

## 2. Thông tin máy

Thông tin đọc từ ảnh hệ thống:

- Device: Acer Nitro AN515-58
- CPU: 12th Gen Intel Core i5-12500H, 3.10 GHz được hiển thị
- Installed RAM: 24.0 GB
- GPU memory được giao diện hiển thị: 4 GB
- Multiple GPUs installed
- Storage: 943 GB, đã dùng khoảng 660 GB
- System: 64-bit Windows, x64 processor

### Chưa xác định

- Model GPU rời chính xác
- NVIDIA driver/CUDA availability
- Phiên bản Windows
- Phiên bản Python
- Có WSL2 hay không
- Tình trạng TensorFlow/ONNX Runtime hiện tại

Không được giả định GPU là RTX 3050 chỉ dựa vào dòng laptop.

---

## 3. Quyết định kiến trúc hiện tại

### Model

**EfficientNetB0 pretrained ImageNet**

Lý do:

- Cân bằng accuracy, model size và latency.
- Kích thước nhỏ hơn ResNet50/InceptionV3.
- Input 224 x 224.
- Phù hợp CPU i5-12500H, RAM 24 GB và VRAM 4 GB.
- Có trong Keras Applications.
- Hỗ trợ đường chuyển TensorFlow → TFLite trực tiếp.
- Có thể chuyển TensorFlow/Keras → ONNX bằng tf2onnx.

### Framework gốc

**TensorFlow/Keras**

Lý do chính: TFLite là format native của TensorFlow; ONNX được tạo qua tf2onnx. Cách này giảm số bước trung gian so với lấy PyTorch làm model nguồn rồi cố chuyển tiếp sang TFLite.

### Benchmark device

**CPU-first**

Mọi phiên bản chính được benchmark trên CPU để đảm bảo so sánh cùng phần cứng:

- TensorFlow FP32
- ONNX FP32
- TFLite FP32
- ONNX INT8
- TFLite INT8

GPU là phần tùy chọn sau khi xác minh thiết bị và provider.

### Tối ưu bổ sung

**Static INT8 Post-Training Quantization**

Áp dụng cho:

- ONNX INT8
- TFLite INT8

QAT không được chọn làm mặc định vì dự án dùng model pretrained và chỉ có tập sample nhỏ; QAT làm tăng đáng kể phạm vi triển khai.

---

## 4. Các giả định đang dùng

- Người dùng muốn inference, không muốn training từ đầu.
- 500 ảnh dùng để benchmark và classify.
- Accuracy nên được đo nếu có ground truth.
- Top-1 agreement được dùng thêm để kiểm tra conversion.
- Batch size chính là 1.
- FPS chính được tính từ model-only inference.
- End-to-end FPS được báo cáo riêng.
- Model size là kích thước artifact trên disk của từng phiên bản.
- Các runtime dùng cùng preprocessing logic.
- Dữ liệu calibration có cùng phân phối với dữ liệu benchmark.
- Notebook được chạy khi máy cắm sạc và không có workload nặng khác.

---

## 5. Điểm chưa rõ

### Dataset

- Có quyền truy cập ImageNet validation hay không?
- 500 ảnh thuộc:
  - 500 class khác nhau
  - 500 ảnh random
  - phân bố đều theo class
  - subset tùy ý
- Có ground truth class index chuẩn không?
- Giáo viên có chấp nhận Imagenette hoặc dataset khác không?

### Benchmark

- FPS là:
  - chỉ model inference
  - hay gồm đọc ảnh và preprocessing
- GPU usage bắt buộc phải khác 0 hay chỉ cần ghi nhận?
- Có bắt buộc benchmark cả CPU và GPU không?
- Có quy định batch size không?
- Có cần chạy nhiều lần và lấy trung bình không?

### Model artifact

- Baseline size tính theo `.keras`, SavedModel hay tổng thư mục?
- ONNX/TFLite có cần metadata/labels nhúng vào file không?

### Báo cáo

- Có yêu cầu accuracy/top-5 hay chỉ classify?
- Có giới hạn accuracy drop sau tối ưu không?
- Có yêu cầu biểu đồ không?
- Có yêu cầu power consumption không?

---

## 6. Rủi ro kỹ thuật chính

### R-01 — GPU không rõ model

Ảnh chỉ hiển thị GPU memory 4 GB và multiple GPUs. Không đủ để chọn CUDA hoặc DirectML.

**Hành động:** xác minh bằng `nvidia-smi`, Device Manager hoặc công cụ hệ thống.

### R-02 — TensorFlow GPU trên Windows

Các bản TensorFlow mới không dùng native-Windows CUDA như trước. GPU TensorFlow thường cần WSL2; điều này có thể làm notebook phức tạp hơn.

**Quyết định:** hoàn thành dự án trên CPU trước.

### R-03 — tf2onnx compatibility

tf2onnx chỉ test một số dải TensorFlow/Python/opset. Dùng package quá mới có thể lỗi conversion.

**Hành động:** test export ở Milestone 0 và pin versions ngay khi thành công.

### R-04 — TFLite INT8

Full INT8 cần representative dataset và có thể gặp operator/dtype compatibility.

**Hành động:** chuẩn bị fallback dynamic range quantization.

### R-05 — So sánh không công bằng

TensorFlow, ONNX Runtime và TFLite có thread scheduler khác nhau.

**Hành động:** thiết lập thread count rõ ràng, batch size 1, cùng warm-up và cùng dataset.

### R-06 — Resource monitoring sai nghĩa

CPU percent có thể khác cách tính theo process hoặc toàn hệ thống; GPU usage có thể phản ánh desktop rendering chứ không phải model.

**Hành động:** ghi rõ metric source và tách process/system.

### R-07 — Accuracy không đáng tin với sample nhỏ

500 ảnh chỉ là sample nhỏ so với ImageNet validation.

**Hành động:** gọi kết quả là “accuracy trên subset 500 ảnh”, không khái quát thành accuracy ImageNet đầy đủ.

---

## 7. Bộ biến thể phải có

| ID | Model/runtime | Precision | Thiết bị chính |
|---|---|---|---|
| M0 | TensorFlow/Keras baseline | FP32 | CPU |
| M1 | ONNX Runtime | FP32 | CPU |
| M2 | TFLite Interpreter | FP32 | CPU |
| M3 | ONNX Runtime quantized | INT8 | CPU |
| M4 | TFLite Interpreter quantized | INT8 | CPU |

### Biến thể bonus

| ID | Model/runtime | Mục đích |
|---|---|---|
| B1 | ONNX Runtime OpenVINO EP | Tận dụng Intel CPU |
| B2 | ONNX Runtime CUDA EP | GPU benchmark nếu NVIDIA |
| B3 | TensorFlow GPU trong WSL2 | GPU appendix |

Các biến thể bonus chỉ làm sau khi M0–M4 hoàn thành.

---

## 8. Schema metric thống nhất

```text
run_id
model_id
runtime
precision
device
provider
batch_size
thread_count
num_images
warmup_runs
model_size_mb
load_time_s
total_model_only_time_s
total_end_to_end_time_s
mean_latency_ms
median_latency_ms
p95_latency_ms
fps_model_only
fps_end_to_end
top1_accuracy
top5_accuracy
top1_agreement
max_abs_output_diff
mean_abs_output_diff
ram_avg_mb
ram_peak_mb
cpu_process_avg_pct
cpu_process_peak_pct
cpu_system_avg_pct
cpu_system_peak_pct
gpu_avg_pct
gpu_peak_pct
vram_avg_mb
vram_peak_mb
speedup_vs_baseline
size_reduction_pct
accuracy_delta
notes
```

---

## 9. Quy tắc quyết định model tốt nhất

Không chọn model chỉ dựa trên FPS.

Đánh giá theo thứ tự:

1. Conversion/inference đúng.
2. Accuracy drop trong ngưỡng chấp nhận.
3. Latency/FPS.
4. Peak RAM.
5. Model size.
6. Độ ổn định p95.
7. Khả năng triển khai.

### Kết luận dự kiến cần viết theo mẫu

- Nhanh nhất: ...
- Nhỏ nhất: ...
- Chính xác nhất: ...
- Tiết kiệm RAM nhất: ...
- Trade-off tốt nhất trên máy thử nghiệm: ...
- Hạn chế: ...

Không điền kết quả trước khi benchmark thực tế.

---

## 10. Trạng thái hiện tại

### Đã hoàn thành

- Phân tích đề bài.
- Chọn EfficientNetB0.
- Chọn TensorFlow/Keras làm source framework.
- Chọn CPU-first benchmark.
- Chọn static INT8 PTQ làm tối ưu bổ sung.
- Lập milestone.
- Lập cấu trúc file/folder.
- Định nghĩa metric và acceptance criteria.

### Chưa thực hiện

- Chưa viết code.
- Chưa tạo notebook.
- Chưa tải dataset.
- Chưa xác minh GPU.
- Chưa chốt package versions.
- Chưa benchmark.
- Chưa có số liệu thực nghiệm.

### Bước tiếp theo

Chốt 5 thông tin:

1. GPU chính xác.
2. Nguồn dataset.
3. Ground truth labels.
4. Có thêm calibration set hay không.
5. Chỉ CPU hay thêm GPU appendix.

## Context Delta — Milestone 0

- Date/time: 2026-07-13 (Asia/Ho_Chi_Minh)
- Files created: `.venv-slot17/`, `requirements.in`, `requirements-lock.txt`, `reports/environment.json`
- Files modified: `CONTEXT.md` (append only)
- Python interpreter: Python 3.11.9
- Venv path: `D:\DAT301m\slot17\.venv-slot17`
- Jupyter kernel: ID `slot17`, display name `Python (slot17)`, correct project interpreter
- Package versions: pinned direct dependencies installed; actual transitive versions recorded in `requirements-lock.txt`
- Smoke-test results: pip check PASS; TensorFlow import/CPU PASS; EfficientNetB0 construction and forward PASS; ONNX conversion/checker/CPU inference PASS; TFLite conversion/inference PASS; NVML PASS
- ONNX Runtime providers: `AzureExecutionProvider`, `CPUExecutionProvider`; smoke test explicitly used CPUExecutionProvider
- TensorFlow devices: CPU only; CUDA build false; no GPU configuration performed
- Decisions confirmed: native Windows CPU-only benchmark; Python 3.11.9; `.venv-slot17`; model-size baseline is `.keras`/`.onnx`/`.tflite`; no pretrained weights or dataset downloaded in Milestone 0
- Blockers: source of 500 images, ImageNet ground-truth class indices, independent calibration dataset, dataset access rights
- Next milestone: Milestone 1 only, after dataset decisions are confirmed

## Context Delta — Milestone 1

- Date/time: 2026-07-13 (Asia/Ho_Chi_Minh)
- Files created: `notebooks/efficientnet_onnx_tflite_benchmark.ipynb`
- Files modified: `CONTEXT.md` (append only)
- Python interpreter: `.venv-slot17\Scripts\python.exe` / Python 3.11.9
- Virtual environment: `D:\DAT301m\slot17\.venv-slot17`
- Jupyter kernel: `slot17` / `Python (slot17)` / `D:\DAT301m\slot17\.venv-slot17\Scripts\python.exe`
- Package versions: pinned versions from `requirements.in` loaded successfully in the notebook environment
- Notebook execution result: PASS; executed end-to-end; final status `BLOCKED_ON_DATASET`
- Dataset/manifest status: `data/raw/imagenet_val` missing, `data/metadata/validation_labels.txt` missing, `data/labels/imagenet_class_index.json` missing; `sample_500.csv` and `calibration_100.csv` not generated; no fake rows or artifacts
- Utility smoke-test results: environment validation PASS; dataset validation BLOCKED_ON_DATASET; shared preprocessing PASS; dataset cache strategy PASS; resource monitor PASS; timing/benchmark harness PASS; empty result schema PASS
- Blockers: dataset source, validation labels, class index mapping, and manifest inputs are still unavailable
- Next milestone: Milestone 2 only after dataset inputs are provided and manifests can be generated
## Context Delta ? Dataset Milestone 1

- Date/time: 2026-07-13 (Asia/Ho_Chi_Minh)
- Dataset: Imagenette 320px ch?nh th?c c?a fast.ai, archive `data/raw/imagenette2-320.tgz`, gi?i n?n t?i `data/raw/imagenette2-320`
- Dataset structure: `train/` v? `val/`, 10 synset folders; ground truth suy ra t? t?n folder
- ImageNet mapping: `data/labels/imagenet_class_index.json` t? ngu?n TensorFlow/Keras storage ch?nh th?c; mapping 10 synset sang output index 0?999 ?? x?c minh
- Manifests: `data/manifests/sample_500.csv` g?m 500 ?nh validation, 50 ?nh/l?p; `data/manifests/calibration_100.csv` g?m 100 ?nh train, 10 ?nh/l?p; seed 42; kh?ng overlap
- Manifest validation: PASS; ?? c?t, SHA-256, c?n b?ng l?p v? 500/100 rows
- Notebook modified: h? tr? Imagenette, b? b?t bu?c `validation_labels.txt`, ki?m tra manifest v? preprocessing tr?n ?nh th?t
- Notebook execution: PASS b?ng kernel `slot17`; kh?ng load pretrained weights, kh?ng baseline, kh?ng conversion, kh?ng INT8
- Utility smoke tests: environment validation PASS; dataset validation PASS; manifest validation PASS; shared preprocessing PASS; cache strategy PASS; resource monitor PASS; benchmark harness PASS; empty result schema PASS
- Status: `READY_FOR_MILESTONE_2`
- Blockers: kh?ng c?n blocker dataset trong Milestone 1
- Next milestone: Milestone 2 ? Baseline EfficientNetB0; ch? b?t ??u khi ???c y?u c?u

## Context Delta - Repository Management

- Date/time: 2026-07-13 (Asia/Ho_Chi_Minh)
- `.gitignore` da duoc cap nhat de bo qua `.venv-slot17/`, `.codex/`, `.agents/`, `data/raw/`, file tam, archive, model artifacts, va `reports/environment.json`
- Notebook da duoc kiem tra lai va phat hien loi font Tieng Viet trong cell Markdown dau; can chuan hoa lai UTF-8 truoc khi tiep tuc neu muon giu noi dung giai thich tieng Viet trong notebook
- Repository da la Git repository hop le; khong can `git init`
- Trang thai tiep theo: kiem tra `git status --short`, dam bao `data/manifests/*.csv` va `data/labels/imagenet_class_index.json` duoc track neu can, va khong track `data/raw/`/venv/model artifacts

## Context Delta - Milestone 2

- Date/time: 2026-07-13 (Asia/Ho_Chi_Minh)
- Files created: `models/tensorflow/efficientnetb0_fp32.keras`, `results/predictions_tensorflow.csv`, `results/benchmark_results.csv`
- Files modified: `notebooks/efficientnet_onnx_tflite_benchmark.ipynb`, `CONTEXT.md`
- Model path: `D:\DAT301m\slot17\models\tensorflow\efficientnetb0_fp32.keras`
- Model size: 21.04 MB
- Dataset: 500/500 anh tu `data/manifests/sample_500.csv`; calibration 100 anh khong dung trong milestone nay
- Baseline results: top-1 accuracy 0.836, top-5 accuracy 0.974
- Latency: model-only mean 203.353 ms, median 140.290 ms, p95 376.778 ms; end-to-end mean 142.159 ms, median 140.236 ms, p95 157.491 ms
- FPS: model-only 4.918; end-to-end 7.034
- Resource monitor: RAM peak 976.73 MB; process CPU peak 100.0%; system CPU peak 33.1%; GPU/VRAM chi la background monitor, khong ket luan TensorFlow dung GPU
- Notebook execution: PASS bang kernel `slot17`
- Status: `READY_FOR_MILESTONE_3`
- Blockers: khong co blocker moi cho Milestone 2
- Next milestone: Milestone 3, ONNX FP32, chi khi duoc yeu cau ro rang

## Context Delta - Milestone 3

- Date/time: 2026-07-13 (Asia/Ho_Chi_Minh)
- Files created: `models/onnx/efficientnetb0_fp32.onnx`, `results/predictions_onnx.csv`
- Files modified: `notebooks/efficientnet_onnx_tflite_benchmark.ipynb`, `results/benchmark_results.csv`, `CONTEXT.md`
- ONNX opset: 13
- Provider: `CPUExecutionProvider`
- Conversion/checker status: conversion PASS; `onnx.load` PASS; `onnx.checker.check_model` PASS
- Numerical equivalence: subset max abs diff `1.3709068298339844e-06`; subset mean abs diff `1.0633635555423738e-09`; subset top-1 agreement `1.0`; subset top-5 agreement `1.0`
- Accuracy: top-1 `0.836`; top-5 `0.974`; top-1 agreement with TensorFlow `1.0`
- Latency/FPS: model-only mean `10.342 ms`, median `10.104 ms`, p95 `12.586 ms`, FPS `96.692`; end-to-end mean `14.661 ms`, median `13.937 ms`, p95 `19.126 ms`, FPS `68.207`
- RAM/CPU: RAM peak `1252.55 MB`; process CPU peak `1666.7%`; system CPU peak `100.0%`; GPU/VRAM chi la background monitor, khong ket luan ONNX dung GPU
- Size reduction: `4.134981%` so voi `.keras`
- Speedup: `21.964587x` so voi TensorFlow baseline
- Notebook output status: saved in-place; code cells `13/13` executed; `13/13` code cells co output; cell cuoi hien `READY_FOR_MILESTONE_4`
- Blockers: khong co blocker; benchmark va validation deu PASS
- Next milestone: Milestone 4, TFLite FP32, chi khi duoc yeu cau ro rang

