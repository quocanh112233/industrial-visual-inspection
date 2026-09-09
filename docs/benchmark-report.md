# Báo cáo benchmark — ba định dạng runtime trên Jetson Orin Nano

*Sinh tự động bởi `ivid.benchmark.report` từ `results/benchmark.json`. Số liệu đo lúc 2026-09-09T10:40:25+00:00. Mọi con số đều truy được về file JSON đó.*

## Điều kiện đo (FR-14)

| Hạng mục | Giá trị |
|---|---|
| Thiết bị | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| L4T / JetPack | R36 (release), REVISION: 5.2, GCID: 46426093, BOARD: generic, EABI: aarch64, DATE: Thu Jul 16 18:56:22 UTC 2026 / (none) |
| CUDA | Cuda compilation tools, release 12.6, V12.6.68 |
| TensorRT | 10.3.0 |
| PyTorch | 2.10.0 |
| ONNX Runtime | 1.24.0 |
| Chế độ nguồn | **15W** (id=0), các chế độ có sẵn: 15W, 25W, MAXN_SUPER |
| Nhiệt độ trung bình trước khi đo | 59.1 °C |
| Warm-up / số phiên | 20 lần / 3 phiên |
| Tập đo | 270 ảnh (test), cùng thứ tự cho mọi định dạng |

> Chế độ nguồn được cố định trong suốt phiên đo (ràng buộc C3). Đổi chế độ nguồn là đổi kết quả — mọi so sánh trong báo cáo này chỉ có giá trị ở chế độ ghi trên.

## Bảng kết quả chính

| Model | Runtime | mAP@0.5 | mAP@0.5:0.95 | p50 (ms) | p95 (ms) | FPS | Model size | Bộ nhớ tăng thêm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8n | PyTorch | 0.7317 | 0.3547 | 34.73 | 35.17 | 28.8 | 6.3 MB | 2 MB |
| yolov8n | ONNX Runtime | 0.7356 | 0.3549 | 18.69 | 18.93 | 53.5 | 12.3 MB | 6 MB |
| yolov8n | TensorRT FP16 | 0.7357 | 0.3553 | 14.61 | 14.82 | 68.4 | 9.0 MB | 2 MB |
| yolov8s | PyTorch | 0.7270 | 0.3335 | 42.94 | 43.26 | 23.3 | 22.5 MB | 1 MB |
| yolov8s | ONNX Runtime | 0.7271 | 0.3331 | 28.40 | 28.61 | 35.2 | 44.8 MB | 2 MB |
| yolov8s | TensorRT FP16 | 0.7270 | 0.3332 | 18.70 | 19.02 | 53.5 | 25.6 MB | 41 MB |

## Thời gian đi đâu (FR-11)

| Model | Runtime | Tiền xử lý | Inference | Hậu xử lý (NMS) | Tổng | Inference chiếm |
|---|---|---:|---:|---:|---:|---:|
| yolov8n | PyTorch | 5.62 | 28.00 | 1.08 | 34.73 | 81% |
| yolov8n | ONNX Runtime | 5.68 | 11.88 | 1.11 | 18.69 | 64% |
| yolov8n | TensorRT FP16 | 5.76 | 7.58 | 1.25 | 14.61 | 52% |
| yolov8s | PyTorch | 5.84 | 35.75 | 1.35 | 42.94 | 83% |
| yolov8s | ONNX Runtime | 5.68 | 21.58 | 1.11 | 28.40 | 76% |
| yolov8s | TensorRT FP16 | 5.77 | 11.67 | 1.26 | 18.70 | 62% |

> Tiền xử lý và hậu xử lý dùng **cùng một hàm NumPy** cho cả ba định dạng (`ivid.preprocess`, `ivid.postprocess`). Nhờ vậy chênh lệch giữa các dòng chỉ đến từ bước inference — đúng mục đích của bảng này.

## Tài nguyên (FR-12)

| Model | Runtime | Model size | RSS đỉnh | RAM hệ thống tăng thêm | Bộ cấp phát của torch |
|---|---|---:|---:|---:|---:|
| yolov8n | PyTorch | 6.3 MB | 1132 MB | 2 MB | 37 MB |
| yolov8n | ONNX Runtime | 12.3 MB | 2675 MB | 6 MB | 0 MB |
| yolov8n | TensorRT FP16 | 9.0 MB | 2257 MB | 2 MB | 6 MB |
| yolov8s | PyTorch | 22.5 MB | 2299 MB | 1 MB | 89 MB |
| yolov8s | ONNX Runtime | 44.8 MB | 3137 MB | 2 MB | 0 MB |
| yolov8s | TensorRT FP16 | 25.6 MB | 2676 MB | 41 MB | 6 MB |

> **Đọc hai cột cuối thế nào.** Jetson dùng *bộ nhớ hợp nhất*: CPU và GPU chia nhau cùng 8 GB DRAM, không có VRAM rời. Vì vậy **RAM hệ thống tăng thêm** mới là con số phản ánh chi phí bộ nhớ thật của mỗi runtime.
>
> Cột cuối chỉ đếm phần do **chính PyTorch** cấp phát. Với ONNX Runtime nó gần bằng 0 vì ORT tự quản lý bộ nhớ GPU; với TensorRT nó chỉ đếm buffer vào/ra chứ không đếm bộ nhớ của engine. Cột này để chẩn đoán, **không dùng để so sánh giữa các runtime**.

## Tính lặp lại (NFR-02)

| Model | Runtime | Độ lệch p50 giữa các phiên | Đạt ≤ 10%? |
|---|---|---:|:---:|
| yolov8n | PyTorch | 0.12% | ✅ |
| yolov8n | ONNX Runtime | 0.66% | ✅ |
| yolov8n | TensorRT FP16 | 0.24% | ✅ |
| yolov8s | PyTorch | 0.02% | ✅ |
| yolov8s | ONNX Runtime | 0.05% | ✅ |
| yolov8s | TensorRT FP16 | 0.09% | ✅ |

## Biểu đồ

![latency_comparison.png](images/latency_comparison.png)

![latency_breakdown.png](images/latency_breakdown.png)

![map_vs_latency.png](images/map_vs_latency.png)

## Kết luận (SRS §7.3)

### 1. TensorRT FP16 nhanh hơn PyTorch bao nhiêu, đổi lấy bao nhiêu mAP?

- **yolov8n**: nhanh hơn **2.38×** (34.73 ms → 14.61 ms), mAP@0.5 0.7317 → 0.7357 (**+0.0040**, +0.5%) — gần như không mất độ chính xác
- **yolov8s**: nhanh hơn **2.30×** (42.94 ms → 18.70 ms), mAP@0.5 0.7270 → 0.7270 (**+0.0000**, +0.0%) — gần như không mất độ chính xác

### 2. Với nhịp dây chuyền 200 ms/sản phẩm, cấu hình nào đáp ứng?

Dùng **p95** chứ không phải p50: dây chuyền hỏng vì trường hợp chậm nhất, không phải vì trường hợp trung bình.

| Cấu hình | p95 (ms) | Đáp ứng? | Nhịp nhanh nhất chịu được |
|---|---:|:---:|---:|
| yolov8n / TensorRT FP16 | 14.82 | ✅ | 67.5 sp/giây (15 ms) |
| yolov8n / ONNX Runtime | 18.93 | ✅ | 52.8 sp/giây (19 ms) |
| yolov8s / TensorRT FP16 | 19.02 | ✅ | 52.6 sp/giây (19 ms) |
| yolov8s / ONNX Runtime | 28.61 | ✅ | 35.0 sp/giây (29 ms) |
| yolov8n / PyTorch | 35.17 | ✅ | 28.4 sp/giây (35 ms) |
| yolov8s / PyTorch | 43.26 | ✅ | 23.1 sp/giây (43 ms) |

**Khuyến nghị: `yolov8n / TensorRT FP16`** — mAP@0.5 0.7357, p95 14.82 ms, còn dư **185 ms** biên an toàn cho chụp ảnh, truyền dữ liệu và dao động tải.
  Các cấu hình `yolov8n / PyTorch`, `yolov8n / ONNX Runtime` có mAP chênh dưới 0.005 so với cấu hình tốt nhất nhưng chậm hơn, nên không được chọn.

### 3. YOLOv8s có đáng đổi độ trễ lấy mAP không?

- Trên TensorRT FP16: YOLOv8s chậm hơn **+4.09 ms** (+28%) và cho mAP@0.5 **-0.0087**.
- **Không đáng.** Chênh lệch mAP quá nhỏ so với chi phí độ trễ.

## mAP theo lớp

| Cấu hình | `crazing` | `inclusion` | `patches` | `pitted_surface` | `rolled-in_scale` | `scratches` |
|---|---:|---:|---:|---:|---:|---:|
| yolov8n / PyTorch | 0.464 | 0.735 | 0.952 | 0.806 | 0.599 | 0.835 |
| yolov8n / ONNX Runtime | 0.463 | 0.734 | 0.952 | 0.804 | 0.599 | 0.862 |
| yolov8n / TensorRT FP16 | 0.462 | 0.734 | 0.952 | 0.805 | 0.599 | 0.863 |
| yolov8s / PyTorch | 0.416 | 0.718 | 0.939 | 0.837 | 0.588 | 0.864 |
| yolov8s / ONNX Runtime | 0.417 | 0.716 | 0.939 | 0.839 | 0.588 | 0.864 |
| yolov8s / TensorRT FP16 | 0.417 | 0.717 | 0.939 | 0.838 | 0.587 | 0.864 |

> Lớp nào thấp hẳn thì xem `docs/dataset.md`: `pitted_surface` có khung phủ hơn nửa ảnh và `crazing` là dạng vết nứt lan toả không có biên rõ — mAP thấp ở các lớp này là đặc tính dataset, không phải lỗi pipeline.

