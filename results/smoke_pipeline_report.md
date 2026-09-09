# Báo cáo benchmark — ba định dạng runtime trên Jetson Orin Nano

*Sinh tự động bởi `ivid.benchmark.report` lúc 2026-09-09T05:03:13+00:00. Mọi con số đọc từ `results/benchmark.json`.*

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
| Nhiệt độ trung bình trước khi đo | 59.0 °C |
| Warm-up / số phiên | 20 lần / 1 phiên |
| Tập đo | 30 ảnh (test), cùng thứ tự cho mọi định dạng |

> Chế độ nguồn được cố định trong suốt phiên đo (ràng buộc C3). Đổi chế độ nguồn là đổi kết quả — mọi so sánh trong báo cáo này chỉ có giá trị ở chế độ ghi trên.

## Bảng kết quả chính

| Model | Runtime | mAP@0.5 | mAP@0.5:0.95 | p50 (ms) | p95 (ms) | FPS | Model size | Bộ nhớ tăng thêm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| _smoke | PyTorch | — | — | 36.99 | 37.51 | 27.0 | 6.5 MB | 0 MB |
| _smoke | ONNX Runtime | — | — | 21.67 | 21.73 | 46.2 | 12.8 MB | 0 MB |
| _smoke | TensorRT FP16 | — | — | 17.34 | 17.45 | 57.7 | 9.4 MB | 0 MB |

## Thời gian đi đâu (FR-11)

| Model | Runtime | Tiền xử lý | Inference | Hậu xử lý (NMS) | Tổng | Inference chiếm |
|---|---|---:|---:|---:|---:|---:|
| _smoke | PyTorch | 5.68 | 28.90 | 2.33 | 36.99 | 78% |
| _smoke | ONNX Runtime | 5.71 | 13.35 | 2.59 | 21.67 | 62% |
| _smoke | TensorRT FP16 | 5.82 | 8.79 | 2.75 | 17.34 | 51% |

> Tiền xử lý và hậu xử lý dùng **cùng một hàm NumPy** cho cả ba định dạng (`ivid.preprocess`, `ivid.postprocess`). Nhờ vậy chênh lệch giữa các dòng chỉ đến từ bước inference — đúng mục đích của bảng này.

## Tài nguyên (FR-12)

| Model | Runtime | Model size | RSS đỉnh | RAM hệ thống tăng thêm | Bộ cấp phát của torch |
|---|---|---:|---:|---:|---:|
| _smoke | PyTorch | 6.5 MB | 1132 MB | 0 MB | 38 MB |
| _smoke | ONNX Runtime | 12.8 MB | 2709 MB | 0 MB | 0 MB |
| _smoke | TensorRT FP16 | 9.4 MB | 2304 MB | 0 MB | 11 MB |

> **Đọc hai cột cuối thế nào.** Jetson dùng *bộ nhớ hợp nhất*: CPU và GPU chia nhau cùng 8 GB DRAM, không có VRAM rời. Vì vậy **RAM hệ thống tăng thêm** mới là con số phản ánh chi phí bộ nhớ thật của mỗi runtime.
>
> Cột cuối chỉ đếm phần do **chính PyTorch** cấp phát. Với ONNX Runtime nó gần bằng 0 vì ORT tự quản lý bộ nhớ GPU; với TensorRT nó chỉ đếm buffer vào/ra chứ không đếm bộ nhớ của engine. Cột này để chẩn đoán, **không dùng để so sánh giữa các runtime**.

## Tính lặp lại (NFR-02)

| Model | Runtime | Độ lệch p50 giữa các phiên | Đạt ≤ 10%? |
|---|---|---:|:---:|
| _smoke | PyTorch | 0.00% | ✅ |
| _smoke | ONNX Runtime | 0.00% | ✅ |
| _smoke | TensorRT FP16 | 0.00% | ✅ |

## Biểu đồ

![latency_comparison.png](images/latency_comparison.png)

![latency_breakdown.png](images/latency_breakdown.png)

## Kết luận (SRS §7.3)

### 1. TensorRT FP16 nhanh hơn PyTorch bao nhiêu, đổi lấy bao nhiêu mAP?

- **_smoke**: nhanh hơn **2.13×** (36.99 ms → 17.34 ms), *(chưa có số mAP)*

### 2. Với nhịp dây chuyền 200 ms/sản phẩm, cấu hình nào đáp ứng?

Dùng **p95** chứ không phải p50: dây chuyền hỏng vì trường hợp chậm nhất, không phải vì trường hợp trung bình.

| Cấu hình | p95 (ms) | Đáp ứng? | Nhịp nhanh nhất chịu được |
|---|---:|:---:|---:|
| _smoke / TensorRT FP16 | 17.45 | ✅ | 57.3 sp/giây (17 ms) |
| _smoke / ONNX Runtime | 21.73 | ✅ | 46.0 sp/giây (22 ms) |
| _smoke / PyTorch | 37.51 | ✅ | 26.7 sp/giây (38 ms) |

**Khuyến nghị: `_smoke / TensorRT FP16`** — mAP@0.5 —, p95 17.45 ms, còn dư **183 ms** biên an toàn cho chụp ảnh, truyền dữ liệu và dao động tải.
  Các cấu hình `_smoke / PyTorch`, `_smoke / ONNX Runtime` có mAP chênh dưới 0.005 so với cấu hình tốt nhất nhưng chậm hơn, nên không được chọn.

### 3. YOLOv8s có đáng đổi độ trễ lấy mAP không?

*(cần cả yolov8n và yolov8s ở định dạng TensorRT)*
## Phần còn thiếu

- `_smoke / PyTorch`: mAP (chua do)
- `_smoke / ONNX Runtime`: mAP (chua do)
- `_smoke / TensorRT FP16`: mAP (chua do)

