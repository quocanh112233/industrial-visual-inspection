# IVID — Industrial Visual Inspection & Deployment Benchmark

Phát hiện lỗi bề mặt thép cán nóng (NEU-DET, 6 lớp) bằng YOLO, rồi **đo hiệu năng
trên ba runtime** — PyTorch / ONNX Runtime / TensorRT FP16 — trên NVIDIA Jetson Orin Nano.

> 🚧 Đang phát triển. Bảng benchmark sẽ nằm ở đây khi có số liệu.

**Đã xong:** dữ liệu (FR-01→03) · kiểm chứng chuỗi export TensorRT trên Jetson (rủi ro R1)
**Đang làm:** train YOLOv8n/s · export + benchmark ba runtime

## Phần cứng đích

| | |
|---|---|
| Thiết bị | Jetson Orin Nano 8GB |
| JetPack | 6.2 (L4T R36.5.2) |
| CUDA / TensorRT | 12.6 / 10.3.0 |
| Python | 3.10.12 |

## Quick start

```bash
# --- trên Jetson ---
bash scripts/check_jetson.sh              # kiểm tra môi trường
bash scripts/setup_jetson.sh              # dựng venv (không phá torch CUDA)
bash scripts/smoke_tensorrt.sh            # ★ chạy NGAY: kiểm chứng rủi ro R1
bash scripts/download_dataset.sh          # tải NEU-DET
make data                                 # chuẩn bị + kiểm tra + thống kê
```

**Kết quả smoke test R1** (Jetson Orin Nano @15W, YOLOv8n pretrained, 640×640, chỉ inference thuần):

| ONNX opset | Engine | Latency p50 | FPS |
|---|---:|---:|---:|
| 12 | 9.0 MB | 7.13 ms | 140 |
| 17 | 9.0 MB | **6.25 ms** | **160** |

Chuỗi `.pt → .onnx → .engine (FP16)` chạy thông trên TensorRT 10.3 / CUDA 12.6 — rủi ro R1 đã loại bỏ.

Train: xem [docs/colab-training.md](docs/colab-training.md).

## Tài liệu

- [docs/colab-training.md](docs/colab-training.md) — train trên Colab (hoặc trên Jetson, Phụ lục B)
- [docs/dataset.md](docs/dataset.md) — phân bố lớp, hình dạng bbox, nhận xét
- `results/data_report.md` — báo cáo toàn vẹn dữ liệu (FR-02)
- `docs/benchmark-report.md` — ★ báo cáo so sánh ba runtime *(sắp có)*
- `docs/deployment.md` — dựng dịch vụ trên Jetson *(sắp có)*

## Dataset

NEU-DET — 1800 ảnh xám 200×200, 4189 bounding box, 6 loại lỗi bề mặt thép:
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.

## Chia dữ liệu

70/15/15 phân tầng theo lớp, seed `1337` — chạy lại cho kết quả giống hệt.

| Tập | Ảnh | Bbox | Mỗi lớp |
|---|---:|---:|---:|
| train | 1260 | 2950 | 210 |
| val | 270 | 624 | 45 |
| test | 270 | 615 | 45 |

Kiểm tra toàn vẹn: **0 bất thường** (không có ảnh thiếu nhãn, bbox vượt biên,
ảnh hỏng hay rò rỉ giữa các tập). 123/1800 ảnh chứa nhiều hơn một loại lỗi.
