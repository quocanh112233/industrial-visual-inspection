# IVID — Industrial Visual Inspection & Deployment Benchmark

Phát hiện lỗi bề mặt thép cán nóng (NEU-DET, 6 lớp) bằng YOLO, rồi **đo hiệu năng
trên ba runtime** — PyTorch / ONNX Runtime / TensorRT FP16 — trên NVIDIA Jetson Orin Nano.

> 🚧 Đang phát triển. Bảng benchmark sẽ nằm ở đây khi có số liệu.

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
```

Train: xem [docs/colab-training.md](docs/colab-training.md).

## Tài liệu

- [docs/colab-training.md](docs/colab-training.md) — train trên Colab (hoặc trên Jetson, Phụ lục B)
- `docs/benchmark-report.md` — ★ báo cáo so sánh ba runtime *(sắp có)*
- `docs/dataset.md` — mô tả NEU-DET, phân bố lớp *(sắp có)*
- `docs/deployment.md` — dựng dịch vụ trên Jetson *(sắp có)*

## Dataset

NEU-DET — 1800 ảnh xám 200×200, 4189 bounding box, 6 loại lỗi bề mặt thép:
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
