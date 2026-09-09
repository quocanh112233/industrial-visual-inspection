# Tính lặp lại của quá trình train (FR-06)

FR-06 yêu cầu: *"Từ file cấu hình lưu lại, chạy lại cho kết quả chênh lệch mAP ≤ 0.02"*.
Dưới đây là phép thử thật, không phải tuyên bố.

## Thiết lập

Hai lần train YOLOv8n độc lập, **cùng seed 1337**, cùng dataset (hash `757d9bcc…`), cùng máy (Colab T4). Khác biệt duy nhất: `cache: ram` → `cache: disk`.
Dữ liệu train hiệu dụng giống hệt nhau — 3 bbox trùng lặp bị loại ở cả hai lần,
chỉ khác là lần 1 do ultralytics âm thầm loại, lần 2 do `ivid.data.prepare` loại.

## Kết quả trên tập test

| | Lần 1 | Lần 2 | Δ |
|---|---:|---:|---:|
| mAP@0.5 | 0.7749 | 0.7634 | **-0.0115** |
| mAP@0.5:0.95 | 0.4378 | 0.4352 | -0.0026 |
| precision | 0.6862 | 0.6770 | |
| recall | 0.7310 | 0.7445 | |
| thời gian train | 33.2 phút | 36.3 phút | +9% |

✅ **Đạt** — |Δ| = 0.0115 ≤ 0.02.

## mAP@0.5 theo từng lớp

| Lớp | Lần 1 | Lần 2 | Δ |
|---|---:|---:|---:|
| `inclusion` | 0.8220 | 0.7987 | -0.0233 |
| `pitted_surface` | 0.8381 | 0.8174 | -0.0207 |
| `crazing` | 0.4781 | 0.4576 | -0.0205 |
| `patches` | 0.9634 | 0.9555 | -0.0079 |
| `scratches` | 0.9434 | 0.9488 | +0.0054 |
| `rolled-in_scale` | 0.6044 | 0.6026 | -0.0018 |

## Điều quan trọng hơn con số tổng

Độ lệch tổng thể là 0.0115, nhưng **độ lệch theo từng lớp lên tới 0.0233** (`inclusion`) — gấp đôi. Sai số trung bình che mất dao động ở từng lớp.

Hệ quả cho phần benchmark, và đây là chỗ dễ kết luận sai:

- **So sánh YOLOv8n với YOLOv8s** là so sánh hai model được train riêng. Chênh lệch mAP nhỏ hơn **~0.02** nằm trong nhiễu của quá trình train và **không được tuyên bố là thật**.
- **So sánh PyTorch / ONNX / TensorRT (FR-13)** thì khác hẳn: cả ba dùng **cùng một file trọng số**. Không có nhiễu train ở đó, nên chênh lệch dù nhỏ vẫn có ý nghĩa — nó đến từ độ chính xác số học của runtime.

Nhập nhằng hai loại so sánh này sẽ dẫn tới kết luận sai theo cả hai hướng: hoặc thổi phồng khác biệt giữa hai model, hoặc bỏ qua khác biệt thật giữa các runtime.

## Ghi chú

Lần 2 được giữ làm bản chính thức, vì nó là bản mà `configs/train_yolov8n.yaml` hiện tại tái tạo được. Lần 1 giữ lại làm bằng chứng đối chứng.

Số liệu gốc: `results/fr06_reproducibility.json`, `results/train_yolov8n_manifest.json`.
