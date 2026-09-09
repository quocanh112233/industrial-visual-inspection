# Khung ảnh đầu vào: chỗ 0.085 mAP bị bỏ quên

> **Kết luận trước.** Thu ảnh về 640 rồi đệm xám ra khung **672×672** — thay vì cho ảnh
> phủ kín khung 640×640 — làm tăng **+0.085 mAP@0.5:0.95** trên YOLOv8n và **+0.104**
> trên YOLOv8s, đổi lấy **6%** độ trễ. Không sửa model, không train lại, không đổi dữ
> liệu. Chỉ đổi một con số trong tiền xử lý.

## Nó lộ ra thế nào

Không phải do đi tìm. FR-13 có một bước đối chiếu: tự tính mAP rồi so với
`ultralytics.val()` trên cùng bản `.pt`, để chắc phép tính của mình không sai. Bước
đó báo lệch **0.0304** — quá lớn để bỏ qua.

Ba khả năng, loại dần bằng đo đạc:

| Nghi vấn | Cách kiểm | Kết quả |
|---|---|---|
| Phép tính mAP sai | Chấm *cùng một tập detection* bằng chính mã ultralytics (`match_predictions` + `ap_per_class`) — `scripts/diag_map.py` | Lệch 0.0004. **Phép tính đúng.** |
| Tiền xử lý phóng ảnh sai thang | Chạy lại với `scaleup=False` | mAP sập còn 0.5145. **Ngược lại hoàn toàn** — giả thuyết sai. |
| Khác chế độ khung ảnh | `scripts/diag_rect.py`: gọi `val()` ba lần, chỉ đổi `rect` | **Đúng.** |

Chỗ chốt nằm trong mã nguồn ultralytics: `engine/validator.py` đặt `rect=True` cho bản
`.pt` nhưng ép `rect=False` cho mọi định dạng xuất. Ở chế độ rect với `pad=0.5`, khung
ảnh không phải 640 mà là

```
ceil(640 / 32 + 0.5) × 32 = 21 × 32 = 672
```

Ba lần chạy trên **cùng một bản `.pt`**, chỉ đổi tham số của trình đánh giá:

| Cấu hình | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| `rect=True` imgsz=640 → khung 672 | 0.7621 | 0.4348 |
| `rect=False` imgsz=640 | 0.7310 | 0.3521 |
| `rect=False` imgsz=672 | 0.7279 | 0.3410 |

Dòng thứ ba quan trọng: phóng ảnh **thẳng** lên 672 lại *kém hơn*. Vậy thứ tạo ra
khoảng cách không phải số pixel, mà là **viền xám**.

## Quét tham số

672 là con số ultralytics tình cờ dùng, không ai kiểm nó có tối ưu không. Quét trên
YOLOv8n / PyTorch, tập test 270 ảnh, chỉ đổi kích thước khung (`results/framing_*.json`):

| khung | mAP@0.5 | mAP@0.5:0.95 | pixel so với 640 |
|---:|---:|---:|---:|
| 640 *(không đệm)* | 0.7317 | 0.3547 | 1.00× |
| **672** | 0.7648 | 0.4401 | **1.10×** |
| 704 | 0.7730 | 0.4421 | 1.21× |
| 736 | 0.7691 | 0.4410 | 1.32× |
| 768 | 0.7726 | 0.4403 | 1.44× |

Toàn bộ bước nhảy nằm ở **640 → 672**. Từ 672 trở đi, mAP@0.5 dao động trong 0.0082 và
mAP@0.5:0.95 trong 0.0020 — đều dưới ngưỡng nhiễu **0.0115** đo được từ hai lần train
cùng seed (`reproducibility.md`), và không có xu hướng tăng.

**Hiệu ứng nhị phân, không tỉ lệ: cứ có viền là được, thêm viền không được thêm gì.**

Vì vậy chọn **672** — khung nhỏ nhất bắt được hiệu ứng, rẻ nhất về tính toán. Chọn 704
vì nó "cao nhất trong bảng" sẽ là bịa tín hiệu từ nhiễu.

## Kết quả trên phần cứng thật

Đo lại toàn bộ sau khi export ONNX ở 672 và dựng lại engine TensorRT (Jetson Orin Nano,
15W, 270 ảnh × 3 phiên):

| Cấu hình | mAP@0.5 | mAP@0.5:0.95 | p50 |
|---|---:|---:|---:|
| YOLOv8n / TensorRT @ 640 | 0.7357 | 0.3553 | 14.19 ms |
| YOLOv8n / TensorRT @ 672/640 | **0.7644** | **0.4405** | 15.04 ms |
| | +0.0287 | **+0.0852 (+24%)** | +0.85 ms (+6%) |
| YOLOv8s / TensorRT @ 640 | 0.7270 | 0.3332 | 18.36 ms |
| YOLOv8s / TensorRT @ 672/640 | **0.7705** | **0.4376** | 19.45 ms |
| | +0.0435 | **+0.1044 (+31%)** | +1.09 ms (+6%) |

Số liệu khung 640 giữ nguyên trong `results/benchmark_640.json` để đối chiếu.

## Cơ chế: chưa giải thích được

Giả thuyết đầu của tôi: viền cho các anchor ở rìa đủ ngữ cảnh, nên vật thể chạm mép ảnh
được định vị tốt hơn. Dữ liệu theo lớp **bác bỏ nó** (YOLOv8n / TensorRT):

| lớp | Δ mAP@0.5 | Δ mAP@0.5:0.95 | đặc điểm |
|---|---:|---:|---|
| `scratches` | +0.0861 | **+0.2395** | vết mảnh, dài |
| `inclusion` | +0.0734 | +0.1248 | đốm nhỏ |
| `patches` | +0.0039 | +0.0635 | mảng vừa |
| `rolled-in_scale` | +0.0049 | +0.0404 | vảy nhỏ rải rác |
| `pitted_surface` | +0.0151 | +0.0247 | rỗ, khung lớn |
| `crazing` | **−0.0110** | +0.0185 | nứt lan toả, chạm mép ảnh |

`crazing` là lớp chạm mép nhiều nhất và là lớp **duy nhất không hưởng lợi**. Lớp hưởng
lợi nhất là các vật thể **nhỏ và mảnh**. Giả thuyết ngữ cảnh rìa không đứng vững.

Một giả thuyết còn lại, chưa kiểm chứng: `best.pt` được **chọn** bằng vòng validation
chạy ở đúng chế độ rect 672. Nghĩa là trong số các checkpoint, ta đã giữ lại cái tốt
nhất *ở khung 672*. Đánh giá nó ở khung 640 là đánh giá ngoài chế độ mà nó được chọn ra.
Điều này giải thích được vì sao 672 là "sân nhà", nhưng chưa giải thích được vì sao
`scratches` hưởng lợi gấp mười lần `crazing`.

**Chưa biết cơ chế không làm kết quả kém tin cậy hơn.** Hiệu ứng được đo trên đúng đường
ống triển khai, tái lập ở hai model, ba runtime, và đối chiếu độc lập với `ultralytics.val()`
(lệch 0.0027 và 0.0002). Nhưng cũng vì chưa biết cơ chế, **không nên suy rộng** sang
dataset khác mà không đo lại.

## Điều rút ra

Con số mAP trong log huấn luyện được sinh ra bởi một cấu hình đánh giá cụ thể, và cấu
hình đó **không tự động đi theo model** khi xuất sang ONNX hay TensorRT. Ở đây khoảng
cách giữa hai cấu hình là 0.085 mAP@0.5:0.95 — lớn hơn mọi khác biệt giữa ba runtime
(≤0.0013) và lớn hơn khác biệt giữa YOLOv8n và YOLOv8s (0.006, dưới ngưỡng nhiễu).

Nói cách khác: **cách đóng khung ảnh đầu vào ảnh hưởng tới độ chính xác nhiều hơn cả
việc chọn runtime lẫn chọn model.** Nếu bước đối chiếu chéo trong FR-13 không tồn tại,
dự án này đã báo cáo 0.355 mAP@0.5:0.95 và không ai biết có 0.085 đang nằm trên bàn.

## Tái lập

```bash
make diag-map      # phép tính mAP có đúng không
make diag-rect     # 672 hay 640 — bằng chứng trực tiếp

for K in 640 672 704 736 768; do
  PYTHONPATH=src python -m ivid.benchmark.accuracy \
    --models yolov8n --backends pytorch --imgsz $K --resize-to 640 \
    --out results/framing_$K.json
done
```
