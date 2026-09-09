# Tính lặp lại của quá trình train (FR-06)

FR-06 yêu cầu: *"Từ file cấu hình lưu lại, chạy lại cho kết quả chênh lệch mAP ≤ 0.02"*.
Dưới đây là phép thử thật, không phải tuyên bố.

## Thiết lập

Hai lần train YOLOv8n độc lập, **cùng seed 1337**, cùng dataset (hash `757d9bcc…`), cùng máy (Colab T4). Khác biệt duy nhất: `cache: ram` → `cache: disk`.
Dữ liệu train hiệu dụng giống hệt nhau — 3 bbox trùng lặp bị loại ở cả hai lần,
chỉ khác là lần 1 do ultralytics âm thầm loại, lần 2 do `ivid.data.prepare` loại.

## Kết quả trên tập test

Đo bằng `ultralytics.val()` ở chế độ mặc định cho bản `.pt` (`rect=True`, khung
672×672). Đó cũng là khung mà đường ống triển khai đang dùng, nên các con số ở
đây so sánh trực tiếp được với bảng benchmark — xem
[input-framing.md](input-framing.md) để biết vì sao khung 672 được chọn.

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

---

# Bộ đo có thiên vị theo thứ tự chạy không?

Một bảng so sánh ba runtime chỉ có giá trị nếu **thứ tự đo không ảnh hưởng kết quả**.
Đây là phép thử.

## Cách làm

Cùng một model (YOLOv8n pretrained COCO), cùng 30 ảnh, cùng thứ tự ảnh, chạy qua cả
ba runner — một lần theo thứ tự `pytorch → onnx → tensorrt`, một lần đảo ngược.
Jetson Orin Nano @15W.

## Kết quả (ms, p50)

| Backend | pre thuận | pre đảo | infer thuận | infer đảo | Lệch infer |
|---|---:|---:|---:|---:|---:|
| PyTorch | 5.68 | 5.74 | 28.90 | 29.25 | 1.2% |
| ONNX Runtime | 5.71 | 5.68 | 13.35 | 13.35 | 0.0% |
| TensorRT FP16 | 5.82 | 5.70 | 8.79 | 8.82 | 0.3% |

**Tiền xử lý giống nhau ở cả ba backend và cả hai thứ tự** — 5.68 đến 5.82 ms, biên độ
2.5%. Đúng như mong đợi, vì cả ba gọi chung `ivid.preprocess`. Nếu con số này lệch nhau
thì hoặc bộ đo sai, hoặc chúng không thật sự dùng chung hàm.

## Vì sao phép thử này cần thiết

Trước khi sửa, PyTorch — backend được đo đầu tiên — hiện `pre` = **7.96–10.24 ms** trong
khi hai backend sau chỉ 5.5–5.8 ms. Nguyên nhân: `warmup()` chỉ gọi `infer()`, bỏ qua
`preprocess`, nên backend đầu tiên gánh toàn bộ chi phí khởi tạo OpenCV (lần `cv2.resize`
đầu tiên chậm gấp ~5 lần lúc ổn định).

Nếu không phát hiện, báo cáo sẽ kết luận *"PyTorch tiền xử lý chậm hơn ONNX 1.8×"* —
sai hoàn toàn, vì cả ba dùng đúng một hàm NumPy. Đó là loại sai số nhìn rất hợp lý và
không ai chất vấn.

## Thành phần kém ổn định nhất

Hậu xử lý (NMS) dao động 1.78–2.94 ms giữa hai lần đo — mạnh hơn hẳn hai giai đoạn kia.
Nó là thao tác ngắn và phụ thuộc số detection, nên khi đọc bảng benchmark cần nhớ rằng
cột này nhiễu hơn cột inference.

Số liệu gốc: `results/harness_validation.json`, `results/smoke_pipeline.json`,
`results/smoke_reversed.json`.
