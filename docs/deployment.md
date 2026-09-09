# Triển khai trên Jetson Orin Nano

## 0. Môi trường đã kiểm chứng

| | |
|---|---|
| L4T | R36.5.2 |
| JetPack | 6.2 |
| CUDA | 12.6.68 |
| TensorRT | 10.3.0.30 |
| PyTorch | 2.10.0 (bản NVIDIA, `cuda=True`) |
| Python | 3.10.12 |

## 1. Dựng môi trường

```bash
git clone https://github.com/quocanh112233/industrial-visual-inspection.git
cd industrial-visual-inspection
bash scripts/check_jetson.sh          # xem môi trường
bash scripts/setup_jetson.sh          # dựng venv
source .venv/bin/activate
```

`setup_jetson.sh` làm hai việc mà `pip install -r requirements.txt` **không** làm được:

1. Tạo venv với `--system-site-packages` để thấy `torch` và `tensorrt` bản NVIDIA đã cài sẵn trong hệ thống.
2. Cài `ultralytics` với `--no-deps`. Nếu cài bình thường, pip sẽ kéo `torch` bản CPU từ PyPI về **ghi đè** bản CUDA và bạn mất GPU mà không có lỗi nào báo ra.

Nó cũng gỡ `onnxruntime` bản CPU và cài bản GPU cho aarch64 từ `pypi.jetson-ai-lab.io`. Kiểm tra:

```bash
python3 -c "import onnxruntime as o; print(o.get_available_providers())"
# phải có CUDAExecutionProvider hoặc TensorrtExecutionProvider
```

## 2. Cố định điều kiện đo (ràng buộc C3)

Bắt buộc trước mọi phiên benchmark. Không làm bước này thì số liệu vô giá trị.

```bash
sudo nvpmodel -q                # xem chế độ hiện tại
sudo nvpmodel -m 0              # 0=15W, 1=25W, 2=MAXN_SUPER
sudo jetson_clocks              # khoá xung nhịp ở mức tối đa của chế độ đó
```

Chờ **5 phút** cho nhiệt độ ổn định rồi mới đo. Theo dõi ở một phiên SSH khác:

```bash
sudo tegrastats --interval 2000
```

`make device-info` in ra đúng những gì sẽ được ghi vào file kết quả (FR-14).

## 3. Chuỗi lệnh đầy đủ

```bash
make data                         # tải + chuẩn bị dataset
# train ở Colab rồi copy best.pt về models/yolov8n/  (xem docs/colab-training.md)
make export-onnx MODEL=yolov8n    # FR-07
make export-trt  MODEL=yolov8n    # FR-08 — bắt buộc trên chính máy này
make parity      MODEL=yolov8n    # FR-09
make bench                        # FR-10..12
make accuracy                     # FR-13
make report                       # FR-15
```

Hoặc `make all` chạy tất cả.

> ⏱️ `make bench` với cấu hình mặc định mất khá lâu: 5 phút chờ ổn định nhiệt + 6 cấu hình × 3 phiên × 270 ảnh + nghỉ 60s giữa các phiên. Chạy thử nhanh bằng `make bench-quick`.

### Luôn `git pull` TRƯỚC khi chạy

`make report --update-readme` ghi bảng benchmark thẳng vào `README.md`, và các
bước đo ghi vào `results/`. Đó đều là file được commit. Nếu máy dev cũng vừa sửa
một trong số đó thì `git pull` sẽ bị **hủy giữa chừng**:

```
error: Your local changes to the following files would be overwritten by merge:
	README.md
Aborting
```

Nguy hiểm ở chỗ `git pull` dừng nhưng các lệnh phía sau trong cùng dòng vẫn chạy
— bằng code cũ, và kết quả trông vẫn bình thường. Đã xảy ra thật: cross-check
báo `LECH LON` chỉ vì bản sửa chưa kịp về máy.

Vì vậy trên Jetson:

```bash
git pull                       # trước tiên, và kiểm tra nó thành công
make bench && make accuracy && make report
git add -A && git commit -m "ket qua do tren jetson" && git push
```

Nếu `git pull` báo xung đột ở file **sinh tự động** (`README.md`, `results/*.json`,
`docs/benchmark-report.md`, `docs/images/*.png`) thì cứ bỏ bản địa phương rồi kéo
lại — chạy lại lệnh sinh sẽ cho ra đúng nội dung đó:

```bash
git checkout -- README.md
git pull
```

## 4. Chạy dịch vụ

### Trực tiếp

```bash
IVID_BACKEND=tensorrt make serve
curl http://localhost:8000/health
curl -F "file=@data/processed/test/images/scratches_10.jpg" \
     http://localhost:8000/predict
```

### Docker (FR-19)

Kiểm tra runtime trước:

```bash
docker info | grep -i 'default runtime'     # phải ra 'nvidia'
```

Nếu chưa phải, thêm vào `/etc/docker/daemon.json`:

```json
{ "default-runtime": "nvidia",
  "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] } } }
```

rồi `sudo systemctl restart docker`.

```bash
make docker-build
make docker-up
curl http://localhost:8000/health
```

> ⚠️ **Bẫy phiên bản TensorRT.** Engine `.engine` gắn chặt với phiên bản TensorRT đã build ra nó. Nếu TensorRT trong container khác với trên host, engine build ở host sẽ không nạp được. Hoặc chọn tag `L4T_TAG` có TensorRT trùng host, hoặc build lại engine bên trong container:
> ```bash
> docker compose -f docker/docker-compose.yml run --rm ivid \
>     python3 -m ivid.export.to_tensorrt --name yolov8n
> ```

## 5. Biến môi trường (FR-17)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `IVID_BACKEND` | `tensorrt` | `pytorch` / `onnx` / `tensorrt` / `mock` |
| `IVID_MODEL` | `yolov8n` | tên thư mục trong `models/` |
| `IVID_IMGSZ` | `640` | phải khớp với lúc export |
| `IVID_CONF` | `0.25` | ngưỡng tin cậy |
| `IVID_IOU` | `0.7` | ngưỡng IoU của NMS |
| `IVID_DEVICE` | `cuda` | `cuda` hoặc `cpu` |
| `IVID_MAX_UPLOAD_MB` | `10` | giới hạn kích thước file (FR-20) |

Đổi backend không cần build lại image — `GET /health` báo lại backend đang dùng.

## 6. Sự cố thường gặp

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `torch.cuda.is_available()` là `False` | pip đã ghi đè torch bằng bản CPU | chạy lại `scripts/setup_jetson.sh` |
| ORT chỉ có `CPUExecutionProvider` | bản CPU và GPU đè lên nhau trong cùng thư mục | gỡ cả hai ở `~/.local`, cài lại bản GPU |
| Engine không nạp được | build ở máy khác hoặc bản TensorRT khác | `make export-trt` lại trên chính máy này |
| Số đo dao động > 10% giữa các phiên | throttling nhiệt (rủi ro R3) | cố định `nvpmodel`, chờ nguội lâu hơn, kiểm tra quạt |
| OOM khi train | 8GB dùng chung CPU+GPU | dùng `configs/train_yolov8n_jetson.yaml`, giảm `batch` |
| `docker: unknown runtime nvidia` | Docker chưa cấu hình nvidia runtime | sửa `/etc/docker/daemon.json` như mục 4 |
