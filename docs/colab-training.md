# Hướng dẫn train trên Google Colab — cho người chưa dùng bao giờ

> ## 🚀 Đường tắt
> Có sẵn notebook chạy được ngay, không phải copy từng ô:
> **[Mở `ivid_colab.ipynb` trong Colab](https://colab.research.google.com/github/quocanh112233/industrial-visual-inspection/blob/main/notebooks/ivid_colab.ipynb)**
>
> Tài liệu này giải thích *vì sao* từng bước tồn tại. Nếu chỉ muốn train thì
> dùng notebook; quay lại đây khi có gì không hiểu hoặc bị lỗi.
>
> Repo đang **private**, nên notebook sẽ hỏi token GitHub ở Phần 3. Muốn bỏ qua
> bước đó thì đổi repo sang public — dự án này rồi cũng sẽ public vì bạn dùng nó
> để ứng tuyển.

> **Tại sao không train thẳng trên Jetson?**
> Train được, nhưng chậm ~20 lần. Ước lượng trên Orin Nano 8GB ở chế độ 15W:
> YOLOv8n ≈ **4–6 giờ**, YOLOv8s ≈ **12–16 giờ** cho 100 epoch.
> Trên Colab (GPU T4 miễn phí): YOLOv8n ≈ **40 phút**, YOLOv8s ≈ **1.5 giờ**.
>
> Vấn đề không phải một lần train, mà là **bạn sẽ train lại nhiều lần** (chỉnh
> epoch, imgsz, augmentation cho tới khi mAP@0.5 ≥ 0.65 theo FR-05). Trên Jetson
> mỗi lần thử là một đêm. Trên Colab là một buổi chiều.
>
> SRS ràng buộc **C1** chỉ yêu cầu *benchmark* chạy trên Jetson — không nói gì về
> nơi train. **C2** yêu cầu ba định dạng dùng cùng một `best.pt`; điều đó vẫn
> đúng khi file `best.pt` được train ở nơi khác rồi copy sang.
> Colab bản miễn phí không tốn tiền nên không vi phạm **C4**.
>
> Nếu bạn vẫn muốn train trên Jetson: xem mục [Phụ lục B](#phụ-lục-b--train-thẳng-trên-jetson) ở cuối.

---

## Phần 0 — Colab là gì (30 giây)

Google Colab là một **máy tính ảo Linux có GPU, chạy miễn phí trong trình duyệt**.
Bạn viết code vào từng "ô" (cell) và bấm ▶ để chạy. Nó giống một terminal Linux
cộng với editor, chỉ khác là máy đó **không phải của bạn** — Google cho mượn.

Ba điều phải nhớ vì chúng sẽ cắn bạn:

| Sự thật | Hậu quả | Cách xử lý |
|---|---|---|
| Máy bị **xoá sạch** khi ngắt kết nối | Mất `best.pt` nếu chỉ lưu trong máy ảo | Lưu vào Google Drive (Phần 3) |
| Ngắt sau **~90 phút không thao tác**, và tối đa ~4–12 giờ/phiên | Train dở dang | Bật `resume=True` (Phần 6) |
| GPU miễn phí **có hạn mức**, dùng nhiều bị tạm khoá vài giờ | Không xin được GPU | Train YOLOv8n trước, YOLOv8s sau, đừng chạy song song |

---

## Phần 1 — Mở Colab và xin GPU

1. Vào **https://colab.research.google.com** (đăng nhập bằng tài khoản Google của bạn).
2. Menu **File → New notebook** (Tệp → Sổ tay mới). Một trang trắng có một ô code hiện ra.
3. **Đây là bước quan trọng nhất, làm sai là train bằng CPU chậm gấp 50 lần:**
   Menu **Runtime → Change runtime type** (Thời gian chạy → Thay đổi loại thời gian chạy)
   → mục **Hardware accelerator** chọn **T4 GPU** → **Save**.
4. Bấm nút **Connect** góc trên bên phải, chờ hiện ✓ và thanh RAM/Disk.

**Kiểm tra ngay** — dán vào ô code đầu tiên, bấm ▶ (hoặc `Shift+Enter`):

```python
!nvidia-smi
```

Phải thấy dòng có `Tesla T4` và `CUDA Version: 12.x`.
Nếu báo `command not found` → bạn chưa chọn GPU ở bước 3, quay lại làm lại.

---

## Phần 2 — Kiểm tra bạn được cấp GPU loại gì

```python
import torch, subprocess
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("GPU  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ")
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv"],
                     capture_output=True, text=True).stdout)
```

Colab miễn phí thường cho **T4 16GB**. Thỉnh thoảng được **L4** (nhanh hơn). Cả hai đều thừa sức.

---

## Phần 3 — Nối Google Drive (để không mất kết quả)

Chạy ô này. Sẽ hiện popup xin quyền → bấm **Cho phép** → chọn tài khoản Google của bạn.

```python
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE = '/content/drive/MyDrive/ivid'      # thư mục lưu kết quả train
os.makedirs(SAVE, exist_ok=True)
print("Sẽ lưu vào:", SAVE)
```

Sau bước này, mọi thứ ghi vào `/content/drive/MyDrive/ivid` sẽ nằm trong Google Drive
của bạn và **sống sót qua mọi lần ngắt kết nối**. Mọi thứ ghi ở chỗ khác sẽ mất.

---

## Phần 4 — Lấy code và dataset

```python
%cd /content
!rm -rf industrial-visual-inspection
!git clone https://github.com/quocanh112233/industrial-visual-inspection.git
%cd /content/industrial-visual-inspection
!ls
```

Cài thư viện (Colab đã có sẵn torch với CUDA, nên chỉ cần ultralytics):

```python
!pip install -q ultralytics
import ultralytics; ultralytics.checks()
```

> Trên Colab **không** cần ghim numpy như trên Jetson. Ràng buộc numpy 1.26.4 chỉ
> áp dụng cho Jetson, nơi JetPack cài sẵn một bộ thư viện build khớp quanh phiên
> bản đó. Colab tự lo phần này.

Tải dataset bằng đúng script mà bạn cũng sẽ chạy trên Jetson:

```python
!bash scripts/download_dataset.sh
```

Kết quả mong đợi: `1800 anh`, `4189 bbox`, `bbox loi bien: 0`, `[ok] du 1800 anh`.

Chuẩn hoá + chia train/val/test 70/15/15 với seed cố định (FR-01):

```python
!PYTHONPATH=src python -m ivid.data.prepare --config configs/data.yaml
!PYTHONPATH=src python -m ivid.data.validate      # FR-02
!PYTHONPATH=src python -m ivid.data.stats         # FR-03
```

---

## Phần 5 — Train

```python
!PYTHONPATH=src python -m ivid.train.train \
    --config configs/train_yolov8n.yaml \
    --project /content/drive/MyDrive/ivid/runs \
    --name yolov8n
```

Ghi thẳng vào Drive (`--project`) để mất kết nối cũng không mất gì.

Trong lúc chạy bạn sẽ thấy bảng tiến độ từng epoch với `box_loss`, `cls_loss`, `mAP50`.
**Đừng đóng tab.** Thỉnh thoảng bấm vào trang để Colab biết bạn còn đó.

---

## Phần 5b — Đánh giá và **sao lưu ngay** (đừng bỏ qua)

Train xong, `best.pt` đã nằm trong Drive nhờ `--project`. Nhưng `results/` và
`models/` thì **vẫn nằm trong máy ảo** và sẽ mất khi ngắt kết nối. Đó là nơi chứa
`train_<name>_manifest.json` — file ghi seed, phiên bản thư viện, siêu tham số và
hash dataset, tức là toàn bộ bằng chứng cho FR-06. Mất nó thì con số mAP trong
báo cáo không truy nguyên được nữa.

```python
# đánh giá trên tập test (FR-05)
!PYTHONPATH=src python -m ivid.train.evaluate --name yolov8n

# sao lưu MỌI thứ cần giữ
!mkdir -p /content/drive/MyDrive/ivid/artifacts
!cp -r results /content/drive/MyDrive/ivid/artifacts/
!cp -r models  /content/drive/MyDrive/ivid/artifacts/
!ls -R /content/drive/MyDrive/ivid/artifacts | head -30
```

Kiểm tra nhanh kết quả:

```python
import json
d = json.load(open('results/train_eval.json'))
for k, v in d.items():
    o = v['overall']
    print(f"{k}: mAP@0.5={o['mAP50']:.4f}  mAP@0.5:0.95={o['mAP50_95']:.4f}")
    for c, m in v['per_class'].items():
        print(f"    {c:18s} mAP50={m.get('mAP50', 0):.4f}")
```

Ngưỡng FR-05 là **mAP@0.5 ≥ 0.65** cho YOLOv8n. Nếu thấp hơn, xem cột theo lớp
trước khi kết luận là hỏng: `pitted_surface` và `crazing` vốn khó (xem
[dataset.md](dataset.md)), một mình chúng kéo xuống là chuyện bình thường.

---

## Phần 6 — Nếu bị ngắt giữa chừng

Đây là chuyện bình thường, không phải hỏng. Kết nối lại (Connect), chạy lại
**Phần 3** (mount Drive) và **Phần 4** (clone + cài), rồi:

```python
from ultralytics import YOLO
m = YOLO('/content/drive/MyDrive/ivid/runs/yolov8n/weights/last.pt')
m.train(resume=True)
```

Nó chạy tiếp từ epoch dang dở, không train lại từ đầu.

---

## Phần 7 — Lấy `best.pt` về máy

Sau khi train xong, file nằm ở `/content/drive/MyDrive/ivid/runs/yolov8n/weights/best.pt`
(khoảng 6 MB cho YOLOv8n).

**Cách 1 — tải thẳng về máy dev:**

```python
from google.colab import files
files.download('/content/drive/MyDrive/ivid/runs/yolov8n/weights/best.pt')
```

**Cách 2 — mở Google Drive trên trình duyệt**, vào `MyDrive/ivid/runs/yolov8n/weights/`,
chuột phải `best.pt` → Tải xuống. Cách này chắc chắn hơn khi file lớn.

**Đưa lên Jetson** (chạy trên máy dev, sau khi đã tải file về `~/Downloads`):

```bash
ssh jetson@<IP> "mkdir -p ~/quoc_anh/industrial-visual-inspection/models/yolov8n"
scp ~/Downloads/best.pt \
    jetson@<IP>:~/quoc_anh/industrial-visual-inspection/models/yolov8n/best.pt

# nhớ mang theo cả manifest — nếu không, số mAP mất đường truy nguyên (FR-06)
scp ~/Downloads/train_yolov8n_manifest.json \
    jetson@<IP>:~/quoc_anh/industrial-visual-inspection/results/
```

Từ đây trở đi mọi thứ (export ONNX, build engine, benchmark, serve) chạy **trên Jetson**.

---

## Phần 8 — Checklist trước khi rời Colab

- [ ] `results/train_eval.json` đã copy vào Drive
- [ ] `results/train_<name>_manifest.json` đã copy vào Drive ← **quan trọng nhất cho FR-06**
- [ ] `best.pt` của **cả** yolov8n và yolov8s đã nằm trong Drive
- [ ] `results/train_<name>_curve.csv` (đường cong loss) đã lưu — cần cho báo cáo
- [ ] Đã chạy `ivid.train.evaluate` và xem mAP theo từng lớp

---

## Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `nvidia-smi: command not found` | Chưa bật GPU | Runtime → Change runtime type → T4 GPU |
| `You are not subscribed... GPU unavailable` | Hết hạn mức GPU miễn phí | Chờ vài giờ, hoặc dùng Kaggle Notebooks (30h GPU/tuần miễn phí) |
| Mất hết file sau khi quay lại | Không lưu vào Drive | Luôn dùng `--project /content/drive/MyDrive/ivid/runs` |
| `CUDA out of memory` | Batch quá lớn | Giảm `batch` trong `configs/train_yolov8n.yaml` xuống 16 hoặc 8 |
| Train xong mAP rất thấp (~0.1) | Sai đường dẫn dataset trong `data.yaml` | Chạy lại `ivid.data.validate` xem báo lỗi gì |

---

## Phụ lục A — Kaggle Notebooks (phương án 2)

Nếu Colab hết hạn mức GPU, **Kaggle Notebooks** cho **30 giờ GPU mỗi tuần** miễn phí
và phiên chạy tối đa 12 giờ (dài hơn Colab). Cách dùng gần như giống hệt:
vào kaggle.com → Code → New Notebook → Settings → Accelerator → **GPU T4 x2**.
Khác biệt duy nhất: không có Google Drive, dùng "Save Version" để giữ output.

Bạn cũng sẽ cần tài khoản Kaggle nếu muốn tải dataset bản gốc VOC XML
(`bash scripts/download_dataset.sh --source kaggle`), nên tạo luôn cũng tiện.

---

## Phụ lục B — Train thẳng trên Jetson

Vẫn hỗ trợ đầy đủ, dùng config riêng đã hạ tham số cho vừa 8GB RAM:

```bash
ssh jetson@<IP>
cd ~/industrial-visual-inspection
source .venv/bin/activate

sudo nvpmodel -m 0            # cố định chế độ nguồn
sudo jetson_clocks            # khoá xung nhịp tối đa

# chạy trong tmux để đóng SSH không mất tiến trình
tmux new -s train
PYTHONPATH=src python -m ivid.train.train --config configs/train_yolov8n_jetson.yaml
# Ctrl+B rồi D để thoát tmux;  tmux attach -t train  để quay lại
```

Khác biệt trong `configs/train_yolov8n_jetson.yaml`: `imgsz: 512`, `batch: 8`,
`workers: 2`, `cache: false`, `amp: true` — để không bị OOM.

Theo dõi nhiệt độ và RAM ở một SSH khác:

```bash
sudo tegrastats --interval 5000
```

Nếu thấy `RAM 7000/7620MB` và swap tăng nhanh → giảm `batch` xuống 4.

> ⚠️ Nếu train trên Jetson, **để máy nguội ít nhất 15 phút** trước khi chạy benchmark.
> Chạy benchmark ngay sau 6 giờ train sẽ dính throttling nhiệt và làm sai số đo (rủi ro R3).
