# Dataset — NEU-DET

*Sinh tự động bởi `ivid.data.stats` lúc 2026-09-09T05:58:06+00:00*

Ảnh xám bề mặt thép cán nóng, kích thước gốc **200×200**, 6 loại lỗi. Chia train/val/test 70/15/15 phân tầng theo lớp, seed cố định (`1337`) — xem `results/dataset_manifest.json`.

## Số ảnh mỗi tập

| Tập | Ảnh | Bbox | Bbox/ảnh |
|---|---:|---:|---:|
| train | 1260 | 2947 | 2.34 |
| val | 270 | 624 | 2.31 |
| test | 270 | 615 | 2.28 |
| **tổng** | **1800** | **4186** | |

## Số bbox theo lớp (FR-03)

| Lớp | train | val | test | tổng | % |
|---|---:|---:|---:|---:|---:|
| `crazing` | 480 | 100 | 108 | **688** | 16.4% |
| `inclusion` | 715 | 158 | 137 | **1010** | 24.1% |
| `patches` | 628 | 122 | 130 | **880** | 21.0% |
| `pitted_surface` | 308 | 64 | 60 | **432** | 10.3% |
| `rolled-in_scale` | 438 | 93 | 97 | **628** | 15.0% |
| `scratches` | 378 | 87 | 83 | **548** | 13.1% |
| **tổng** | | | | **4186** | |

## Số ảnh chứa mỗi lớp

| Lớp | train | val | test | tổng |
|---|---:|---:|---:|---:|
| `crazing` | 210 | 45 | 45 | **300** |
| `inclusion` | 273 | 58 | 51 | **382** |
| `patches` | 244 | 48 | 50 | **342** |
| `pitted_surface` | 211 | 45 | 45 | **301** |
| `rolled-in_scale` | 210 | 45 | 45 | **300** |
| `scratches` | 210 | 45 | 45 | **300** |

> Tổng cột này lớn hơn 1800 vì một ảnh có thể chứa nhiều loại lỗi.

## Hình dạng bounding box (tập train)

| Lớp | Diện tích trung vị (tỉ lệ ảnh) | Tỉ lệ rộng/cao trung vị |
|---|---:|---:|
| `crazing` | 0.213 | 1.80 |
| `inclusion` | 0.041 | 0.38 |
| `patches` | 0.092 | 0.72 |
| `pitted_surface` | 0.530 | 0.76 |
| `rolled-in_scale` | 0.122 | 1.01 |
| `scratches` | 0.078 | 0.17 |

## Nhận xét

- **Mất cân bằng lớp: 2.34×** giữa lớp nhiều nhất (`inclusion`, 1010 bbox) và ít nhất (`pitted_surface`, 432 bbox).
  Mức này đáng chú ý — nếu mAP của lớp ít mẫu thấp hẳn thì đây là một nguyên nhân, cần nói rõ trong báo cáo thay vì chỉ báo con số.
- Các lớp `pitted_surface` có khung chiếm hơn 25% diện tích ảnh. Khung rất lớn và chồng lấn khiến IoU nhạy cảm, mAP@0.5:0.95 sẽ thấp hơn mAP@0.5 nhiều — đây là đặc tính dataset, không phải lỗi model.
- Các lớp `inclusion`, `scratches` có khung rất dẹt. Anchor-free như YOLOv8 xử lý được, nhưng đáng theo dõi recall của chúng.
- Ảnh gốc **200×200** trong khi benchmark chạy ở 640×640 (phóng to 3.2×). Điều này phải nêu rõ trong báo cáo: nó mở ra khả năng giảm `imgsz` để tăng tốc mà gần như không mất mAP — một hướng tối ưu đáng đo.
