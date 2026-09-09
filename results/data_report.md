# Báo cáo kiểm tra dữ liệu (FR-02)

*Sinh tự động lúc 2026-09-09T03:26:10+00:00 bởi `ivid.data.validate`*

Thư mục kiểm tra: `/home/quocanh/quoc_anh/industrial-visual-inspection/data/processed`

## Tổng quan

| Tập | Ảnh | File nhãn | Bbox | Ảnh nhiều loại lỗi |
|---|---:|---:|---:|---:|
| train | 1260 | 1260 | 2950 | 96 |
| val | 270 | 270 | 624 | 16 |
| test | 270 | 270 | 615 | 11 |
| **tổng** | **1800** | | **4189** | |

## Bất thường

| Loại bất thường | train | val | test | tổng |
|---|---:|---:|---:|---:|
| Ảnh không có file nhãn | 0 | 0 | 0 | **0** |
| Nhãn trỏ tới ảnh không tồn tại | 0 | 0 | 0 | **0** |
| File nhãn rỗng (ảnh không có bbox nào) | 0 | 0 | 0 | **0** |
| Bounding box vượt biên ảnh | 0 | 0 | 0 | **0** |
| Bounding box rỗng hoặc kích thước âm | 0 | 0 | 0 | **0** |
| Class id ngoài danh sách lớp | 0 | 0 | 0 | **0** |
| Dòng nhãn sai định dạng | 0 | 0 | 0 | **0** |
| Ảnh hỏng / không đọc được | 0 | 0 | 0 | **0** |
| Ảnh trùng giữa các tập (rò rỉ) | | | | **0** |

## Phân bố bbox theo lớp

| Lớp | train | val | test | tổng |
|---|---:|---:|---:|---:|
| crazing | 481 | 100 | 108 | **689** |
| inclusion | 716 | 158 | 137 | **1011** |
| patches | 629 | 122 | 130 | **881** |
| pitted_surface | 308 | 64 | 60 | **432** |
| rolled-in_scale | 438 | 93 | 97 | **628** |
| scratches | 378 | 87 | 83 | **548** |

## Kết luận

✅ **Không phát hiện bất thường nào.** Dữ liệu sẵn sàng để train.
