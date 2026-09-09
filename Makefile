PY      := PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python3
CONFIG  := configs/data.yaml
MODEL   ?= yolov8n
PORT    ?= 8000

.PHONY: help data prepare validate stats train train-jetson eval check-stack smoke-pipeline \
        export-onnx export-trt parity bench accuracy report all \
        serve docker-build docker-up docker-down smoke test lint clean diag-map diag-rect demo-images mem

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

data: ## Tải NEU-DET + chuẩn bị + kiểm tra + thống kê (FR-01..03)
	bash scripts/download_dataset.sh
	$(MAKE) prepare validate stats

prepare: ## FR-01 chuẩn hoá về YOLO, chia 70/15/15 seed cố định
	$(PY) -m ivid.data.prepare --config $(CONFIG)

validate: ## FR-02 kiểm tra toàn vẹn -> results/data_report.md
	$(PY) -m ivid.data.validate --config $(CONFIG)

stats: ## FR-03 thống kê phân bố lớp -> docs/dataset.md
	$(PY) -m ivid.data.stats --config $(CONFIG)

train: ## FR-04 train (MODEL=yolov8s để train bản s)
	$(PY) -m ivid.train.train --config configs/train_$(MODEL).yaml

train-jetson: ## FR-04 train ngay trên Jetson (imgsz 512, batch 8)
	$(PY) -m ivid.train.train --config configs/train_yolov8n_jetson.yaml

eval: ## FR-05 đánh giá trên tập test -> results/train_eval.json
	$(PY) -m ivid.train.evaluate --name $(MODEL)

export-onnx: ## FR-07 export ONNX + kiểm tra số học
	$(PY) -m ivid.export.to_onnx --name $(MODEL)

export-trt: ## FR-08 build TensorRT engine FP16 (CHỈ TRÊN JETSON)
	$(PY) -m ivid.export.to_tensorrt --name $(MODEL)

parity: ## FR-09 so sánh detection của ba định dạng
	$(PY) -m ivid.export.verify_parity --name $(MODEL)

bench: ## FR-10..12 đo latency + tài nguyên (CHỈ TRÊN JETSON)
	$(PY) -m ivid.benchmark.latency --config configs/benchmark.yaml

bench-quick: ## Chạy thử nhanh: 1 phiên, 20 ảnh -> results/benchmark_quick.json
	$(PY) -m ivid.benchmark.latency --sessions 1 --settle 0 --cooldown 0 \
		--max-images 20 --out results/benchmark_quick.json

accuracy: ## FR-13 đo mAP cho từng định dạng
	$(PY) -m ivid.benchmark.accuracy

diag-map: ## Truy tìm chênh lệch mAP giữa đường ống của ta và ultralytics.val()
	$(PY) scripts/diag_map.py

diag-rect: ## Kiểm chứng: ultralytics chấm .pt ở 672x672 còn engine chạy 640x640
	$(PY) scripts/diag_rect.py

device-info: ## FR-14 in điều kiện đo hiện tại
	$(PY) -m ivid.benchmark.device_info

mem: ## FR-12 đo RAM mỗi runtime, mỗi cấu hình một tiến trình riêng
	$(PY) -m ivid.benchmark.memprobe

demo-images: ## Vẽ detection lên ảnh test -> docs/images/sample_detections.png
	$(PY) -m ivid.visualize --backend $(or $(BACKEND),tensorrt) --model $(or $(MODEL),yolov8n)

report: ## FR-15 sinh bảng + biểu đồ + docs/benchmark-report.md
	$(PY) -m ivid.benchmark.report --update-readme

all: ## Toàn bộ pipeline từ dữ liệu tới báo cáo (NFR-03)
	$(MAKE) data
	$(MAKE) train MODEL=yolov8n && $(MAKE) eval MODEL=yolov8n
	$(MAKE) export-onnx MODEL=yolov8n && $(MAKE) export-trt MODEL=yolov8n
	$(MAKE) parity MODEL=yolov8n
	$(MAKE) bench accuracy report

serve: ## FR-16..18 chạy API (IVID_BACKEND=pytorch|onnx|tensorrt)
	PYTHONPATH=src python3 -m uvicorn ivid.serve.app:app --host 0.0.0.0 --port $(PORT)

serve-mock: ## Chạy API với runner giả (không cần GPU hay trọng số)
	IVID_BACKEND=mock $(MAKE) serve

docker-build: ## FR-19 build image Jetson
	docker compose -f docker/docker-compose.yml build

docker-up: ## FR-19 chạy dịch vụ bằng Docker
	docker compose -f docker/docker-compose.yml up -d
	@echo "API: http://localhost:$(PORT)/docs"

docker-down:
	docker compose -f docker/docker-compose.yml down

smoke: ## Kiểm chứng rủi ro R1 trên Jetson (.pt -> .onnx -> .engine)
	bash scripts/smoke_tensorrt.sh

smoke-pipeline: ## Kiểm chứng cả bộ đo bằng model pretrained (trước khi train)
	bash scripts/smoke_pipeline.sh

check-stack: ## Kiểm tra các thư viện làm việc được với nhau
	bash scripts/check_stack.sh

test: ## Chạy unit test
	$(PY) -m pytest tests -q

lint: ## Kiểm tra style
	ruff check src tests

clean: ## Xoá dữ liệu đã xử lý (giữ raw đã tải)
	rm -rf data/processed
