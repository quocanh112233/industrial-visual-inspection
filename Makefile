# IVID — moi buoc cua pipeline la mot lenh (rang buoc C5)
#
# PYTHONUTF8=1: Jetson chay locale C/POSIX, nen Python mac dinh dung ascii cho
# stdout va cho file. Bao cao co ky tu tieng Viet va dau '—' se lam no
# UnicodeEncodeError. Moi cho doc/ghi file da khai bao encoding tuong minh;
# bien nay lo not phan in ra man hinh.
PY      := PYTHONPATH=src PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python3
CONFIG  := configs/data.yaml
MODEL   ?= yolov8n
PORT    ?= 8000

.PHONY: help data prepare validate stats train train-jetson eval check-stack smoke-pipeline \
        export-onnx export-trt parity bench accuracy report all \
        serve docker-build docker-up docker-down smoke test lint clean diag-map diag-rect demo-images mem

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- du lieu
data: ## Tai NEU-DET + chuan bi + kiem tra + thong ke (FR-01..03)
	bash scripts/download_dataset.sh
	$(MAKE) prepare validate stats

prepare: ## FR-01 chuan hoa ve YOLO, chia 70/15/15 seed co dinh
	$(PY) -m ivid.data.prepare --config $(CONFIG)

validate: ## FR-02 kiem tra toan ven -> results/data_report.md
	$(PY) -m ivid.data.validate --config $(CONFIG)

stats: ## FR-03 thong ke phan bo lop -> docs/dataset.md
	$(PY) -m ivid.data.stats --config $(CONFIG)

# ---------------------------------------------------------------- huan luyen
train: ## FR-04 train (MODEL=yolov8s de train ban s)
	$(PY) -m ivid.train.train --config configs/train_$(MODEL).yaml

train-jetson: ## FR-04 train ngay tren Jetson (imgsz 512, batch 8)
	$(PY) -m ivid.train.train --config configs/train_yolov8n_jetson.yaml

eval: ## FR-05 danh gia tren tap test -> results/train_eval.json
	$(PY) -m ivid.train.evaluate --name $(MODEL)

# ---------------------------------------------------------------- chuyen doi
export-onnx: ## FR-07 export ONNX + kiem tra so hoc
	$(PY) -m ivid.export.to_onnx --name $(MODEL)

export-trt: ## FR-08 build TensorRT engine FP16 (CHI TREN JETSON)
	$(PY) -m ivid.export.to_tensorrt --name $(MODEL)

parity: ## FR-09 so sanh detection cua ba dinh dang
	$(PY) -m ivid.export.verify_parity --name $(MODEL)

# ---------------------------------------------------------------- benchmark
bench: ## FR-10..12 do latency + tai nguyen (CHI TREN JETSON)
	$(PY) -m ivid.benchmark.latency --config configs/benchmark.yaml

bench-quick: ## Chay thu nhanh: 1 phien, 20 anh -> results/benchmark_quick.json
	$(PY) -m ivid.benchmark.latency --sessions 1 --settle 0 --cooldown 0 \
		--max-images 20 --out results/benchmark_quick.json

accuracy: ## FR-13 do mAP cho tung dinh dang
	$(PY) -m ivid.benchmark.accuracy

diag-map: ## Truy tim chenh lech mAP giua duong ong cua ta va ultralytics.val()
	$(PY) scripts/diag_map.py

diag-rect: ## Kiem chung: ultralytics cham .pt o 672x672 con engine chay 640x640
	$(PY) scripts/diag_rect.py

device-info: ## FR-14 in dieu kien do hien tai
	$(PY) -m ivid.benchmark.device_info

mem: ## FR-12 do RAM moi runtime, moi cau hinh mot tien trinh rieng
	$(PY) -m ivid.benchmark.memprobe

demo-images: ## Ve detection len anh test -> docs/images/sample_detections.png
	$(PY) -m ivid.visualize --backend $(or $(BACKEND),tensorrt) --model $(or $(MODEL),yolov8n)

report: ## FR-15 sinh bang + bieu do + docs/benchmark-report.md
	$(PY) -m ivid.benchmark.report --update-readme

all: ## Toan bo pipeline tu du lieu toi bao cao (NFR-03)
	$(MAKE) data
	$(MAKE) train MODEL=yolov8n && $(MAKE) eval MODEL=yolov8n
	$(MAKE) export-onnx MODEL=yolov8n && $(MAKE) export-trt MODEL=yolov8n
	$(MAKE) parity MODEL=yolov8n
	$(MAKE) bench accuracy report

# ---------------------------------------------------------------- dich vu
serve: ## FR-16..18 chay API (IVID_BACKEND=pytorch|onnx|tensorrt)
	PYTHONPATH=src python3 -m uvicorn ivid.serve.app:app --host 0.0.0.0 --port $(PORT)

serve-mock: ## Chay API voi runner gia (khong can GPU hay trong so)
	IVID_BACKEND=mock $(MAKE) serve

docker-build: ## FR-19 build image Jetson
	docker compose -f docker/docker-compose.yml build

docker-up: ## FR-19 chay dich vu bang Docker
	docker compose -f docker/docker-compose.yml up -d
	@echo "API: http://localhost:$(PORT)/docs"

docker-down:
	docker compose -f docker/docker-compose.yml down

# ---------------------------------------------------------------- khac
smoke: ## Kiem chung rui ro R1 tren Jetson (.pt -> .onnx -> .engine)
	bash scripts/smoke_tensorrt.sh

smoke-pipeline: ## Kiem chung ca bo do bang model pretrained (truoc khi train)
	bash scripts/smoke_pipeline.sh

check-stack: ## Kiem tra cac thu vien lam viec duoc voi nhau
	bash scripts/check_stack.sh

test: ## Chay unit test
	$(PY) -m pytest tests -q

lint: ## Kiem tra style
	ruff check src tests

clean: ## Xoa du lieu da xu ly (giu raw da tai)
	rm -rf data/processed
