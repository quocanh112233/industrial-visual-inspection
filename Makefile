# IVID — moi buoc cua pipeline la mot lenh (rang buoc C5)
PY      := PYTHONPATH=src python3
CONFIG  := configs/data.yaml
MODEL   ?= yolov8n
PORT    ?= 8000

.PHONY: help data prepare validate stats train train-jetson eval \
        export-onnx export-trt parity bench accuracy report all \
        serve docker-build docker-up docker-down smoke test lint clean

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

bench-quick: ## Chay thu nhanh: 1 phien, 20 anh, khong cho on dinh nhiet
	$(PY) -m ivid.benchmark.latency --sessions 1 --settle 0 --cooldown 0 --max-images 20

accuracy: ## FR-13 do mAP cho tung dinh dang
	$(PY) -m ivid.benchmark.accuracy

device-info: ## FR-14 in dieu kien do hien tai
	$(PY) -m ivid.benchmark.device_info

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

test: ## Chay unit test
	$(PY) -m pytest tests -q

lint: ## Kiem tra style
	ruff check src tests

clean: ## Xoa du lieu da xu ly (giu raw da tai)
	rm -rf data/processed
