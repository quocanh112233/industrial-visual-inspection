# IVID — moi buoc cua pipeline la mot lenh (rang buoc C5)
PY      := PYTHONPATH=src python3
CONFIG  := configs/data.yaml

.PHONY: help data prepare validate stats test lint smoke clean
help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

data: ## Tai NEU-DET + chuan bi + kiem tra + thong ke (FR-01..03)
	bash scripts/download_dataset.sh
	$(MAKE) prepare validate stats

prepare: ## FR-01 chuan hoa ve YOLO, chia 70/15/15 seed co dinh
	$(PY) -m ivid.data.prepare --config $(CONFIG)

validate: ## FR-02 kiem tra toan ven -> results/data_report.md
	$(PY) -m ivid.data.validate --config $(CONFIG)

stats: ## FR-03 thong ke phan bo lop -> docs/dataset.md
	$(PY) -m ivid.data.stats --config $(CONFIG)

smoke: ## Kiem chung rui ro R1 tren Jetson (.pt -> .onnx -> .engine)
	bash scripts/smoke_tensorrt.sh

test: ## Chay unit test
	$(PY) -m pytest tests -q

lint: ## Kiem tra style
	ruff check src tests

clean: ## Xoa du lieu da xu ly (giu raw da tai)
	rm -rf data/processed
