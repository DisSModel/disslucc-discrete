.PHONY: install test lint format run-moju benchmark

install:
	pip install -e ".[dev]" --break-system-packages

test:
	pytest tests/

lint:
	ruff check src/ benchmark/
	black --check src/ benchmark/

format:
	black src/ benchmark/
	ruff check --fix src/ benchmark/

run-moju:
	python3 src/disslucc_discrete/executors/clue_s_vector_executor.py run \
		--input data/cs_moju.zip \
		--output outputs/resultado_moju.gpkg \
		--toml examples/moju_model.toml \
		--param demand_csv=examples/data/demand_moju.csv \
		--param n_steps=6

benchmark:
	cd benchmark && python3 validate_lab6.py ../data/cs_moju.zip data/Lab6_2004.zip
