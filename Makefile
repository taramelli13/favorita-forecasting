.PHONY: install prepare train predict tune evaluate test lint clean

install:
	poetry install

prepare:
	poetry run favorita prepare

train:
	poetry run favorita train

predict:
	poetry run favorita predict

tune:
	poetry run favorita tune

evaluate:
	poetry run favorita evaluate

test:
	poetry run pytest tests/ -v

test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v -m integration

lint:
	poetry run ruff check src/ tests/
	poetry run ruff format --check src/ tests/

format:
	poetry run ruff check --fix src/ tests/
	poetry run ruff format src/ tests/

clean:
	rm -rf data/interim/* data/processed/* data/submissions/* models/*.txt mlruns/
