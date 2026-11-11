# Thin wrappers around common project commands. `make help` lists targets.
.PHONY: help sync test lint format report demo-data

help:
	@echo "sync    - install/sync dependencies (uv sync --all-extras)"
	@echo "test    - run the test suite"
	@echo "lint    - ruff check with auto-fix"
	@echo "format  - ruff format"
	@echo "demo-data - download public data and build the Antakya demo (Path 1; needs .env)"
	@echo "report  - refresh the results table in README.md from reports/*.json"

sync:
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

report:
	uv run python scripts/make_report.py

demo-data:
	uv run python scripts/download_demo_area.py
	uv run python scripts/download_docs.py
	uv run python scripts/download_hotosm.py
	uv run python scripts/extract_rag_text.py
	uv run python scripts/build_rag_index.py
	uv run python scripts/build_buildings.py
	uv run python scripts/check_data.py
