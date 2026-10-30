# Thin wrappers around the commands in CLAUDE.md. `make help` lists targets.
.PHONY: help sync test lint format report

help:
	@echo "sync    - install/sync dependencies (uv sync)"
	@echo "test    - run the test suite"
	@echo "lint    - ruff check with auto-fix"
	@echo "format  - ruff format"
	@echo "report  - refresh the results table in README.md from reports/*.json"

sync:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

report:
	uv run python scripts/make_report.py
