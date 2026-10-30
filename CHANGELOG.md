# Changelog

All notable changes are listed here, newest first. Format: [Keep a Changelog](https://keepachangelog.com/).
One section per project week until the first tagged release.

## [Unreleased]

### Week 1 — Data and exploration (in progress)
- KATE-CD download script and EDA (class balance, sample grid, coordinate check).
- Documentation scaffolding: data card, model card, glossary, docs map.
- Tooling: CI workflow, pre-commit config (ruff, gitleaks), Makefile, config loader, provenance helper,
  results-table generator.

### Week 0 — Setup
- Repository skeleton, `uv` environment, `configs/default.yaml`, `.env.example`.
- Switched to the `google-genai` SDK; default model `gemini-2.5-flash`.
- Smoke tests: package import, PyTorch device (MPS/CPU), project CRS.
