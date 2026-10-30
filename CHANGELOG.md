# Changelog

All notable changes are listed here, newest first. Format: [Keep a Changelog](https://keepachangelog.com/).
One section per project week until the first tagged release.

## [Unreleased]

### Week 1 — Data and exploration
- KATE-CD downloaded (446 MB, 404/44/38 train/val/test). EDA: no geographic metadata; ~3.7% damaged pixels;
  every sample contains damage (`pct_with_any_damage = 1.0`). Sample grid figure → `reports/figures/`.
- Vantor STAC scanned (76 collections): single overlapping pre/post pair found (Kilis, 1 HOTOSM building).
  Demo area set to **Antakya** with fallback plan: HOTOSM destroyed buildings as simulated detector output.
- Downloaded for Antakya: 1876 OSM building footprints, 717 road edges (OSMnx), WorldPop Turkey 100 m.
- Downloaded RAG documents: AFAD TAMP (65 pp, text layer OK), INSARAG Vol II Man B + Vol III (scanned, no text layer — OCR needed in Week 6).
- `scripts/check_data.py`: verifies 10 files; all green.
- Key decisions logged in `docs/decisions.md`; `docs/notes/week-01.md` written.
- Documentation scaffolding: data card (TBDs partially filled), model card, glossary, docs map.
- Tooling: CI workflow, pre-commit config (ruff, gitleaks), Makefile, config loader, provenance helper,
  results-table generator (`scripts/make_report.py`).

### Week 0 — Setup
- Repository skeleton, `uv` environment, `configs/default.yaml`, `.env.example`.
- Switched to the `google-genai` SDK; default model `gemini-2.5-flash`.
- Smoke tests: package import, PyTorch device (MPS/CPU), project CRS.
