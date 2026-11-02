# Changelog

All notable changes are listed here, newest first. Format: [Keep a Changelog](https://keepachangelog.com/).
One section per project week until the first tagged release.

## [Unreleased]

### Week 4 — Building scores, population, RAG (in progress)
- Found and fixed a wrong-bbox bug in `download_demo_area.py` (osmnx 2.x argument order): Week-1 buildings/roads
  were ~10 km from the HOTOSM area. Corrected data pending an Overpass retry (public endpoints timed out).
- `buildings/zonal.py`: per-building zonal statistics (mean probability/uncertainty, damaged fraction, NaN-aware,
  `no_valid_pixels` flag for the verification queue); unit-tested on synthetic rasters.
- RAG: section-aware chunker, local e5-small embeddings + BM25 hybrid search (`quaketriage.rag`), 10-question gold
  set, retrieval ablation; recall@1 0.8, MRR 0.80 (`reports/metrics_rag.json`).

### Week 3 — Siamese model, uncertainty, guard, RAG text
- Siamese trained on Kaggle (23 epochs, best epoch 13); paired bootstrap vs baseline on test (thr 0.2, val-tuned):
  F1 0.602 vs 0.502, ΔF1 +0.104 [+0.038, +0.166] (`scripts/compare_models.py`, `reports/comparison_*`).
- `scripts/evaluate.py --threshold-mode {ckpt,f1,f2}`: thresholds chosen on validation only.
- Uncertainty decision: 8-view TTA (`docs/decisions.md`); `infer_area.py` uses it; operating threshold 0.2 in config.
- `SiameseUNet` (shared encoder, |pre-post| skip features) and `--model` flag in `train.py`/`evaluate.py`;
  separate Kaggle kernel under `kaggle/siamese/` (baseline notebook untouched).
- Image-quality guard (`data/quality.py`): blank/bright ratios + block SSIM; SSIM threshold recalibrated 0.4 -> 0.05
  from KATE-CD pairs.
- `scripts/infer_area.py`: sliding-window inference with guard, GeoTIFF probability/uncertainty output;
  Kilis Vantor crop (`scripts/download_kilis_crop.py`) used as a real-imagery pipeline demo.
- Uncertainty framework: MC Dropout (forward hooks) vs 8-view TTA, ECE + reliability diagrams (`scripts/eval_uncertainty.py`).
- RAG raw text extraction (`scripts/extract_rag_text.py`): all PDFs have a text layer, no OCR needed.
- Fixes: `uv sync --all-extras` documented; notebooks under `kaggle/**` are tracked.
- CI installs CPU-only PyTorch on Linux (`[tool.uv.sources]`), so all unit tests run there; actions bumped to
  Node-24 versions; `tests/conftest.py` still skips torch tests if torch is absent.
- Refactor: shared `quaketriage/metrics.py` (train/evaluate/eval_uncertainty); outputs verified bit-identical.

### Week 2 — Baseline model
- `KateCDDataset`, early-fusion U-Net (ResNet34, 6 channels), Dice + Focal loss, Kaggle GPU training notebook.
- Test (38 images, threshold 0.5): F1 0.490 [0.370-0.582], precision 0.657, recall 0.391, IoU 0.325
  (`reports/metrics_baseline.json`).

### Week 1 — Data and exploration
- KATE-CD downloaded (446 MB, 404/44/38 train/val/test). EDA: no geographic metadata; ~3.7% damaged pixels;
  every sample contains damage (`pct_with_any_damage = 1.0`). Sample grid figure → `reports/figures/`.
- Vantor STAC scanned (76 collections): single overlapping pre/post pair found (Kilis, 1 HOTOSM building).
  Demo area set to **Antakya** with fallback plan: HOTOSM destroyed buildings as simulated detector output.
- Downloaded for Antakya: 1876 OSM building footprints, 717 road edges (OSMnx), WorldPop Turkey 100 m.
- Downloaded RAG documents: AFAD TAMP (65 pp, text layer OK), INSARAG Vol II Man B + Vol III (originally noted as scanned; corrected in Week 3 — text layer is present).
- `scripts/check_data.py`: verifies 10 files; all green.
- Key decisions logged in `docs/decisions.md`; `docs/notes/week-01.md` written.
- Documentation scaffolding: data card (TBDs partially filled), model card, glossary, docs map.
- Tooling: CI workflow, pre-commit config (ruff, gitleaks), Makefile, config loader, provenance helper,
  results-table generator (`scripts/make_report.py`).

### Week 0 — Setup
- Repository skeleton, `uv` environment, `configs/default.yaml`, `.env.example`.
- Switched to the `google-genai` SDK; default model `gemini-2.5-flash`.
- Smoke tests: package import, PyTorch device (MPS/CPU), project CRS.
