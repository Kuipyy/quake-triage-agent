# QuakeTriage — instructions for Claude

Agentic post-earthquake building damage assessment + rescue prioritization. Full plan: `PLAN.md`. Status: `PROGRESS.md`.

## Session protocol
- Start: read `PROGRESS.md`, then ONLY the current week's section of `PLAN.md` (not the whole file).
- After EVERY completed checklist item in `PLAN.md`: tick it (`[x]`), run `uv run pytest -q` (if code changed), then `git add -A && git commit -m "<type>: <short message>" && git push origin main`. Small, frequent commits; never batch a whole week into one commit. If push fails, fix it (or tell the project owner in numbered steps) before continuing.
- End: update `PROGRESS.md` (done / metrics / decisions / next step), commit, push. When a week finishes, write `docs/notes/week-XX.md` (Turkish, 1 page: what we did, why, how to explain it in an interview).
- Stay within the current week's scope. Ask before changing scope, deleting data, or using anything paid.

## Language
- Talk to the project owner in Turkish, briefly. When you need her to do something manually, give numbered steps.
- Code, comments, commit messages, README: English.

## Commands
- Install/sync: `uv sync --all-extras` (plain `uv sync` REMOVES torch/agent extras and breaks tests)
- Tests: `uv run pytest -q`
- Lint: `uv run ruff check . --fix`
- Run scripts: `uv run python scripts/<name>.py`

## Rules
- Secrets live only in `.env` (gitignored). Never print, log, or commit keys. Never ask her to paste keys into chat.
- `data/`, `models/`, `outputs/`, `*.tif`, `*.pt`, `*.parquet` are gitignored. Commit only small figures/JSON under `reports/`.
- Scripts print short summaries (one line per epoch). Write long logs to `outputs/logs/` and read only the tail.
- Before any run expected to exceed 10 minutes, state the estimate. Prefer Mac MPS; move training to Kaggle (`kaggle/`) if an epoch takes > 3 min.
- Config values (paths, thresholds, TOPSIS weights, LLM model) come from `configs/default.yaml` / `.env`, not hardcoded.
- One CRS for all geo work (demo area's UTM zone); use helpers in `src/quaketriage/data/geo.py`.
- Every agent tool is a plain, unit-tested Python function with a pydantic input schema.
- Record non-obvious decisions in `docs/decisions.md` (date, decision, reason).
- Data licenses matter: Vantor/Maxar imagery is CC BY-NC 4.0. Keep attribution in README.
