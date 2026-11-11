# Release checklist (checked 2026-11-11)

| Check | Result | Evidence / action |
|---|---|---|
| Secrets in git history | **Clean** | `gitleaks git` (v8.30.1): 87 commits, ~12 MB scanned, no leaks. The actual values of the four secrets in `.env` (Gemini, Groq, HF, Kaggle) were searched for in all commits (`git log -S`) and in the tracked tree: 0 hits. |
| `.env` ever committed | **No** | `git log --all -- .env` is empty; only `.env.example` is tracked (placeholders). |
| Large files in the tree | **OK** | largest tracked: `reports/map_preview.html` 3.1 MB, `uv.lock` 1 MB, `docs/images/ui_map.png` 0.7 MB. |
| Model weights (`*.pt`) | **Never committed** | no `.pt/.pth/.ckpt` in any commit; `*.pt` and `/models/` are git-ignored. |
| Vantor/Maxar GeoTIFFs | **Never committed** | no `.tif` in any commit; `*.tif` and `/data/` git-ignored. |
| Satellite imagery in the current tree | **None** | tracked images: 5 UI screenshots + 1 GIF (UI over OSM tiles, no satellite imagery) and 4 reliability plots; `tests/test_no_imagery.py` guards it. |
| Satellite imagery in git **history** | **FAIL, needs action** | KATE-CD figures were committed earlier (`reports/figures/predictions_*.png` ×6, `kate_cd_samples.png`, `worst10_siamese.png`). They are removed from the tree but remain in history. See "History" below. |
| Personal data in history | **Note** | 73 commits carry the author email of the repository owner (a personal address); 14 carry Claude's noreply address. Decide whether that address should be public. See "History". |
| Data licenses | **OK** | README "Data licenses" and `docs/data_card.md` agree: KATE-CD no licence found, images not redistributed; Vantor CC BY-NC 4.0; OSM/HOTOSM ODbL; WorldPop CC BY 4.0; code MIT. |
| Links | **Fixed 2 / noted 5** | 24 external URLs checked. The two INSARAG PDF URLs in `scripts/download_docs.py` returned 404; the current official URLs were added (files verified byte-identical, same SHA-256). `mebakub.meb.gov.tr/.../TAMP.pdf` returns 500 (it is only the second mirror; the first AFAD URL works). The Maxar S3 `events` URLs are bucket prefixes, not pages (404 is expected). |
| CI | **Green** | last runs on `main`: success (job installs all extras). |
| Repository visibility | PRIVATE | `gh repo view`: `Kuipyy/quake-triage-agent`. Not changed by Claude. |

## History (decision for the repository owner)

The tree is publishable; the history is not, because of the KATE-CD figures (no licence found; Maxar CC BY-NC and Airbus Pleiades imagery) and, optionally, the personal email.

- **Option A (recommended, simplest):** publish a *new* repository with a single commit made from the current tree; keep the old private repository as the archive. No imagery, no old emails, no history surprises. Cost: the 87-commit history is not public.
- **Option B:** rewrite history with `git filter-repo` (remove the paths above, optionally map the author email to the GitHub noreply address), then force-push. Keeps history, but rewrites every SHA and needs a fresh clone and a backup first.

Numbered steps for both were given in the Week 8 session (the local working note is not published).
