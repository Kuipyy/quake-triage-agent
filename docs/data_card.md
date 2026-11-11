# Data card

One entry per dataset. Facts below come from `reports/metrics_eda.json`, `data/manifest.json` and
`docs/decisions.md`; an item that could not be verified says so.

> Licenses matter: Vantor/Maxar imagery is **CC BY-NC 4.0** (non-commercial). Keep attribution in `README.md`.

## Summary

| Dataset | Role | License | Status |
|---|---|---|---|
| KATE-CD | Model training / evaluation | No license found; imagery from Maxar Open Data (CC BY-NC) and Airbus Pleiades; cite the paper | Downloaded, EDA done; images not redistributed |
| Vantor (Maxar) Open Data | Georeferenced demo imagery | CC BY-NC 4.0 | Fallback — no overlapping pre/post tiles for Antakya |
| HOTOSM destroyed buildings | Building-level validation labels | ODbL | Downloaded; 3239 buildings Turkey-wide, 732 in Antakya demo area |
| HOTOSM / OSM buildings + roads | Footprints, floors, road network | ODbL | Downloaded for Antakya bbox |
| WorldPop Turkey 100 m | Population estimate | CC BY 4.0 | Downloaded (20.5 MB); zonal statistics per building (Week 4) |
| AFAD TAMP | RAG corpus (TR) | Public document | Downloaded; text layer OK (Turkish chars correct) |
| INSARAG Guidelines 2020 | RAG corpus (EN) | Public document | Downloaded; text layer OK (re-checked 2026-11-02, see `docs/decisions.md`) — extracted with `pypdf`, no OCR |

## KATE-CD

- **Source:** `huggingface.co/datasets/CSCRS/kate-cd` (Parquet). Paper: *Earthquake Damage Assessment with SAMCD*; code: `cscrs/kate-cd`.
- **Content:** 512×512 pre-event image, post-event image, binary damage mask.
- **License / provenance (checked 2026-11-11):** no license declared on the sources we could read; the dataset
  description says its imagery comes from Maxar Open Data (CC BY-NC 4.0) and Airbus Pleiades (proprietary imagery,
  redistribution terms unknown). Therefore this repository does **not** redistribute KATE-CD images: the sample grid,
  the prediction figures and the worst-10 figure are generated locally and are git-ignored; only metrics (JSON) are
  committed. Users download the dataset themselves (`scripts/download_kate_cd.py`). Not verified: the exact terms on the
  Hugging Face card, the CodeOcean capsule and the GitHub repo (those hosts were not reachable from the cloud session).
  Sources: [ITU CSCRS publication page](https://web.cscrs.itu.edu.tr/kate-cd-high-resolution-change-detection-dataset-from-the-2023-earthquake-in-kahramanmaras-turkiye/),
  [ITU research portal](https://research.itu.edu.tr/en/publications/kate-cd-high-resolution-change-detection-dataset-from-the-2023-ea/).
- **Size:** train 404 / validation 44 / test 38 samples.
- **Georeferencing:** none. Columns are `pre_image`, `post_image`, `label` only (see `docs/decisions.md`).
- **Label semantics:** binary (damaged / not damaged). No damage grades.
- **Class balance:** mean damaged-pixel ratio 3.7 % (train), 3.2 % (val), 2.8 % (test).
- **Known issues:**
  - Every sample in every split contains some damage (`pct_with_any_damage = 1.0`), so the model never sees a fully
    undamaged tile. Expect false positives on undamaged windows of the demo area; evaluate that explicitly.
  - Test set is small (38 images): report bootstrap confidence intervals, not point estimates only.
  - Possible spatial leakage between neighbouring tiles across splits: not checkable (no coordinates); treat the test numbers as optimistic.
  - Sensor / ground sampling distance differs from the Vantor demo imagery (0.31 m/px crop): not measured; no labels on the crop.
- **Checksums:** `data/manifest.json` (local, git-ignored).

## Vantor (Maxar) Open Data — Kahramanmaraş event

- **Source:** STAC `https://maxar-opendata.s3.amazonaws.com/events/catalog.json`, event `Kahramanmaras-turkey-earthquake-23`.
- **Use:** pre/post GeoTIFF window (about 1–2 km²) for the demo area; read only the window from the COG.
- **License:** CC BY-NC 4.0. Attribution required; no commercial use.
- **Demo area:** fallback plan activated — no overlapping pre/post tile pair covers Antakya (see `docs/decisions.md`). Kilis pre/post pair exists (2023-01-26 / 2023-02-07, bbox ≈ [37.082, 36.714, 37.098, 36.733]) but only 1 HOTOSM destroyed building in that area.
- **Kilis crop (local, not in the repository):** `data/raw/demo/kilis_pre.tif` / `kilis_post.tif`, 2048×2048 px, 3 bands, EPSG:32637, 0.305 m/px, acquisitions 2023-01-26 (pre) and 2023-02-07 (post). Checksums: see `data/manifest.json`. Attribution: imagery © Vantor/Maxar Open Data, CC BY-NC 4.0. The imagery is never committed (`*.tif` is git-ignored; a test guards tracked images).

## HOTOSM destroyed buildings

- **Source:** `data.humdata.org/dataset/hotosm_tur_destroyed_buildings`.
- **Use:** building-level labels for validating the detector on the demo area.
- **Known issue:** crowd-sourced and incomplete. An unlabeled building is not necessarily intact, so precision is
  under-estimated; lead with recall and say so in the results.
- **Feature count (total):** 3239 buildings Turkey-wide; 732 in Antakya demo bbox (36.10–36.25°E, 36.15–36.30°N).
- **SHA-256 (first 16 hex):** `1fad1825b8019bce...` (see `data/manifest.json` once generated).
- **Download date:** 2026-10-31.

## HOTOSM / OSM buildings and roads

- **Source:** HDX `hotosm_tur_buildings` or OSMnx/Overpass.
- **Use:** footprints, floor counts (`building:levels`, often missing), drivable road graph.
- **Correction (2026-11-02):** the Week-1 download used a wrong bbox order and was replaced; the correct OSM query returned 53,028 building footprints and a road graph of 11,768 nodes / 30,718 edges inside the demo bbox (old, wrong files kept in `data/raw/demo/_wrong_bbox/`). `check_data.py` now asserts the extent. Today's OSM contains almost none of the buildings HOTOSM lists as destroyed (8 overlap), so the planning demo set is built from HOTOSM footprints + nearby OSM buildings (`docs/decisions.md`).
- **Known issue:** floor count coverage is partial; the fallback rule is documented in the occupancy step (Week 4).
- **Query date:** 2026-10-31. **Cache paths:** `data/raw/demo/antakya_buildings.gpkg` (53,028 buildings after the 2026-11-02 re-download; the original 1876 were the wrong area), `data/raw/demo/antakya_roads.graphml` (314 nodes / 717 edges, 356 KB).
- **Demo bbox:** 36.10–36.25°E, 36.15–36.30°N (Antakya, Hatay). CRS: WGS84 (EPSG:4326) from OSMnx; reprojected to EPSG:32637 in pipeline.
- **SHA-256 (first 16 hex):** buildings `51e75a0da61fd6c9...`, roads `ee74e58ce98f80b6...`.

## WorldPop Turkey 100 m

- **Source:** `hub.worldpop.org/geodata/summary?id=75773`.
- **Known issue:** pre-earthquake population at 100 m resolution; a coarse approximation per building.
- **Vintage:** 2020 (UN-adjusted, constrained). **Resolution:** 100 m. **File:** `data/raw/demo/worldpop_turkey_100m.tif` (20.5 MB, full Turkey).
- **SHA-256 (first 16 hex):** `97b3703d5727bbbd...`. **Download date:** 2026-10-31.

## AFAD TAMP and INSARAG Guidelines 2020

- **Source:** see `PLAN.md` section 4.
- **Use:** RAG corpus with `[source#section]` citations.
- **Known issues:** check that the PDF has a usable text layer and that Turkish characters survive extraction.
  INSARAG publishes a new version effective 1 January 2027 (noted in `README.md`).
- **AFAD TAMP:** 65 pages. Text layer present; Turkish characters extracted correctly with `pypdf`. SHA-256 `f0237b70b552a61e...`. Download date 2026-10-31.
- **INSARAG Vol II Manual B (Operations):** 58 pages. Text layer present (2 low-text pages; ~120k characters via `pypdf`). The Week-1 note that the PDF was scanned was wrong. Extracted to `data/interim/rag_text/`. SHA-256 `21956d2b9557f678...`. Download date 2026-10-31.
- **INSARAG Vol III (Field Guide):** 46 pages. Text layer present (1 low-text page; ~72k characters). SHA-256 `e37e157cf7473dff...`. Download date 2026-10-31.

## Derived: Antakya demo set (`data/processed/buildings.gpkg`, not committed)

- **Content:** 5,895 buildings: the 732 HOTOSM destroyed footprints plus nearby OSM buildings (100 m buffer), with floors, estimated persons (night factor 0.9, non-residential 0.1: scenario assumptions in `configs/default.yaml`), and `damage_prob` / `uncertainty` from a **simulated detector** (miss 20 %, false alarm 3 %, seed 0).
- **Why simulated:** today's OSM contains almost none of the destroyed buildings (8 overlap) and no overlapping pre/post imagery exists for Antakya, so no real detector output can be produced there (`docs/decisions.md`).
- **Licenses of the sources:** HOTOSM / OSM ODbL, WorldPop CC BY 4.0; attribution in `README.md`.
- **Known issue:** HOTOSM is incomplete; "false positive" overstates real errors.
