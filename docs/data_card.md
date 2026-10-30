# Data card

One entry per dataset. Fill the `TBD` cells as each data step lands (Week 1 downloads, later weeks derive
products). Facts below come from `reports/metrics_eda.json` and `docs/decisions.md`; everything else is
marked `TBD` until verified.

> Licenses matter: Vantor/Maxar imagery is **CC BY-NC 4.0** (non-commercial). Keep attribution in `README.md`.

## Summary

| Dataset | Role | License | Status |
|---|---|---|---|
| KATE-CD | Model training / evaluation | No license on the HF page; cite the paper | Downloaded, EDA done |
| Vantor (Maxar) Open Data | Georeferenced demo imagery | CC BY-NC 4.0 | TBD |
| HOTOSM destroyed buildings | Building-level validation labels | ODbL | TBD |
| HOTOSM / OSM buildings + roads | Footprints, floors, road network | ODbL | TBD |
| WorldPop Turkey 100 m | Population estimate | CC BY 4.0 | TBD |
| AFAD TAMP | RAG corpus (TR) | Public document | TBD |
| INSARAG Guidelines 2020 | RAG corpus (EN) | Public document | TBD |

## KATE-CD

- **Source:** `huggingface.co/datasets/CSCRS/kate-cd` (Parquet). Paper: *Earthquake Damage Assessment with SAMCD*; code: `cscrs/kate-cd`.
- **Content:** 512×512 pre-event image, post-event image, binary damage mask.
- **Size:** train 404 / validation 44 / test 38 samples.
- **Georeferencing:** none. Columns are `pre_image`, `post_image`, `label` only (see `docs/decisions.md`).
- **Label semantics:** binary (damaged / not damaged). No damage grades.
- **Class balance:** mean damaged-pixel ratio 3.7 % (train), 3.2 % (val), 2.8 % (test).
- **Known issues:**
  - Every sample in every split contains some damage (`pct_with_any_damage = 1.0`), so the model never sees a fully
    undamaged tile. Expect false positives on undamaged windows of the demo area; evaluate that explicitly.
  - Test set is small (38 images): report bootstrap confidence intervals, not point estimates only.
  - Possible spatial leakage between neighbouring tiles across splits: TBD (cannot be checked without coordinates).
  - Sensor / ground sampling distance differs from the Vantor demo imagery: TBD.
- **Checksums:** TBD (`data/manifest.json` once the manifest exists).

## Vantor (Maxar) Open Data — Kahramanmaraş event

- **Source:** STAC `https://maxar-opendata.s3.amazonaws.com/events/catalog.json`, event `Kahramanmaras-turkey-earthquake-23`.
- **Use:** pre/post GeoTIFF window (about 1–2 km²) for the demo area; read only the window from the COG.
- **License:** CC BY-NC 4.0. Attribution required; no commercial use.
- **Demo area, acquisition dates, GSD, CRS, checksums:** TBD.

## HOTOSM destroyed buildings

- **Source:** `data.humdata.org/dataset/hotosm_tur_destroyed_buildings`.
- **Use:** building-level labels for validating the detector on the demo area.
- **Known issue:** crowd-sourced and incomplete. An unlabeled building is not necessarily intact, so precision is
  under-estimated; lead with recall and say so in the results.
- **Checksums, feature count in demo area:** TBD.

## HOTOSM / OSM buildings and roads

- **Source:** HDX `hotosm_tur_buildings` or OSMnx/Overpass.
- **Use:** footprints, floor counts (`building:levels`, often missing), drivable road graph.
- **Known issue:** floor count coverage is partial; the fallback rule is documented in the occupancy step (Week 4).
- **Query date and cache path:** TBD.

## WorldPop Turkey 100 m

- **Source:** `hub.worldpop.org/geodata/summary?id=75773`.
- **Known issue:** pre-earthquake population at 100 m resolution; a coarse approximation per building.
- **Vintage, checksums:** TBD.

## AFAD TAMP and INSARAG Guidelines 2020

- **Source:** see `PLAN.md` section 4.
- **Use:** RAG corpus with `[source#section]` citations.
- **Known issues:** check that the PDF has a usable text layer and that Turkish characters survive extraction.
  INSARAG publishes a new version effective 1 January 2027 (noted in `README.md`).
- **Checksums, page counts:** TBD.
