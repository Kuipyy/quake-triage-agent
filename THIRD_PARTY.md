# Third-party data, models and software

This file lists what the project uses from others and under which terms. It was compiled with reasonable care
on 2026-11-11 from the sources named below; it is **not legal advice**. Check the linked terms yourself before
redistributing or using anything commercially. The code of this repository is MIT-licensed ([`LICENSE`](LICENSE));
that licence covers the code only, not the data, models or documents below.

## Data and documents

| Source | Used for | Licence / terms | Redistributed here? |
|---|---|---|---|
| **KATE-CD** (Hugging Face `CSCRS/kate-cd`) | training and evaluation of the damage model | No licence found on the sources we could read. The dataset's imagery comes from Maxar Open Data (CC BY-NC 4.0) and Airbus Pleiades (proprietary; redistribution terms unknown). | **No.** Images are not distributed; users download the dataset themselves (`scripts/download_kate_cd.py`). Only metrics (JSON) are committed. |
| **Vantor (formerly Maxar) Open Data**, event Kahramanmaras-turkey-earthquake-23 | Kilis demo crop (pre 2023-01-26, post 2023-02-07) | CC BY-NC 4.0, non-commercial only, attribution required | **No** (git-ignored; a test guards the repository against tracked images) |
| **OpenStreetMap** buildings and roads (via OSMnx / Overpass) | footprints, floors, road graph, map tiles | ODbL 1.0; attribution "© OpenStreetMap contributors" | derived numbers and map screenshots only |
| **HOTOSM** Turkey destroyed buildings (HDX `hotosm_tur_destroyed_buildings`) | labels for the simulated detector | ODbL; attribution "© OpenStreetMap contributors, Humanitarian OpenStreetMap Team" | derived numbers only |
| **WorldPop** Turkey 100 m, 2020, UN-adjusted, constrained (`tur_ppp_2020_UNadj_constrained.tif`) | population per building | CC BY 4.0 | derived numbers only |
| **AFAD** Turkey Disaster Response Plan (TAMP) | RAG corpus (Turkish) | public document of a government agency; source `afad.gov.tr`, downloaded 2026-10-31 | short cited excerpts only (at most about 300 characters each, with page) |
| **INSARAG Guidelines 2020**, Vol. II Manual B (Operations) and Vol. III (Operational Field Guide) | RAG corpus (English) | published by INSARAG / UN OCHA on `insarag.org`; reuse terms not verified here; a new edition applies from 2027-01-01 | short cited excerpts only (at most about 300 characters each, with page) |

**Attribution texts**
- "© OpenStreetMap contributors" (ODbL, <https://www.openstreetmap.org/copyright>): shown on every map in the UI and in the generated HTML maps.
- Imagery: © Vantor (formerly Maxar) Open Data Program, CC BY-NC 4.0 (<https://creativecommons.org/licenses/by-nc/4.0/>).
- Population: WorldPop (<https://www.worldpop.org>), University of Southampton, CC BY 4.0. Constrained individual-country population, Turkey, 2020, 100 m, UN-adjusted. Copy the exact "recommended citation" (with DOI) from the dataset page when you cite it; we did not verify the citation text for this vintage.
- KATE-CD: Musaoglu N., Ozaydin E., Amirgan B., Taskin G., Nayir H., Kaya H., "KATE-CD: high-resolution change detection dataset from the 2023 Earthquake in Kahramanmaraş, Türkiye", *Journal of Applied Remote Sensing* 19(4), 044508 (2025), doi:10.1117/1.JRS.19.044508.

**Map tiles.** The UI loads OpenStreetMap standard tiles from the OSM tile servers for demonstration at low volume. The [OSM tile usage policy](https://operations.osmfoundation.org/policies/tiles/) applies (attribution, no heavy use, valid Referer). Deployments with more than occasional use should use their own tile source.

## Pretrained models

| Model | Used for | Licence | Redistributed here? |
|---|---|---|---|
| ResNet34 ImageNet weights (loaded by `segmentation-models-pytorch` from the Hugging Face repo `smp-hub/resnet34.imagenet`) | encoder initialisation of the damage models | the Hub repo is tagged `license: other` (torchvision ResNet weights trained on ImageNet; ImageNet's terms are for non-commercial research and education) | No. Downloaded at training time. |
| `intfloat/multilingual-e5-small` (Wang et al., arXiv:2402.05672) | embeddings for the RAG index | MIT | No. Downloaded at index-build time. |
| Our trained checkpoints `baseline_best.pt`, `siamese_best.pt` | damage detection | derived from the models above and from KATE-CD (licence unclear); **not distributed** | No |

## External services (not redistributed)

- Google Gemini API (`google-genai`) and Groq API (open-weight `openai/gpt-oss-120b`, Apache-2.0) are called with the user's own free-tier keys; their terms apply. Keys live only in the user's `.env`.

## Software dependencies

Direct dependencies as installed on 2026-11-11 (`uv sync --all-extras`), licences from `pip-licenses`. The full transitive set (166 packages in the development environment) contains **no GPL, AGPL or LGPL** licence. Weak, file-level copyleft (MPL-2.0) appears in `certifi`, `orjson` and `tqdm`; they are used unmodified as libraries. Regenerate with `uv run --with pip-licenses pip-licenses --format=markdown`.

| Package | Version | Licence |
|---|---|---|
| albumentations | 2.0.8 | MIT |
| folium | 0.20.0 | MIT |
| geopandas | 1.2.0 | BSD-3-Clause |
| google-genai | 2.34.0 | Apache-2.0 |
| gradio | 6.31.0 | Apache-2.0 |
| groq | 1.7.0 | Apache-2.0 |
| langchain-core | 1.7.3 | MIT |
| langgraph | 1.2.19 | MIT |
| matplotlib | 3.11.2 | Python Software Foundation License |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| ortools | 9.12.4544 | Apache-2.0 |
| osmnx | 2.1.1 | MIT |
| pandas | 3.1.0 | BSD-3-Clause |
| pillow | 12.3.0 | MIT-CMU |
| pyarrow | 26.0.0 | Apache-2.0 |
| pydantic | 2.14.2 | MIT |
| pydantic-settings | 2.16.2 | MIT |
| pypdf | 6.21.4 | BSD-3-Clause |
| pyproj | 3.7.2 | MIT |
| pystac-client | 0.9.1 | Apache-2.0 |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| rasterio | 1.4.4 | BSD |
| segmentation_models_pytorch | 0.6.0 | MIT |
| sentence-transformers | 6.2.0 | Apache-2.0 |
| shapely | 2.2.0 | BSD-3-Clause |
| torch | 2.15.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib |
| torchvision | 0.29.2 | BSD |

Algorithms implemented here from published descriptions rather than copied code: TOPSIS and AHP weighting, block-wise SSIM for the image-quality guard, the prize-collecting VRP model on OR-Tools, 8-view test-time augmentation. No code was copied from other projects.
