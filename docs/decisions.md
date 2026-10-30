# Architecture & Data Decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-10-31 | Use UTM zone 37N (EPSG:32637) as the single CRS | Kahramanmaraş demo area falls in this zone; metric units simplify buffer/distance calculations |
| 2026-10-31 | KATE-CD has NO geographic metadata | Columns are only pre_image / post_image / label — no lat/lon/city. Model trained on KATE-CD; georeferenced inference done on Vantor demo area only. |
| 2026-10-31 | KATE-CD is heavily imbalanced (~3.7% damaged pixels) | Use Dice + Focal loss; tune threshold on validation F1, not accuracy. |
| 2026-10-31 | Demo area: Antakya — fallback plan activated | Vantor catalog has only 1 overlapping pre/post tile pair (Kilis area), but Kilis has 1 HOTOSM destroyed building vs Antakya's 732. No matching Vantor post-earthquake tile exists for Antakya. Decision: use Antakya with HOTOSM destroyed buildings as simulated detector output for the planning demo. Model performance reported on KATE-CD only. README will note this clearly. |
| 2026-10-31 | LLM provider configured via .env (LLM_PROVIDER / LLM_MODEL) | Avoids hardcoding; allows switching Gemini ↔ Groq without code changes |
| 2026-10-31 | KATE-CD for model training, Vantor GeoTIFF for georeferenced demo area | KATE-CD likely lacks coordinates; Vantor Open Data provides CC BY-NC 4.0 imagery with spatial reference |
