# PROGRESS — QuakeTriage

> Her oturum sonunda Claude günceller. En yeni kayıt en üstte.

## Durum özeti
- **Aktif hafta:** Hafta 1 — Veri ve keşif (2–8 Kasım)
- **Sıradaki adım:** KATE-CD veri setini indir, EDA yap
- **Bekleyen manuel işler (the project owner):** —

## Önemli metrikler
| Metrik | Değer | Tarih |
|---|---|---|
| — | — | — |

## Oturum kayıtları

### 2026-10-31 — Hafta 0 tamamlandı
- Konu seçildi: deprem sonrası hasar tespiti ve ekip önceliklendirme ajanı (QuakeTriage).
- Git repo init + GitHub private repo bağlandı (Kuipyy/quake-triage-agent).
- pyproject.toml, uv.lock, .gitignore, .env.example, configs/default.yaml oluşturuldu.
- Tam paket iskeleti oluşturuldu (src/quaketriage altında 7 alt paket).
- SDK kararı: `google-generativeai` deprecated → `google-genai`'ya geçildi; model `gemini-2.5-flash`.
- 3 smoke test yeşil (import, MPS/CPU, CRS). Gemini API çağrısı çalışıyor.
- Sıradaki: Hafta 1 — KATE-CD indirme ve EDA.
