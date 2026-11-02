# PROGRESS — QuakeTriage

> Her oturum sonunda Claude günceller. En yeni kayıt en üstte.

## Durum özeti
- **Aktif hafta:** Hafta 3 — Siamese model, belirsizlik, guard, RAG ön işlemesi
- **Sıradaki adım:** Siamese eğitimi bitince (Kaggle, 17/50 epoch'ta val F1=0.555) `evaluate.py --model siamese`, `eval_uncertainty.py --model siamese`; karşılaştırma tablosu; dropout'lu 3. Kaggle koşusu; `infer_area.py` Siamese ile; week-03 notu
- **Bekleyen manuel işler (the project owner):** `models/siamese_best.pt` indirip `models/` altına koy (Kaggle kernel: quaketriage-siamese-training)

## Önemli metrikler

Metrikler `reports/*.json` dosyalarından otomatik üretilir ve `README.md` sonuç tablosunda gösterilir.
Güncel değerler için: `uv run python scripts/make_report.py --check` → [README.md](../README.md#results)

## Oturum kayıtları

### 2026-11-02 — Hafta 3 (devam ediyor)
- Siamese U-Net (`SiameseUNet`, paylaşılan encoder, |pre−post| skip) + `--model` bayrağı (train/evaluate). Ayrı Kaggle kernel: `kaggle/siamese/` (baseline notebook'una dokunulmadı).
- Kalite guard'ı `data/quality.py` (boş/parlak oran + blok SSIM). SSIM eşiği KATE-CD ölçümüyle 0.4 → 0.05'e çekildi (`decisions.md`).
- RAG ham metin: 3 PDF'te de metin katmanı var, OCR gerekmedi (`scripts/extract_rag_text.py`).
- `scripts/infer_area.py` iskeleti (kayan pencere, guard, prob/belirsizlik GeoTIFF). Kilis kesiti (`download_kilis_crop.py`) ile gerçek görüntüde denendi; sadece pipeline gösterimi.
- Belirsizlik çerçevesi: `uncertainty.py`, `eval_uncertainty.py`. Baseline (val): tek geçiş ECE 0.0247, MC Dropout 0.0163, TTA 0.0159; hata/belirsizlik oranı MC 7.2, TTA 13.9. Karar bekliyor: dropout'suz eğitilmiş modelde MC adil değil.
- Kararlar: Siamese sonrası dropout'lu 3. koşu (ücretsiz Kaggle kotası), final modelde eşik F2 ile seçilecek (recall odaklı) — the project owner onayladı.
- Siamese eğitimi: epoch 17/50, val F1=0.5548 (baseline 0.5196), 47 sn/epoch.
- Not: `uv sync` ekstraları siler → `uv sync --all-extras`.

### 2026-11-02 — Hafta 2 tamamlandı
- `KateCDDataset` sınıfı yazıldı (Parquet → tensör, flip/rot90/renk augmentasyonu).
- Early-fusion U-Net (6 kanal, ResNet34, ImageNet): `src/quaketriage/models/unet.py`.
- Kayıp: Dice + Focal (`DiceFocalLoss`); eşik val F1'e göre threshold sweep ile seçildi.
- Eğitim MPS'te 409s/epoch → Kaggle GPU'ya taşındı. Notebook: `kaggle/train.ipynb` (internet + HF_TOKEN Kaggle Secret gerekti).
- Kaggle eğitimi: epoch 33'te erken durdurma, val F1=0.520. Checkpoint `models/baseline_best.pt`.
- `scripts/evaluate.py` yazıldı: bootstrap CI, figür, `reports/metrics_baseline.json`.
- **Test sonuçları (threshold=0.5):** F1=0.490 [0.370–0.582], Precision=0.657, Recall=0.391, IoU=0.325.
- Hasarsız pencere değerlendirmesi Hafta 3'e taşındı (`infer_area.py` ile birlikte yapılacak).
- Kaggle kernel slug: `YOUR_KAGGLE_USERNAME/quaketriage-training` (metadata'daki `quaketriage-train`'den farklı — Kaggle otomatik değiştirdi).
- Sıradaki: Hafta 3 — Siamese model, MC Dropout + TTA, quality guard, RAG metin çıkarımı.

### 2026-10-31 — Hafta 1 tamamlandı
- KATE-CD indirildi (446 MB, 404+44+38 örnek). EDA: yalnızca pre_image/post_image/label sütunları, koordinat yok. Ortalama hasarlı piksel oranı %3.7 (train) — ciddi sınıf dengesizliği.
- Demo bölgesi kararı: **Antakya, yedek plan**. Vantor STAC'ta 76 koleksiyon tarandı; tek örtüşen pre/post çifti Kilis bölgesinde ama orada yalnızca 1 HOTOSM yıkılmış bina var. Antakya'da 732 yıkılmış bina → HOTOSM etiketleri simüle edilmiş dedektör çıktısı olarak kullanılacak.
- Antakya demo verisi indirildi: 1876 OSM binası, 717 yol kenarı, WorldPop 100m (20 MB).
- AFAD TAMP (3 MB), INSARAG Vol II Man B (2.8 MB), INSARAG Vol III (1.1 MB) indirildi.
- `scripts/check_data.py` yeşil (10/10 dosya OK).
- Kararlar `docs/decisions.md`'ye yazıldı.
- Sıradaki: Hafta 2 — baseline U-Net (early-fusion, 6 kanal).
- **Not (B11):** `kaggle/kernel-metadata.json` henüz olmayan `train.ipynb`'yi gösteriyor. Hafta 2'de Kaggle notebook oluşturulunca birlikte düzeltilecek.

### 2026-10-30 — Hafta 0 tamamlandı
- Konu seçildi: deprem sonrası hasar tespiti ve ekip önceliklendirme ajanı (QuakeTriage).
- Git repo init + GitHub private repo bağlandı (Kuipyy/quake-triage-agent).
- pyproject.toml, uv.lock, .gitignore, .env.example, configs/default.yaml oluşturuldu.
- Tam paket iskeleti oluşturuldu (src/quaketriage altında 7 alt paket).
- SDK kararı: `google-generativeai` deprecated → `google-genai`'ya geçildi; model `gemini-2.5-flash`.
- 3 smoke test yeşil (import, MPS/CPU, CRS). Gemini API çağrısı çalışıyor.
- Sıradaki: Hafta 1 — KATE-CD indirme ve EDA.
