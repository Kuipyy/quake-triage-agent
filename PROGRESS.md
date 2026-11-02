# PROGRESS — QuakeTriage

> Her oturum sonunda Claude günceller. En yeni kayıt en üstte.

## Durum özeti
- **Aktif hafta:** Hafta 4 — Bina bazında skor, nüfus, RAG chunk/embed (Hafta 3 tamamlandı)
- **Sıradaki adım:** Hafta 4 başında açık soru: Antakya'da gerçek dedektör çıktısı yok; HOTOSM etiketleri girdi *ve* doğrulama olursa bina bazında precision/recall döngüsel olur → the project owner ile çözüm seç (ör. HOTOSM'u gürültü/kayma ekleyerek simüle et, ya da bina doğrulamasını Kilis/KATE-CD dışında raporlama)
- **Bekleyen manuel işler (the project owner):** —

## Önemli metrikler

Metrikler `reports/*.json` dosyalarından otomatik üretilir ve `README.md` sonuç tablosunda gösterilir.
Güncel değerler için: `uv run python scripts/make_report.py --check` → [README.md](../README.md#results)

## Oturum kayıtları

### 2026-11-02 — Hafta 3 tamamlandı
- **Siamese:** Kaggle'da eğitildi (23 epoch, en iyi 13), `models/siamese_best.pt` CLI ile indirildi. Test (thr 0.2, val'de seçildi): F1 0.602 vs baseline 0.502, ΔF1 +0.104 [+0.038, +0.166] (eşleştirilmiş bootstrap, `reports/comparison_siamese_vs_baseline_f1.json`).
- **Eşik kuralı:** val'de seçilir (`evaluate.py --threshold-mode`); operasyon eşiği 0.2. F2 kararsız (ızgara kenarı) → "yüksek recall profili", varsayılan değil.
- **Belirsizlik kararı:** 8 görünümlü TTA > MC Dropout (4 model×split'te Brier ve hata farkındalığı). Dropout'lu 3. Kaggle koşusu atlandı (the project owner onayı).
- **Guard:** SSIM eşiği 0.05 (KATE-CD dağılımı). **infer_area.py:** Siamese + TTA, Kilis kesitinde çalışıyor (doğruluk kanıtı değil).
- **RAG:** ham metin `data/interim/rag_text/` (OCR gerekmedi).
- **Altyapı:** CI yeşil (CPU torch), `metrics.py` birleştirildi (çıktılar birebir aynı), `compare_models.py`, `models/loading.py`.
- Ayrı Kaggle kernel `kaggle/siamese/` (baseline notebook'una dokunulmadı). Not: düz `uv sync` ekstraları siler → `uv sync --all-extras`.
- Notlar: `docs/notes/week-03.md`. Sınırlama: model başına tek koşu, 38 test görüntüsü.

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
- Antakya demo verisi indirildi (**not:** 1876 bina/717 kenar yanlış bölgeydi, Hafta 4'te düzeltildi: 53.028 bina, 30.718 kenar), WorldPop 100m (20 MB).
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
