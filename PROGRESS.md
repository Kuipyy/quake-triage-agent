# PROGRESS — QuakeTriage

> Her oturum sonunda Claude günceller. En yeni kayıt en üstte.

## Durum özeti
- **Aktif hafta:** Hafta 7 — Değerlendirme (senaryo seti hazır)
- **Sıradaki adım:** Hafta 7 madde 2: `eval/run_agent_eval.py` (senaryoları gerçek LLM ile koşturur; paylaşılan yol grafı/RAG, çözücü süresi kısa; metrikler → `reports/agent_eval.json`: görev başarısı, araç doğruluğu, yasak davranış, atıf geçerliliği/destek, gecikme, LLM çağrısı). Tahmini ~45-60 LLM çağrısı: koşmadan önce the project owner'a söyle
- **YAYIN ENGELİ:** KATE-CD görselleri (kate_cd_samples, predictions_*, worst10_*) ağaçtan kaldırıldı ama git GEÇMİŞİNDE duruyor. Repo public yapılmadan önce geçmiş temizlenmeli (git filter-repo + force push, ya da sıkıştırılmış ağaçtan yeni public repo) — the project owner onayıyla, en son adım olarak. Bkz. docs/decisions.md (2026-11-11).
- **Bekleyen manuel işler (the project owner):** yok. Damage gate sonrası tüm planlama çıktıları ve 38 senaryo eval'i 2026-11-11'de yeniden üretildi (eski sürümler `reports/archive/pre_gate/`). Elle protokol incelemesi yeni koşu için TEKRARLANMADI (yalnız sıkı 7/8 var; P04 hâlâ gold sayfayı atıflamıyor)

## Önemli metrikler

Metrikler `reports/*.json` dosyalarından otomatik üretilir ve `README.md` sonuç tablosunda gösterilir.
Güncel değerler için: `uv run python scripts/make_report.py --check` → [README.md](../README.md#results)

## Oturum kayıtları

### 2026-11-11 — Damage gate sonrası yeniden üretim
- Planlama (400 aday, 4 ekip): VRP öncelik 27,80 → 26,03; kişi 3.969 → 3.106; ziyaret 44 → 42. Baselines: öncelik-önce 22,94/2.749; greedy 12,14/430; rastgele 11,72/899. VRP, öncelik-önce kuralından %13 iyi (eskiden %19). Aday kümesindeki kişi 16.372 → 10.309 (hasarsız kalabalık binalar çıktı).
- Depo duyarlılığı: Spearman 0,72/0,84 → 0,9997; ilk-100 örtüşme 0,97/0,93, ilk-500 0,94/0,91 (eskiden 0,99/0,98). Belirsizlik filtresi: FP 153→150 aynı; plan düzeyinde 1 vs 1 FP (eskiden 18 vs 17) — fayda yok. Hata analizi: en kötü 10 karo yine hep kaçırılmış hasar.
- Eval (38 senaryo, soğuk, 163 istek, 1134 sn): görev başarısı 30 → 34/38; araç seçimi 45/48 aynı; ihlal 0; altın sayfa atfı 3/8 → 7/8; destek 12/18 → 21/26. Sağlayıcı: 27 Groq / 25 Gemini cevap (senaryo bazında JSON'da). Başarısız: W04 (yeni), W08, A03, P04. Düzelen: A04, P03, P06, P07, P08. Not: bu fark gate+atıf+protokol düzeltmesinin birleşimi, tek koşu, LLM varyansı var; eski v1 koşusuyla karşılaştırma adil olmayabilir (protokol v2 düzeltmesi de içeride).
- P01/P05 atıf: P01 4/5 destekli (eskiden 3/3), P05 1/1.

### 2026-11-10 — Hafta 6 tamamlandı
- **llm.py** (Gemini + Groq yedeği, backoff, önbellek), **tools.py** (9 araç; guard ve belirsizlik kuralı araç kodunda; her çıktıda SIMULATED_DETECTOR + belirsizlik), **graph.py** (router, koşullu kenarlar, kalite kapısı, what-if'te yalnız bayat adımlar), **CLI** `python -m quaketriage.agent`.
- **Gerçek LLM demosu:** 5 konuşma, 6 LLM çağrısı, hepsi Gemini (Groq yedeği ayrıca tek çağrıyla doğrulandı); `reports/agent_demo_transcripts.md`. Atıflı sitrep: 5 atıf, hepsi indekste (`valid: true`; destek kontrolü Hafta 7).
- **Sitrep satırı:** doğrulama kuyruğu plan dışı, Hafta 5 metrikleri kuyruk dahil.
- Varsayılan protokol sorgusu hasar/triyaj odaklı yapıldı (ilk sorgu ilgisiz pasajlar getiriyordu).
- Notlar: `docs/notes/week-06.md`. Testler yeşil.
- Düzeltmeler: sitrep protokol bölümü (ilgili pasaj, cümle sınırı, İngilizce kaynak notu), yerelleştirilmiş uyarılar, doğrulama kuyruğundaki kişi sayısı, kaba atıf-destek kontrolü, konuşma 4 denetimi; Hafta 8'e plan önbelleği notu.

### 2026-11-10 — Hafta 5 tamamlandı
- **TOPSIS + AHP** (`planning/topsis.py`); demo depo = OSM "AFAD" binası (senaryo varsayımı), duyarlılık `reports/depot_sensitivity.json` (ilk 500 örtüşme 0,97+).
- **Enkaz riski:** hasar > 0,5 ve ≥ 3 kat → 50 m içindeki yol ×3, ≥ 5 kaynak → kapalı (kenarların %7,3 / %1,1'i).
- **VRP:** 4 ekip, 6 sa, 30 dk/bina, ilk N=400 aday, önbellekli mesafe matrisi. Kapanma 1.139 binayı (%19) kopardı → yürüme bacağı (3 km/h, ≤ 500 m) ile 0'a indi.
- **Metrikler** (`reports/metrics_planning.json`, SIMULATED): VRP öncelik 27,80 / 3.969 kişi; greedy 18,54 / 1.453; rastgele 15,43 / 1.455. Varyantlar: kapanma yok 27,82, kapanma var/yürüme yok 26,80, kapanma var/yürüme var 27,80. N=200/400/500/1000 → 27,75/27,80/27,75/27,26.
- **Çıktılar:** `reports/plan_demo.json`, `reports/route_map.html`. Notlar: `docs/notes/week-05.md`. 104 test.
- Karar kayıtları: `docs/decisions.md` (depo, enkaz, VRP/N, yürüme + baselines).
- Rota haritası proje sahibi tarafından doğrulandı (rotalar sokakları takip ediyor). Eklendi: ekip renk anahtarı, araç bacaklarında kapalı kenar olmadığını doğrulayan test, "en yüksek öncelik önce" baseline'ı (23,29 / 3.273 kişi, 36 ziyaret).

### 2026-11-02 — Hafta 4 tamamlandı
- **Veri hatası:** OSM indirme `bbox` sırası yanlıştı (osmnx 2.x) → Hafta 1 verisi yanlış bölgeydi. Düzeltildi: 53.028 bina, 30.718 yol kenarı; `check_data.py` sınır doğrulaması eklendi; indirme scripti retry/backoff ile (Overpass kapasite sorunu yaşandı).
- **Zonal istatistik** (`buildings/zonal.py`), **nüfus** (`buildings/population.py`, gece 0.9 / konut dışı 0.1 varsayımları config'te), **doğrulama kuyruğu** (`buildings/queue.py`); hepsi testli. YAML'da tırnaksız `yes` boolean oluyordu — düzeltildi.
- **Bina bazlı değerlendirme:** bugünkü OSM'de yıkık binalar yok (8 örtüşme) → demo kümesi 732 HOTOSM yıkık + 5.163 yakın OSM binası; dedektör simüle (`buildings/simulate.py`); `reports/metrics_building.json` = gürültü duyarlılığı (SIMULATED işaretli), model doğruluğu iddiası değil.
- **RAG:** bölüm farkında chunk, e5-small + BM25 hibrit; altın sette recall@1 0.8, MRR 0.80 (`reports/metrics_rag.json`, `reports/rag_ablation.json`). Bilinen kayıplar: Vol III tablo metni, İng→Türkçe.
- **DoD:** `data/processed/buildings.gpkg`, `reports/metrics_building.json`, `reports/map_preview.html` (3.2 MB; tarayıcıda doğrulandı).
- Notlar: `docs/notes/week-04.md`. CI yeşil, 76 test.

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

## Hafta 7 — ajan değerlendirme koşusu (2026-11-10)
- Soğuk koşu, 38 senaryo, tek koşu: görev başarısı 30/38, araç seçimi 45/48 tur, yasak davranış ihlali 0, atıf (altın sayfa) 3/8, sarkan atıf 0.
- Başarısız: W08, A03, A04, P03, P04, P06, P07, P08 (`reports/agent_eval.json` → `failed_scenarios`). Beklentiler sonuç görüldükten sonra değiştirilmedi.
- Uyarı: senaryoları ajanı yazan taraf hazırladı; tek koşu; LLM deterministik değil. 236 LLM isteği / 52 başarılı cevap (429 yeniden denemeleri).
- Sıradaki: ablasyonlar (baseline vs Siamese; belirsizlik filtresi açık/kapalı), hata analizi, README sınırlamalar/etik, week-07 notu.

## Hafta 7 — ablasyonlar ve değerlendirme düzeltmeleri (2026-11-10)
- README tablosu yalnız özet metrik; agent_eval JSON'da ayrıntı. Groq 44/52 not düşüldü. Protokol: sıkı 3/8, elle incelenmiş 4/8 (P06 yanlış cevap; kök neden: router sorguyu kısa İngilizce anahtar kelimeye çeviriyor — henüz düzeltilmedi).
- `damage_threshold` yalnız düşürülebilir (test var).
- Ablasyon: Siamese F1 0.60 vs 0.50 (KATE-CD, Hafta 3). Belirsizlik filtresi (SİMÜLE): FP 153→150; hata-duyarlı varsayım (k=5) ile 153→116. Plan düzeyinde fark ~1 bina.
- Sıradaki: hata analizi (en kötü 10 bina + görsel inceleme), README sınırlamalar/etik, week-07 notu.

## Hafta 7 — tamamlandı (2026-11-10)
- Router sorgu düzeltmesi (sorgu = orijinal mesaj) + 8 protokol senaryosu yeniden: katı 3/8 → 7/8, elle 4/8 → 7/8 (hepsi Groq, 80 istek). v1 `reports/archive/agent_eval_v1.json`, yan yana `reports/protocol_v1_vs_v2.json`.
- Belirsizlik filtresi ablasyonu README'de "etkisi ihmal edilebilir, gerçek etiket olmadan ölçülemez" olarak yazıldı; fayda iddiası yok.
- Hata analizi: `reports/error_analysis.json`, `reports/figures/worst10_siamese.png`. README sınırlamalar + etik eklendi. `docs/notes/week-07.md` yazıldı.
- Açık: 【1】 atıf biçimi, UCC açılımı hatası, hasarsız-kalabalık bina önceliği, tam eval yeniden koşusu. Sonraki: Hafta 8 (UI, plan önbelleği, README/LICENSE, release).
