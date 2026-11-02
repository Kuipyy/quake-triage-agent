# QuakeTriage — Proje Planı

> **Agentic post-earthquake damage assessment & rescue prioritization**
> Uydu görüntüsünden bina bazında hasar tespiti yapan, belirsizliği ölçen, kurtarma ekiplerini önceliklendirip rota planlayan ve kararlarını AFAD/INSARAG belgelerine dayandıran bir LLM ajanı.

- **Sahip:** proje sahibi
- **Süre:** 8 hafta + kurulum (2 Kasım – 27 Aralık 2026), haftada 10–15 saat
- **Bütçe:** 0 TL (yalnızca ücretsiz kaynaklar)
- **Çalışma şekli:** Kodu Claude Code yazar ve çalıştırır. the project owner yönlendirir, karar verir, haftalık notları okur.

---

## 1. Problem ve motivasyon

6 Şubat 2023 Kahramanmaraş depremlerinde ön hasar tespitleri bir aydan uzun sürdü. Akademide uydu görüntüsünden hasar tespiti iyi çalışılmış bir konu, ancak çoğu çalışma bir **hasar haritası** üretip orada duruyor.

QuakeTriage şu soruyu cevaplar:

> "Hasar haritası çıktı. Elimdeki sınırlı ekibi **önce nereye**, **hangi yoldan** göndermeliyim? Model hangi binalarda emin değil?"

**Benzer projelerden farkı** (referans: `Kuipyy/plant-disease-agent`):
1. **Karar üretir.** Önceliklendirme (TOPSIS) ve rota optimizasyonu (OR-Tools) yapar, sadece tavsiye vermez.
2. **Belirsizliği ölçer.** MC Dropout ile ölçülür, emin olunmayan binalar "saha doğrulaması" kuyruğuna gider (human-in-the-loop).
3. **What-if sorularına cevap verir.** "2 ekip daha geldi", "şu yol kapalı" gibi değişikliklerde planı yeniden kurar.
4. **Ajanın kendisi ölçülür.** Senaryo seti üzerinde araç seçimi, görev başarısı ve atıf doğruluğu raporlanır.
5. **Model gerçekçi koşullarda test edilir.** KATE-CD dışındaki demo bölgesinde HOTOSM etiketleriyle değerlendirilir.

---

## 2. Kapsam

### MVP (mutlaka bitecek)
- KATE-CD üzerinde değişim tespiti modeli ve test metrikleri
- MC Dropout belirsizliği ve görüntü kalitesi guard'ı
- Demo bölgesinde bina bazında hasar skoru (OSM/HOTOSM bina sınırlarıyla)
- Nüfus tahmini ve TOPSIS önceliklendirme
- OR-Tools ile ekip rotaları ve basit kurallara karşı kıyaslama
- AFAD TAMP ve INSARAG üzerinde atıflı RAG
- LangGraph ajanı, what-if senaryoları ve araç izi (tool trace)
- Gradio arayüzü (harita, öncesi/sonrası görüntü, öncelik tablosu, rapor)
- Ajan değerlendirmesi ve İngilizce README

### Zaman kalırsa (stretch)
- Araçları **MCP sunucusu** olarak dışa açmak
- xBD alt kümesiyle ön eğitim ve karşılaştırma
- SAR görüntüsü (BRIGHT veri seti) ile çok kaynaklı analiz
- Hugging Face Space'te canlı demo

### Kapsam dışı
- Gerçek zamanlı, operasyonel kullanım. Bu bir **karar destek prototipidir.**
- Bina hasarının çok sınıflı sınıflandırması. KATE-CD ikili etiket içeriyor (hasarlı / hasarsız).

### Geride kalınırsa çıkarılacaklar (sırasıyla)
1. Siamese model (baseline ile devam edilir)
2. Rota optimizasyonu (sadece önceliklendirme kalır)
3. Canlı demo (yerel demo ve video yeterli)

---

## 3. Mimari

```
Kullanıcı (TR/EN): "Bu bölgede ilk 6 saat için 5 ekibi nasıl dağıtayım?"
        │
        ▼
┌─────────────────────── LangGraph Ajanı ───────────────────────┐
│  check_image_quality  → bulut/boş alan, öncesi-sonrası hizalama  │
│        │ kötü kalite → uyarı + "doğrulama kuyruğu"             │
│        ▼                                                     │
│  detect_damage        → değişim tespiti modeli → olasılık + belirsizlik │
│        ▼                                                     │
│  score_buildings      → bina sınırları üzerinden hasar skoru     │
│        ▼                                                     │
│  estimate_occupancy   → WorldPop + kat sayısı → tahmini kişi │
│        ▼                                                     │
│  prioritize_sites     → TOPSIS (ağırlıklar ayarlanabilir)    │
│        ▼                                                     │
│  plan_routes          → OSMnx yol ağı + kapalı yollar → OR-Tools │
│        ▼                                                     │
│  lookup_protocol      → AFAD TAMP / INSARAG RAG (atıflı)     │
│        ▼                                                     │
│  write_sitrep         → durum raporu (Markdown)              │
│                                                              │
│  update_constraints   → what-if: ekip, kapalı yol, ağırlık   │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
Gradio: Harita | Öncesi/Sonrası | Öncelik tablosu | Rapor | Araç izi
```

**Ajan kuralları**
- Kesinlik iddia etmez. Her bina skoru belirsizliğiyle birlikte verilir.
- Protokol önerilerinde mutlaka `[kaynak#bölüm]` formatında atıf yapar.
- Görüntü kalitesi kötü veya belirsizlik yüksekse karar vermez, doğrulama ister.
- Kullanıcının dilinde cevap verir (TR/EN).

---

## 4. Veri kaynakları

Hepsi ücretsiz. Lisanslar README'de mutlaka belirtilecek.

| Kaynak | Kullanım | Erişim | Lisans / not |
|---|---|---|---|
| **KATE-CD** | Model eğitimi ve testi | `huggingface.co/datasets/CSCRS/kate-cd` (Parquet; 404 eğitim / 44 doğrulama / 38 test; 512×512 öncesi/sonrası/maske) | Sayfada lisans yok, makalesine atıf yapılacak (Earthquake Damage Assessment with SAMCD; GitHub: `cscrs/kate-cd`) |
| **Vantor (eski Maxar) Open Data** | Demo bölgesinin koordinatlı öncesi/sonrası görüntüsü | STAC: `https://maxar-opendata.s3.amazonaws.com/events/catalog.json`, `aws s3 ls --no-sign-request s3://maxar-opendata/`, olay listesi: `github.com/opengeos/maxar-open-data` (`Kahramanmaras-turkey-earthquake-23`) | **CC BY-NC 4.0**, ticari kullanım yasak. Arşiv Vantor'a taşınıyor, erişilebilirlik 1. haftada doğrulanacak |
| **HOTOSM yıkılmış binalar** | Bina bazında doğrulama etiketleri | `data.humdata.org/dataset/hotosm_tur_destroyed_buildings` | ODbL (OSM) |
| **HOTOSM / OSM binalar ve yollar** | Bina sınırları, kat sayısı, yol ağı | HDX `hotosm_tur_buildings` veya OSMnx/Overpass | ODbL |
| **WorldPop Türkiye 100 m** | Nüfus tahmini | `hub.worldpop.org/geodata/summary?id=75773` | CC BY 4.0 |
| **AFAD TAMP** | RAG (TR) | `https://www.afad.gov.tr/kurumlar/afad.gov.tr/e_Kutuphane/Planlar/TAMP.pdf` (yedek: `https://mebakub.meb.gov.tr/akub/dosyalar/TAMP.pdf`) | Kamu belgesi |
| **INSARAG Guidelines 2020** | RAG (EN): Vol. II Man. B Operations, Vol. III Field Guide | `https://www.insarag.org/guidelines` | Kamu belgesi. Yeni sürüm 1 Ocak 2027'de yürürlüğe giriyor, README'de belirtilecek |
| xBD (opsiyonel) | Ön eğitim / karşılaştırma | xview2.org (eğitim paketi ~30 GB) | CC BY-NC-SA, sadece stretch hedefte |

**Kritik risk:** KATE-CD görüntüleri büyük olasılıkla koordinat bilgisi içermiyor. Bu yüzden:
- Model **KATE-CD** ile eğitilir.
- Bina, nüfus ve rota adımları **koordinatlı demo bölgesinde** (Vantor GeoTIFF, ~1–2 km², ör. Antakya veya Kahramanmaraş merkez) çalışır.
- **Yedek plan:** Vantor görüntüsü erişilemezse demo bölgesinde HOTOSM "yıkılmış" etiketleri **simüle edilmiş detektör çıktısı** olarak kullanılır. Model başarısı yalnızca KATE-CD'de raporlanır. README'de bu durum açıkça belirtilir.

---

## 5. Teknoloji seti

| Katman | Seçim |
|---|---|
| Ortam | Python 3.11, `uv` (paket ve sanal ortam yönetimi) |
| Derin öğrenme | PyTorch (Mac'te MPS, gerekirse Kaggle GPU), `segmentation-models-pytorch`, `albumentations` |
| Coğrafi veri | `rasterio`, `geopandas`, `shapely`, `pyproj`, `osmnx`, `pystac-client` |
| Optimizasyon | `ortools` (prize-collecting VRP), kendi TOPSIS/AHP kodumuz (numpy) |
| RAG | `pypdf`, `sentence-transformers` (çok dilli: `paraphrase-multilingual-MiniLM-L12-v2`), numpy kosinüs benzerliği |
| Ajan | `langgraph`; LLM katmanı değiştirilebilir: **Gemini** (Google AI Studio ücretsiz katmanı), yedek **Groq** |
| Arayüz | `gradio` (ImageSlider, HTML ile folium haritası) |
| Test ve kalite | `pytest`, `ruff` |
| Eğitim yedeği | Kaggle Notebooks (haftada ~30 saat ücretsiz GPU) + `kaggle` CLI |

**LLM notu:** Model adı `.env` içindeki `LLM_MODEL` değişkeninden okunur, kodda sabit yazılmaz. Ücretsiz katmandaki güncel modeller Google AI Studio'dan kontrol edilir. 429 (rate limit) hatası alınırsa üstel bekleme ile tekrar denenir, olmazsa Groq'a geçilir.

---

## 6. Repo yapısı

```
quake-triage-agent/
├── CLAUDE.md               # Claude Code için kısa kurallar
├── PLAN.md                 # bu dosya
├── PROGRESS.md             # ilerleme günlüğü (her oturum sonunda güncellenir)
├── README.md               # İngilizce, portföy vitrini (8. haftada son hali)
├── pyproject.toml
├── .env.example            # anahtar şablonu (gerçek .env commit EDİLMEZ)
├── .gitignore
├── configs/default.yaml    # yollar, eşikler, TOPSIS ağırlıkları, model ayarları
├── data/                   # gitignored: raw/ interim/ processed/
├── models/                 # gitignored: checkpoint'ler
├── reports/                # metrik JSON'ları + küçük figürler (commit edilir)
├── docs/
│   ├── decisions.md        # önemli kararlar ve gerekçeleri
│   └── notes/week-XX.md    # the project owner için haftalık öğrenme notu (TR)
├── kaggle/                 # kernel-metadata.json + eğitim notebook'u
├── scripts/                # download_*.py, check_data.py, train.py, evaluate.py, infer_area.py
├── src/quaketriage/
│   ├── data/               # kate_cd.py, geo.py, quality.py
│   ├── models/             # unet.py, siamese.py, uncertainty.py
│   ├── buildings/          # zonal.py, occupancy.py
│   ├── planning/           # topsis.py, routing.py, baselines.py
│   ├── rag/                # ingest.py, retrieve.py
│   ├── agent/              # graph.py, tools.py, llm.py, prompts.py
│   └── ui/                 # app.py, maps.py
├── eval/                   # agent_scenarios.yaml, run_agent_eval.py
└── tests/
```

---

## 7. Haftalık plan

Her haftanın sonunda **tamamlanma kriterleri (DoD)** sağlanmalı. Sağlanamazsa PROGRESS.md'ye nedeni yazılır ve "çıkarılacaklar" listesine bakılır.

### Hafta 0 — Kurulum (ilk oturum, ~1–2 saat)
- [x] Git bağlantısı: `gh auth status` kontrolü → `git init -b main` → `git remote add origin https://github.com/<kullanıcı>/quake-triage-agent.git` → ilk commit → `git push -u origin main`. Bundan sonra **her tamamlanan madde ayrı bir commit olarak push edilir** (CLAUDE.md'deki kural).
- [x] `uv` kurulumu, `pyproject.toml`, temel bağımlılıklar
- [x] Repo iskeleti (bölüm 6), `.gitignore` (data/, models/, .env, outputs/, *.tif, *.pt)
- [x] `.env.example`: `GEMINI_API_KEY`, `GROQ_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`, `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
- [x] the project owner `.env` dosyasını kendisi doldurur. Anahtarlar sohbete yazılmaz.
- [x] Smoke test: `uv run pytest -q` geçer, PyTorch MPS/CPU kontrolü yapılır, Gemini'ye tek bir test çağrısı gider
- **DoD:** Repo GitHub'da, testler yeşil, LLM çağrısı çalışıyor. ✓

### Hafta 1 (2–8 Kasım) — Veri ve keşif
- [x] KATE-CD → `data/raw/kate_cd/`. EDA: sınıf dengesi (hasarlı piksel oranı), örnek ızgara figürü → `reports/figures/`
- [x] KATE-CD'de şehir veya koordinat bilgisi var mı kontrol et, sonucu `docs/decisions.md`'ye yaz
- [x] Vantor STAC'ta Kahramanmaraş olayını sorgula. Öncesi ve sonrası görüntüsü olan, HOTOSM yıkılmış bina yoğunluğu yüksek 1–2 km²'lik bir **demo bölgesi** seç. COG'dan sadece o pencereyi oku (tam dosya indirme).
- [x] Demo bölgesi için: HOTOSM yıkılmış binalar ve bina sınırları, WorldPop raster kırpması, OSMnx yol ağı (önbelleğe alınmış)
- [x] TAMP ve INSARAG PDF'leri → `data/raw/docs/`
- [x] `scripts/check_data.py`: tüm dosyaların varlığını ve boyutlarını doğrular
- **DoD:** `check_data.py` geçer, demo bölgesi kararı (veya yedek plan) `docs/decisions.md`'de yazılı. ✓

### Hafta 2 (9–15 Kasım) — Baseline model
- [x] Dataset sınıfı (Parquet → tensör), augmentasyon (flip, rot90, hafif renk oynaması; aynı geometrik dönüşüm öncesi/sonrası/maskeye birlikte uygulanır)
- [x] **Baseline:** early-fusion U-Net (6 kanal girdi, ResNet34 encoder, ImageNet ağırlıkları)
- [x] Kayıp: Dice + Focal. Eşik doğrulama setinde F1'e göre seçilir.
- [x] Eğitim: önce Mac MPS'te. Epoch başına 3 dakikadan uzun sürerse Kaggle GPU'ya geçilir (`kaggle/` + `kaggle kernels push`). → MPS 409s/epoch → Kaggle notebook hazırlandı.
- [x] Değerlendirme: hasar sınıfı için IoU, F1, precision, recall → `reports/metrics_baseline.json`, tahmin örnekleri figürü
- [ ] **Ek değerlendirme (hasarsız pencereler):** Demo bölgesinde HOTOSM etiketlenmeyen alanları "hasarsız" varsayarak sliding window uygula; yanlış pozitif oranını ayrıca raporla. *Eğitim veri setini değiştirme, yalnızca bu ek metriği ekle.*
- **DoD:** Test metrikleri JSON'da, figür hazır, checkpoint `models/` altında. ✓ (yalnızca hasarsız pencere değerlendirmesi Hafta 3'e taşındı)

### Hafta 3 (16–22 Kasım) — Siamese model, belirsizlik, guard, RAG ön işlemesi
- [x] **Siamese U-Net** (eğitildi; `reports/comparison_siamese_vs_baseline_f1.json`: ΔF1 +0.104 [+0.038, +0.166]): paylaşılan encoder, her ölçekte |f_öncesi − f_sonrası| farkı decoder'a verilir. Baseline ile karşılaştırma tablosu.
- [x] **Belirsizlik** (karar: 8 görünümlü TTA, bkz. `docs/decisions.md`): SMP decoder'ına dropout katmanı ekle. MC Dropout (T=10) ve TTA (flip/rot90 kombinasyonları) yöntemlerini ECE ve güvenilirlik diyagramıyla karşılaştır; kazanan yöntem `docs/decisions.md`'ye yazılır.
- [x] **Görüntü kalitesi guard'ı** (`quality.py`): boş/siyah alan oranı, aşırı parlak/bulutlu piksel oranı, değişmemesi beklenen alanlarda SSIM ile hizalama skoru. Birim testleri yazılır.
- [x] `scripts/infer_area.py` (Kilis kesitinde gerçek görüntüyle denendi; Antakya'da görüntü yok, bkz. karar): demo bölgesinde kayan pencere (sliding window) ile çıkarım → hasar olasılığı ve belirsizlik GeoTIFF'leri
- [x] **RAG ön işlemesi — başlangıç:** INSARAG Vol II/III için metin katmanlı PDF alternatif ara (insarag.org güncel sürüm veya HTML). Bulunamazsa `pytesseract` + İngilizce dil paketi ile OCR uygula. TAMP için `pypdf` doğrudan yeterli (metin katmanı doğrulandı). Ham metin ve metadata (belge, bölüm, sayfa) `data/interim/rag_text/` altına kaydedilir.
- **DoD:** Karşılaştırma tablosu, demo bölgesi rasterları, guard testleri yeşil, ham RAG metni hazır.

### Hafta 4 (23–29 Kasım) — Bina bazında skor, nüfus, RAG chunk/embed
- [x] Bina sınırları içinde zonal istatistik (`buildings/zonal.py`, testli; Antakya'da görüntü yok, gerçek raster'a yalnızca Kilis kesitiyle uygulanabilir): `damage_prob` (ortalama), `uncertainty` (ortalama std), `damaged` (eşik)
- [x] Bina bazında değerlendirme: HOTOSM etiketleri doğru kabul edildi, dedektör **simüle edildi** (gürültülü); precision/recall/kişi-ağırlıklı recall duyarlılık ızgarası `reports/metrics_building.json` (model doğruluğu değil, planlama hattının dayanıklılığı; bkz. `docs/decisions.md`).
- [x] Nüfus tahmini (`buildings/population.py`; varsayımlar `configs/default.yaml` → `population.*`): WorldPop hücre nüfusu, binalara taban alanı × kat sayısı oranında dağıtılır. Gece katsayısı eklenir (deprem 04:17'de oldu, konutlar dolu).
- [x] Doğrulama kuyruğu kuralı (`buildings/queue.py`, testli; gerçek veriye bağlanınca uygulanacak): belirsizlik > τ veya guard uyarısı olan bina "saha doğrulaması gerekli" olarak işaretlenir.
- [x] **RAG chunk + embed** (e5-small + BM25 hibrit; recall@1 0.8, MRR 0.80; bkz. `docs/decisions.md`): Hafta 3'te hazırlanan ham metni bölüm bazında parçala (chunk). Metadata: belge, bölüm, sayfa. Çok dilli embedding (`paraphrase-multilingual-MiniLM-L12-v2`), numpy kosinüs benzerliği ile arama. Retrieval doğruluğu küçük altın setinde (10 soru-cevap) ölçülür.
- **DoD:** `data/processed/buildings.gpkg` (tüm alanlarla; GeoJSON yalnızca `reports/` altında harita çıktısı için WGS84'e çevrilerek üretilir), `reports/metrics_building.json`, harita önizleme HTML'i, RAG retrieval testi yeşil.

### Hafta 5 (30 Kasım – 6 Aralık) — Önceliklendirme ve rota
- [ ] **TOPSIS** kriterleri: hasar olasılığı (fayda), tahmini kişi (fayda), kat sayısı (fayda), depoya mesafe (maliyet). Ağırlıklar `configs/default.yaml`'dan gelir. AHP ikili karşılaştırma ile ağırlık üretme opsiyonu eklenir.
- [ ] Yol ağı: OSMnx sürüş grafı. Yüksek hasarlı ve çok katlı binalara X metreden yakın yol parçaları "enkaz riski" ile kapalı veya cezalı sayılır.
- [ ] **OR-Tools prize-collecting VRP:** K ekip, depo noktaları, süre limiti T (ör. 6 saat), bina başına müdahale süresi. Ziyaret edilmeyen binanın cezası = öncelik skoru. Yani amaç, süre içinde toplanan öncelik skorunu en büyük yapmak.
- [ ] Karşılaştırma için basit kurallar: "en yakına git" (greedy) ve rastgele. Metrik: T saatte ulaşılan tahmini kişi sayısı ve toplanan öncelik.
- **DoD:** Plan JSON'ı + rota haritası, kıyaslama tablosu `reports/metrics_planning.json`, testler yeşil.

### Hafta 6 (7–13 Aralık) — Ajan
- [ ] `llm.py`: Gemini/Groq adaptörü, yeniden deneme (retry), basit önbellek
- [ ] `tools.py`: 9 araç (mimari bölümündeki liste), her biri saf Python fonksiyonu ve pydantic şeması ile
- [ ] `graph.py`: LangGraph StateGraph. Durum: bölge, sonuçlar, kısıtlar, araç izi.
- [ ] What-if: `update_constraints` aracı yalnızca değişen adımları yeniden çalıştırır (ör. ekip sayısı değiştiyse sadece rota)
- [ ] CLI: `uv run python -m quaketriage.agent "..."`. 5 örnek konuşma uçtan uca çalışır.
- **DoD:** 5 senaryo çalışıyor, durum raporu (sitrep) atıflı üretiliyor.

### Hafta 7 (14–20 Aralık) — Değerlendirme
- [ ] `eval/agent_scenarios.yaml`: 30–40 senaryo. Her biri için beklenen araçlar, kısıtlar ve "yasak davranış" (ör. guard uyarısına rağmen karar verme). Normal, what-if, kötü görüntü, protokol sorusu ve konu dışı soru tiplerini kapsar.
- [ ] Ajan metrikleri: görev başarısı, araç seçim doğruluğu, atıf geçerliliği (atıf yapılan bölüm var mı ve iddiayı destekliyor mu), ortalama gecikme, LLM çağrı sayısı
- [ ] Ablasyonlar: baseline vs Siamese; belirsizlik filtresi açık vs kapalı (yanlış pozitif bina sayısına etkisi)
- [ ] Hata analizi: en kötü 10 bina ve görsel inceleme
- [ ] README sonuç bölümü, sınırlamalar ve etik notu
- **DoD:** `reports/agent_eval.json` hazır, README sonuçları dolu.

### Hafta 8 (21–27 Aralık) — Arayüz ve yayın
- [ ] Gradio arayüzü:
  - sol: demo bölgesi seçimi + sohbet
  - sağ sekmeler: Harita (hasar katmanı, öncelik işaretleri, rotalar) / Öncesi-Sonrası (ImageSlider) / Öncelik tablosu / Durum raporu (indirilebilir) / Araç izi / Model kartı
- [ ] Ekran görüntüleri, 60–90 saniyelik demo GIF/video
- [ ] README (İngilizce): problem, mimari diyagramı, sonuç tabloları, kurulum, veri lisansları, sınırlamalar, etik
- [ ] `LICENSE` (kod: MIT; veri lisansları ayrıca belirtilir), model kartı
- [ ] Yayın kontrol listesi: geçmişte sır yok (`git log -p | grep -i key`), büyük dosya yok, lisanslar doğru
- [ ] (Opsiyonel) Hugging Face Space (ZeroGPU). Ücretsiz hesap 30 günden eski olmalı.
- **DoD:** `uv run python -m quaketriage.ui.app` çalışıyor, README tam, repo public yapılmaya hazır.

---

## 8. Değerlendirme özeti

| Seviye | Metrikler | Veri |
|---|---|---|
| Piksel | IoU, F1, precision, recall (hasar sınıfı); ECE; belirsizlik yöntemi karşılaştırması (MC Dropout vs TTA) | KATE-CD test |
| Piksel (hasarsız) | Yanlış pozitif oranı (demo bölgesi, etiketlenmemiş alanlar) | Demo bölgesi sliding window |
| Bina | Precision, recall, F1 | Demo bölgesi + HOTOSM yıkılmış etiketleri |
| Planlama | T saatte ulaşılan tahmini kişi, toplanan öncelik, çözüm süresi | Greedy ve rastgeleye karşı |
| Ajan | Görev başarısı, araç doğruluğu, atıf geçerliliği, gecikme, LLM çağrı sayısı | 30–40 senaryo |

---

## 9. Riskler ve önlemler

| Risk | Önlem |
|---|---|
| Vantor arşivinde Kahramanmaraş görüntüsü yok | Yedek plan (bölüm 4): HOTOSM etiketleriyle simüle edilmiş detektör + KATE-CD metrikleri |
| KATE-CD küçük (404 eğitim çifti), aşırı öğrenme | Güçlü augmentasyon, ImageNet ön eğitimi, erken durdurma |
| Görüntü hizalama hatası | Guard'daki SSIM kontrolü, kötü bölgeler doğrulama kuyruğuna gider |
| Gemini ücretsiz limiti | Retry + önbellek + Groq yedeği; değerlendirme koşuları gece ve parça parça yapılır |
| Mac'te eğitim yavaş | Kaggle GPU (kernel push) |
| Koordinat sistemi hataları | Tüm vektör/raster işlemleri tek CRS'de (demo bölgesinin UTM zonu). `geo.py`'de yardımcı fonksiyonlar ve testler |
| Zaman aşımı | Bölüm 2'deki "çıkarılacaklar" sırası |

---

## 10. Etik ve sınırlamalar (README'ye taşınacak)
- Bu bir **araştırma ve karar destek prototipidir**, gerçek arama-kurtarma operasyonlarında kullanılmak üzere doğrulanmamıştır.
- Nüfus tahminleri kaba bir yaklaşıktır. WorldPop 100 m çözünürlükte ve deprem öncesi döneme ait.
- Vantor görüntüleri CC BY-NC 4.0 lisanslıdır, ticari kullanım için başka bir görüntü kaynağı gerekir.
- Model tek bir depremin verisiyle eğitildi. Başka şehir veya yapı tiplerine genellenmesi garanti değildir.

---

## 11. Çalışma ritmi (the project owner + Claude)

**Her oturumun başında** the project owner şöyle yazar:
> `PROGRESS.md'yi ve PLAN.md'de sıradaki haftanın bölümünü oku, kaldığımız yerden devam et.`

**Her oturumun sonunda Claude:**
1. `PROGRESS.md`'yi günceller: yapılanlar, metrikler, kararlar, sıradaki adım
2. Hafta bittiyse `docs/notes/week-XX.md` yazar: 1 sayfa, Türkçe, "ne yaptık, neden, mülakatta nasıl anlatılır"
3. Commit ve push yapar

**Token ve hız tasarrufu**
- Konu değiştiğinde `/clear`, uzun oturumda `/compact` kullanılır.
- Planlama ve zor hatalarda güçlü model, rutin kodlamada Sonnet tercih edilir (`/model sonnet`).
- Scriptler özet basar. Uzun loglar dosyaya yazılır, gerekirse sadece son satırlar okunur.
- PLAN.md her oturumda baştan sona okunmaz, sadece ilgili hafta okunur.

---

## 12. Proje sahibinin yapacakları (manuel adımlar)

### İlk oturumdan önce (~45 dakika)
- [ ] GitHub'da **private** ve **boş** repo aç: `quake-triage-agent` (README, .gitignore ve lisans **ekleme**)
- [ ] Bilgisayarda `git`, `gh` (GitHub CLI) ve Claude Code kurulu olmalı. `gh auth login` ile giriş yap.
- [ ] Google AI Studio → **Get API key** (ücretsiz, kart gerekmez)
- [ ] Kaggle: hesap aç → Settings → **Phone verification** (GPU için zorunlu) → Settings → API → **Create New Token**
- [ ] Hugging Face: hesap aç (**şimdi aç**, Space için hesabın 30 günden eski olması gerekiyor) → Settings → Access Tokens → Read token
- [ ] (Opsiyonel) Groq: console.groq.com → API key

### Hafta 0 sırasında
- [ ] Claude `.env` dosyasını oluşturunca anahtarları **bir metin editöründe kendin** yapıştır. Sohbete yazma.

### Her hafta (~20 dakika)
- [ ] `docs/notes/week-XX.md`'yi oku. Anlamadığın yeri Claude'a sor.
- [ ] Claude'un sorduğu kararları ver (demo bölgesi, TOPSIS ağırlıkları vb.)

### Hafta 8
- [ ] Demo videosunu izle veya kaydet, README'yi gözden geçir, repoyu **public** yap.
