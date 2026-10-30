# Sözlük (TR/EN)

Projede geçen terimlerin kısa açıklamaları. Yeni terim geçtikçe buraya eklenir.

| Terim | Açıklama |
|---|---|
| **Change detection / değişim tespiti** | Aynı yerin öncesi ve sonrası görüntüsünü karşılaştırıp neyin değiştiğini bulma. Burada: hasar. |
| **KATE-CD** | Kahramanmaraş depremi için öncesi/sonrası görüntü çiftleri ve ikili hasar maskeleri içeren veri seti. |
| **U-Net** | Görüntü segmentasyonu için encoder-decoder ağı. Her piksel için bir sınıf/olasılık üretir. |
| **Early fusion** | Öncesi ve sonrası görüntüleri girişte kanal olarak birleştirmek (3+3 = 6 kanal). |
| **Siamese ağ** | İki görüntüyü aynı (paylaşılan) encoder'dan geçirip öznitelik farkını kullanan yapı. |
| **Dice / Focal loss** | Dengesiz sınıflarda (hasarlı piksel azınlık) işe yarayan kayıp fonksiyonları. |
| **IoU / F1 / precision / recall** | Segmentasyon başarısı. Recall: gerçek hasarlıların kaçı bulundu. Precision: bulunanların kaçı gerçekten hasarlı. |
| **MC Dropout** | Test sırasında dropout'u açık bırakıp aynı girdiyi T kez çalıştırmak. Çıktıların ortalaması tahmin, standart sapması belirsizliktir. |
| **TTA (test-time augmentation)** | Girdiyi çevirip/döndürüp tahminleri karşılaştırmak. Belirsizlik için ucuz bir alternatif. |
| **ECE (expected calibration error)** | Model "%80 eminim" dediğinde gerçekten %80 doğru mu? Kalibrasyon hatasının ortalaması. |
| **Guard (kalite bekçisi)** | Bulutlu, boş veya hizasız görüntüde karar vermeyi engelleyen kontrol. |
| **SSIM** | İki görüntünün yapısal benzerliği. Değişmemesi gereken alanlarda hizalamayı kontrol etmek için kullanılır. |
| **CRS / UTM 37N** | Koordinat sistemi. UTM metre cinsindendir; mesafe/tampon hesapları için gerekli. Proje: `EPSG:32637`. |
| **COG (Cloud Optimized GeoTIFF)** | Parça parça okunabilen GeoTIFF. Tüm dosyayı indirmeden sadece bir pencere alınır. |
| **STAC** | Uydu görüntüsü kataloglarının standart arama arayüzü. |
| **Zonal istatistik** | Bir poligonun (bina) içine düşen raster piksellerinin özeti (ortalama olasılık gibi). |
| **WorldPop** | Izgara hücresi başına tahmini nüfus veren açık veri seti. |
| **TOPSIS** | Çok kriterli sıralama: en iyi çözüme en yakın, en kötüye en uzak seçenek öne çıkar. Ağırlıklar ayarlanabilir. |
| **AHP** | İkili karşılaştırmalarla kriter ağırlığı üretme yöntemi. |
| **Rank reversal** | Seçenek eklenince/çıkınca TOPSIS sıralamasının beklenmedik değişmesi. Duyarlılık analiziyle izlenir. |
| **VRP / prize-collecting VRP** | Araç rotalama problemi. "Prize-collecting": her noktayı ziyaret etmek zorunlu değil, süre içinde toplanan ödül (öncelik) en büyütülür. |
| **OR-Tools** | Google'ın optimizasyon kütüphanesi (rotalama çözücüsü). |
| **Greedy baseline** | Her adımda en yakın noktaya gitme kuralı. Optimizasyonun gerçekten fayda sağladığını göstermek için kıyas. |
| **RAG** | Cevap üretmeden önce belgelerden ilgili parçaları getirip modele vermek. Atıf (`[kaynak#bölüm]`) bu parçalara bağlanır. |
| **Chunk** | RAG için bölünmüş belge parçası (başlık/bölüm bazlı). |
| **Embedding** | Metnin sayı vektörüne çevrilmiş hali; benzer anlamlar yakın vektörler olur. |
| **Hit@k** | Doğru parça ilk k sonuç içinde mi? Retrieval kalitesi ölçüsü. |
| **LangGraph** | Ajan iş akışını düğümler ve kenarlar olarak tanımlayan kütüphane. |
| **Tool trace (araç izi)** | Ajanın hangi aracı hangi girdiyle çağırdığının kaydı. |
| **What-if** | "Ekip sayısı 2 artarsa?" gibi kısıt değişikliğiyle planı yeniden kurma. |
| **Human-in-the-loop** | Model emin değilken kararı insana (saha doğrulaması) bırakma. |
| **ODbL / CC BY-NC 4.0 / CC BY 4.0** | Veri lisansları. BY-NC: ticari kullanım yasak. ODbL: atıf + aynı lisansla paylaşım. |
| **Bootstrap güven aralığı** | Küçük test setinde metriğin ne kadar oynayabileceğini örnekleme tekrarıyla tahmin etme. |
| **Provenance / lineage** | Bir sonucun hangi kod, config ve girdiden çıktığının kaydı. |
