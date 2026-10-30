# Mülakat hazırlığı

Her hafta sonunda ilgili bölümü gerçek sayılarla doldurun. Sayı uydurmayın: kaynak `reports/*.json`.

## Proje özeti (30 saniye)
> Depremden sonra hasar haritası çıkarmak yetmiyor; kısıtlı ekibi nereye, hangi sırayla ve hangi yoldan göndereceğimizi de bilmek gerekiyor. Uydu görüntüsünden bina bazında hasar tespiti yapıyorum, modelin emin olmadığı binaları saha doğrulamasına yönlendiriyorum, kalanları TOPSIS ile önceliklendirip OR-Tools ile rota planlıyorum. Ajan, protokol önerilerini AFAD TAMP ve INSARAG belgelerine atıfla veriyor.

## Sık sorulacak sorular

**Neden sadece segmentasyon değil de ajan?**
Kullanıcı sorusuna göre farklı araçlar gerekiyor (hasar, nüfus, rota, protokol). Ajan hangi aracı ne zaman çağıracağına karar veriyor; sayısal işleri LLM değil, test edilmiş Python fonksiyonları yapıyor.

**LLM sayıları uydurur mu?**
Sayılar araçlardan gelir; LLM sadece araç seçer ve sonucu anlatır. Atıflar otomatik doğrulanır (atıf yapılan bölüm gerçekten var mı, iddiayı destekliyor mu).

**Belirsizliği nasıl ölçtün, neden önemli?**
TBD (Hafta 3). Yanlış öncelik can kaybına yol açabilir; belirsiz binalar karar yerine doğrulama kuyruğuna gider.

**Veri seti küçük (404 eğitim çifti), nasıl aşırı öğrenmeyi önledin?**
TBD (Hafta 2): ImageNet ön eğitimi, augmentasyon, erken durdurma. Test seti 38 görüntü olduğu için güven aralığı raporlanır.

**KATE-CD'de koordinat yok, bina bazında nasıl çalışıyor?**
Model KATE-CD ile eğitildi; koordinatlı Vantor demo bölgesine uygulanıp HOTOSM etiketleriyle doğrulandı. Bu bir alan kayması (domain shift) riski, açıkça raporlandı.

**Neden TOPSIS?**
Çok kriterli sıralama için basit, açıklanabilir. Ağırlıklar config'den gelir ve duyarlılık analiziyle kararlılığı ölçülür. TBD (Hafta 5).

**Rota optimizasyonu gerçekten fayda sağlıyor mu?**
Greedy ve rastgele kurallara karşı T saatte ulaşılan tahmini kişi sayısı kıyaslanır. TBD (Hafta 5).

**Sınırlamalar neler?**
Tek deprem, deprem öncesi nüfus verisi, eksik HOTOSM etiketleri, hasarsız karo olmayan eğitim verisi. Operasyonel kullanım için doğrulanmadı.

**Lisans / etik?**
Vantor görüntüleri CC BY-NC 4.0: ticari kullanım yok. Proje bir karar destek prototipi.
