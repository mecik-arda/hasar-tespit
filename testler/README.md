# Hades Scanner Test Altyapısı

Bu klasör, Hades Scanner sisteminin tüm bileşenlerini (Donanım, Veri İşleme, Model Eğitimi, Çıkarım vb.) uçtan uca doğrulamak ve hatalara karşı genel direncini ölçmek amacıyla oluşturulmuştur.

## Testleri Çalıştırma

Oluşturulan tüm birim (unit) ve entegrasyon testlerini, CLI arayüzünden (`main.py`) "8" numaralı **Sistem Testlerini Çalıştır** seçeneğiyle topluca tetikleyebilir veya terminalden manuel olarak koşturabilirsiniz:

```bash
python -m unittest discover -s testler -p "test_*.py"
```

## Mevcut Test Modülleri

Sistemde toplamda 17 alt test koşulunu barındıran aşağıdaki modüller yer almaktadır:

### 1. Temel İşlevler
* **`test_donanim.py`:** CPU/RAM/GPU değerlerinin donanım tespit sistemleri tarafından başarıyla yakalanıp optimizer önerileri (cihaz ve batch_size) sunduğunu doğrular.
* **`test_veri_araclari.py`:** `config.yaml` dosyasının bütünlüğünü, veri seti parametrelerini ve tanımlanan hasar sınıflarının tutarlılığını test eder.

### 2. İleri Düzey Entegrasyon ve Kararlılık
* **`test_performans.py`:** Modelin optimize edilmesi (farklı formatlara dönüştürülmesi) sırasındaki fonksiyon hızlarını ve başarı durumunu denetler.
* **`test_dayaniklilik.py`:** Modele yapay olarak zifiri karanlık, yüksek gürültülü(noise) görseller besleyerek zorlu ortamlardaki kararlılığını test eder. Türkçe karakterli yol (path) kaynaklı sistem çökmelerinin yaşanmamasını denetler. (Geçici sanal test resimleri kullanılır).
* **`test_gecersiz_girdi.py`:** Sahte uzantılı, boş veya silinmiş/bulunamayan dosyalar girildiğinde uygulamanın kilitlenmek yerine uyarı vererek güvenli şekilde `None` döndürmesini denetler.
* **`test_egitim_akisi.py`:** Sanal (dummy) bir veri seti üzerinden eğitim döngüsünün baştan sona çalışabildiğini ve en sonunda `best.pt` ağırlığını kaydedip kaydedemediğini denetler.
* **`test_veri_artirimi_dagilimi.py`:** Albumentations fonksiyonlarının ürettiği arttırılmış (augmented) görsellerdeki koordinat kutularının (bounding box) görsel sınırları dışına taşıp taşmadığını test eder (0.0 ile 1.0 sınırları arasında kaldığını kanıtlar).
* **`test_cikarim_tutarliligi.py`:** Aynı görsel üzerinden standart PyTorch formatındaki `.pt` modeli ve çıkarım (inference) altyapısının başarıyla tepki verip çalışabildiğini test eder.
* **`test_yuk_ve_es_zamanlilik.py`:** Sistemi Multithreading (çoklu iş parçacığı) mantığıyla aşırı yükleyip eşzamanlı olarak birçok çıkarım talebi gönderir. Sistemin kaynak sızıntısı yaşatmadan tepki verebilmesini ve kitlenmemesini stres testine sokar.
* **`test_limitler.py`:** Programın parametrelerine negatif ve sınır dışı konfigürasyonlar (-5 epoch, 0 image_size vb.) pompalandığında kodun bunu fark edip çökmek (crash) yerine durumu tolere ederek varsayılan (default) değerlerine döndüğünü kanıtlar.
* **`test_gateway.py`:** CLIP tabanlı Akıllı Yönlendirici (AI Router) modülünün çöp filtresi, kanal yönlendirme, yedek (fallback) mod ve config entegrasyonunu mock'lanmış CLIP modeli ile test eder.

> **Güvenlik Notu:** Testler çalışırken `test_gecici`, `test_stres` gibi çeşitli sanal klasörler oluşturulur ve testlerin başarı durumundan bağımsız olarak `tearDown` metotları vasıtasıyla ortamdan anında otomatik temizlenir. Testler sırasında hiçbir orijinal verinize kalıcı zarar verilmez.
