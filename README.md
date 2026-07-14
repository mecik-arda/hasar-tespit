# Hades Scanner (Hasar Tespit Sistemi)

> **Bu proje, Soft İş Çözümleri bünyesinde hazırlanmış bir staj projesidir.**  
> **Oluşturulma Tarihi: 14.07.2026**

Bu proje, görüntü işleme ve derin öğrenme (YOLO) algoritmaları kullanarak araçlar üzerindeki fiziksel hasarları (Çizik, Göçük, Cam Kırığı, Pas vb.) tespit etmek amacıyla geliştirilmiş uçtan uca bir yapay zeka sistemidir.

## Proje Dizini ve Modüller

Projede yer alan temel modüller ve görevleri aşağıda açıklanmıştır:

### `main.py`
Projenin ana giriş noktasıdır. Kullanıcıya interaktif bir Komut Satırı Arayüzü (CLI) sunar. Bütün alt modüllere (donanım testi, veri etiketleme, veri artırımı, veri bölme, eğitim ve çıkarım) buradan tek tuşla erişilir. Menü seçenekleri `1` ile `9` arasında numaralandırılmıştır.

### `src/hardware_check.py`
Sistem kaynaklarını optimize etmekle görevlidir. İçerisinde yer alan fonksiyonlar şunlardır:
* `wmic_gpu_bilgisi_al()`: Sistemdeki harici GPU donanımını tespit eder.
* `donanim_profili_olustur()`: CPU çekirdek sayısını, RAM miktarını ve GPU verilerini toplayarak o anki sistemin kapasitesine uygun optimum batch size (işlem boyutu) ve cihaz önerisinde bulunur.
* `donanim_ozeti_yazdir()`: Toplanan bilgileri kullanıcıya CLI üzerinden formatlı şekilde sunar.

### `src/data_tools.py`
Veri hazırlama ve işleme süreçlerinin omurgasıdır. Şu fonksiyonları içerir:
* `yapilandirma_yukle()`: `config.yaml` dosyasını ayrıştırır.
* `etiketleme_baslat()`: Kullanıcının resimleri etiketleyebilmesi için arka planda `labelImg` aracını başlatır.
* `augmentation_uygula()`: Albumentations kütüphanesi yardımıyla mevcut eğitim setini bulanıklaştırma, karanlıklaştırma ve çevirme teknikleriyle çoğaltır.
* `veri_bol()`: Etiketlenen veri setini eğitim (%80) ve doğrulama (%20) olmak üzere ikiye ayırarak modelin eğitilmesine hazır hale getirir. Klasörler arasında `shutil.move` ile hızlı taşıma yapar.

### `src/train.py`
Modelin eğitilmesi ve raporlanmasından sorumludur:
* `egitim_baslat()`: Girdi parametrelerini (epoch, batch) yapılandırır ve YOLO modelini transfer öğrenimi (transfer learning) yöntemiyle eğitmeye başlar. Çökmeleri önlemek için negatif girdilerde varsayılan değerlere döner.
* `egitim_raporu_goster()`: Tamamlanan eğitimin ardından oluşan metrik dosyalarını bularak terminale yazdırır.

### `src/pipeline.py`
Eğitilmiş model üzerinden çıkarım (inference) işlemlerini yürütür:
* `egitilmis_model_yolu_bul()`: Son çalıştırılan eğitimden kalan en iyi model ağırlığını (`best.pt`) arar.
* `hasar_tespiti_yap()`: Verilen tekil bir görüntüyü modele sokarak tespit edilen hasar koordinatlarını (bounding box) çizer ve sonuçları kaydeder.
* `toplu_hasar_tespiti_yap()`: Özel olarak tasarlanan klasör tarama modülüdür. **`hasar-ornek`** klasöründeki fotoğrafları topluca okuyup otomatik işler ve etiketlenmiş sonuçları tek bir genel JSON raporu eşliğinde **`hasar-sonucu`** klasörüne yazar.

### `src/export.py`
* `optimize_edilmis_model_olustur()`: Eğitilmiş PyTorch modelini donanıma daha hızlı yanıt verecek olan ONNX, OpenVINO veya TensorRT gibi formatlara dönüştürerek dışa aktarır.

### `testler/` Klasörü
Projenin sınırlarını ve hata yönetimini doğrulayan unittest modüllerini barındırır:
* `test_donanim.py`, `test_veri_araclari.py`, `test_performans.py`
* `test_dayaniklilik.py` (Karanlık/Bozuk görsellerde kararlılık)
* `test_gecersiz_girdi.py` (Sahte dosya formatları)
* `test_limitler.py` (Negatif ve geçersiz konfigürasyon girdileri)
* `test_yuk_ve_es_zamanlilik.py` (Paralel işlem/stres toleransı)
* `test_egitim_akisi.py` (Sanal verilerle eğitim döngüsü doğrulaması)

## Kullanım

Sistemi başlatmak için terminalinizde aşağıdaki komutu çalıştırmanız yeterlidir:
```bash
python main.py
```
Açılan menü üzerinden donanımınızı test edebilir, veri setinizi oluşturup bölebilir, modeli eğitebilir ve hasar tespitine başlayabilirsiniz. Tüm testleri (modül stabilitesini) ana menüdeki `8` numaralı seçeneği kullanarak koşturabilirsiniz.
