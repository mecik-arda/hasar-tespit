# HADES DETECTOR (Araç Hasar Tespiti) - Proje Anayasası

## Mimari ve Kapsam
Bu proje, araç hasarlarını tespit eden gelişmiş bir Çoklu-Model (YOLOv12x, RT-DETR-v2, SAM 2, Florence-2) orkestrasyon sistemidir. 
Sistemde "CLIP Akıllı Yönlendirici (AI Router)" kullanılarak görseller önce filtrelerden geçer, ardından uygun kanallara (Tek Model veya Çoklu Model) yönlendirilir.

## Kesin ve Değiştirilemez Kurallar
- **Sıfır Yorum Satırı:** Python, C++, Bash vb. hiçbir dosyada #, //, /* */ gibi yorum satırları KESİNLİKLE kullanılamaz. Sadece docstring (`"""`) kullanılabilir.
- **Self-Documenting (Öz-Açıklayıcı) Kod:** Kodun ne yaptığını değişken ve fonksiyon isimleri anlatmalıdır. Yorum satırına ihtiyaç duyuyorsan, kodu kötü yazmışsındır.
- **%100 Türkçe İsimlendirme:** image_path yerine gorsel_yolu, 	rain_model yerine model_egit gibi tamamen Türkçe ve anlaşılır isimlendirmeler zorunludur.
- **Placeholder (Geçici Kod) Yasaktır:** // TODO: burayı doldur gibi yarım bırakılmış bloklar yazılamaz. Kod daima çalıştırılabilir ve eksiksiz olmalıdır.
- **Emoji ve Tablo Yasakları:** Teknik metinlerde emoji kullanılmaz. Ayrıca Medium veya benzeri platform yazılarında markdown tabloları (|---|) yerine hiyerarşik (nested) listeler kullanılmalıdır.

## Çalışma Prensibi
Büyük batch taramalarında (Çoklu-Model), VRAM taşmasını önlemek için **Horizontal Batching (Yatay Toplu Tarama)** ve **Chunking (50'şerlik paketler)** yapısı kullanılır. Modeller dikey (her görsel için yükle-sil) değil, yatay (yükle -> tüm paketi tara -> sil) çalışır.
