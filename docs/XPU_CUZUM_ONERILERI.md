# Intel XPU / Arc GPU Çözüm Önerileri

> **Tarih:** 14 Temmuz 2026
> **Kaynak:** `XPU_INTEL_ARC_RAPORU.md` analizine dayanmaktadır
> **Hedef:** Intel Arc 140V GPU ile eğitim ve çıkarım performansını maksimize etmek

---

## Özet

Rapor, Windows'ta Intel XPU'nun pip ile kurulamadığını yapısal bir sorun olarak tespit etmiştir.
DirectML alternatif olarak çalışsa da, mevcut kod entegrasyonunda **kritik hatalar** vardır.
Aşağıdaki çözüm önerileri 3 kategoriye ayrılır: **acil düzeltmeler**, **kısa vadeli iyileştirmeler** ve **uzun vadeli strateji**.

---

## 1. Tespit Edilen Kritik Sorunlar (Kod İncelemesi)

### Sorun A: DirectML Eğitim Entegrasyonu Çalışmıyor (KRİTİK)

`src/train.py` dosyasındaki `_directml_ortamini_hazirla()` fonksiyonu şu mantığı kullanıyor:

```python
torch.set_default_device(dml)   # DirectML'i varsayılan cihaz yap
# Ultralytics'e "cpu" olarak geç  ← HATALI
return "cpu", dml, msg
```

**Bu yaklaşım çalışmaz çünkü:**
- Ultralytics YOLO'nun `model.train()` metodu içeride `model.to(device)` çağırır
- `device="cpu"` geçirildiğinde model ve veriler **açıkça CPU'ya taşınır**
- `torch.set_default_device()`, açık `.to("cpu")` çağrılarını geçersiz kılmaz
- Sonuç: Eğitim GPU'da değil CPU'da çalışır, kullanıcı yanıltılır

### Sorun B: Çıkarım (Inference) Pipeline'ı GPU Kullanmıyor

`src/pipeline.py` dosyasında DirectML/OpenVINO entegrasyonu yoktur.
`model.predict()` varsayılan olarak CPU'da çalışır.
Intel Arc GPU çıkarım için hiç kullanılmaz.

### Sorun C: OpenVINO Export Ediliyor Ama Kullanılmıyor

`src/export.py` Intel Arc için OpenVINO formatına export yapıyor,
ancak `pipeline.py` bu export edilen modeli çıkarım için kullanmıyor.
Sadece `.pt` dosyaları ile çıkarım yapılıyor.

### Sorun D: `requirements.txt` Eksik

`torch_directml` paketi `requirements.txt` dosyasında yok.
Yeni kurulumlarda DirectML otomatik kurulmaz.

### Sorun E: NPU (Intel AI Boost) Entegre Değil

Raporda NPU'nun OpenVINO ile çıkarım için kullanılabileceği belirtilmiş,
ancak kodda NPU çıkarım yolu implement edilmemiş.

---

## 2. Çözüm Önerileri

### Çözüm 1: DirectML Eğitim Yaklaşımını Düzelt (Acil)

**Yaklaşım:** DirectML + Ultralytics `train()` kombinasyonu güvenilmezdir.
DirectML, `torch.set_default_device()` üzerinden Ultralytics ile düzgün çalışmaz.

**Önerilen strateji:**

| Senaryo | Eğitim Cihazı | Açıklama |
|---------|--------------|----------|
| Intel Arc (Windows) | **CPU** | OpenVINO optimizasyonu ile CPU eğitimi |
| Intel Arc + İnternet | **Google Colab** | Ücretsiz CUDA T4 GPU |
| İleride XPU desteği gelirse | **XPU** | Intel torch wheel yayınlandığında |

**Kod değişikliği:**
- `_directml_ortamini_hazirla()` fonksiyonundaki yanıltıcı "DirectML GPU'da eğitim" mesajını kaldır
- DirectML seçildiğinde kullanıcıya dürüst bilgi ver: "DirectML çıkarım içindir, eğitim CPU'da çalışacak"
- Eğitim için OpenVINO-aware CPU optimizasyonları uygula (Intel OpenMP thread ayarı)

```python
# Önerilen düzeltme:
def _directml_ortamini_hazirla(hedef_cihaz):
    if hedef_cihaz != "directml":
        return hedef_cihaz, None, None

    # DirectML çıkarım için kullanılacak, eğitim CPU'da çalışacak
    print("[i] DirectML seçildi. Eğitim CPU'da çalışacak.")
    print("[i] Çıkarım (inference) DirectML GPU ile hızlandırılabilir.")
    
    # Intel CPU optimizasyonu
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))
    
    return "cpu", None, "DirectML modu - eğitim CPU, çıkarım GPU"
```

### Çözüm 2: OpenVINO ile GPU Çıkarım Pipeline'ı Ekle (Yüksek Öncelik)

Intel Arc GPU'yu çıkarımda kullanmanın **en doğru yolu** OpenVINO'dur:

```
.pt modeli → export(format="openvino") → OpenVINO Runtime ile çıkarım
                                           ↓
                                    Intel Arc GPU / NPU üzerinde çalışır
```

**Önerilen implementasyon (`src/pipeline.py`):**

```python
def _model_yukle_optimize(model_yolu, yapilandirma):
    """Modeli yükle, OpenVINO formatı varsa GPU ile çıkarım yap."""
    model_tur = yapilandirma.get("model", {}).get("tur", "yolo")
    
    openvino_yolu = Path(model_yolu).with_suffix("")
    openvino_yolu = openvino_yolu.parent / (openvino_yolu.name + "_openvino_model")
    
    # OpenVINO modeli varsa, GPU çıkarım için kullan
    if openvino_yolu.exists():
        from ultralytics import YOLO, RTDETR
        ModelSinifi = RTDETR if model_tur == "rtdetr" else YOLO
        model = ModelSinifi(str(openvino_yolu))
        # Ultralytics OpenVINO backend otomatik Intel GPU'yu kullanır
        return model, "OpenVINO (Intel Arc GPU)"
    
    # Yoksa normal .pt ile devam et
    from ultralytics import YOLO, RTDETR
    ModelSinifi = RTDETR if model_tur == "rtdetr" else YOLO
    model = ModelSinifi(str(model_yolu))
    return model, "PyTorch (CPU)"
```

### Çözüm 3: NPU Çıkarım Desteği Ekle (Orta Öncelik)

Intel AI Boost NPU, OpenVINO INT8 modeller ile çıkarım yapabilir:

```python
def npu_ile_cikarim_yap(gorsel_yolu, model_yolu):
    """NPU üzerinde çıkarım yap (OpenVINO INT8 gerekir)."""
    from openvino.runtime import Core
    
    core = Core()
    # NPU cihazını bul
    available_devices = core.available_devices
    npu_device = None
    for dev in available_devices:
        if "NPU" in dev:
            npu_device = dev
            break
    
    if npu_device is None:
        print("[-] NPU bulunamadı. CPU kullanılacak.")
        npu_device = "CPU"
    
    # Modeli yükle
    model_path = str(model_yolu).replace(".pt", "_openvino_model")
    compiled = core.compile_model(model_path, npu_device)
    
    # Çıkarım yap...
    # (implementasyon detayları aşağıda)
```

**NPU çıkarım adımları:**
1. Modeli `model.export(format="openvino", int8=True)` ile INT8 olarak export et
2. OpenVINO Runtime ile NPU cihazında yükle
3. Çıkarımı NPU üzerinde çalıştır

### Çözüm 4: Otomatik Export + Çıkarım Akışı (Yüksek Öncelik)

Kullanıcı eğitimden sonra otomatik olarak en uygun çıkarım formatına export yapılmalı:

```
Eğitim tamamlandı
    ↓
Donanım profili kontrol et
    ↓
Intel Arc var? → OpenVINO export (GPU çıkarım)
NPU var? → OpenVINO INT8 export (NPU çıkarım)
NVIDIA var? → TensorRT export (CUDA çıkarım)
Sadece CPU? → ONNX export (CPU çıkarım)
```

### Çözüm 5: `requirements.txt` Güncelle (Acil)

```txt
# Mevcut paketler...
torch_directml>=0.2.5  # Intel Arc / AMD GPU çıkarım için (opsiyonel)
openvino>=2024.0       # Intel GPU/NPU çıkarım için (opsiyonel)
```

**Not:** Bu paketler opsiyonel olmalı, `try/except` ile import edilmeli.

### Çözüm 6: WSL2 Üzerinden XPU Denemesi (Deneysel)

Windows Subsystem for Linux (WSL2) üzerinden Intel XPU denenebilir:

```bash
# WSL2 Ubuntu kur
wsl --install -d Ubuntu-22.04

# WSL2 içinde
pip install torch torchvision intel_extension_for_pytorch \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

**Riskler:**
- WSL2'de GPU geçişi (GPU passthrough) Intel Arc için sınırlı destek
- Lunar Lake (mobile) GPU'lar için WSL2 GPU desteği deneyseldir
- Performans kaybı olabilir

**Öneri:** Bu yol deneysel olarak denenebilir, ancak ana çözüm olarak önerilmez.

### Çözüm 7: Intel oneAPI Base Toolkit Kurulumu (Ağır Sıklet)

Raporda belirtilen DLL bağımlılıklarını çözmek için Intel oneAPI Base Toolkit kurulabilir:

- **Boyut:** 6+ GB
- **İçerik:** MKL, SYCL runtime, OpenCL runtime, compiler araçları
- **Sonuç:** `mkl_core.2.dll`, `mkl_sycl_blas.5.dll`, `sycl8.dll` bağımlılıkları karşılanır

**Ancak:** `torch_xpu.dll` ve `c10_xpu.dll` yine eksik kalır (bu DLL'ler sadece XPU-destekli torch build'inde bulunur).

**Sonuç:** oneAPI Toolkit tek başına XPU'yu çalıştırmaz. XPU-destekli torch wheel'i de gerekli.

### Çözüm 8: Intel AI Analytics Toolkit (Alternatif)

Intel, AI iş yükleri için ayrı bir toolkit sunar:
- Intel Distribution for OpenVINO
- Intel Extension for PyTorch (önceden derlenmiş)
- Intel Neural Compressor

```bash
# Anaconda üzerinden
conda install -c intel intel-extension-for-pytorch
```

**Durum:** Rapor, Intel conda kanalının 403 Forbidden döndüğünü belirtiyor.
Bu yol şu an kapalı görünüyor.

---

## 3. Önerilen İş Akışı (Intel Arc Sistemi İçin)

### Eğitim Akışı:
```
[1] Donanım Kontrolü → Intel Arc + NPU tespit edilir
    ↓
[5] Model Eğitimi → CPU üzerinde (OpenMP optimize)
    VEYA
    Google Colab (CUDA T4 GPU - önerilen)
    ↓
[7] Model export → OpenVINO formatına (Otomatik)
    ↓
    .pt → .xml + .bin (OpenVINO IR)
```

### Çıkarım Akışı:
```
[6] Hasar Tespiti
    ↓
OpenVINO modeli var mı?
    EVET → Intel Arc GPU üzerinde çıkarım (OpenVINO Runtime)
    HAYIR → .pt ile CPU çıkarım (uyarı: export öner)
    ↓
NPU kullanılsın mı?
    EVET → OpenVINO INT8 modeli ile NPU çıkarım
    HAYIR → GPU çıkarım devam et
```

---

## 4. Performans Beklentileri

| Yöntem | Eğitim | Çıkarım | Karmaşıklık |
|--------|--------|---------|-------------|
| CPU (mevcut) | 1x (baz) | 1x (baz) | Düşük |
| DirectML | ❌ Çalışmıyor | ~2-3x | Orta |
| OpenVINO (GPU) | ❌ N/A | ~5-10x | Orta |
| OpenVINO (NPU) | ❌ N/A | ~3-5x | Yüksek |
| Google Colab (CUDA) | ~10-20x | N/A | Düşük |
| XPU (gelecek) | ~8-15x | ~8-15x | Düşük (tek zaman gelirse) |

---

## 5. Öncelik Sıralaması

| Öncelik | Çözüm | Efor | Etki |
|---------|-------|------|------|
| 🔴 Acil | DirectML eğitim mesajını düzelt (Sorun A) | Düşük | Kullanıcı yanıltmasını önler |
| 🔴 Acil | `requirements.txt` güncelle (Sorun D) | Düşük | Kurulum hatasını önler |
| 🟡 Yüksek | OpenVINO çıkarım pipeline'ı (Çözüm 2) | Orta | Çıkarım 5-10x hızlanma |
| 🟡 Yüksek | Otomatik export akışı (Çözüm 4) | Orta | Kullanıcı deneyimi |
| 🟢 Orta | NPU çıkarım desteği (Çözüm 3) | Yüksek | NPU kullanımı |
| 🔵 Düşük | WSL2 XPU denemesi (Çözüm 6) | Yüksek | Deneysel |
| ⚪ İzle | Intel XPU torch wheel bekle (Çözüm 8) | Yok | Yapısal çözüm |

---

## 6. Sonuç

Intel Arc 140V GPU'nun **eğitim için** Windows'ta XPU ile kullanılması şu an mümkün değildir.
Bu Intel'in dağıtım eksikliğidir, donanım yetersizliği değildir.

**Pratik öneri:**
- **Eğitim:** Google Colab (ücretsiz CUDA GPU) veya CPU
- **Çıkarım:** OpenVINO formatına export edip Intel Arc GPU/NPU üzerinde çalıştır
- **Gelecek:** Intel XPU torch wheel yayınlandığında otomatik geçiş yap

OpenVINO çıkarım entegrasyonu, Intel Arc GPU'nun gücünü çıkarımda kullanmanın
en doğru ve desteklenen yoludur.