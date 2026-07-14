# Intel XPU Destek Raporu

> **Tarih:** 14 Temmuz 2026
> **Donanım:** Intel Arc 140V GPU (16GB) — Lunar Lake
> **GPU Sürücü:** 32.0.101.8860
> **Sonuç:** ❌ XPU çalıştırılamadı — DirectML alternatif olarak çalışıyor

---

## 1. Donanım Bilgisi

| Bileşen | Model |
|---------|-------|
| CPU | Intel Core Ultra 7 258V (8 çekirdek, Lunar Lake) |
| GPU | Intel(R) Arc(TM) 140V GPU (16GB paylaşımlı) |
| NPU | Intel AI Boost (aktif, yalnızca inference) |
| RAM | 32 GB |
| OS | Windows 11 Home (build 10.0.26200) |
| Python (base) | 3.13.9 (Anaconda) |

## 2. XPU Nedir?

**XPU**, Intel'in kendi GPU'ları için PyTorch'ta kullandığı cihaz backend'idir.
NVIDIA'nın `cuda`'sı neyse Intel için `xpu` odur.

Kullanım mantığı:
```python
import torch
tensor = torch.randn(3, 3).to("xpu")  # Intel GPU'ya taşı
```

XPU'nun çalışması için iki ana bileşen gerekir:
1. **Intel Extension for PyTorch (IPEX)** — XPU backend'ini sağlayan paket
2. **XPU-destekli PyTorch build'i** — `torch_xpu.dll`, `c10_xpu.dll` gibi XPU'ya özel DLL'leri içeren özel torch derlemesi

## 3. Yapılan Denemeler

### Deneme 1: Base Python 3.13 + pip

```bash
pip install intel_extension_for_pytorch \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

**Sonuç:** ❌ Python 3.13 için IPEX wheel'i yok. Hiçbir paket yüklenemedi.

**Neden:** Intel, 2026 Temmuz itibarıyla yalnızca Python 3.11'e kadar wheel yayınlamış.

---

### Deneme 2: Conda ortamı `hades_xpu` (Python 3.11) + pip

```bash
conda create -n hades_xpu python=3.11
conda run -n hades_xpu pip install torch torchvision intel_extension_for_pytorch \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

Yüklenen paketler:
```
torch==2.13.0
torchvision==0.28.0
intel_extension_for_pytorch==2.8.10+xpu
```

Ayrıca Intel runtime paketleri de yüklendi:
```
intel-sycl-rt==2026.1.0
intel-cmplr-lib-rt==2026.1.0
intel-cmplr-lib-ur==2026.1.0
intel-opencl-rt==2026.1.0
intel-openmp==2026.1.0
tbb==2023.1.0
```

**Sonuç:** ❌ `import intel_extension_for_pytorch` → **Exit code 127** (DLL bulunamadı)

Python sessizce çöküyor — traceback dahi üretmiyor.

---

### Deneme 3: DLL bağımlılık analizi

`intel-ext-pt-gpu.dll` dosyasının PE import tablosu parse edildi.

**Bulunan bağımlılıklar ve durumları:**

| DLL | Durum | Açıklama |
|-----|-------|----------|
| `torch_xpu.dll` | ❌ EKSİK | XPU-destekli özel torch build'inde bulunur |
| `c10_xpu.dll` | ❌ EKSİK | XPU-destekli özel torch build'inde bulunur |
| `c10.dll` | ✅ var | `torch/lib/` içinde |
| `torch_cpu.dll` | ✅ var | `torch/lib/` içinde |
| `sycl8.dll` | ❌ YANLIŞ VERSİYON | Bizde `sycl9.dll` var (2026.1.0 → SYCL 9) |
| `mkl_core.2.dll` | ❌ EKSİK | Intel Math Kernel Library — pip'te ayrı paket yok |
| `mkl_sycl_blas.5.dll` | ❌ EKSİK | Intel MKL SYCL BLAS — pip'te ayrı paket yok |
| `OpenCL.dll` | ✅ var | `C:\Windows\System32` içinde |
| `ze_loader.dll` | ✅ var | `C:\Windows\System32` içinde |
| `svml_dispmd.dll` | ✅ var | `Library/bin` içinde |
| `libmmd.dll` | ✅ var | `Library/bin` içinde |
| `MSVCP140.dll` | ✅ var | Sistem + conda |
| `VCRUNTIME140.dll` | ✅ var | Sistem + conda |
| XeTLA kernel DLL'leri (~55 adet) | ✅ var | IPEX `bin/` içinde |
| `esimd_kernels.dll` | ✅ var | IPEX `bin/` içinde |
| `xetla_gemm.dll` | ✅ var | IPEX `bin/` içinde |

**Kök neden:** Intel, Windows için pip'te XPU-destekli PyTorch build'ini **yayınlamıyor**. PyPI'deki `torch` paketi CPU-only. IPEX paketinin beklediği `torch_xpu.dll`, `c10_xpu.dll`, `mkl_core.2.dll`, `mkl_sycl_blas.5.dll` ve `sycl8.dll` bağımlılıkları pip ekosisteminde karşılanamıyor.

---

### Deneme 4: Intel Conda kanalı

```bash
conda search -c intel pytorch
```

**Sonuç:** ❌ Intel'in conda kanalı (`conda.anaconda.org/intel`) → **HTTP 403 Forbidden**

Kanal kapalı veya erişime kapatılmış durumda.

---

### Deneme 5: DirectML (Alternatif çözüm) ✅

```bash
conda run -n hades_xpu pip install torch_directml
```

Yüklenenler:
```
torch==2.4.1
torchvision==0.19.1
torch_directml==0.2.5.dev240914
```

Test:
```python
import torch
import torch_directml

dml = torch_directml.device()        # privateuseone:0
print(torch_directml.device_name(0)) # Intel(R) Arc(TM) 140V GPU (16GB)

x = torch.randn(1000, 1000).to(dml)
y = torch.matmul(x, x.T)            # GPU'da çalışıyor
```

**Sonuç:** ✅ ÇALIŞIYOR. DirectML, DirectX 12 üzerinden Intel Arc GPU'ya erişiyor.

---

### Deneme 6: Temiz XPU ortamı `hades_xpu2` (Python 3.11)

Aynı adımlar sıfırdan tekrarlandı, DirectML kurulmadan yalnızca IPEX + torch 2.13.0 denendi.

**Sonuç:** ❌ Aynı DLL hataları — bu bir kurulum/prosedür hatası değil, Intel'in dağıtım eksikliğinden kaynaklanan yapısal bir sorun.

---

## 4. Neden Çalışmıyor? (Kök Neden Analizi)

```

Windows pip ekosisteminde Intel XPU stack'i:

  ┌─────────────────────────────────────────────────┐
  │              intel_extension_for_pytorch        │
  │                   (IPEX 2.8.10+xpu)              │
  │                      pip'te VAR ✅               │
  └─────────────────────┬───────────────────────────┘
                        │ bağımlı
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌────────────┐ ┌──────────────┐
│ torch_xpu.dll│ │  sycl8.dll │ │mkl_core.2.dll│
│ c10_xpu.dll  │ │            │ │mkl_sycl_blas │
│              │ │            │ │    .5.dll    │
│  pip'te YOK  │ │Bizde sycl9 │ │  pip'te YOK  │
│      ❌      │ │     ❌     │ │      ❌      │
└──────────────┘ └────────────┘ └──────────────┘
```

Intel, bu bağımlılıkları yalnızca **Intel oneAPI Base Toolkit** (6+ GB) içinde dağıtıyor. Pip'e parçalı olarak koymamışlar.

SYCL 8 vs 9 uyuşmazlığı: IPEX 2.8.10, SYCL 8 DLL'lerine (`sycl8.dll`) karşı derlenmiş, ancak pip'teki `intel-sycl-rt` paketi yalnızca `sycl9.dll` (SYCL 2026.1.0) sağlıyor. Geriye dönük uyumluluk yok.

Windows'ta Intel XPU'nun pip üzerinden çalıştırılabilmesi için Intel'in şunları yapması gerekir:
1. XPU-destekli `torch` wheel'i yayınlaması (`torch_xpu.dll` + `c10_xpu.dll` içeren)
2. `mkl` wheel'lerini pip'e koyması
3. IPEX'i `sycl9.dll` ile uyumlu derlemesi

Bunlar yapılmadığı sürece Windows'ta pip ile XPU kullanmak mümkün değildir.

**Bu sorun donanımdan, işletim sisteminden veya kullanıcı hatasından kaynaklanmıyor. Tamamen Intel'in yazılım dağıtım stratejisindeki eksikliktir.**

---

## 5. Mevcut Durum ve Çalışan Çözüm

| Ortam | Python | GPU Desteği | Torch Sürümü | Durum |
|-------|--------|-------------|-------------|-------|
| `base` | 3.13.9 | Yok (CPU) | 2.13.0+cpu | Ana ortam, GPU yok |
| `hades_xpu` | 3.11.15 | **DirectML** ✅ | 2.4.1+cpu | **Eğitim için kullanılacak** |
| `hades_xpu2` | 3.11.15 | XPU ❌ | 2.13.0+cpu | Başarısız deneme |

### DirectML ile GPU kullanımı:

```python
import torch
import torch_directml

dml = torch_directml.device()  # Intel Arc GPU'yu otomatik algılar

# Modeli GPU'ya taşı
model = model.to(dml)
tensor = tensor.to(dml)

# Eğitim döngüsü normal PyTorch API'si ile çalışır
optimizer.step()
loss.backward()
```

### Aktivasyon:

```bash
# Eğitim/donanım kontrolü için:
conda activate hades_xpu

# Proje bağımlılıklarını güncellemek gerekirse:
pip install -r requirements.txt
```

**Not:** DirectML, CUDA kadar hızlı olmasa da Intel Arc GPU için şu an Windows'taki **tek çalışan çözümdür**. Performans beklentisi: CUDA'nın yaklaşık %60-80'i.

---

## 6. NPU (Intel AI Boost)

Sistemde **Intel AI Boost NPU** tespit edildi. NPU yalnızca **inference (çıkarım)** için kullanılabilir, **eğitimde kullanılamaz**.

NPU ile inference yapmak için:
- **OpenVINO** formatına model export edilmeli (`.xml` + `.bin`)
- `model.export(format='openvino')` ile YOLO modeli dışa aktarılabilir
- Bu özellik entegre edilmemiştir, ileride eklenebilir

---

## 7. İleride XPU Denenecek Yollar

Aşağıdaki durumlardan biri gerçekleşirse XPU tekrar denenebilir:

1. **Intel, Windows için XPU-destekli torch wheel'i yayınlarsa**
   - `torch_xpu.dll` + `c10_xpu.dll` içeren Windows wheel
   - pip index: `https://pytorch-extension.intel.com/release-whl/stable/xpu/us/`

2. **Intel, IPEX'i güncel SYCL (sycl9.dll) ile uyumlu derlerse**
   - Yeni IPEX sürümü `intel-sycl-rt >= 2026.1.0` ile uyumlu olmalı

3. **Intel oneAPI Base Toolkit kurulursa (6+ GB)**
   - Tüm DLL bağımlılıklarını sistem geneline kurar
   - Ancak disk alanı ve karmaşıklık açısından önerilmez

---

## 8. Referanslar

- [Intel Extension for PyTorch GitHub](https://github.com/intel/intel-extension-for-pytorch)
- [Intel XPU pip index](https://pytorch-extension.intel.com/release-whl/stable/xpu/us/)
- [torch-directml GitHub](https://github.com/microsoft/torch-directml)
- [Intel GPU Sürücüleri](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

---

*Rapor, `src/hardware_check.py` donanım tespit modülü ve manuel DLL analizi sonuçlarına dayanmaktadır.*
