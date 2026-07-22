import copy
import gc
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from src.utils import PROJE_KOKU, yapilandirma_yukle


GÖRSEL_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
DESTEKLENMIYOR = "Atlandı (Desteklenmiyor)"
VRAM_ERISILEMIYOR = "Erişilemiyor (Sürücü Kısıtı)"


def _json_uyumlu_yap(deger):
    if isinstance(deger, dict):
        return {str(anahtar): _json_uyumlu_yap(alt_deger) for anahtar, alt_deger in deger.items()}
    if isinstance(deger, (list, tuple, set)):
        return [_json_uyumlu_yap(alt_deger) for alt_deger in deger]
    if isinstance(deger, Path):
        return str(deger)
    if isinstance(deger, np.generic):
        return deger.item()
    if isinstance(deger, float) and (np.isnan(deger) or np.isinf(deger)):
        return None
    return deger


MEVCUT_SUREC = psutil.Process(os.getpid())


def bellek_olcu_al():
    sanal_bellek = psutil.virtual_memory()
    sonuc = {
        "toplam_ram_mb": round(sanal_bellek.total / 1048576, 2),
        "kullanilan_ram_mb": round(sanal_bellek.used / 1048576, 2),
        "ram_kullanim_yuzdesi": round(sanal_bellek.percent, 2),
        "surec_ram_mb": round(MEVCUT_SUREC.memory_info().rss / 1048576, 2),
        "cuda_vram": DESTEKLENMIYOR,
        "directml_openvino_vram": VRAM_ERISILEMIYOR,
    }
    try:
        import torch
        if torch.cuda.is_available():
            sonuc["cuda_vram"] = {
                "ayrilmis_mb": round(torch.cuda.memory_allocated() / 1048576, 2),
                "rezerve_mb": round(torch.cuda.memory_reserved() / 1048576, 2),
                "tepe_ayrilmis_mb": round(torch.cuda.max_memory_allocated() / 1048576, 2),
            }
    except (ImportError, RuntimeError):
        pass
    return sonuc


def ortam_profili_olustur():
    cpu_adi = platform.processor() or "Bilinmiyor"
    try:
        import cpuinfo
        cpu_adi = cpuinfo.get_cpu_info().get("brand_raw", cpu_adi)
    except (ImportError, OSError, AttributeError):
        pass
    torch_surumu = "Yüklü değil"
    cuda_surumu = DESTEKLENMIYOR
    cuda_gpu_modelleri = []
    try:
        import torch
        torch_surumu = torch.__version__
        if torch.cuda.is_available():
            cuda_surumu = torch.version.cuda or "Bilinmiyor"
            cuda_gpu_modelleri = [torch.cuda.get_device_name(indeks) for indeks in range(torch.cuda.device_count())]
    except (ImportError, RuntimeError):
        pass
    directml_durumu = DESTEKLENMIYOR
    try:
        import torch_directml
        directml_durumu = torch_directml.device_name(0)
    except (ImportError, RuntimeError, IndexError, AttributeError):
        pass
    openvino_surumu = "Yüklü değil"
    try:
        import openvino
        openvino_surumu = getattr(openvino, "__version__", "Bilinmiyor")
    except ImportError:
        pass
    return {
        "isletim_sistemi": platform.platform(),
        "python_surumu": sys.version.split()[0],
        "torch_surumu": torch_surumu,
        "cpu_modeli": cpu_adi,
        "mantiksal_cekirdek": psutil.cpu_count(logical=True),
        "fiziksel_cekirdek": psutil.cpu_count(logical=False),
        "ram": bellek_olcu_al(),
        "cuda_surumu": cuda_surumu,
        "cuda_gpu_modelleri": cuda_gpu_modelleri,
        "directml": directml_durumu,
        "openvino_surumu": openvino_surumu,
        "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _gecikme_istatistikleri(sureler):
    temiz_sureler = [float(sure) for sure in sureler if sure is not None and float(sure) >= 0]
    if not temiz_sureler:
        return {
            "ornek_sayisi": 0,
            "ortalama_saniye": None,
            "minimum_saniye": None,
            "maksimum_saniye": None,
            "standart_sapma_saniye": None,
            "p95_saniye": None,
            "fps": None,
        }
    ortalama = statistics.fmean(temiz_sureler)
    return {
        "ornek_sayisi": len(temiz_sureler),
        "ortalama_saniye": round(ortalama, 6),
        "minimum_saniye": round(min(temiz_sureler), 6),
        "maksimum_saniye": round(max(temiz_sureler), 6),
        "standart_sapma_saniye": round(statistics.pstdev(temiz_sureler), 6),
        "p95_saniye": round(float(np.percentile(temiz_sureler, 95)), 6),
        "fps": round(1.0 / ortalama, 4) if ortalama > 0 else None,
    }


def _gorsel_orijinal_mi(gorsel_yolu):
    yol = Path(gorsel_yolu)
    if any(parca.lower() == "augmented" for parca in yol.parts):
        return False
    return "_aug" not in yol.stem.lower()


def _gorselleri_listele(klasor, artirilmis_dahil=True):
    klasor = Path(klasor).resolve()
    if not klasor.exists():
        return []
    gorseller = [yol for yol in klasor.rglob("*") if yol.is_file() and yol.suffix.lower() in GÖRSEL_UZANTILARI]
    if not artirilmis_dahil:
        gorseller = [yol for yol in gorseller if _gorsel_orijinal_mi(yol)]
    return sorted(gorseller, key=lambda yol: str(yol).lower())


def _miktari_uygula(gorseller, miktar):
    if miktar is None:
        return gorseller
    return gorseller[:max(0, int(miktar))]


def etiketli_veri_kaynagi_bul():
    val_gorsel_klasoru = PROJE_KOKU / "data" / "images" / "val"
    val_etiket_klasoru = PROJE_KOKU / "data" / "labels" / "val"
    if _gorselleri_listele(val_gorsel_klasoru) and val_etiket_klasoru.exists():
        return val_gorsel_klasoru, val_etiket_klasoru, "data/images/val + data/labels/val"
    etiketli_klasor = PROJE_KOKU / "hasar-ornek-labelli"
    return etiketli_klasor, etiketli_klasor, "hasar-ornek-labelli"


def _etiket_yolu_bul(gorsel_yolu, gorsel_koku, etiket_koku):
    gorsel_yolu = Path(gorsel_yolu)
    etiket_koku_cozulum = Path(etiket_koku).resolve()
    try:
        goreli_yol = gorsel_yolu.relative_to(gorsel_koku).with_suffix(".txt")
        aday = (etiket_koku_cozulum / goreli_yol).resolve()
    except ValueError:
        aday = (etiket_koku_cozulum / f"{gorsel_yolu.stem}.txt").resolve()
    return aday


def _gorsel_boyutu_al(gorsel_yolu):
    import cv2
    gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
    gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
    if gorsel is None:
        raise ValueError(f"Görsel okunamadı: {gorsel_yolu}")
    yukseklik, genislik = gorsel.shape[:2]
    return genislik, yukseklik


def _yolo_etiketlerini_oku(etiket_yolu, gorsel_id, genislik, yukseklik):
    etiketler = []
    etiket_yolu = Path(etiket_yolu)
    if not etiket_yolu.exists():
        return etiketler
    for satir in etiket_yolu.read_text(encoding="utf-8").splitlines():
        parcalar = satir.strip().split()
        if len(parcalar) != 5:
            continue
        try:
            sinif_id = int(float(parcalar[0]))
            x_merkez, y_merkez, kutu_genisligi, kutu_yuksekligi = map(float, parcalar[1:])
        except ValueError:
            continue
        x1 = (x_merkez - kutu_genisligi / 2) * genislik
        y1 = (y_merkez - kutu_yuksekligi / 2) * yukseklik
        x2 = (x_merkez + kutu_genisligi / 2) * genislik
        y2 = (y_merkez + kutu_yuksekligi / 2) * yukseklik
        etiketler.append({
            "gorsel_id": str(gorsel_id),
            "sinif_id": sinif_id,
            "kutucuk": [x1, y1, x2, y2],
        })
    return etiketler


def _kutu_dizisine_cevir(kutu):
    if isinstance(kutu, dict):
        return [float(kutu.get("x1", 0)), float(kutu.get("y1", 0)), float(kutu.get("x2", 0)), float(kutu.get("y2", 0))]
    return [float(deger) for deger in kutu]


def kutu_iou_hesapla(birinci_kutu, ikinci_kutu):
    ax1, ay1, ax2, ay2 = _kutu_dizisine_cevir(birinci_kutu)
    bx1, by1, bx2, by2 = _kutu_dizisine_cevir(ikinci_kutu)
    kesisim_genisligi = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    kesisim_yuksekligi = max(0.0, min(ay2, by2) - max(ay1, by1))
    kesisim = kesisim_genisligi * kesisim_yuksekligi
    birinci_alan = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ikinci_alan = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    birlesim = birinci_alan + ikinci_alan - kesisim
    return kesisim / birlesim if birlesim > 0 else 0.0


def _sinif_ortalama_hassasiyet(tahminler, gercekler, sinif_id, iou_esigi):
    sinif_gercekleri = [gercek for gercek in gercekler if int(gercek["sinif_id"]) == int(sinif_id)]
    sinif_tahminleri = sorted(
        [tahmin for tahmin in tahminler if int(tahmin["sinif_id"]) == int(sinif_id)],
        key=lambda tahmin: float(tahmin.get("guven", 0.0)),
        reverse=True,
    )
    gercek_sayisi = len(sinif_gercekleri)
    if gercek_sayisi == 0:
        return {
            "ap": None,
            "tp": 0,
            "fp": len(sinif_tahminleri),
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
        }
    gorsel_gercekleri = {}
    for gercek in sinif_gercekleri:
        gorsel_gercekleri.setdefault(str(gercek["gorsel_id"]), []).append(gercek)
    kullanilanlar = {gorsel_id: set() for gorsel_id in gorsel_gercekleri}
    dogru_pozitifler = []
    yanlis_pozitifler = []
    for tahmin in sinif_tahminleri:
        gorsel_id = str(tahmin["gorsel_id"])
        adaylar = gorsel_gercekleri.get(gorsel_id, [])
        en_iyi_iou = 0.0
        en_iyi_indeks = None
        for indeks, gercek in enumerate(adaylar):
            if indeks in kullanilanlar.get(gorsel_id, set()):
                continue
            iou = kutu_iou_hesapla(tahmin["kutucuk"], gercek["kutucuk"])
            if iou > en_iyi_iou:
                en_iyi_iou = iou
                en_iyi_indeks = indeks
        eslesti = en_iyi_indeks is not None and en_iyi_iou >= iou_esigi
        if eslesti:
            kullanilanlar[gorsel_id].add(en_iyi_indeks)
            dogru_pozitifler.append(1.0)
            yanlis_pozitifler.append(0.0)
        else:
            dogru_pozitifler.append(0.0)
            yanlis_pozitifler.append(1.0)
    if not sinif_tahminleri:
        return {"ap": 0.0, "tp": 0, "fp": 0, "fn": gercek_sayisi, "precision": 0.0, "recall": 0.0}
    birikimli_tp = np.cumsum(dogru_pozitifler)
    birikimli_fp = np.cumsum(yanlis_pozitifler)
    recall_dizisi = birikimli_tp / gercek_sayisi
    precision_dizisi = birikimli_tp / np.maximum(birikimli_tp + birikimli_fp, np.finfo(float).eps)
    ornek_recall = np.linspace(0.0, 1.0, 101)
    ornek_precision = [float(np.max(precision_dizisi[recall_dizisi >= recall])) if np.any(recall_dizisi >= recall) else 0.0 for recall in ornek_recall]
    ap = float(np.mean(ornek_precision))
    tp = int(birikimli_tp[-1])
    fp = int(birikimli_fp[-1])
    fn = gercek_sayisi - tp
    return {
        "ap": ap,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / gercek_sayisi,
    }


def dogruluk_metriklerini_hesapla(tahminler, gercekler, siniflar=None):
    siniflar = siniflar or {}
    sinif_idleri = sorted(
        {int(gercek["sinif_id"]) for gercek in gercekler}
        | {int(tahmin["sinif_id"]) for tahmin in tahminler}
    )
    iou_esikleri = [round(deger, 2) for deger in np.arange(0.5, 1.0, 0.05)]
    esik_ap_degerleri = {}
    esik_sonuclari = {}
    for iou_esigi in iou_esikleri:
        sinif_sonuclari = {}
        ap_degerleri = []
        for sinif_id in sinif_idleri:
            sonuc = _sinif_ortalama_hassasiyet(tahminler, gercekler, sinif_id, iou_esigi)
            sinif_sonuclari[sinif_id] = sonuc
            if sonuc["ap"] is not None:
                ap_degerleri.append(sonuc["ap"])
        esik_sonuclari[iou_esigi] = sinif_sonuclari
        esik_ap_degerleri[iou_esigi] = statistics.fmean(ap_degerleri) if ap_degerleri else 0.0
    elli_sonuclari = esik_sonuclari.get(0.5, {})
    toplam_tp = sum(sonuc["tp"] for sonuc in elli_sonuclari.values())
    toplam_fp = sum(sonuc["fp"] for sonuc in elli_sonuclari.values())
    toplam_fn = sum(sonuc["fn"] for sonuc in elli_sonuclari.values())
    sinif_bazli = {}
    for sinif_id, sonuc in elli_sonuclari.items():
        sinif_adi = siniflar.get(sinif_id, siniflar.get(str(sinif_id), f"Sinif_{sinif_id}"))
        sinif_bazli[str(sinif_adi)] = {
            "sinif_id": sinif_id,
            "ap50": round(sonuc["ap"], 6) if sonuc["ap"] is not None else None,
            "precision": round(sonuc["precision"], 6),
            "recall": round(sonuc["recall"], 6),
            "tp": sonuc["tp"],
            "fp": sonuc["fp"],
            "fn": sonuc["fn"],
        }
    return {
        "mAP50": round(esik_ap_degerleri.get(0.5, 0.0), 6),
        "mAP50_95": round(statistics.fmean(esik_ap_degerleri.values()), 6) if esik_ap_degerleri else 0.0,
        "precision": round(toplam_tp / (toplam_tp + toplam_fp), 6) if toplam_tp + toplam_fp else 0.0,
        "recall": round(toplam_tp / (toplam_tp + toplam_fn), 6) if toplam_tp + toplam_fn else 0.0,
        "tp": toplam_tp,
        "fp": toplam_fp,
        "fn": toplam_fn,
        "tahmin_sayisi": len(tahminler),
        "gercek_kutu_sayisi": len(gercekler),
        "degerlendirilen_sinif_sayisi": len(sinif_idleri),
        "iou_esikleri": iou_esikleri,
        "sinif_bazli": sinif_bazli,
    }


def _benchmark_yapilandirmasi(yapilandirma=None):
    sonuc = copy.deepcopy(yapilandirma or yapilandirma_yukle())
    sonuc.setdefault("cikarim", {})["gorsel_kaydet"] = False
    sonuc["cikarim"]["json_kaydet"] = False
    sonuc.setdefault("multi_model", {})["ram_optimizasyonu"] = False
    return sonuc


def _cuda_tepe_bellegini_sifirla():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def tekil_model_benchmark_calistir(gorseller, yapilandirma=None):
    from src.pipeline import _model_bosalt, _model_yukle_optimize, egitilmis_model_yolu_bul, hasar_tespiti_yap
    gorseller = [Path(gorsel) for gorsel in gorseller]
    if not gorseller:
        return {"durum": "Atlandı (Görsel bulunamadı)", "_sonuclar": []}
    benchmark_yapilandirmasi = _benchmark_yapilandirmasi(yapilandirma)
    model_yolu = egitilmis_model_yolu_bul()
    if model_yolu is None:
        return {"durum": "Atlandı (Model bulunamadı)", "_sonuclar": []}
    baslangic_bellek = bellek_olcu_al()
    _cuda_tepe_bellegini_sifirla()
    model = None
    try:
        soguk_baslangic = time.perf_counter()
        model, backend = _model_yukle_optimize(model_yolu, benchmark_yapilandirmasi)
        ilk_sonuc = hasar_tespiti_yap(gorseller[0], json_kaydet=False, model=model, yapilandirma=benchmark_yapilandirmasi)
        soguk_sure = time.perf_counter() - soguk_baslangic
        sicak_gorseller = gorseller[1:] if len(gorseller) > 1 else gorseller[:1]
        sicak_sureler = []
        sonuclar = [ilk_sonuc] if ilk_sonuc else []
        for indeks, gorsel_yolu in enumerate(sicak_gorseller):
            baslangic = time.perf_counter()
            sonuc = hasar_tespiti_yap(gorsel_yolu, json_kaydet=False, model=model, yapilandirma=benchmark_yapilandirmasi)
            sicak_sureler.append(time.perf_counter() - baslangic)
            if sonuc and (len(gorseller) > 1 or indeks > 0):
                sonuclar.append(sonuc)
        return {
            "durum": "Tamamlandı",
            "backend": backend,
            "model_yolu": str(model_yolu),
            "soguk_baslangic_saniye": round(soguk_sure, 6),
            "sicak_cikarim": _gecikme_istatistikleri(sicak_sureler),
            "bellek_baslangic": baslangic_bellek,
            "bellek_bitis": bellek_olcu_al(),
            "_sonuclar": sonuclar,
        }
    except Exception as hata:
        return {
            "durum": f"Hata: {hata}",
            "model_yolu": str(model_yolu),
            "bellek_baslangic": baslangic_bellek,
            "bellek_bitis": bellek_olcu_al(),
            "_sonuclar": [],
        }
    finally:
        if model is not None:
            del model
        _model_bosalt(True)


def _modeli_directml_tasi(model):
    try:
        import torch_directml
        cihaz = torch_directml.device()
        if hasattr(model, "model") and model.model is not None:
            model.model.to(cihaz)
            return True
    except (ImportError, RuntimeError, AttributeError):
        pass
    return False


def _coklu_modelleri_yukle(yapilandirma):
    from src.pipeline import egitilmis_model_yolu_bul
    from ultralytics import RTDETR, SAM, YOLO
    agirliklar = yapilandirma.get("multi_model", {}).get("agirliklar", {})
    siralama = yapilandirma.get("multi_model", {}).get("siralama", [])
    modeller = {"rtdetr": None, "yolo": None, "sam": None}
    hatalar = []
    rtdetr_yolu = PROJE_KOKU / agirliklar.get("rtdetr", "rtdetr-v2-x.pt")
    if "rt-detr-v2-x" in siralama:
        if not rtdetr_yolu.exists():
            rtdetr_yolu = egitilmis_model_yolu_bul()
        try:
            if rtdetr_yolu is not None:
                modeller["rtdetr"] = RTDETR(str(rtdetr_yolu))
                _modeli_directml_tasi(modeller["rtdetr"])
        except Exception as hata:
            hatalar.append(f"RT-DETR yüklenemedi: {hata}")
    yolo_yolu = PROJE_KOKU / agirliklar.get("yolo", "yolov12x.pt")
    if "yolov12x" in siralama:
        if not yolo_yolu.exists():
            yolo_yolu = egitilmis_model_yolu_bul()
        try:
            if yolo_yolu is not None:
                modeller["yolo"] = YOLO(str(yolo_yolu))
                _modeli_directml_tasi(modeller["yolo"])
        except Exception as hata:
            hatalar.append(f"YOLO yüklenemedi: {hata}")
    sam_yolu = PROJE_KOKU / agirliklar.get("sam", "sam2_s.pt")
    if "sam2_small" in siralama:
        try:
            if sam_yolu.exists():
                modeller["sam"] = SAM(str(sam_yolu))
                _modeli_directml_tasi(modeller["sam"])
            else:
                hatalar.append(f"SAM ağırlığı bulunamadı: {sam_yolu}")
        except Exception as hata:
            hatalar.append(f"SAM yüklenemedi: {hata}")
    return modeller, hatalar


def _asama_istatistiklerini_hesapla(sonuclar):
    anahtarlar = ["rtdetr_saniye", "yolo_saniye", "wbf_saniye", "sam_saniye", "florence_saniye"]
    return {
        anahtar: _gecikme_istatistikleri([
            sonuc.get("asama_sureleri", {}).get(anahtar)
            for sonuc in sonuclar
            if sonuc and sonuc.get("asama_sureleri", {}).get(anahtar) is not None
        ])
        for anahtar in anahtarlar
    }


def coklu_model_benchmark_calistir(gorseller, yapilandirma=None):
    from src.pipeline import _model_bosalt, coklu_model_hasar_tespiti_yap
    gorseller = [Path(gorsel) for gorsel in gorseller]
    if not gorseller:
        return {"durum": "Atlandı (Görsel bulunamadı)", "_sonuclar": []}
    benchmark_yapilandirmasi = _benchmark_yapilandirmasi(yapilandirma)
    if not benchmark_yapilandirmasi.get("multi_model", {}).get("aktif", False):
        return tekil_model_benchmark_calistir(gorseller, benchmark_yapilandirmasi)
    baslangic_bellek = bellek_olcu_al()
    _cuda_tepe_bellegini_sifirla()
    modeller = {"rtdetr": None, "yolo": None, "sam": None}
    try:
        yukleme_baslangici = time.perf_counter()
        modeller, yukleme_hatalari = _coklu_modelleri_yukle(benchmark_yapilandirmasi)
        model_yukleme_suresi = time.perf_counter() - yukleme_baslangici
        soguk_baslangic = time.perf_counter()
        ilk_sonuc = coklu_model_hasar_tespiti_yap(
            gorseller[0], json_kaydet=False, yapilandirma=benchmark_yapilandirmasi, hazir_modeller=modeller
        )
        ilk_cikarim_suresi = time.perf_counter() - soguk_baslangic
        sicak_gorseller = gorseller[1:] if len(gorseller) > 1 else gorseller[:1]
        sicak_sureler = []
        sonuclar = [ilk_sonuc] if ilk_sonuc else []
        for indeks, gorsel_yolu in enumerate(sicak_gorseller):
            baslangic = time.perf_counter()
            sonuc = coklu_model_hasar_tespiti_yap(
                gorsel_yolu, json_kaydet=False, yapilandirma=benchmark_yapilandirmasi, hazir_modeller=modeller
            )
            sicak_sureler.append(time.perf_counter() - baslangic)
            if sonuc and (len(gorseller) > 1 or indeks > 0):
                sonuclar.append(sonuc)
        return {
            "durum": "Tamamlandı" if sonuclar else "Hata (Çıkarım sonucu üretilemedi)",
            "model_yukleme_saniye": round(model_yukleme_suresi, 6),
            "ilk_cikarim_saniye": round(ilk_cikarim_suresi, 6),
            "soguk_baslangic_toplam_saniye": round(model_yukleme_suresi + ilk_cikarim_suresi, 6),
            "sicak_cikarim": _gecikme_istatistikleri(sicak_sureler),
            "sicak_asama_istatistikleri": _asama_istatistiklerini_hesapla(sonuclar[1:] if len(sonuclar) > 1 else sonuclar),
            "ilk_cikarim_asama_sureleri": ilk_sonuc.get("asama_sureleri", {}) if ilk_sonuc else {},
            "model_yukleme_uyarilari": yukleme_hatalari,
            "bellek_baslangic": baslangic_bellek,
            "bellek_bitis": bellek_olcu_al(),
            "_sonuclar": sonuclar,
        }
    except Exception as hata:
        return {
            "durum": f"Hata: {hata}",
            "bellek_baslangic": baslangic_bellek,
            "bellek_bitis": bellek_olcu_al(),
            "_sonuclar": [],
        }
    finally:
        modeller.clear()
        try:
            from src.inspector_florence import _florence_modelini_bosalt
            _florence_modelini_bosalt()
        except (ImportError, RuntimeError):
            pass
        _model_bosalt(True)


def _tahminleri_sonuclardan_oku(sonuclar, siniflar=None):
    siniflar = siniflar or {}
    ad_karsiliklari = {str(sinif_adi).lower(): int(sinif_id) for sinif_id, sinif_adi in siniflar.items()}
    tahminler = []
    for sonuc in sonuclar:
        if not sonuc:
            continue
        gorsel_id = str(Path(sonuc.get("gorsel_yolu", "")).resolve())
        for tespit in sonuc.get("tespitler", []):
            sinif_adi = str(tespit.get("sinif_adi", "")).lower()
            sinif_id = ad_karsiliklari.get(sinif_adi, tespit.get("sinif_id"))
            if sinif_id is None or not tespit.get("kutucuk"):
                continue
            tahminler.append({
                "gorsel_id": gorsel_id,
                "sinif_id": int(sinif_id),
                "guven": float(tespit.get("guven", 0.0)),
                "kutucuk": _kutu_dizisine_cevir(tespit.get("kutucuk", {})),
            })
    return tahminler


def etiketli_dogruluk_benchmark_calistir(miktar=10, artirilmis_dahil=False, yapilandirma=None, gorseller=None):
    benchmark_yapilandirmasi = _benchmark_yapilandirmasi(yapilandirma)
    gorsel_koku, etiket_koku, kaynak_adi = etiketli_veri_kaynagi_bul()
    if gorseller is None:
        gorseller = _miktari_uygula(_gorselleri_listele(gorsel_koku, artirilmis_dahil), miktar)
    else:
        gorseller = _miktari_uygula([Path(gorsel) for gorsel in gorseller if artirilmis_dahil or _gorsel_orijinal_mi(gorsel)], miktar)
    if not gorseller:
        return {"durum": "Atlandı (Etiketli görsel bulunamadı)", "veri_kaynagi": kaynak_adi}
    if benchmark_yapilandirmasi.get("multi_model", {}).get("aktif", False):
        performans = coklu_model_benchmark_calistir(gorseller, benchmark_yapilandirmasi)
    else:
        performans = tekil_model_benchmark_calistir(gorseller, benchmark_yapilandirmasi)
    sonuclar = performans.pop("_sonuclar", [])
    tahminler = _tahminleri_sonuclardan_oku(sonuclar, benchmark_yapilandirmasi.get("siniflar", {}))
    gercekler = []
    etiketsiz_gorseller = []
    okunamayan_gorseller = []
    for gorsel_yolu in gorseller:
        etiket_yolu = _etiket_yolu_bul(gorsel_yolu, gorsel_koku, etiket_koku)
        if not etiket_yolu.exists():
            etiketsiz_gorseller.append(str(gorsel_yolu))
            continue
        try:
            genislik, yukseklik = _gorsel_boyutu_al(gorsel_yolu)
            gercekler.extend(_yolo_etiketlerini_oku(etiket_yolu, str(Path(gorsel_yolu).resolve()), genislik, yukseklik))
        except ValueError:
            okunamayan_gorseller.append(str(gorsel_yolu))
    metrikler = dogruluk_metriklerini_hesapla(tahminler, gercekler, benchmark_yapilandirmasi.get("siniflar", {}))
    return {
        "durum": "Tamamlandı" if gercekler else "Hata (Geçerli etiket bulunamadı)",
        "veri_kaynagi": kaynak_adi,
        "artirilmis_gorseller_dahil": bool(artirilmis_dahil),
        "degerlendirilen_gorsel_sayisi": len(gorseller) - len(etiketsiz_gorseller) - len(okunamayan_gorseller),
        "etiketsiz_gorsel_sayisi": len(etiketsiz_gorseller),
        "okunamayan_gorsel_sayisi": len(okunamayan_gorseller),
        "metrikler": metrikler,
        "cikarim_performansi": performans,
    }


def _backend_model_sinifi(yapilandirma):
    if yapilandirma.get("model", {}).get("tur", "yolo") == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR
    from ultralytics import YOLO
    return YOLO


def _backend_tahmin_yap(model, gorsel_yolu, backend, yapilandirma):
    cikarim = yapilandirma.get("cikarim", {})
    parametreler = {
        "source": str(gorsel_yolu),
        "conf": cikarim.get("guven_esigi", 0.25),
        "iou": cikarim.get("iou_esigi", 0.7),
        "save": False,
        "verbose": False,
    }
    if backend == "PyTorch CPU":
        parametreler["device"] = "cpu"
    elif backend == "CUDA":
        parametreler["device"] = 0
    sonuclar = model.predict(**parametreler)
    if backend == "CUDA":
        import torch
        torch.cuda.synchronize()
    return sum(len(sonuc.boxes) for sonuc in sonuclar if sonuc.boxes is not None)


def _backend_kullanilabilirlik(backend, model_yolu):
    if backend == "PyTorch CPU":
        return True, model_yolu
    if backend == "CUDA":
        try:
            import torch
            return (torch.cuda.is_available(), model_yolu)
        except ImportError:
            return False, model_yolu
    if backend == "DirectML":
        try:
            import torch_directml
            torch_directml.device()
            return True, model_yolu
        except (ImportError, RuntimeError):
            return False, model_yolu
    if backend == "OpenVINO":
        from src.pipeline import _openvino_model_yolu_bul
        openvino_yolu = _openvino_model_yolu_bul(model_yolu)
        return openvino_yolu is not None, openvino_yolu
    return False, model_yolu


def donanim_backend_benchmark_calistir(gorseller, yapilandirma=None):
    from src.pipeline import _model_bosalt, egitilmis_model_yolu_bul
    benchmark_yapilandirmasi = _benchmark_yapilandirmasi(yapilandirma)
    gorseller = [Path(gorsel) for gorsel in gorseller]
    model_yolu = egitilmis_model_yolu_bul()
    if not gorseller or model_yolu is None:
        return {"durum": "Atlandı (Görsel veya model bulunamadı)", "backendler": {}}
    model_sinifi = _backend_model_sinifi(benchmark_yapilandirmasi)
    backend_sonuclari = {}
    for backend in ["PyTorch CPU", "CUDA", "DirectML", "OpenVINO"]:
        kullanilabilir, backend_model_yolu = _backend_kullanilabilirlik(backend, model_yolu)
        if not kullanilabilir:
            backend_sonuclari[backend] = {"durum": DESTEKLENMIYOR}
            continue
        model = None
        try:
            _cuda_tepe_bellegini_sifirla()
            bellek_baslangic = bellek_olcu_al()
            soguk_baslangic = time.perf_counter()
            model = model_sinifi(str(backend_model_yolu))
            if backend == "DirectML" and not _modeli_directml_tasi(model):
                raise RuntimeError("Model DirectML cihazına taşınamadı")
            ilk_tespit_sayisi = _backend_tahmin_yap(model, gorseller[0], backend, benchmark_yapilandirmasi)
            soguk_sure = time.perf_counter() - soguk_baslangic
            sicak_gorseller = gorseller[1:] if len(gorseller) > 1 else gorseller[:1]
            sicak_sureler = []
            toplam_tespit = ilk_tespit_sayisi
            for gorsel_yolu in sicak_gorseller:
                baslangic = time.perf_counter()
                toplam_tespit += _backend_tahmin_yap(model, gorsel_yolu, backend, benchmark_yapilandirmasi)
                sicak_sureler.append(time.perf_counter() - baslangic)
            backend_sonuclari[backend] = {
                "durum": "Tamamlandı",
                "model_yolu": str(backend_model_yolu),
                "soguk_baslangic_saniye": round(soguk_sure, 6),
                "sicak_cikarim": _gecikme_istatistikleri(sicak_sureler),
                "toplam_tespit": toplam_tespit,
                "bellek_baslangic": bellek_baslangic,
                "bellek_bitis": bellek_olcu_al(),
                "vram_olcumu": "CUDA sayaçları" if backend == "CUDA" else VRAM_ERISILEMIYOR,
            }
        except Exception as hata:
            backend_sonuclari[backend] = {"durum": f"Hata: {hata}"}
        finally:
            if model is not None:
                del model
            _model_bosalt(True)
    return {"durum": "Tamamlandı", "backendler": backend_sonuclari}


def _ozel_alanlari_temizle(veri):
    if isinstance(veri, dict):
        return {anahtar: _ozel_alanlari_temizle(deger) for anahtar, deger in veri.items() if not str(anahtar).startswith("_")}
    if isinstance(veri, list):
        return [_ozel_alanlari_temizle(deger) for deger in veri]
    return veri


def _markdown_degerini_yaz(satirlar, veri, girinti=0):
    bosluk = "  " * girinti
    if isinstance(veri, dict):
        for anahtar, deger in veri.items():
            etiket = str(anahtar).replace("_", " ").title()
            if isinstance(deger, (dict, list)):
                satirlar.append(f"{bosluk}- **{etiket}:**")
                _markdown_degerini_yaz(satirlar, deger, girinti + 1)
            else:
                satirlar.append(f"{bosluk}- **{etiket}:** {deger}")
    elif isinstance(veri, list):
        if not veri:
            satirlar.append(f"{bosluk}- Yok")
        for deger in veri:
            if isinstance(deger, (dict, list)):
                satirlar.append(f"{bosluk}-")
                _markdown_degerini_yaz(satirlar, deger, girinti + 1)
            else:
                satirlar.append(f"{bosluk}- {deger}")
    else:
        satirlar.append(f"{bosluk}- {veri}")


def markdown_raporu_olustur(rapor):
    satirlar = [
        "# HADES Hyper Benchmark Raporu",
        "",
        f"Oluşturulma zamanı: {rapor.get('zaman_damgasi', 'Bilinmiyor')}",
        "",
    ]
    bolumler = [
        ("Çalıştırma Özeti", "calistirma"),
        ("Donanım ve Ortam", "ortam"),
        ("Başlangıç Belleği", "bellek_baslangic"),
        ("Tekil Model Performansı", "tekil_model_performansi"),
        ("Çoklu Model Performansı", "coklu_model_performansi"),
        ("Etiketli Doğruluk", "etiketli_dogruluk"),
        ("Backend Karşılaştırması", "backend_karsilastirmasi"),
        ("Bitiş Belleği", "bellek_bitis"),
    ]
    for baslik, anahtar in bolumler:
        if anahtar not in rapor:
            continue
        satirlar.extend([f"## {baslik}", ""])
        _markdown_degerini_yaz(satirlar, rapor[anahtar])
        satirlar.append("")
    return "\n".join(satirlar).rstrip() + "\n"


def rapor_kaydet(rapor, cikti_klasoru=None):
    cikti_klasoru = Path(cikti_klasoru or PROJE_KOKU / "runs" / "benchmark")
    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    zaman = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    json_yolu = cikti_klasoru / f"benchmark_raporu_{zaman}.json"
    markdown_yolu = cikti_klasoru / f"benchmark_raporu_{zaman}.md"
    temiz_rapor = _json_uyumlu_yap(_ozel_alanlari_temizle(rapor))
    temiz_rapor["rapor_dosyalari"] = {"json": str(json_yolu), "markdown": str(markdown_yolu)}
    json_yolu.write_text(json.dumps(temiz_rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_yolu.write_text(markdown_raporu_olustur(temiz_rapor), encoding="utf-8")
    return {"json": str(json_yolu), "markdown": str(markdown_yolu)}


def benchmark_suitini_calistir(hedef="tam", miktar=10, artirilmis_dahil=False, backend_karsilastir=True, yapilandirma=None):
    benchmark_yapilandirmasi = _benchmark_yapilandirmasi(yapilandirma)
    hedef = str(hedef).lower()
    gecerli_hedefler = {"gercek", "etiketli", "tam"}
    if hedef not in gecerli_hedefler:
        raise ValueError(f"Geçersiz benchmark hedefi: {hedef}")
    rapor = {
        "rapor_surumu": "1.0",
        "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
        "calistirma": {
            "hedef": hedef,
            "istenen_gorsel_sayisi": "Tümü" if miktar is None else int(miktar),
            "artirilmis_gorseller_dahil": bool(artirilmis_dahil),
            "backend_karsilastirmasi": bool(backend_karsilastir),
        },
        "ortam": ortam_profili_olustur(),
        "bellek_baslangic": bellek_olcu_al(),
    }
    gercek_gorseller = _miktari_uygula(_gorselleri_listele(PROJE_KOKU / "hasar-ornek"), miktar)
    if hedef in {"gercek", "tam"}:
        rapor["calistirma"]["gercek_gorsel_sayisi"] = len(gercek_gorseller)
        rapor["tekil_model_performansi"] = _ozel_alanlari_temizle(
            tekil_model_benchmark_calistir(gercek_gorseller, benchmark_yapilandirmasi)
        )
        if benchmark_yapilandirmasi.get("multi_model", {}).get("aktif", False):
            rapor["coklu_model_performansi"] = _ozel_alanlari_temizle(
                coklu_model_benchmark_calistir(gercek_gorseller, benchmark_yapilandirmasi)
            )
    if hedef in {"etiketli", "tam"}:
        rapor["etiketli_dogruluk"] = etiketli_dogruluk_benchmark_calistir(
            miktar=miktar,
            artirilmis_dahil=artirilmis_dahil,
            yapilandirma=benchmark_yapilandirmasi,
        )
    if backend_karsilastir and hedef in {"gercek", "tam"}:
        rapor["backend_karsilastirmasi"] = donanim_backend_benchmark_calistir(
            gercek_gorseller, benchmark_yapilandirmasi
        )
    rapor["bellek_bitis"] = bellek_olcu_al()
    rapor_dosyalari = rapor_kaydet(rapor)
    rapor["rapor_dosyalari"] = rapor_dosyalari
    return rapor
