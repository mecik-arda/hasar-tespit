import copy
import gc
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

from src.benchmark import (
    _gorselleri_listele,
    _kutu_dizisine_cevir,
    _miktari_uygula,
    _yolo_etiketlerini_oku,
    bellek_olcu_al,
    dogruluk_metriklerini_hesapla,
    etiketli_veri_kaynagi_bul,
    kutu_iou_hesapla,
)
from src.utils import PROJE_KOKU, yapilandirma_kaydet, yapilandirma_yukle


BOZULMA_TURLERI = ("karanlik", "parlama", "hareket_bulanikligi", "sis", "gauss_gurultusu")
SİDDET_SEVIYELERI = (1, 2, 3)
MEVCUT_SUREC = psutil.Process()


def _json_uyumlu_yap(deger):
    if isinstance(deger, dict):
        return {str(anahtar): _json_uyumlu_yap(alt_deger) for anahtar, alt_deger in deger.items()}
    if isinstance(deger, (list, tuple, set)):
        return [_json_uyumlu_yap(alt_deger) for alt_deger in deger]
    if isinstance(deger, Path):
        return str(deger)
    if isinstance(deger, np.generic):
        return deger.item()
    if isinstance(deger, float) and (math.isnan(deger) or math.isinf(deger)):
        return None
    return deger


def _gorseli_oku(gorsel_yolu):
    gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
    gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
    if gorsel is None:
        raise ValueError(f"Görsel okunamadı: {gorsel_yolu}")
    return gorsel


def _etiket_yolu_bul(gorsel_yolu, gorsel_koku, etiket_koku):
    gorsel_yolu = Path(gorsel_yolu).resolve()
    gorsel_koku = Path(gorsel_koku).resolve()
    etiket_koku = Path(etiket_koku).resolve()
    try:
        goreli_yol = gorsel_yolu.relative_to(gorsel_koku).with_suffix(".txt")
    except ValueError:
        goreli_yol = Path(f"{gorsel_yolu.stem}.txt")
    aday = (etiket_koku / goreli_yol).resolve()
    try:
        aday.relative_to(etiket_koku)
    except ValueError as hata:
        raise ValueError("Etiket yolu veri kökü dışında") from hata
    return aday


def _etiketli_veriyi_hazirla(miktar=50):
    gorsel_koku, etiket_koku, kaynak_adi = etiketli_veri_kaynagi_bul()
    gorseller = _miktari_uygula(_gorselleri_listele(gorsel_koku, artirilmis_dahil=False), miktar)
    kayitlar = []
    gercekler = []
    for gorsel_yolu in gorseller:
        etiket_yolu = _etiket_yolu_bul(gorsel_yolu, gorsel_koku, etiket_koku)
        if not etiket_yolu.exists():
            continue
        try:
            gorsel = _gorseli_oku(gorsel_yolu)
        except ValueError:
            continue
        yukseklik, genislik = gorsel.shape[:2]
        gorsel_id = str(Path(gorsel_yolu).resolve())
        etiketler = _yolo_etiketlerini_oku(etiket_yolu, gorsel_id, genislik, yukseklik)
        if not etiketler:
            continue
        kayitlar.append({
            "gorsel_yolu": Path(gorsel_yolu),
            "gorsel_id": gorsel_id,
            "gorsel": gorsel,
            "genislik": genislik,
            "yukseklik": yukseklik,
            "etiketler": etiketler,
        })
        gercekler.extend(etiketler)
    return kayitlar, gercekler, kaynak_adi


def bozulma_uygula(gorsel, bozulma_turu, siddet=1, tohum=42):
    if bozulma_turu not in BOZULMA_TURLERI:
        raise ValueError(f"Geçersiz bozulma türü: {bozulma_turu}")
    if siddet not in SİDDET_SEVIYELERI:
        raise ValueError(f"Geçersiz şiddet seviyesi: {siddet}")
    kaynak = np.asarray(gorsel, dtype=np.uint8)
    if kaynak.ndim != 3 or kaynak.shape[2] != 3:
        raise ValueError("Görsel üç kanallı BGR biçiminde olmalıdır")
    rastgele = np.random.default_rng(int(tohum) + int(siddet) * 1009)
    if bozulma_turu == "karanlik":
        katsayi = {1: 0.65, 2: 0.40, 3: 0.20}[siddet]
        return np.clip(kaynak.astype(np.float32) * katsayi, 0, 255).astype(np.uint8)
    if bozulma_turu == "parlama":
        yukseklik, genislik = kaynak.shape[:2]
        merkez_x = int(genislik * (0.65 - siddet * 0.05))
        merkez_y = int(yukseklik * (0.30 + siddet * 0.04))
        yaricap = max(12, int(min(genislik, yukseklik) * (0.18 + siddet * 0.08)))
        y_grid, x_grid = np.ogrid[:yukseklik, :genislik]
        uzaklik = np.sqrt((x_grid - merkez_x) ** 2 + (y_grid - merkez_y) ** 2)
        maske = np.clip(1.0 - uzaklik / yaricap, 0.0, 1.0)[..., None]
        yogunluk = 0.35 + siddet * 0.15
        return np.clip(kaynak.astype(np.float32) + maske * 255.0 * yogunluk, 0, 255).astype(np.uint8)
    if bozulma_turu == "hareket_bulanikligi":
        cekirdek_boyutu = {1: 7, 2: 15, 3: 25}[siddet]
        cekirdek = np.zeros((cekirdek_boyutu, cekirdek_boyutu), dtype=np.float32)
        cekirdek[cekirdek_boyutu // 2, :] = 1.0
        donus = cv2.getRotationMatrix2D((cekirdek_boyutu / 2, cekirdek_boyutu / 2), 12 * siddet, 1.0)
        cekirdek = cv2.warpAffine(cekirdek, donus, (cekirdek_boyutu, cekirdek_boyutu))
        cekirdek /= max(float(cekirdek.sum()), np.finfo(np.float32).eps)
        return cv2.filter2D(kaynak, -1, cekirdek)
    if bozulma_turu == "sis":
        yogunluk = {1: 0.20, 2: 0.38, 3: 0.55}[siddet]
        gurultu = rastgele.normal(210.0, 18.0, kaynak.shape[:2]).astype(np.float32)
        gurultu = cv2.GaussianBlur(gurultu, (0, 0), sigmaX=15 + siddet * 10)
        sis_katmani = np.repeat(gurultu[..., None], 3, axis=2)
        return np.clip(kaynak.astype(np.float32) * (1.0 - yogunluk) + sis_katmani * yogunluk, 0, 255).astype(np.uint8)
    standart_sapma = {1: 8.0, 2: 20.0, 3: 35.0}[siddet]
    gurultu = rastgele.normal(0.0, standart_sapma, kaynak.shape)
    return np.clip(kaynak.astype(np.float32) + gurultu, 0, 255).astype(np.uint8)


def _ultralytics_sonuclarini_cevir(sonuclar, gorsel_id):
    tahminler = []
    for sonuc in sonuclar or []:
        if getattr(sonuc, "boxes", None) is None:
            continue
        for kutu in sonuc.boxes:
            koordinatlar = kutu.xyxy[0].cpu().numpy().astype(float).tolist()
            tahminler.append({
                "gorsel_id": str(gorsel_id),
                "sinif_id": int(kutu.cls[0].cpu().numpy()),
                "guven": float(kutu.conf[0].cpu().numpy()),
                "kutucuk": koordinatlar,
            })
    return tahminler


def _tahminleri_standartlastir(tahminler, gorsel_id):
    standart = []
    for tahmin in tahminler or []:
        if tahmin.get("sinif_id") is None or tahmin.get("kutucuk") is None:
            continue
        standart.append({
            "gorsel_id": str(gorsel_id),
            "sinif_id": int(tahmin["sinif_id"]),
            "guven": float(tahmin.get("guven", 0.0)),
            "kutucuk": _kutu_dizisine_cevir(tahmin["kutucuk"]),
        })
    return standart


def _gercek_tahmin_ureticisini_hazirla(yapilandirma):
    from src.pipeline import _model_bosalt, _model_yukle_optimize, egitilmis_model_yolu_bul
    model_yolu = egitilmis_model_yolu_bul()
    if model_yolu is None:
        raise FileNotFoundError("Eğitilmiş model bulunamadı")
    model, backend = _model_yukle_optimize(model_yolu, yapilandirma)
    cikarim = yapilandirma.get("cikarim", {})

    def tahmin_uret(gorsel, gorsel_id):
        sonuclar = model.predict(
            source=gorsel,
            conf=0.10,
            iou=cikarim.get("iou_esigi", 0.7),
            save=False,
            verbose=False,
        )
        tahminler = _ultralytics_sonuclarini_cevir(sonuclar, gorsel_id)
        guven_esigi = cikarim.get("guven_esigi", 0.25)
        sinif_esikleri = cikarim.get("sinif_guven_esikleri", {})
        return [
            tahmin
            for tahmin in tahminler
            if tahmin["guven"] >= float(sinif_esikleri.get(tahmin["sinif_id"], guven_esigi))
        ]

    def kapat():
        nonlocal model
        model = None
        _model_bosalt(True)

    return tahmin_uret, kapat, backend


def _rapor_dosyalari_ekle(rapor, rapor_uret, rapor_adi):
    if not rapor_uret:
        return rapor
    rapor["rapor_dosyalari"] = gelismis_rapor_kaydet(rapor, rapor_adi=rapor_adi)
    return rapor


def dayaniklilik_benchmark_calistir(miktar=50, siddetler=SİDDET_SEVIYELERI, tahmin_uretici=None, yapilandirma=None, rapor_uret=True):
    yapilandirma = copy.deepcopy(yapilandirma or yapilandirma_yukle())
    kayitlar, gercekler, kaynak_adi = _etiketli_veriyi_hazirla(miktar)
    if not kayitlar:
        return _rapor_dosyalari_ekle({"durum": "Atlandı (Etiketli görsel bulunamadı)", "veri_kaynagi": kaynak_adi}, rapor_uret, "dayaniklilik")
    def kapat():
        return None
    backend = "Özel tahmin üreticisi"
    try:
        if tahmin_uretici is None:
            tahmin_uretici, kapat, backend = _gercek_tahmin_ureticisini_hazirla(yapilandirma)
        temel_tahminler = []
        for kayit in kayitlar:
            temel_tahminler.extend(_tahminleri_standartlastir(tahmin_uretici(kayit["gorsel"], kayit["gorsel_id"]), kayit["gorsel_id"]))
        temel_metrikler = dogruluk_metriklerini_hesapla(temel_tahminler, gercekler, yapilandirma.get("siniflar", {}))
        bozulma_sonuclari = {}
        for bozulma_turu in BOZULMA_TURLERI:
            bozulma_sonuclari[bozulma_turu] = {}
            for siddet in siddetler:
                tahminler = []
                for indeks, kayit in enumerate(kayitlar):
                    bozulmus = bozulma_uygula(kayit["gorsel"], bozulma_turu, int(siddet), tohum=42 + indeks)
                    tahminler.extend(_tahminleri_standartlastir(tahmin_uretici(bozulmus, kayit["gorsel_id"]), kayit["gorsel_id"]))
                metrikler = dogruluk_metriklerini_hesapla(tahminler, gercekler, yapilandirma.get("siniflar", {}))
                temel_map = float(temel_metrikler.get("mAP50_95", 0.0))
                guncel_map = float(metrikler.get("mAP50_95", 0.0))
                kayip = max(0.0, temel_map - guncel_map)
                bozulma_sonuclari[bozulma_turu][str(siddet)] = {
                    "metrikler": metrikler,
                    "map50_95_mutlak_kayip": round(kayip, 6),
                    "map50_95_kayip_yuzdesi": round(kayip / temel_map * 100.0, 4) if temel_map > 0 else None,
                }
        rapor = {
            "durum": "Tamamlandı",
            "benchmark": "dayaniklilik",
            "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
            "veri_kaynagi": kaynak_adi,
            "gorsel_sayisi": len(kayitlar),
            "backend": backend,
            "sabit_tohum": 42,
            "temel_metrikler": temel_metrikler,
            "bozulmalar": bozulma_sonuclari,
        }
        return _rapor_dosyalari_ekle(rapor, rapor_uret, "dayaniklilik")
    except Exception as hata:
        return _rapor_dosyalari_ekle({"durum": f"Hata: {hata}", "benchmark": "dayaniklilik"}, rapor_uret, "dayaniklilik")
    finally:
        kapat()


def _aralik_degerleri(baslangic, bitis, adim):
    basamak = max(0, len(str(adim).split(".")[-1]))
    degerler = []
    guncel = float(baslangic)
    while guncel <= float(bitis) + adim / 10:
        degerler.append(round(guncel, basamak))
        guncel += adim
    return degerler


def _ham_tespit_onbellegi_olustur(kayitlar, yapilandirma, ham_tespit_uretici=None):
    if ham_tespit_uretici is not None:
        return [
            {
                "gorsel_id": kayit["gorsel_id"],
                "genislik": kayit["genislik"],
                "yukseklik": kayit["yukseklik"],
                "boxes": list(ham_tespit_uretici(kayit["gorsel_yolu"]) or []),
            }
            for kayit in kayitlar
        ], "Özel ham tespit üreticisi"
    from src.benchmark import _coklu_modelleri_yukle
    from src.pipeline import _model_bosalt, _tek_model_tara
    from ultralytics import RTDETR, YOLO
    modeller, hatalar = _coklu_modelleri_yukle(yapilandirma)
    onbellek = []
    try:
        for kayit in kayitlar:
            havuz = {"boxes": [], "masks": []}
            if modeller.get("rtdetr") is not None:
                _tek_model_tara(RTDETR, "", "rt-detr-v2-x", kayit["gorsel"], havuz, 0.10, 0.70, {}, yapilandirma.get("siniflar", {}), False, 640, True, False, hazir_model=modeller["rtdetr"])
            if modeller.get("yolo") is not None:
                _tek_model_tara(YOLO, "", "yolov12x", kayit["gorsel"], havuz, 0.10, 0.70, {}, yapilandirma.get("siniflar", {}), False, 640, True, False, hazir_model=modeller["yolo"])
            onbellek.append({
                "gorsel_id": kayit["gorsel_id"],
                "genislik": kayit["genislik"],
                "yukseklik": kayit["yukseklik"],
                "boxes": havuz["boxes"],
            })
        return onbellek, hatalar
    finally:
        modeller.clear()
        _model_bosalt(True)


def _wbf_parametrelerini_degerlendir(onbellek, gercekler, yapilandirma, iou_esigi, guven_esigi):
    from src.pipeline import _wbf_kutu_birlestir
    tahminler = []
    for kayit in onbellek:
        havuz = {"boxes": kayit["boxes"], "masks": []}
        birlesmis = _wbf_kutu_birlestir(havuz, kayit["genislik"], kayit["yukseklik"], iou_esigi=iou_esigi, guven_esigi=guven_esigi, yapilandirma=yapilandirma)
        tahminler.extend(_tahminleri_standartlastir(birlesmis, kayit["gorsel_id"]))
    metrikler = dogruluk_metriklerini_hesapla(tahminler, gercekler, yapilandirma.get("siniflar", {}))
    precision = float(metrikler.get("precision", 0.0))
    recall = float(metrikler.get("recall", 0.0))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou_esigi": round(float(iou_esigi), 2),
        "guven_esigi": round(float(guven_esigi), 2),
        "mAP50": metrikler.get("mAP50", 0.0),
        "mAP50_95": metrikler.get("mAP50_95", 0.0),
        "precision": metrikler.get("precision", 0.0),
        "recall": metrikler.get("recall", 0.0),
        "f1": round(f1, 6),
        "tp": metrikler.get("tp", 0),
        "fp": metrikler.get("fp", 0),
        "fn": metrikler.get("fn", 0),
    }


def _grid_sonuclarini_sirala(sonuclar):
    return sorted(sonuclar, key=lambda sonuc: (sonuc["mAP50_95"], sonuc["mAP50"], sonuc["f1"], sonuc["recall"]), reverse=True)


def wbf_grid_search_calistir(miktar=50, ince_ayar=False, ham_tespit_uretici=None, yapilandirma=None, rapor_uret=True):
    yapilandirma = copy.deepcopy(yapilandirma or yapilandirma_yukle())
    kayitlar, gercekler, kaynak_adi = _etiketli_veriyi_hazirla(miktar)
    if not kayitlar:
        return _rapor_dosyalari_ekle({"durum": "Atlandı (Etiketli görsel bulunamadı)", "veri_kaynagi": kaynak_adi}, rapor_uret, "wbf_grid_search")
    try:
        onbellek_baslangici = time.perf_counter()
        onbellek, onbellek_bilgisi = _ham_tespit_onbellegi_olustur(kayitlar, yapilandirma, ham_tespit_uretici)
        onbellek_suresi = time.perf_counter() - onbellek_baslangici
        kaba_sonuclar = []
        arama_baslangici = time.perf_counter()
        for iou_esigi in _aralik_degerleri(0.30, 0.80, 0.05):
            for guven_esigi in _aralik_degerleri(0.10, 0.50, 0.05):
                kaba_sonuclar.append(_wbf_parametrelerini_degerlendir(onbellek, gercekler, yapilandirma, iou_esigi, guven_esigi))
        sirali_kaba_sonuclar = _grid_sonuclarini_sirala(kaba_sonuclar)
        en_iyi = sirali_kaba_sonuclar[0]
        ince_sonuclar = []
        if ince_ayar:
            iou_baslangic = max(0.30, en_iyi["iou_esigi"] - 0.04)
            iou_bitis = min(0.80, en_iyi["iou_esigi"] + 0.04)
            guven_baslangic = max(0.10, en_iyi["guven_esigi"] - 0.04)
            guven_bitis = min(0.50, en_iyi["guven_esigi"] + 0.04)
            for iou_esigi in _aralik_degerleri(iou_baslangic, iou_bitis, 0.01):
                for guven_esigi in _aralik_degerleri(guven_baslangic, guven_bitis, 0.01):
                    ince_sonuclar.append(_wbf_parametrelerini_degerlendir(onbellek, gercekler, yapilandirma, iou_esigi, guven_esigi))
            en_iyi = _grid_sonuclarini_sirala(ince_sonuclar + [en_iyi])[0]
        rapor = {
            "durum": "Tamamlandı",
            "benchmark": "wbf_grid_search",
            "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
            "veri_kaynagi": kaynak_adi,
            "gorsel_sayisi": len(kayitlar),
            "dedektor_calistirma_sayisi": len(kayitlar),
            "onbellek_olusturma_saniye": round(onbellek_suresi, 6),
            "arama_saniye": round(time.perf_counter() - arama_baslangici, 6),
            "kaba_kombinasyon_sayisi": len(kaba_sonuclar),
            "ince_ayar_uygulandi": bool(ince_ayar),
            "ince_kombinasyon_sayisi": len(ince_sonuclar),
            "onbellek_bilgisi": onbellek_bilgisi,
            "onerilen_parametreler": {
                "wbf_iou_esigi": en_iyi["iou_esigi"],
                "guven_esigi": en_iyi["guven_esigi"],
                "metrikler": en_iyi,
                "config_yaml_degistirildi": False,
            },
            "en_iyi_10_kaba_sonuc": sirali_kaba_sonuclar[:10],
            "en_iyi_10_ince_sonuc": _grid_sonuclarini_sirala(ince_sonuclar)[:10],
        }
        return _rapor_dosyalari_ekle(rapor, rapor_uret, "wbf_grid_search")
    except Exception as hata:
        return _rapor_dosyalari_ekle({"durum": f"Hata: {hata}", "benchmark": "wbf_grid_search"}, rapor_uret, "wbf_grid_search")


def wbf_onerisini_yapilandirmaya_uygula(rapor):
    wbf_raporu = rapor.get("wbf_grid_search", rapor)
    oneri = wbf_raporu.get("onerilen_parametreler", {})
    iou_esigi = oneri.get("wbf_iou_esigi")
    guven_esigi = oneri.get("guven_esigi")
    if iou_esigi is None or guven_esigi is None:
        raise ValueError("Raporda uygulanabilir WBF önerisi bulunamadı")
    iou_esigi = float(iou_esigi)
    guven_esigi = float(guven_esigi)
    if not 0.0 < iou_esigi <= 1.0 or not 0.0 <= guven_esigi <= 1.0:
        raise ValueError("WBF önerisi geçerli eşik aralığında değil")
    yapilandirma = copy.deepcopy(yapilandirma_yukle())
    multi_model = yapilandirma.setdefault("multi_model", {})
    onceki_degerler = {
        "wbf_iou_esigi": multi_model.get("wbf_iou_esigi"),
        "guven_esigi": multi_model.get("guven_esigi"),
    }
    multi_model["wbf_iou_esigi"] = round(iou_esigi, 2)
    multi_model["guven_esigi"] = round(guven_esigi, 2)
    yapilandirma_kaydet(yapilandirma)
    uygulama = {
        "durum": "Uygulandı",
        "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
        "onceki_degerler": onceki_degerler,
        "yeni_degerler": {
            "wbf_iou_esigi": multi_model["wbf_iou_esigi"],
            "guven_esigi": multi_model["guven_esigi"],
        },
    }
    oneri["config_yaml_degistirildi"] = True
    oneri["uygulama"] = uygulama
    if rapor.get("rapor_dosyalari"):
        gelismis_rapor_dosyalarini_guncelle(rapor)
    return uygulama


def karisiklik_matrisi_hesapla(tahminler, gercekler, siniflar=None, iou_esigi=0.50):
    siniflar = siniflar or {}
    sinif_idleri = sorted({int(k) for k in siniflar} | {int(t["sinif_id"]) for t in tahminler} | {int(g["sinif_id"]) for g in gercekler})
    indeksler = {sinif_id: indeks for indeks, sinif_id in enumerate(sinif_idleri)}
    arka_plan_indeksi = len(sinif_idleri)
    matris = np.zeros((arka_plan_indeksi + 1, arka_plan_indeksi + 1), dtype=np.int64)
    gorsel_idleri = sorted({str(t["gorsel_id"]) for t in tahminler} | {str(g["gorsel_id"]) for g in gercekler})
    karisikliklar = []
    for gorsel_id in gorsel_idleri:
        gorsel_gercekleri = [g for g in gercekler if str(g["gorsel_id"]) == gorsel_id]
        gorsel_tahminleri = sorted([t for t in tahminler if str(t["gorsel_id"]) == gorsel_id], key=lambda t: float(t.get("guven", 0.0)), reverse=True)
        kullanilan_gercekler = set()
        for tahmin in gorsel_tahminleri:
            adaylar = []
            for indeks, gercek in enumerate(gorsel_gercekleri):
                if indeks not in kullanilan_gercekler:
                    adaylar.append((kutu_iou_hesapla(tahmin["kutucuk"], gercek["kutucuk"]), indeks, gercek))
            en_iyi = max(adaylar, default=(0.0, None, None), key=lambda aday: aday[0])
            if en_iyi[1] is not None and en_iyi[0] >= iou_esigi:
                kullanilan_gercekler.add(en_iyi[1])
                gercek_sinif = int(en_iyi[2]["sinif_id"])
                tahmin_sinif = int(tahmin["sinif_id"])
                matris[indeksler[gercek_sinif], indeksler[tahmin_sinif]] += 1
                if gercek_sinif != tahmin_sinif:
                    karisikliklar.append({"gercek_sinif_id": gercek_sinif, "tahmin_sinif_id": tahmin_sinif, "iou": round(en_iyi[0], 6)})
            else:
                matris[arka_plan_indeksi, indeksler[int(tahmin["sinif_id"])]] += 1
        for indeks, gercek in enumerate(gorsel_gercekleri):
            if indeks not in kullanilan_gercekler:
                matris[indeksler[int(gercek["sinif_id"])], arka_plan_indeksi] += 1
    satir_toplamlari = matris.sum(axis=1, keepdims=True)
    normalize = np.divide(matris, satir_toplamlari, out=np.zeros_like(matris, dtype=float), where=satir_toplamlari != 0)
    etiketler = [str(siniflar.get(sinif_id, siniflar.get(str(sinif_id), f"Sinif_{sinif_id}"))) for sinif_id in sinif_idleri] + ["Arka Plan"]
    sinif_bazli = {}
    for sinif_id, indeks in indeksler.items():
        sinif_adi = etiketler[indeks]
        sinif_bazli[sinif_adi] = {
            "sinif_id": sinif_id,
            "tp": int(matris[indeks, indeks]),
            "fp": int(matris[:, indeks].sum() - matris[indeks, indeks]),
            "fn": int(matris[indeks, :].sum() - matris[indeks, indeks]),
        }
    cift_sayaclari = {}
    for kayit in karisikliklar:
        anahtar = (kayit["gercek_sinif_id"], kayit["tahmin_sinif_id"])
        cift_sayaclari.setdefault(anahtar, []).append(kayit["iou"])
    karisiklik_ozeti = [
        {
            "gercek": str(siniflar.get(gercek_id, siniflar.get(str(gercek_id), f"Sinif_{gercek_id}"))),
            "tahmin": str(siniflar.get(tahmin_id, siniflar.get(str(tahmin_id), f"Sinif_{tahmin_id}"))),
            "adet": len(iou_degerleri),
            "ortalama_iou": round(statistics.fmean(iou_degerleri), 6),
        }
        for (gercek_id, tahmin_id), iou_degerleri in cift_sayaclari.items()
    ]
    karisiklik_ozeti.sort(key=lambda kayit: (kayit["adet"], kayit["ortalama_iou"]), reverse=True)
    return {
        "etiketler": etiketler,
        "matris": matris.tolist(),
        "normalize_matris": np.round(normalize, 6).tolist(),
        "sinif_bazli": sinif_bazli,
        "sinif_karisikliklari": karisiklik_ozeti,
        "iou_esigi": float(iou_esigi),
    }


def sinif_karisiklik_matrisi_calistir(miktar=50, tahminler=None, gercekler=None, tahmin_uretici=None, yapilandirma=None, rapor_uret=True):
    yapilandirma = copy.deepcopy(yapilandirma or yapilandirma_yukle())
    def kapat():
        return None
    try:
        kaynak_adi = "Sağlanan tahmin ve etiketler"
        kayitlar = []
        if gercekler is None:
            kayitlar, gercekler, kaynak_adi = _etiketli_veriyi_hazirla(miktar)
        if not gercekler:
            return _rapor_dosyalari_ekle({"durum": "Atlandı (Etiket bulunamadı)", "veri_kaynagi": kaynak_adi}, rapor_uret, "sinif_karisiklik")
        backend = "Sağlanan tahminler"
        if tahminler is None:
            if tahmin_uretici is None:
                tahmin_uretici, kapat, backend = _gercek_tahmin_ureticisini_hazirla(yapilandirma)
            else:
                backend = "Özel tahmin üreticisi"
            tahminler = []
            for kayit in kayitlar:
                tahminler.extend(_tahminleri_standartlastir(tahmin_uretici(kayit["gorsel"], kayit["gorsel_id"]), kayit["gorsel_id"]))
        sonuc = karisiklik_matrisi_hesapla(tahminler, gercekler, yapilandirma.get("siniflar", {}))
        rapor = {
            "durum": "Tamamlandı",
            "benchmark": "sinif_karisiklik_matrisi",
            "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
            "veri_kaynagi": kaynak_adi,
            "backend": backend,
            "tahmin_sayisi": len(tahminler),
            "gercek_sayisi": len(gercekler),
            **sonuc,
        }
        return _rapor_dosyalari_ekle(rapor, rapor_uret, "sinif_karisiklik")
    except Exception as hata:
        return _rapor_dosyalari_ekle({"durum": f"Hata: {hata}", "benchmark": "sinif_karisiklik_matrisi"}, rapor_uret, "sinif_karisiklik")
    finally:
        kapat()


def sinif_karasizlik_matrisi_calistir(*args, **kwargs):
    return sinif_karisiklik_matrisi_calistir(*args, **kwargs)


def _vram_guvenlik_bilgisi():
    try:
        import torch
        if torch.cuda.is_available():
            bos_bayt, toplam_bayt = torch.cuda.mem_get_info()
            return {"bos_vram_mb": round(bos_bayt / 1048576, 2), "toplam_vram_mb": round(toplam_bayt / 1048576, 2)}
    except (ImportError, RuntimeError):
        pass
    return None


def _stres_on_kontrol(isci_sayisi, tahmini_isci_ram_mb):
    sanal_bellek = psutil.virtual_memory()
    gereken_ram_mb = float(tahmini_isci_ram_mb) * int(isci_sayisi) * 1.25
    bos_ram_mb = sanal_bellek.available / 1048576
    vram = _vram_guvenlik_bilgisi()
    nedenler = []
    if sanal_bellek.percent >= 85:
        nedenler.append("Sistem RAM kullanımı yüzde 85 veya üzerinde")
    if gereken_ram_mb > bos_ram_mb:
        nedenler.append("Tahmini RAM gereksinimi kullanılabilir RAM'i aşıyor")
    if vram is not None and vram["bos_vram_mb"] < min(1024.0, float(tahmini_isci_ram_mb)):
        nedenler.append("Kullanılabilir CUDA VRAM güvenlik eşiğinin altında")
    return {
        "guvenli": not nedenler,
        "nedenler": nedenler,
        "bos_ram_mb": round(bos_ram_mb, 2),
        "tahmini_gereken_ram_mb": round(gereken_ram_mb, 2),
        "vram": vram,
    }


def eszamanlilik_stres_testi_calistir(gorseller=None, isci_seviyeleri=(5, 10, 20), is_fonksiyonu=None, is_sayisi_katsayisi=2, tahmini_isci_ram_mb=None, yapilandirma=None, rapor_uret=True):
    yapilandirma = copy.deepcopy(yapilandirma or yapilandirma_yukle())
    if gorseller is None:
        gorseller = _gorselleri_listele(PROJE_KOKU / "hasar-ornek", artirilmis_dahil=False)
    gorseller = [Path(gorsel) for gorsel in gorseller]
    if not gorseller:
        return _rapor_dosyalari_ekle({"durum": "Atlandı (Görsel bulunamadı)"}, rapor_uret, "eszamanlilik_stres")
    if is_fonksiyonu is None:
        from src.pipeline import coklu_model_hasar_tespiti_yap, hasar_tespiti_yap
        benchmark_yapilandirmasi = copy.deepcopy(yapilandirma)
        benchmark_yapilandirmasi.setdefault("cikarim", {})["gorsel_kaydet"] = False
        benchmark_yapilandirmasi["cikarim"]["json_kaydet"] = False

        def is_fonksiyonu(gorsel_yolu):
            if benchmark_yapilandirmasi.get("multi_model", {}).get("aktif", False):
                return coklu_model_hasar_tespiti_yap(gorsel_yolu, json_kaydet=False, yapilandirma=benchmark_yapilandirmasi)
            return hasar_tespiti_yap(gorsel_yolu, json_kaydet=False, yapilandirma=benchmark_yapilandirmasi)

    if tahmini_isci_ram_mb is None:
        tahmini_isci_ram_mb = max(512.0, MEVCUT_SUREC.memory_info().rss / 1048576)
    seviye_sonuclari = {}
    for isci_sayisi in isci_seviyeleri:
        isci_sayisi = int(isci_sayisi)
        on_kontrol = _stres_on_kontrol(isci_sayisi, tahmini_isci_ram_mb)
        if not on_kontrol["guvenli"]:
            seviye_sonuclari[str(isci_sayisi)] = {"durum": "Atlandı (Kaynak güvenliği)", "on_kontrol": on_kontrol}
            continue
        toplam_is = max(isci_sayisi, isci_sayisi * int(is_sayisi_katsayisi))
        is_listesi = [gorseller[indeks % len(gorseller)] for indeks in range(toplam_is)]
        bellek_once = MEVCUT_SUREC.memory_info().rss / 1048576
        baslangic = time.perf_counter()
        gecikmeler = []
        hata_sayisi = 0

        def zamanli_is(gorsel_yolu):
            is_baslangici = time.perf_counter()
            sonuc = is_fonksiyonu(gorsel_yolu)
            return sonuc, time.perf_counter() - is_baslangici

        with ThreadPoolExecutor(max_workers=isci_sayisi, thread_name_prefix="hades-stres") as havuz:
            gelecekler = [havuz.submit(zamanli_is, gorsel_yolu) for gorsel_yolu in is_listesi]
            for gelecek in as_completed(gelecekler):
                try:
                    sonuc, gecikme = gelecek.result()
                    gecikmeler.append(gecikme)
                    if sonuc is None:
                        hata_sayisi += 1
                except Exception:
                    hata_sayisi += 1
        toplam_sure = time.perf_counter() - baslangic
        gc.collect()
        bellek_sonra = MEVCUT_SUREC.memory_info().rss / 1048576
        bellek_artisi = bellek_sonra - bellek_once
        sizinti_esigi = max(64.0, bellek_once * 0.10)
        seviye_sonuclari[str(isci_sayisi)] = {
            "durum": "Tamamlandı",
            "on_kontrol": on_kontrol,
            "toplam_is": toplam_is,
            "basarili_is": toplam_is - hata_sayisi,
            "hata_sayisi": hata_sayisi,
            "toplam_saniye": round(toplam_sure, 6),
            "istek_saniye": round(toplam_is / toplam_sure, 6) if toplam_sure > 0 else None,
            "ortalama_gecikme_saniye": round(statistics.fmean(gecikmeler), 6) if gecikmeler else None,
            "p95_gecikme_saniye": round(float(np.percentile(gecikmeler, 95)), 6) if gecikmeler else None,
            "bellek_once_mb": round(bellek_once, 2),
            "bellek_sonra_mb": round(bellek_sonra, 2),
            "bellek_artisi_mb": round(bellek_artisi, 2),
            "bellek_sizintisi_suphesi": bellek_artisi > sizinti_esigi,
            "sizinti_esigi_mb": round(sizinti_esigi, 2),
        }
    rapor = {
        "durum": "Tamamlandı",
        "benchmark": "eszamanlilik_stres",
        "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
        "is_parcacigi_modeli": "ThreadPoolExecutor",
        "florence_erisim_modeli": "Seri kilit",
        "tahmini_isci_ram_mb": round(float(tahmini_isci_ram_mb), 2),
        "seviyeler": seviye_sonuclari,
    }
    return _rapor_dosyalari_ekle(rapor, rapor_uret, "eszamanlilik_stres")


def _dengeli_pozitif_ornekleri_sec(kayitlar, ornek_sayisi):
    sinif_havuzlari = {}
    for kayit in kayitlar:
        for etiket in kayit["etiketler"]:
            sinif_havuzlari.setdefault(int(etiket["sinif_id"]), []).append((kayit, etiket))
    secilenler = []
    sinif_idleri = sorted(sinif_havuzlari)
    hedef = sum(len(degerler) for degerler in sinif_havuzlari.values()) if ornek_sayisi is None else max(0, int(ornek_sayisi))
    konumlar = {sinif_id: 0 for sinif_id in sinif_idleri}
    while len(secilenler) < hedef:
        ilerledi = False
        for sinif_id in sinif_idleri:
            konum = konumlar[sinif_id]
            if konum < len(sinif_havuzlari[sinif_id]):
                secilenler.append(sinif_havuzlari[sinif_id][konum])
                konumlar[sinif_id] += 1
                ilerledi = True
                if len(secilenler) >= hedef:
                    break
        if not ilerledi:
            break
    return secilenler


def _negatif_bolge_bul(kayit, rastgele):
    genislik = kayit["genislik"]
    yukseklik = kayit["yukseklik"]
    kutu_genisligi = max(24, int(genislik * 0.22))
    kutu_yuksekligi = max(24, int(yukseklik * 0.22))
    if kutu_genisligi >= genislik or kutu_yuksekligi >= yukseklik:
        return None
    for _ in range(30):
        x1 = int(rastgele.integers(0, genislik - kutu_genisligi))
        y1 = int(rastgele.integers(0, yukseklik - kutu_yuksekligi))
        aday = [x1, y1, x1 + kutu_genisligi, y1 + kutu_yuksekligi]
        if all(kutu_iou_hesapla(aday, etiket["kutucuk"]) < 0.02 for etiket in kayit["etiketler"]):
            return aday
    return None


def _vlm_gercek_sorgulayicisini_hazirla(yapilandirma):
    from src.inspector_florence import _florence_modeli_yukle, _florence_sorgula, _hasar_siniflandir
    multi_model = yapilandirma.get("multi_model", {})
    denetleyici = multi_model.get("denetleyici_ayarlari", {})
    model, islemci, cihaz = _florence_modeli_yukle(denetleyici.get("model", "microsoft/Florence-2-base"), multi_model.get("otomatik_yedekleme_cpu", True))
    ekstra_siniflar = denetleyici.get("ekstra_siniflar", [])

    def sorgula(crop, negatif=False):
        gorev = "<DETAILED_CAPTION>"
        sonuc = _florence_sorgula(model, islemci, cihaz, crop, gorev=gorev)
        bolum = sonuc.get(gorev, sonuc) if isinstance(sonuc, dict) else sonuc
        if isinstance(bolum, dict):
            metin = " ".join(str(etiket) for etiket in bolum.get("labels", []))
        else:
            metin = str(bolum)
        return _hasar_siniflandir(metin, ekstra_siniflar)

    return sorgula, cihaz


def vlm_skorlarini_hesapla(pozitif_sonuclar, negatif_sonuclar):
    dogru = sum(1 for sonuc in pozitif_sonuclar if sonuc["tahmin"] == sonuc["gercek"])
    bilinmeyen = sum(1 for sonuc in pozitif_sonuclar if sonuc["tahmin"] == "Bilinmeyen")
    yanlis = len(pozitif_sonuclar) - dogru - bilinmeyen
    halusinasyon = sum(1 for sonuc in negatif_sonuclar if sonuc["tahmin"] != "Bilinmeyen")
    return {
        "pozitif_ornek_sayisi": len(pozitif_sonuclar),
        "dogru_siniflandirma": dogru,
        "yanlis_siniflandirma": yanlis,
        "bilinmeyen_sayisi": bilinmeyen,
        "dogruluk": round(dogru / len(pozitif_sonuclar), 6) if pozitif_sonuclar else None,
        "negatif_ornek_sayisi": len(negatif_sonuclar),
        "halusinasyon_sayisi": halusinasyon,
        "halusinasyon_orani": round(halusinasyon / len(negatif_sonuclar), 6) if negatif_sonuclar else None,
    }


def vlm_dogrulama_benchmark_calistir(ornek_sayisi=50, negatif_ornek_sayisi=20, vlm_sorgulayici=None, yapilandirma=None, rapor_uret=True):
    yapilandirma = copy.deepcopy(yapilandirma or yapilandirma_yukle())
    toplam_ihtiyac = (ornek_sayisi or 50) + max(0, int(negatif_ornek_sayisi)) + 10
    kayitlar, _, kaynak_adi = _etiketli_veriyi_hazirla(toplam_ihtiyac)
    if not kayitlar:
        return _rapor_dosyalari_ekle({"durum": "Atlandı (Etiketli görsel bulunamadı)", "veri_kaynagi": kaynak_adi}, rapor_uret, "vlm_dogrulama")
    try:
        backend = "Özel VLM sorgulayıcısı"
        if vlm_sorgulayici is None:
            vlm_sorgulayici, backend = _vlm_gercek_sorgulayicisini_hazirla(yapilandirma)
        pozitif_adaylar = _dengeli_pozitif_ornekleri_sec(kayitlar, ornek_sayisi)
        siniflar = yapilandirma.get("siniflar", {})
        pozitif_sonuclar = []
        for kayit, etiket in pozitif_adaylar:
            x1, y1, x2, y2 = [int(deger) for deger in etiket["kutucuk"]]
            crop = kayit["gorsel"][max(0, y1):min(kayit["yukseklik"], y2), max(0, x1):min(kayit["genislik"], x2)]
            if crop.size == 0:
                continue
            gercek_sinif = str(siniflar.get(int(etiket["sinif_id"]), siniflar.get(str(etiket["sinif_id"]), f"Sinif_{etiket['sinif_id']}")))
            tahmin = str(vlm_sorgulayici(crop, False))
            pozitif_sonuclar.append({"gercek": gercek_sinif, "tahmin": tahmin})
        rastgele = np.random.default_rng(42)
        negatif_sonuclar = []
        negatif_hedef = max(0, int(negatif_ornek_sayisi))
        kullanilan_negatifler = set()
        for indeks in range(max(negatif_hedef * 4, len(kayitlar))):
            if len(negatif_sonuclar) >= negatif_hedef:
                break
            kayit = kayitlar[indeks % len(kayitlar)]
            kutu = _negatif_bolge_bul(kayit, rastgele)
            negatif_anahtari = (kayit["gorsel_id"], tuple(kutu) if kutu is not None else None)
            if kutu is None or negatif_anahtari in kullanilan_negatifler:
                continue
            kullanilan_negatifler.add(negatif_anahtari)
            x1, y1, x2, y2 = kutu
            crop = kayit["gorsel"][y1:y2, x1:x2]
            tahmin = str(vlm_sorgulayici(crop, True))
            negatif_sonuclar.append({"gercek": "Hasarsiz Arka Plan", "tahmin": tahmin})
        skorlar = vlm_skorlarini_hesapla(pozitif_sonuclar, negatif_sonuclar)
        rapor = {
            "durum": "Tamamlandı",
            "benchmark": "vlm_dogrulama",
            "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
            "veri_kaynagi": kaynak_adi,
            "backend": backend,
            "artirilmis_gorseller_dahil": False,
            "dengeli_ornekleme": True,
            "skorlar": skorlar,
            "pozitif_sonuclar": pozitif_sonuclar,
            "negatif_sonuclar": negatif_sonuclar,
        }
        return _rapor_dosyalari_ekle(rapor, rapor_uret, "vlm_dogrulama")
    except Exception as hata:
        return _rapor_dosyalari_ekle({"durum": f"Hata: {hata}", "benchmark": "vlm_dogrulama"}, rapor_uret, "vlm_dogrulama")


def gelismis_markdown_raporu_olustur(rapor):
    satirlar = ["# HADES Gelişmiş Benchmark Raporu", ""]

    def yaz(deger, girinti=0):
        bosluk = "  " * girinti
        if isinstance(deger, dict):
            for anahtar, alt_deger in deger.items():
                etiket = str(anahtar).replace("_", " ").title()
                if isinstance(alt_deger, (dict, list)):
                    satirlar.append(f"{bosluk}- **{etiket}:**")
                    yaz(alt_deger, girinti + 1)
                else:
                    satirlar.append(f"{bosluk}- **{etiket}:** {alt_deger}")
        elif isinstance(deger, list):
            if not deger:
                satirlar.append(f"{bosluk}- Yok")
            for alt_deger in deger:
                if isinstance(alt_deger, (dict, list)):
                    satirlar.append(f"{bosluk}-")
                    yaz(alt_deger, girinti + 1)
                else:
                    satirlar.append(f"{bosluk}- {alt_deger}")
        else:
            satirlar.append(f"{bosluk}- {deger}")

    yaz(_json_uyumlu_yap(rapor))
    return "\n".join(satirlar).rstrip() + "\n"


def gelismis_rapor_kaydet(rapor, cikti_klasoru=None, rapor_adi="gelismis_benchmark"):
    cikti_klasoru = Path(cikti_klasoru or PROJE_KOKU / "runs" / "benchmark")
    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    zaman = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    guvenli_ad = "".join(karakter if karakter.isalnum() or karakter in ("-", "_") else "_" for karakter in str(rapor_adi))
    json_yolu = cikti_klasoru / f"{guvenli_ad}_{zaman}.json"
    markdown_yolu = cikti_klasoru / f"{guvenli_ad}_{zaman}.md"
    temiz_rapor = _json_uyumlu_yap(rapor)
    temiz_rapor["rapor_dosyalari"] = {"json": str(json_yolu), "markdown": str(markdown_yolu)}
    json_yolu.write_text(json.dumps(temiz_rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_yolu.write_text(gelismis_markdown_raporu_olustur(temiz_rapor), encoding="utf-8")
    return {"json": str(json_yolu), "markdown": str(markdown_yolu)}


def gelismis_rapor_dosyalarini_guncelle(rapor):
    rapor_dosyalari = rapor.get("rapor_dosyalari", {})
    json_yolu = Path(rapor_dosyalari.get("json", ""))
    markdown_yolu = Path(rapor_dosyalari.get("markdown", ""))
    if not json_yolu.is_file() or not markdown_yolu.is_file():
        raise FileNotFoundError("Güncellenecek gelişmiş benchmark raporu bulunamadı")
    temiz_rapor = _json_uyumlu_yap(rapor)
    json_yolu.write_text(json.dumps(temiz_rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_yolu.write_text(gelismis_markdown_raporu_olustur(temiz_rapor), encoding="utf-8")
    return {"json": str(json_yolu), "markdown": str(markdown_yolu)}


def gelismis_benchmark_suitini_calistir(miktar=50, ince_ayar=False, negatif_ornek_sayisi=20, yapilandirma=None):
    yapilandirma = copy.deepcopy(yapilandirma or yapilandirma_yukle())
    rapor = {
        "durum": "Tamamlandı",
        "benchmark": "gelismis_benchmark_suiti",
        "zaman_damgasi": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bellek_baslangic": bellek_olcu_al(),
        "dayaniklilik": dayaniklilik_benchmark_calistir(miktar=miktar, yapilandirma=yapilandirma, rapor_uret=False),
        "wbf_grid_search": wbf_grid_search_calistir(miktar=miktar, ince_ayar=ince_ayar, yapilandirma=yapilandirma, rapor_uret=False),
        "sinif_karisiklik_matrisi": sinif_karisiklik_matrisi_calistir(miktar=miktar, yapilandirma=yapilandirma, rapor_uret=False),
        "eszamanlilik_stres": eszamanlilik_stres_testi_calistir(yapilandirma=yapilandirma, rapor_uret=False),
        "vlm_dogrulama": vlm_dogrulama_benchmark_calistir(ornek_sayisi=miktar, negatif_ornek_sayisi=negatif_ornek_sayisi, yapilandirma=yapilandirma, rapor_uret=False),
        "bellek_bitis": bellek_olcu_al(),
    }
    rapor["rapor_dosyalari"] = gelismis_rapor_kaydet(rapor, rapor_adi="gelismis_benchmark_suiti")
    return rapor
