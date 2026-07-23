import sys
import json
import cv2
import time
import os
import gc
import math
from pathlib import Path
from colorama import Fore, Style, init

from src.utils import (
    PROJE_KOKU, YAPILANDIRMA_YOLU, EGITIM_KOKU, CIKARIM_KOKU, SINIF_RENKLERI,
    yapilandirma_yukle, yapilandirma_kaydet,
    _directml_cihazini_al, _openvino_kullanilabilir_mi,
)
from src.adaptive_tta import (
    gorsel_kalitesini_analiz_et,
    tta_tahminini_orijinale_tasi,
    tta_varyantlarini_olustur,
)

init()


def _openvino_model_yolu_bul(pt_model_yolu):
    """Verilen .pt modelinin OpenVINO export edilmis halini arar.
    Ultralytics export sonrasi su yapida klasor olusturur:
      runs/train/hades_egitim/weights/best_openvino_model/
    """
    pt_yolu = Path(pt_model_yolu)
    ov_klasor = pt_yolu.parent / (pt_yolu.stem + "_openvino_model")
    if ov_klasor.exists() and (ov_klasor / "best.xml").exists():
        return ov_klasor
    ov_xml = pt_yolu.with_suffix(".xml")
    if ov_xml.exists():
        return ov_xml
    return None


def _model_yukle_optimize(model_yolu, yapilandirma, amac="cikarim"):
    """Modeli en uygun backend ile yukler.

    Oncelik sirasi:
      1. OpenVINO (Intel Arc GPU / NPU hizlandirmasi)
      2. DirectML (Intel Arc / AMD GPU - yalnizca .pt modellerde)
      3. PyTorch CPU (her zaman calisir)

    Returns:
        tuple: (model, backend_adi)
    """
    from ultralytics import YOLO
    model_tur = yapilandirma.get("model", {}).get("tur", "yolo")
    ModelSinifi = YOLO
    if model_tur == "rtdetr":
        from ultralytics import RTDETR
        ModelSinifi = RTDETR

    model_yolu = str(model_yolu)

    ov_yolu = _openvino_model_yolu_bul(model_yolu)
    if ov_yolu is not None:
        try:
            model = ModelSinifi(str(ov_yolu))
            backend = f"OpenVINO (Intel GPU/NPU)"
            print(f"{Fore.GREEN}[+] OpenVINO modeli bulundu, GPU/NPU cikarim kullaniliyor.{Style.RESET_ALL}")
            return model, backend
        except Exception as e:
            print(f"{Fore.YELLOW}[!] OpenVINO model yuklenemedi: {e}{Style.RESET_ALL}")

    try:
        model = ModelSinifi(model_yolu)
    except Exception as e:
        raise RuntimeError(f"Model yuklenemedi: {e}")

    dml = _directml_cihazini_al()
    dml = _directml_cihazini_al()
    if dml is not None:
        try:
            import torch
            if hasattr(model, 'model') and model.model is not None:
                model.model.to(dml)
                print(f"{Fore.GREEN}[+] Model DirectML GPU'ya tasindi.{Style.RESET_ALL}")
                return model, "DirectML GPU"
        except Exception:
            pass

    return model, "PyTorch CPU"


def _openvino_export_oner(model_yolu):
    """Eger OpenVINO exportu yoksa kullaniciya oneri mesaji gosterir."""
    ov_yolu = _openvino_model_yolu_bul(model_yolu)
    if ov_yolu is None and _openvino_kullanilabilir_mi():
        print(f"{Fore.YELLOW}[!] OpenVINO exportu bulunamadi. GPU hizlandirmasi icin:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}    python src/export.py openvino{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}    veya egitim menusunden [5] ile egitimi tekrar calistirin (otomatik export).{Style.RESET_ALL}")
        print()


def egitilmis_model_yolu_bul():
    egitim_klasoru = EGITIM_KOKU / "hades_egitim"
    en_iyi_agirlik = egitim_klasoru / "weights" / "best.pt"
    if en_iyi_agirlik.exists():
        return en_iyi_agirlik

    son_agirlik = egitim_klasoru / "weights" / "last.pt"
    if son_agirlik.exists():
        return son_agirlik

    yapilandirma = yapilandirma_yukle()
    if "model" in yapilandirma and "agirlik" in yapilandirma["model"]:
        varsayilan_model = yapilandirma["model"]["agirlik"]
        model_yolu = PROJE_KOKU / varsayilan_model
        if model_yolu.exists():
            return model_yolu
        return varsayilan_model

    return None


def hasar_tespiti_yap(gorsel_yolu, cikti_klasoru=None, json_kaydet=None, model=None, yapilandirma=None):
    if yapilandirma is None:
        yapilandirma = yapilandirma_yukle()
    cikarim_ayari = yapilandirma.get("cikarim", {})
    siniflar = yapilandirma.get("siniflar", {})

    model_yolu = None
    guven_esigi = cikarim_ayari.get("guven_esigi", 0.25)
    iou_esigi = cikarim_ayari.get("iou_esigi", 0.7)
    if cikti_klasoru is None:
        cikti_klasoru = cikarim_ayari.get("cikti_klasoru", "runs/predict")
    cikti_klasoru = Path(cikti_klasoru)
    gorsel_kaydet = cikarim_ayari.get("gorsel_kaydet", True)
    tta_aktif = cikarim_ayari.get("tta_aktif", False)
    tta_adaptif_ayar = cikarim_ayari.get("tta_adaptif", {})
    sinif_guven_esikleri = cikarim_ayari.get("sinif_guven_esikleri", {})
    sahi_aktif = cikarim_ayari.get("sahi_aktif", False)
    sahi_dilim_boyutu = cikarim_ayari.get("sahi_dilim_boyutu", 640)
    sahi_adaptif_ayar = cikarim_ayari.get("sahi_adaptif", {})
    if json_kaydet is None:
        json_kaydet = cikarim_ayari.get("json_kaydet", True)

    if cikti_klasoru.is_absolute():
        cikarim_klasoru = cikti_klasoru
    else:
        cikarim_klasoru = PROJE_KOKU / cikti_klasoru

    gorsel_yolu = Path(gorsel_yolu)
    if not gorsel_yolu.exists():
        print(f"{Fore.RED}[-] Gorsel bulunamadi: {gorsel_yolu}{Style.RESET_ALL}")
        return None

    if model is None:
        model_yolu = egitilmis_model_yolu_bul()
        if model_yolu is None:
            print(f"{Fore.RED}[-] Egitilmis model bulunamadi.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[*] Once model egitimi yapin (Menu secenek 5).{Style.RESET_ALL}")
            return None

        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  HADES DETECTOR - Hasar Tespiti{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Cikarim Yapilandirmasi{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Gorsel          : {gorsel_yolu}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Model           : {model_yolu}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Guven Esigi     : {guven_esigi}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}IOU Esigi       : {iou_esigi}{Style.RESET_ALL}")

        try:
            model, backend = _model_yukle_optimize(str(model_yolu), yapilandirma)
            print(f"    {Fore.WHITE}Backend         : {Fore.GREEN}{backend}{Style.RESET_ALL}")
        except Exception as hata:
            print(f"{Fore.RED}[-] Model yuklenemedi: {hata}{Style.RESET_ALL}")
            return None

        _openvino_export_oner(str(model_yolu))
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print()

    print(f"{Fore.BLUE}[*] Hasar tespiti yapiliyor...{Style.RESET_ALL}")
    baslangic_zamani = time.time()

    try:
        import numpy as np
        gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
        okunan_gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)

        if okunan_gorsel is None:
            print(f"{Fore.RED}[-] Gorsel okunamadi veya formati desteklenmiyor: {gorsel_yolu}{Style.RESET_ALL}")
            return None

        kalite_raporu = gorsel_kalitesini_analiz_et(okunan_gorsel, tta_adaptif_ayar)
        if tta_adaptif_ayar.get("aktif", False) or tta_aktif:
            tam_gorsel_tahminleri, tta_telemetrisi = _adaptif_tta_tarama(
                model,
                okunan_gorsel,
                kalite_raporu,
                tta_adaptif_ayar,
                guven_esigi=0.10,
                iou_esigi=iou_esigi,
                zorla=tta_aktif,
            )
        else:
            tam_gorsel_tahminleri = _ultralytics_tahminlerini_standartlastir(
                model.predict(
                    source=okunan_gorsel,
                    conf=0.10,
                    iou=iou_esigi,
                    save=False,
                    verbose=False,
                    augment=False,
                )
            )
            tta_telemetrisi = {
                "tta_tetiklendi": False,
                "tta_nedeni": [],
                "uygulanan_varyantlar": ["orijinal"],
                "sinirda_guvenilirlik": bool(kalite_raporu.get("sinirda_guvenilirlik", False)),
                "tta_ek_sure_ms": 0.0,
            }
        if sahi_aktif:
            tahminler = _sahi_tarama(
                model,
                okunan_gorsel,
                guven_esigi=0.10,
                iou_esigi=iou_esigi,
                dilim_boyutu=sahi_dilim_boyutu,
                adaptif_ayar=sahi_adaptif_ayar,
                siniflar=siniflar,
                tam_gorsel_tahminleri=tam_gorsel_tahminleri,
            )
        else:
            tahminler = tam_gorsel_tahminleri
    except Exception as hata:
        print(f"{Fore.RED}[-] Cikarim sirasinda hata: {hata}{Style.RESET_ALL}")
        return None

    gecen_sure = time.time() - baslangic_zamani

    gorsel = okunan_gorsel.copy()

    tespit_edilen_hasarlar = []
    sinif_sayaclari = {}

    for tahmin in tahminler:
        koordinat = tahmin["kutucuk"]
        x1, y1, x2, y2 = koordinat["x1"], koordinat["y1"], koordinat["x2"], koordinat["y2"]
        sinif_id = tahmin["sinif_id"]
        guven = tahmin["guven"]

        guncel_esik = sinif_guven_esikleri.get(sinif_id, guven_esigi)
        if guven < guncel_esik:
            continue

        sinif_adi = siniflar.get(sinif_id, f"Sinif_{sinif_id}")
        renk = SINIF_RENKLERI.get(sinif_id, (255, 255, 255))
        cv2.rectangle(gorsel, (x1, y1), (x2, y2), renk, 3)

        etiket_metni = f"{sinif_adi} {guven:.2f}"
        (metin_genislik, metin_yukseklik), _ = cv2.getTextSize(etiket_metni, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(gorsel, (x1, y1 - metin_yukseklik - 10), (x1 + metin_genislik, y1), renk, -1)
        cv2.putText(gorsel, etiket_metni, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        tespit_edilen_hasarlar.append({
            "sinif_id": sinif_id,
            "sinif_adi": sinif_adi,
            "guven": round(guven, 4),
            "kutucuk": koordinat,
            "adaptif_sahi": tahmin.get("adaptif_sahi", False),
            "sahi_dilim_boyutu": tahmin.get("sahi_dilim_boyutu"),
            "adaptif_tta": tahmin.get("adaptif_tta", False),
            "tta_varyanti": tahmin.get("tta_varyanti", "orijinal"),
        })

        sinif_sayaclari[sinif_adi] = sinif_sayaclari.get(sinif_adi, 0) + 1

    print()
    print(f"{Fore.GREEN}[+] Hasar tespiti tamamlandi!{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Gecen Sure      : {gecen_sure:.3f} saniye{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Tespit Sayisi   : {len(tespit_edilen_hasarlar)}{Style.RESET_ALL}")

    if sinif_sayaclari:
        print(f"    {Fore.WHITE}Hasar Dagilimi  :{Style.RESET_ALL}")
        for sinif, sayi in sinif_sayaclari.items():
            print(f"      {Fore.YELLOW}- {sinif}: {sayi}{Style.RESET_ALL}")
    else:
        print(f"    {Fore.YELLOW}Hasar tespit edilmedi.{Style.RESET_ALL}")
    print()

    cikarim_klasoru.mkdir(parents=True, exist_ok=True)
    zaman_damgasi = int(time.time())

    cikti_verisi = {
        "gorsel_yolu": str(gorsel_yolu),
        "model_yolu": str(model_yolu),
        "gecen_sure_saniye": round(gecen_sure, 4),
        "toplam_tespit": len(tespit_edilen_hasarlar),
        "hasar_dagilimi": sinif_sayaclari,
        "tespitler": tespit_edilen_hasarlar,
        "kalite_telemetrisi": {
            **kalite_raporu,
            **tta_telemetrisi,
        },
    }

    if gorsel_kaydet:
        uzanti = gorsel_yolu.suffix.lower()
        if uzanti not in ['.jpg', '.jpeg', '.png', '.bmp']:
            uzanti = '.jpg'
        
        cikti_gorsel_yolu = cikarim_klasoru / f"{gorsel_yolu.stem}_tespit_{zaman_damgasi}{uzanti}"
        basari, buffer = cv2.imencode(uzanti, gorsel)
        if basari:
            buffer.tofile(str(cikti_gorsel_yolu))
        print(f"{Fore.GREEN}[+] Isaretli gorsel kaydedildi: {cikti_gorsel_yolu}{Style.RESET_ALL}")
        cikti_verisi["cikti_gorsel_yolu"] = str(cikti_gorsel_yolu)

    if json_kaydet:
        json_yolu = cikarim_klasoru / f"{gorsel_yolu.stem}_sonuc_{zaman_damgasi}.json"
        with open(json_yolu, "w", encoding="utf-8") as dosya:
            json.dump(cikti_verisi, dosya, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN}[+] JSON sonuc kaydedildi  : {json_yolu}{Style.RESET_ALL}")
        cikti_verisi["json_yolu"] = str(json_yolu)

    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return cikti_verisi


def toplu_hasar_tespiti_yap(girdi_klasoru, cikti_klasoru, miktar):
    girdi_klasoru = Path(girdi_klasoru)
    if not girdi_klasoru.is_absolute():
        girdi_klasoru = PROJE_KOKU / girdi_klasoru

    cikti_klasoru = Path(cikti_klasoru)
    if not cikti_klasoru.is_absolute():
        cikti_klasoru = PROJE_KOKU / cikti_klasoru

    gorsel_uzantilari = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    gorseller = sorted([f for f in girdi_klasoru.iterdir() if f.suffix.lower() in gorsel_uzantilari])

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Toplu Hasar Tespiti{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] Toplu Tarama Baslatiliyor{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Girdi Klasoru   : {girdi_klasoru}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Cikti Klasoru   : {cikti_klasoru}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Islenecek Adet  : {miktar}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    if not gorseller:
        print(f"{Fore.RED}[-] Klasorde gorsel bulunamadi: {girdi_klasoru}{Style.RESET_ALL}")
        return None

    islenecek_gorseller = gorseller[:miktar]

    yapilandirma = yapilandirma_yukle()
    model_yolu = egitilmis_model_yolu_bul()

    if model_yolu is None:
        print(f"{Fore.RED}[-] Egitilmis model bulunamadi.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Once model egitimi yapin (Menu secenek 5).{Style.RESET_ALL}")
        return None

    try:
        model, backend = _model_yukle_optimize(str(model_yolu), yapilandirma)
        print(f"    {Fore.WHITE}Cikarim Backend : {Fore.GREEN}{backend}{Style.RESET_ALL}")
    except Exception as hata:
        print(f"{Fore.RED}[-] Model yuklenemedi: {hata}{Style.RESET_ALL}")
        return None

    _openvino_export_oner(str(model_yolu))

    print(f"{Fore.GREEN}[+] Model bir kez yuklendi. Toplu tarama basliyor...{Style.RESET_ALL}")
    print()

    toplam_taranan = 0
    toplam_hasar = 0
    hasar_sayaclari = {}
    tum_guven_skorlari = []
    detayli_sonuclar = []

    baslangic_zamani = time.time()

    for i, gorsel in enumerate(islenecek_gorseller, 1):
        print(f"{Fore.BLUE}[*] ({i}/{len(islenecek_gorseller)}) Isleniyor: {gorsel.name}{Style.RESET_ALL}")

        sonuc = hasar_tespiti_yap(
            str(gorsel),
            cikti_klasoru=str(cikti_klasoru),
            json_kaydet=False,
            model=model,
            yapilandirma=yapilandirma,
        )

        if sonuc is not None:
            toplam_taranan += 1
            toplam_hasar += sonuc.get("toplam_tespit", 0)

            for sinif_adi, sayi in sonuc.get("hasar_dagilimi", {}).items():
                hasar_sayaclari[sinif_adi] = hasar_sayaclari.get(sinif_adi, 0) + sayi

            for tespit in sonuc.get("tespitler", []):
                tum_guven_skorlari.append(tespit.get("guven", 0))

            detayli_sonuclar.append({
                "gorsel_adi": gorsel.name,
                "gorsel_yolu": str(gorsel),
                "tespit_sayisi": sonuc.get("toplam_tespit", 0),
                "hasar_dagilimi": sonuc.get("hasar_dagilimi", {}),
                "gecen_sure_saniye": sonuc.get("gecen_sure_saniye", 0),
            })

    gecen_sure = time.time() - baslangic_zamani

    ortalama_guven = sum(tum_guven_skorlari) / len(tum_guven_skorlari) if tum_guven_skorlari else 0.0

    oransal_dagilim = {}
    if toplam_hasar > 0:
        for sinif_adi, sayi in hasar_sayaclari.items():
            oransal_dagilim[sinif_adi] = round((sayi / toplam_hasar) * 100, 2)

    genel_rapor = {
        "toplam_taranan_resim": toplam_taranan,
        "tespit_edilen_toplam_hasar": toplam_hasar,
        "hasar_tipleri_dagilimi": hasar_sayaclari,
        "hasar_tipleri_oransal_dagilim": oransal_dagilim,
        "ortalama_guven_skoru": round(ortalama_guven, 4),
        "toplam_gecen_sure_saniye": round(gecen_sure, 4),
        "detayli_sonuclar": detayli_sonuclar,
    }

    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    zaman_damgasi = int(time.time())
    rapor_yolu = cikti_klasoru / f"genel_rapor_{zaman_damgasi}.json"

    with open(rapor_yolu, "w", encoding="utf-8") as dosya:
        json.dump(genel_rapor, dosya, ensure_ascii=False, indent=2)

    print()
    print(f"{Fore.GREEN}[+] Toplu tarama tamamlandi!{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Toplam Taranan Resim  : {toplam_taranan}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Tespit Edilen Hasar  : {toplam_hasar}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Ortalama Guven Skoru : {ortalama_guven:.4f}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Toplam Gecen Sure    : {gecen_sure:.3f} saniye{Style.RESET_ALL}")

    if hasar_sayaclari:
        print(f"    {Fore.WHITE}Hasar Dagilimi       :{Style.RESET_ALL}")
        for sinif, sayi in hasar_sayaclari.items():
            oran = oransal_dagilim.get(sinif, 0)
            print(f"      {Fore.YELLOW}- {sinif}: {sayi} (%{oran}){Style.RESET_ALL}")

    print(f"{Fore.GREEN}[+] Genel rapor kaydedildi: {rapor_yolu}{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return genel_rapor


def _ram_havuzu_olustur():
    return {
        "boxes": [],
        "masks": [],
        "kalite_telemetrisi": None,
        "tta_model_telemetrisi": {},
    }


def _model_bosalt(ram_optimizasyonu=True):
    gc.collect()
    if ram_optimizasyonu:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _tek_model_tara(model_sinifi, model_yolu, kaynak_etiketi, gorsel, tespitler_havuzu,
                    guven_esigi, iou_esigi, sinif_guven_esikleri, siniflar,
                    sahi_aktif, sahi_dilim_boyutu, sahi_adaptif_ayar, otomatik_yedekleme, ram_optimizasyonu,
                    hazir_model=None, tta_ayar=None, kalite_raporu=None, tta_zorla=False):
    """Tek bir modeli yukler, gorseli tarar, kutulari havuza ekler, modeli bosaltir.

    Returns:
        int: Havuza eklenen kutu sayisi.
    """
    eklenen = 0
    model = hazir_model
    model_sahibi = hazir_model is None
    try:
        if model_sahibi:
            try:
                model = model_sinifi(str(model_yolu))
                dml = _directml_cihazini_al()
                if dml is not None:
                    import torch
                    if hasattr(model, 'model') and model.model is not None:
                        try:
                            model.model.to(dml)
                        except Exception as dml_hata:
                            print(f"{Fore.YELLOW}[!] DirectML cihazina tasima basarisiz: {dml_hata}{Style.RESET_ALL}")
            except RuntimeError as hata:
                if "out of memory" in str(hata).lower() and otomatik_yedekleme:
                    print(f"{Fore.YELLOW}[!] VRAM dolu, {kaynak_etiketi} CPU'ya kaydiriliyor...{Style.RESET_ALL}")
                    model = model_sinifi(str(model_yolu))
                else:
                    raise

        tta_ayar = tta_ayar or {}
        kalite_raporu = kalite_raporu or gorsel_kalitesini_analiz_et(gorsel, tta_ayar)
        tespitler_havuzu["kalite_telemetrisi"] = kalite_raporu
        if tta_ayar.get("aktif", False) or tta_zorla:
            tam_gorsel_tahminleri, tta_telemetrisi = _adaptif_tta_tarama(
                model,
                gorsel,
                kalite_raporu,
                tta_ayar,
                guven_esigi=0.10,
                iou_esigi=iou_esigi,
                zorla=tta_zorla,
            )
        else:
            tam_gorsel_tahminleri = _ultralytics_tahminlerini_standartlastir(
                model.predict(
                    source=gorsel, conf=0.10, iou=iou_esigi, save=False, verbose=False,
                )
            )
            tta_telemetrisi = {
                "tta_tetiklendi": False,
                "tta_nedeni": [],
                "uygulanan_varyantlar": ["orijinal"],
                "tta_ek_sure_ms": 0.0,
            }
        tespitler_havuzu["tta_model_telemetrisi"][kaynak_etiketi] = tta_telemetrisi
        if sahi_aktif:
            tahminler = _sahi_tarama(
                model,
                gorsel,
                guven_esigi=0.10,
                iou_esigi=iou_esigi,
                dilim_boyutu=sahi_dilim_boyutu,
                adaptif_ayar=sahi_adaptif_ayar,
                siniflar=siniflar,
                tam_gorsel_tahminleri=tam_gorsel_tahminleri,
            )
        else:
            tahminler = tam_gorsel_tahminleri

        for tahmin in tahminler:
            sinif_id = tahmin["sinif_id"]
            guven = tahmin["guven"]
            guncel_esik = sinif_guven_esikleri.get(sinif_id, guven_esigi)
            if guven < guncel_esik:
                continue
            sinif_adi = siniflar.get(sinif_id, f"Sinif_{sinif_id}")
            tespitler_havuzu["boxes"].append({
                "sinif_id": sinif_id,
                "sinif_adi": sinif_adi,
                "guven": round(guven, 4),
                "kutucuk": tahmin["kutucuk"],
                "kaynak_model": kaynak_etiketi,
                "adaptif_sahi": tahmin.get("adaptif_sahi", False),
                "sahi_dilim_boyutu": tahmin.get("sahi_dilim_boyutu"),
            })
            eklenen += 1

        if model_sahibi:
            del model
            _model_bosalt(ram_optimizasyonu)
    except Exception as hata:
        print(f"{Fore.RED}[-] {kaynak_etiketi} taramasi basarisiz: {hata}{Style.RESET_ALL}")

    return eklenen


def _wbf_sinif_adi_bul(sinif_id, tespitler_havuzu, yapilandirma=None):
    """WBF sonrasi sinif ID'sine karsilik gelen sinif adini config'den veya havuzdaki kutulardan bulur."""
    if yapilandirma is not None and "siniflar" in yapilandirma:
        siniflar_map = yapilandirma.get("siniflar", {})
        if sinif_id in siniflar_map:
            return siniflar_map[sinif_id]
    for kutu in tespitler_havuzu.get("boxes", []):
        if kutu.get("sinif_id") == sinif_id and "sinif_adi" in kutu:
            return kutu["sinif_adi"]
    return f"Sinif_{sinif_id}"


def _wbf_metrik_degerini_normalize_et(deger):
    try:
        metrik = float(deger)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metrik):
        return None
    if 1.0 < metrik <= 100.0:
        metrik /= 100.0
    if metrik < 0.0 or metrik > 1.0:
        return None
    return metrik


def _wbf_model_metrigini_al(model_ayari, sinif_adi):
    if not isinstance(model_ayari, dict):
        return _wbf_metrik_degerini_normalize_et(model_ayari)
    sinif_metrikleri = model_ayari.get("siniflar", {})
    if isinstance(sinif_metrikleri, dict) and sinif_adi in sinif_metrikleri:
        sinif_metrigi = _wbf_metrik_degerini_normalize_et(sinif_metrikleri[sinif_adi])
        if sinif_metrigi is not None:
            return sinif_metrigi
    return _wbf_metrik_degerini_normalize_et(model_ayari.get("genel"))


def _wbf_sabit_agirliklarini_al(sinif_adi, model_isimleri, yapilandirma):
    sinif_agirliklari = yapilandirma.get("multi_model", {}).get("wbf_sinif_agirliklari", {})
    if sinif_adi not in sinif_agirliklari:
        return None
    agirlik_ayari = sinif_agirliklari[sinif_adi]
    return [float(agirlik_ayari.get(model_adi, 1.0)) for model_adi in model_isimleri]


def _wbf_model_agirliklarini_hesapla(sinif_adi, model_isimleri, yapilandirma):
    sabit_agirliklar = _wbf_sabit_agirliklarini_al(sinif_adi, model_isimleri, yapilandirma)
    dinamik_ayar = yapilandirma.get("multi_model", {}).get("wbf_dinamik_agirliklandirma", {})
    if not dinamik_ayar.get("aktif", False):
        return sabit_agirliklar

    model_metrikleri = dinamik_ayar.get("model_metrikleri", {})
    sinifa_ozel_metrik_var = any(
        isinstance(model_ayari, dict)
        and isinstance(model_ayari.get("siniflar"), dict)
        and sinif_adi in model_ayari["siniflar"]
        for model_ayari in model_metrikleri.values()
    )
    if sabit_agirliklar is not None and not sinifa_ozel_metrik_var:
        return sabit_agirliklar

    metrikler = {
        model_adi: _wbf_model_metrigini_al(model_metrikleri.get(model_adi), sinif_adi)
        for model_adi in model_isimleri
    }
    gecerli_metrikler = [metrik for metrik in metrikler.values() if metrik is not None and metrik > 0.0]
    if not gecerli_metrikler:
        return sabit_agirliklar

    try:
        asgari_agirlik = max(0.01, float(dinamik_ayar.get("asgari_agirlik", 1.0)))
        azami_agirlik = max(asgari_agirlik, float(dinamik_ayar.get("azami_agirlik", 2.5)))
        duyarlilik = max(0.1, float(dinamik_ayar.get("duyarlilik", 4.0)))
    except (TypeError, ValueError):
        return sabit_agirliklar

    en_yuksek_metrik = max(gecerli_metrikler)
    hesaplanan_agirliklar = []
    for indeks, model_adi in enumerate(model_isimleri):
        metrik = metrikler[model_adi]
        if metrik is None or metrik <= 0.0:
            yedek_agirlik = sabit_agirliklar[indeks] if sabit_agirliklar is not None else 1.0
            hesaplanan_agirliklar.append(yedek_agirlik)
            continue
        basari_orani = min(1.0, metrik / en_yuksek_metrik)
        agirlik = asgari_agirlik + (basari_orani ** duyarlilik) * (azami_agirlik - asgari_agirlik)
        hesaplanan_agirliklar.append(round(agirlik, 6))
    return hesaplanan_agirliklar


def _wbf_kutu_birlestir(tespitler_havuzu, gorsel_genisligi, gorsel_yuksekligi, iou_esigi=0.55, guven_esigi=0.25, yapilandirma=None):
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        print(f"{Fore.YELLOW}[!] ensemble-boxes kutuphanesi yuklu degil, WBF atlandi.{Style.RESET_ALL}")
        return tespitler_havuzu["boxes"]

    model_gruplari = {}
    for kutu_bilgisi in tespitler_havuzu["boxes"]:
        kaynak = kutu_bilgisi.get("kaynak_model", "model_0")
        if kaynak not in model_gruplari:
            model_gruplari[kaynak] = []
        model_gruplari[kaynak].append(kutu_bilgisi)

    if not model_gruplari:
        return tespitler_havuzu["boxes"]

    if yapilandirma is None:
        from src.utils import yapilandirma_yukle
        yapilandirma = yapilandirma_yukle()

    tum_siniflar = set()
    for kutu_bilgisi in tespitler_havuzu["boxes"]:
        tum_siniflar.add(int(kutu_bilgisi.get("sinif_id", 0)))

    model_isimleri = list(model_gruplari.keys())
    sonuc_kutular = []

    for sinif_id in tum_siniflar:
        sinif_adi = _wbf_sinif_adi_bul(sinif_id, tespitler_havuzu, yapilandirma=yapilandirma)

        agirlik_listesi = _wbf_model_agirliklarini_hesapla(sinif_adi, model_isimleri, yapilandirma)

        kutu_listeleri = []
        skor_listeleri = []
        etiket_listeleri = []

        for kaynak in model_isimleri:
            kutular = model_gruplari[kaynak]
            k_list = []
            s_list = []
            e_list = []

            for kutu_bilgisi in kutular:
                if int(kutu_bilgisi.get("sinif_id", 0)) != sinif_id:
                    continue

                koordinat = kutu_bilgisi.get("kutucuk", {})
                x1, y1 = koordinat.get("x1", 0), koordinat.get("y1", 0)
                x2, y2 = koordinat.get("x2", 0), koordinat.get("y2", 0)

                normalize_x1 = max(0.0, min(1.0, x1 / float(gorsel_genisligi)))
                normalize_y1 = max(0.0, min(1.0, y1 / float(gorsel_yuksekligi)))
                normalize_x2 = max(0.0, min(1.0, x2 / float(gorsel_genisligi)))
                normalize_y2 = max(0.0, min(1.0, y2 / float(gorsel_yuksekligi)))

                k_list.append([normalize_x1, normalize_y1, normalize_x2, normalize_y2])
                s_list.append(float(kutu_bilgisi.get("guven", 0.0)))
                e_list.append(int(kutu_bilgisi.get("sinif_id", 0)))

            kutu_listeleri.append(k_list)
            skor_listeleri.append(s_list)
            etiket_listeleri.append(e_list)

        bos_kutu_sayisi = sum(len(l) for l in kutu_listeleri)
        if bos_kutu_sayisi == 0:
            continue

        birlesmis_kutular, birlesmis_skorlar, birlesmis_etiketler = weighted_boxes_fusion(
            kutu_listeleri,
            skor_listeleri,
            etiket_listeleri,
            weights=agirlik_listesi,
            iou_thr=iou_esigi,
            skip_box_thr=guven_esigi,
        )

        for i, (kutu, skor, etiket) in enumerate(zip(birlesmis_kutular, birlesmis_skorlar, birlesmis_etiketler)):
            x1 = int(kutu[0] * gorsel_genisligi)
            y1 = int(kutu[1] * gorsel_yuksekligi)
            x2 = int(kutu[2] * gorsel_genisligi)
            y2 = int(kutu[3] * gorsel_yuksekligi)

            sonuc_kutular.append({
                "sinif_id": int(etiket),
                "sinif_adi": sinif_adi,
                "guven": round(float(skor), 4),
                "kutucuk": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "kaynak_model": "wbf",
                "wbf_birlestirildi": True,
                "wbf_model_agirliklari": {
                    model_adi: agirlik_listesi[indeks] if agirlik_listesi is not None else 1.0
                    for indeks, model_adi in enumerate(model_isimleri)
                },
            })

    return sonuc_kutular


def _ultralytics_tahminlerini_standartlastir(sonuclar):
    tahminler = []
    for sonuc in sonuclar or []:
        if sonuc.boxes is None:
            continue
        for kutu in sonuc.boxes:
            x1, y1, x2, y2 = kutu.xyxy[0].cpu().numpy().astype(int)
            tahminler.append({
                "sinif_id": int(kutu.cls[0].cpu().numpy()),
                "guven": float(kutu.conf[0].cpu().numpy()),
                "kutucuk": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "adaptif_sahi": False,
                "sahi_dilim_boyutu": None,
            })
    return tahminler


def _adaptif_tta_tarama(model, gorsel, kalite_raporu, tta_ayar, guven_esigi=0.10, iou_esigi=0.7, zorla=False):
    baslangic = time.perf_counter()
    yukseklik, genislik = gorsel.shape[:2]
    varyantlar = tta_varyantlarini_olustur(gorsel, kalite_raporu, tta_ayar, zorla=zorla)
    tum_tahminler = []
    orijinal_cikarim_suresi = 0.0
    for varyant_indeksi, varyant in enumerate(varyantlar):
        varyant_baslangici = time.perf_counter()
        sonuclar = model.predict(
            source=varyant["gorsel"],
            conf=guven_esigi,
            iou=iou_esigi,
            save=False,
            verbose=False,
            augment=False,
        )
        standart_tahminler = _ultralytics_tahminlerini_standartlastir(sonuclar)
        tum_tahminler.extend(
            tta_tahminini_orijinale_tasi(tahmin, varyant, genislik, yukseklik)
            for tahmin in standart_tahminler
        )
        if varyant_indeksi == 0:
            orijinal_cikarim_suresi = time.perf_counter() - varyant_baslangici
    model_ici_iou = max(0.0, min(1.0, float(tta_ayar.get("model_ici_iou_esigi", 0.55))))
    birlesmis = _sinif_bazli_tahmin_birlestir(tum_tahminler, model_ici_iou)
    tta_tetiklendi = len(varyantlar) > 1
    nedenler = list(kalite_raporu.get("tta_nedeni", []))
    if tta_tetiklendi and zorla and not nedenler:
        nedenler = ["manuel"]
    toplam_tta_suresi = time.perf_counter() - baslangic
    telemetri = {
        "tta_tetiklendi": tta_tetiklendi,
        "tta_nedeni": nedenler,
        "uygulanan_varyantlar": [varyant["ad"] for varyant in varyantlar],
        "sinirda_guvenilirlik": bool(kalite_raporu.get("sinirda_guvenilirlik", False)),
        "tta_ek_sure_ms": round(max(0.0, toplam_tta_suresi - orijinal_cikarim_suresi) * 1000.0, 4),
    }
    return birlesmis, telemetri


def _sahi_tahminlerini_standartlastir(nesne_tahminleri, hedef_sinif_idleri, dilim_boyutu):
    tahminler = []
    for nesne_tahmini in nesne_tahminleri:
        kategori = getattr(nesne_tahmini, "category", None)
        sinif_id = getattr(kategori, "id", getattr(nesne_tahmini, "category_id", None))
        if sinif_id is None:
            continue
        sinif_id = int(sinif_id)
        if hedef_sinif_idleri is not None and sinif_id not in hedef_sinif_idleri:
            continue
        skor = getattr(nesne_tahmini, "score", None)
        guven = getattr(skor, "value", skor)
        kutu = nesne_tahmini.bbox.to_xyxy()
        tahminler.append({
            "sinif_id": sinif_id,
            "guven": float(guven),
            "kutucuk": {
                "x1": int(round(kutu[0])),
                "y1": int(round(kutu[1])),
                "x2": int(round(kutu[2])),
                "y2": int(round(kutu[3])),
            },
            "adaptif_sahi": True,
            "sahi_dilim_boyutu": dilim_boyutu,
        })
    return tahminler


def _tahmin_iou_hesapla(birinci, ikinci):
    birinci_kutu = birinci["kutucuk"]
    ikinci_kutu = ikinci["kutucuk"]
    x1 = max(birinci_kutu["x1"], ikinci_kutu["x1"])
    y1 = max(birinci_kutu["y1"], ikinci_kutu["y1"])
    x2 = min(birinci_kutu["x2"], ikinci_kutu["x2"])
    y2 = min(birinci_kutu["y2"], ikinci_kutu["y2"])
    kesisim = max(0, x2 - x1) * max(0, y2 - y1)
    birinci_alan = max(0, birinci_kutu["x2"] - birinci_kutu["x1"]) * max(0, birinci_kutu["y2"] - birinci_kutu["y1"])
    ikinci_alan = max(0, ikinci_kutu["x2"] - ikinci_kutu["x1"]) * max(0, ikinci_kutu["y2"] - ikinci_kutu["y1"])
    birlesim = birinci_alan + ikinci_alan - kesisim
    return kesisim / birlesim if birlesim > 0 else 0.0


def _sinif_bazli_tahmin_birlestir(tahminler, iou_esigi):
    birlesmis = []
    sinif_idleri = sorted({tahmin["sinif_id"] for tahmin in tahminler})
    for sinif_id in sinif_idleri:
        adaylar = sorted(
            [tahmin for tahmin in tahminler if tahmin["sinif_id"] == sinif_id],
            key=lambda tahmin: tahmin["guven"],
            reverse=True,
        )
        while adaylar:
            secilen = adaylar.pop(0)
            birlesmis.append(secilen)
            adaylar = [aday for aday in adaylar if _tahmin_iou_hesapla(secilen, aday) < iou_esigi]
    return sorted(birlesmis, key=lambda tahmin: tahmin["guven"], reverse=True)


def _adaptif_sahi_dilim_boyutunu_hesapla(gorsel, dilim_boyutu, adaptif_ayar):
    yukseklik, genislik = gorsel.shape[:2]
    if not adaptif_ayar.get("aktif", False):
        sabit_boyut = int(dilim_boyutu or min(genislik, yukseklik, 640))
        return min(sabit_boyut, genislik, yukseklik)

    try:
        minimum_uzun_kenar = max(1, int(adaptif_ayar.get("minimum_uzun_kenar", 1024)))
        dilim_orani = min(1.0, max(0.1, float(adaptif_ayar.get("dilim_orani", 0.5))))
        asgari_boyut = max(32, int(adaptif_ayar.get("asgari_dilim_boyutu", 384)))
        azami_boyut = max(asgari_boyut, int(adaptif_ayar.get("azami_dilim_boyutu", 768)))
    except (TypeError, ValueError):
        return None

    if max(genislik, yukseklik) < minimum_uzun_kenar:
        return None
    hesaplanan = int(round(min(genislik, yukseklik) * dilim_orani / 32.0) * 32)
    hesaplanan = min(azami_boyut, max(asgari_boyut, hesaplanan), genislik, yukseklik)
    if hesaplanan >= genislik and hesaplanan >= yukseklik:
        return None
    return hesaplanan


def _sahi_hedef_sinif_idlerini_al(siniflar, adaptif_ayar):
    if not adaptif_ayar.get("aktif", False):
        return None
    hedef_siniflar = {str(sinif_adi).casefold() for sinif_adi in adaptif_ayar.get("hedef_siniflar", ["Cizik", "Pas"])}
    return {
        int(sinif_id)
        for sinif_id, sinif_adi in siniflar.items()
        if str(sinif_adi).casefold() in hedef_siniflar
    }


def _sahi_tarama(model, gorsel, guven_esigi=0.10, iou_esigi=0.7, dilim_boyutu=None, adaptif_ayar=None, siniflar=None, tam_gorsel_tahminleri=None):
    adaptif_ayar = adaptif_ayar or {}
    siniflar = siniflar or {}
    if tam_gorsel_tahminleri is None:
        tam_gorsel_tahminleri = _ultralytics_tahminlerini_standartlastir(model.predict(
            source=gorsel,
            conf=guven_esigi,
            iou=iou_esigi,
            save=False,
            verbose=False,
        ))
    hesaplanan_dilim_boyutu = _adaptif_sahi_dilim_boyutunu_hesapla(gorsel, dilim_boyutu, adaptif_ayar)
    hedef_sinif_idleri = _sahi_hedef_sinif_idlerini_al(siniflar, adaptif_ayar)
    if hesaplanan_dilim_boyutu is None or hedef_sinif_idleri == set():
        return tam_gorsel_tahminleri

    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        return tam_gorsel_tahminleri

    try:
        bindirme_orani = min(0.5, max(0.0, float(adaptif_ayar.get("bindirme_orani", 0.2))))
        birlestirme_iou_esigi = min(1.0, max(0.0, float(adaptif_ayar.get("birlestirme_iou_esigi", 0.5))))
        detection_model = getattr(model, "_hades_sahi_detection_model", None)
        if detection_model is None:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model=model,
                confidence_threshold=guven_esigi,
            )
            setattr(model, "_hades_sahi_detection_model", detection_model)
        else:
            detection_model.confidence_threshold = guven_esigi
        tum_sinif_idleri = {int(sinif_id) for sinif_id in siniflar}
        haric_sinif_idleri = sorted(tum_sinif_idleri - hedef_sinif_idleri) if hedef_sinif_idleri is not None else None
        sahi_sonucu = get_sliced_prediction(
            gorsel,
            detection_model,
            slice_height=hesaplanan_dilim_boyutu,
            slice_width=hesaplanan_dilim_boyutu,
            overlap_height_ratio=bindirme_orani,
            overlap_width_ratio=bindirme_orani,
            perform_standard_pred=False,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=birlestirme_iou_esigi,
            exclude_classes_by_id=haric_sinif_idleri,
            verbose=0,
        )
        dilimli_tahminler = _sahi_tahminlerini_standartlastir(
            sahi_sonucu.object_prediction_list,
            hedef_sinif_idleri,
            hesaplanan_dilim_boyutu,
        )
        if hedef_sinif_idleri is None:
            return _sinif_bazli_tahmin_birlestir(tam_gorsel_tahminleri + dilimli_tahminler, birlestirme_iou_esigi)
        hedef_tam_gorsel = [tahmin for tahmin in tam_gorsel_tahminleri if tahmin["sinif_id"] in hedef_sinif_idleri]
        diger_tam_gorsel = [tahmin for tahmin in tam_gorsel_tahminleri if tahmin["sinif_id"] not in hedef_sinif_idleri]
        hedef_birlesmis = _sinif_bazli_tahmin_birlestir(hedef_tam_gorsel + dilimli_tahminler, birlestirme_iou_esigi)
        return sorted(diger_tam_gorsel + hedef_birlesmis, key=lambda tahmin: tahmin["guven"], reverse=True)
    except Exception as hata:
        print(f"{Fore.YELLOW}[!] SAHI tarama basarisiz, tam gorsel sonucuna donuluyor: {hata}{Style.RESET_ALL}")
        return tam_gorsel_tahminleri


def coklu_model_hasar_tespiti_yap(gorsel_yolu, cikti_klasoru=None, json_kaydet=None, yapilandirma=None, hazir_modeller=None):
    if yapilandirma is None:
        yapilandirma = yapilandirma_yukle()

    multi_model_ayari = yapilandirma.get("multi_model", {})
    if not multi_model_ayari.get("aktif", False):
        return hasar_tespiti_yap(gorsel_yolu, cikti_klasoru, json_kaydet, yapilandirma=yapilandirma)

    cikarim_ayari = yapilandirma.get("cikarim", {})
    siniflar = yapilandirma.get("siniflar", {})
    guven_esigi = multi_model_ayari.get("guven_esigi", cikarim_ayari.get("guven_esigi", 0.25))
    iou_esigi = cikarim_ayari.get("iou_esigi", 0.7)
    wbf_iou_esigi = multi_model_ayari.get("wbf_iou_esigi", 0.55)
    ram_optimizasyonu = multi_model_ayari.get("ram_optimizasyonu", True)
    otomatik_yedekleme = multi_model_ayari.get("otomatik_yedekleme_cpu", True)
    sinif_guven_esikleri = cikarim_ayari.get("sinif_guven_esikleri", {})
    sahi_aktif = cikarim_ayari.get("sahi_aktif", False)
    sahi_dilim_boyutu = cikarim_ayari.get("sahi_dilim_boyutu", 640)
    sahi_adaptif_ayar = cikarim_ayari.get("sahi_adaptif", {})
    tta_aktif = cikarim_ayari.get("tta_aktif", False)
    tta_adaptif_ayar = cikarim_ayari.get("tta_adaptif", {})
    max_sam_boxes = cikarim_ayari.get("max_sam_boxes", 20)

    if cikti_klasoru is None:
        cikti_klasoru = cikarim_ayari.get("cikti_klasoru", "runs/predict")
    cikti_klasoru = Path(cikti_klasoru)
    if cikti_klasoru.is_absolute():
        cikarim_klasoru = cikti_klasoru
    else:
        cikarim_klasoru = PROJE_KOKU / cikti_klasoru

    gorsel_kaydet = cikarim_ayari.get("gorsel_kaydet", True)
    if json_kaydet is None:
        json_kaydet = cikarim_ayari.get("json_kaydet", True)

    gorsel_yolu = Path(gorsel_yolu)
    if not gorsel_yolu.exists():
        print(f"{Fore.RED}[-] Gorsel bulunamadi: {gorsel_yolu}{Style.RESET_ALL}")
        return None

    import numpy as np
    gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
    gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
    if gorsel is None:
        print(f"{Fore.RED}[-] Gorsel okunamadi: {gorsel_yolu}{Style.RESET_ALL}")
        return None
    kalite_raporu = gorsel_kalitesini_analiz_et(gorsel, tta_adaptif_ayar)

    baslik_aktif = hazir_modeller is None
    if baslik_aktif:
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  HADES DETECTOR - Coklu Model Hasar Tespiti{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Gorsel          : {gorsel_yolu}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Modeller        : {' -> '.join(multi_model_ayari.get('siralama', []))}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}RAM Optimizasyon: {ram_optimizasyonu}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}CPU Yedekleme   : {otomatik_yedekleme}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print()

    tespitler_havuzu = _ram_havuzu_olustur()
    baslangic_zamani = time.perf_counter()
    asama_sureleri = {
        "rtdetr_saniye": 0.0,
        "yolo_saniye": 0.0,
        "wbf_saniye": 0.0,
        "sam_saniye": 0.0,
        "florence_saniye": 0.0,
    }
    agirliklar = multi_model_ayari.get("agirliklar", {})
    hazir_modeller = hazir_modeller or {}

    rtdetr_yolu = agirliklar.get("rtdetr", "rtdetr-v2-x.pt")
    rtdetr_model_yolu = PROJE_KOKU / rtdetr_yolu
    if not rtdetr_model_yolu.exists():
        rtdetr_model_yolu = egitilmis_model_yolu_bul()

    rtdetr_baslangici = time.perf_counter()
    rtdetr_hazir_belirtildi = "rtdetr" in hazir_modeller
    if (not rtdetr_hazir_belirtildi and rtdetr_model_yolu is not None) or hazir_modeller.get("rtdetr") is not None:
        from ultralytics import RTDETR
        rtdetr_eklenen = _tek_model_tara(
            RTDETR, rtdetr_model_yolu, "rt-detr-v2-x", gorsel, tespitler_havuzu,
            guven_esigi, iou_esigi, sinif_guven_esikleri, siniflar,
            sahi_aktif, sahi_dilim_boyutu, sahi_adaptif_ayar, otomatik_yedekleme, ram_optimizasyonu,
            hazir_model=hazir_modeller.get("rtdetr"),
            tta_ayar=tta_adaptif_ayar,
            kalite_raporu=kalite_raporu,
            tta_zorla=tta_aktif,
        )
        print(f"{Fore.GREEN}[+] RT-DETRv2-X Taramasi... [Bitti] ({rtdetr_eklenen} tespit){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[!] RT-DETR modeli bulunamadi, atlandi.{Style.RESET_ALL}")
    asama_sureleri["rtdetr_saniye"] = round(time.perf_counter() - rtdetr_baslangici, 6)

    yolo_yolu = agirliklar.get("yolo", "yolov12x.pt")
    yolo_model_yolu = PROJE_KOKU / yolo_yolu
    if not yolo_model_yolu.exists():
        yolo_model_yolu = egitilmis_model_yolu_bul()

    yolo_baslangici = time.perf_counter()
    yolo_hazir_belirtildi = "yolo" in hazir_modeller
    if (not yolo_hazir_belirtildi and yolo_model_yolu is not None) or hazir_modeller.get("yolo") is not None:
        from ultralytics import YOLO
        yolo_eklenen = _tek_model_tara(
            YOLO, yolo_model_yolu, "yolov12x", gorsel, tespitler_havuzu,
            guven_esigi, iou_esigi, sinif_guven_esikleri, siniflar,
            sahi_aktif, sahi_dilim_boyutu, sahi_adaptif_ayar, otomatik_yedekleme, ram_optimizasyonu,
            hazir_model=hazir_modeller.get("yolo"),
            tta_ayar=tta_adaptif_ayar,
            kalite_raporu=kalite_raporu,
            tta_zorla=tta_aktif,
        )
        print(f"{Fore.GREEN}[+] YOLOv12x Taramasi... [Bitti] ({yolo_eklenen} tespit){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[!] YOLO modeli bulunamadi, atlandi.{Style.RESET_ALL}")
    asama_sureleri["yolo_saniye"] = round(time.perf_counter() - yolo_baslangici, 6)

    gorsel_yuksekligi, gorsel_genisligi = gorsel.shape[:2]
    wbf_baslangici = time.perf_counter()
    birlesmis_kutular = _wbf_kutu_birlestir(tespitler_havuzu, gorsel_genisligi, gorsel_yuksekligi, iou_esigi=wbf_iou_esigi, guven_esigi=guven_esigi, yapilandirma=yapilandirma)
    tespitler_havuzu["boxes"] = birlesmis_kutular
    asama_sureleri["wbf_saniye"] = round(time.perf_counter() - wbf_baslangici, 6)

    sam_yolu = agirliklar.get("sam", "sam2_s.pt")
    sam_model_yolu = PROJE_KOKU / sam_yolu

    sam_baslangici = time.perf_counter()
    sam_hazir_belirtildi = "sam" in hazir_modeller
    try:
        if sam_hazir_belirtildi and hazir_modeller.get("sam") is None:
            raise ImportError
        from ultralytics import SAM
        sam_model = hazir_modeller.get("sam")
        sam_model_sahibi = sam_model is None
        if sam_model_sahibi:
            try:
                sam_model = SAM(str(sam_model_yolu))
                dml = _directml_cihazini_al()
                if dml is not None:
                    import torch
                    if hasattr(sam_model, 'model') and sam_model.model is not None:
                        sam_model.model.to(dml)
            except RuntimeError as hata:
                if "out of memory" in str(hata).lower() and otomatik_yedekleme:
                    print(f"{Fore.YELLOW}[!] VRAM dolu, SAM 2 CPU'ya kaydiriliyor...{Style.RESET_ALL}")
                    sam_model = SAM(str(sam_model_yolu))
                else:
                    raise

        gocuk_kutulari = [b for b in tespitler_havuzu["boxes"] if b.get("sinif_adi", "").lower() in ("gocuk", "dent")]
        masklenecek_kutular = gocuk_kutulari if gocuk_kutulari else tespitler_havuzu["boxes"]
        masklenecek_kutular = sorted(masklenecek_kutular, key=lambda b: b.get("guven", 0), reverse=True)
        masklenecek_kutular = masklenecek_kutular[:max_sam_boxes]

        for kutu_bilgisi in masklenecek_kutular:
            koordinat = kutu_bilgisi.get("kutucuk", {})
            x1 = koordinat.get("x1", 0)
            y1 = koordinat.get("y1", 0)
            x2 = koordinat.get("x2", 0)
            y2 = koordinat.get("y2", 0)

            try:
                sam_sonuclar = sam_model.predict(
                    source=gorsel,
                    bboxes=[float(x1), float(y1), float(x2), float(y2)],
                    save=False,
                    verbose=False,
                )

                for sonuc in sam_sonuclar:
                    if sonuc.masks is not None:
                        for maske in sonuc.masks.data:
                            maske_np = maske.cpu().numpy()
                            tespitler_havuzu["masks"].append({
                                "sinif_adi": kutu_bilgisi.get("sinif_adi", "Bilinmeyen"),
                                "kutucuk": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                                "maske_sekli": list(maske_np.shape),
                                "kaynak_model": "sam2_small",
                            })
            except Exception:
                pass

        if sam_model_sahibi:
            del sam_model
            _model_bosalt(ram_optimizasyonu)
    except ImportError:
        print(f"{Fore.YELLOW}[!] SAM 2 kutuphanesi yuklu degil, maskeleme atlandi.{Style.RESET_ALL}")
    except Exception as hata:
        print(f"{Fore.RED}[-] SAM 2 maskelemesi basarisiz: {hata}{Style.RESET_ALL}")
    asama_sureleri["sam_saniye"] = round(time.perf_counter() - sam_baslangici, 6)

    florence_baslangici = time.perf_counter()
    try:
        from src.inspector_florence import denetle as florence_denetle
        tespitler_havuzu = florence_denetle(tespitler_havuzu, gorsel, yapilandirma=yapilandirma)
    except Exception as hata:
        print(f"{Fore.RED}[-] Florence-2 denetimi basarisiz: {hata}{Style.RESET_ALL}")
    asama_sureleri["florence_saniye"] = round(time.perf_counter() - florence_baslangici, 6)

    gecen_sure = time.perf_counter() - baslangic_zamani

    dogrulanmis_tespitler = tespitler_havuzu.get("boxes", [])
    sinif_sayaclari = {}
    for tespit in dogrulanmis_tespitler:
        sinif_adi = tespit.get("sinif_adi", "Bilinmeyen")
        sinif_sayaclari[sinif_adi] = sinif_sayaclari.get(sinif_adi, 0) + 1

    isaretli_gorsel = gorsel.copy()
    for tespit in dogrulanmis_tespitler:
        koordinat = tespit.get("kutucuk", {})
        x1 = koordinat.get("x1", 0)
        y1 = koordinat.get("y1", 0)
        x2 = koordinat.get("x2", 0)
        y2 = koordinat.get("y2", 0)
        sinif_adi = tespit.get("sinif_adi", "Bilinmeyen")
        guven = tespit.get("guven", 0.0)

        sinif_id = 0
        for sid, sadi in siniflar.items():
            if sadi == sinif_adi:
                sinif_id = sid
                break

        renk = SINIF_RENKLERI.get(sinif_id, (255, 255, 255))
        cv2.rectangle(isaretli_gorsel, (x1, y1), (x2, y2), renk, 3)
        etiket_metni = f"{sinif_adi} {guven:.2f}"
        (metin_genislik, metin_yukseklik), _ = cv2.getTextSize(etiket_metni, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(isaretli_gorsel, (x1, y1 - metin_yukseklik - 10), (x1 + metin_genislik, y1), renk, -1)
        cv2.putText(isaretli_gorsel, etiket_metni, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print()
    print(f"{Fore.GREEN}[+] Coklu model hasar tespiti tamamlandi!{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Gecen Sure      : {gecen_sure:.3f} saniye{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Tespit Sayisi   : {len(dogrulanmis_tespitler)}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Maske Sayisi    : {len(tespitler_havuzu.get('masks', []))}{Style.RESET_ALL}")

    if sinif_sayaclari:
        print(f"    {Fore.WHITE}Hasar Dagilimi  :{Style.RESET_ALL}")
        for sinif, sayi in sinif_sayaclari.items():
            print(f"      {Fore.YELLOW}- {sinif}: {sayi}{Style.RESET_ALL}")
    else:
        print(f"    {Fore.YELLOW}Hasar tespit edilmedi.{Style.RESET_ALL}")
    print()

    cikarim_klasoru.mkdir(parents=True, exist_ok=True)
    zaman_damgasi = int(time.time())

    cikti_verisi = {
        "gorsel_yolu": str(gorsel_yolu),
        "coklu_model": True,
        "model_siralamasi": multi_model_ayari.get("siralama", []),
        "gecen_sure_saniye": round(gecen_sure, 4),
        "asama_sureleri": asama_sureleri,
        "toplam_tespit": len(dogrulanmis_tespitler),
        "toplam_maske": len(tespitler_havuzu.get("masks", [])),
        "hasar_dagilimi": sinif_sayaclari,
        "tespitler": dogrulanmis_tespitler,
        "maskeler": tespitler_havuzu.get("masks", []),
        "kalite_telemetrisi": {
            **kalite_raporu,
            "tta_tetiklendi": any(
                telemetri.get("tta_tetiklendi", False)
                for telemetri in tespitler_havuzu.get("tta_model_telemetrisi", {}).values()
            ),
            "uygulanan_varyantlar": sorted({
                varyant
                for telemetri in tespitler_havuzu.get("tta_model_telemetrisi", {}).values()
                for varyant in telemetri.get("uygulanan_varyantlar", [])
            }),
            "tta_nedeni": sorted({
                neden
                for telemetri in tespitler_havuzu.get("tta_model_telemetrisi", {}).values()
                for neden in telemetri.get("tta_nedeni", [])
            }),
            "tta_ek_sure_ms": round(sum(
                float(telemetri.get("tta_ek_sure_ms", 0.0))
                for telemetri in tespitler_havuzu.get("tta_model_telemetrisi", {}).values()
            ), 4),
            "model_telemetrisi": tespitler_havuzu.get("tta_model_telemetrisi", {}),
        },
    }

    if gorsel_kaydet:
        uzanti = gorsel_yolu.suffix.lower()
        if uzanti not in ['.jpg', '.jpeg', '.png', '.bmp']:
            uzanti = '.jpg'
        cikti_gorsel_yolu = cikarim_klasoru / f"{gorsel_yolu.stem}_tespit_{zaman_damgasi}{uzanti}"
        basari, buffer = cv2.imencode(uzanti, isaretli_gorsel)
        if basari:
            buffer.tofile(str(cikti_gorsel_yolu))
        print(f"{Fore.GREEN}[+] Isaretli gorsel kaydedildi: {cikti_gorsel_yolu}{Style.RESET_ALL}")
        cikti_verisi["cikti_gorsel_yolu"] = str(cikti_gorsel_yolu)

    if json_kaydet:
        json_yolu = cikarim_klasoru / f"{gorsel_yolu.stem}_sonuc_{zaman_damgasi}.json"
        with open(json_yolu, "w", encoding="utf-8") as dosya:
            json.dump(cikti_verisi, dosya, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN}[+] JSON sonuc kaydedildi  : {json_yolu}{Style.RESET_ALL}")
        cikti_verisi["json_yolu"] = str(json_yolu)

    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return cikti_verisi


def coklu_model_toplu_tespiti_yap(girdi_klasoru, cikti_klasoru, miktar, yapilandirma=None):
    """Yatay toplu tarama (Horizontal Batching): Modeller birer kez yuklenir, chunk halinde tum gorsellere uygulanir.

    50 goruntuluk chunk'larda: RT-DETR yukle -> 50 gorsel tara -> YOLO yukle -> 50 gorsel tara ->
    WBF birlestir -> SAM 2 yukle -> 50 gorsel maskele -> Florence-2 yukle -> 50 gorsel denetle.
    Bu sayede 50 gorsel icin 200 olan model yukleme sayisi 4'e duser.
    """
    CHUNK_BOYUTU = 50

    if yapilandirma is None:
        yapilandirma = yapilandirma_yukle()

    girdi_klasoru = Path(girdi_klasoru)
    if not girdi_klasoru.is_absolute():
        girdi_klasoru = PROJE_KOKU / girdi_klasoru

    cikti_klasoru = Path(cikti_klasoru)
    if not cikti_klasoru.is_absolute():
        cikti_klasoru = PROJE_KOKU / cikti_klasoru

    gorsel_uzantilari = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    gorseller = sorted([f for f in girdi_klasoru.iterdir() if f.suffix.lower() in gorsel_uzantilari])

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Coklu Model Toplu Hasar Tespiti (Yatay){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] Toplu Tarama Baslatiliyor{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Girdi Klasoru   : {girdi_klasoru}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Cikti Klasoru   : {cikti_klasoru}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Islenecek Adet  : {miktar}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Chunk Boyutu    : {CHUNK_BOYUTU} gorsel{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    if not gorseller:
        print(f"{Fore.RED}[-] Klasorde gorsel bulunamadi: {girdi_klasoru}{Style.RESET_ALL}")
        return None

    islenecek_gorseller = gorseller[:miktar]
    toplam_taranan = 0
    toplam_hasar = 0
    hasar_sayaclari = {}
    tum_guven_skorlari = []
    detayli_sonuclar = []
    hata_alan_gorseller = []

    baslangic_zamani = time.time()
    cikti_klasoru.mkdir(parents=True, exist_ok=True)

    for chunk_idx in range(0, len(islenecek_gorseller), CHUNK_BOYUTU):
        chunk = islenecek_gorseller[chunk_idx:chunk_idx + CHUNK_BOYUTU]
        chunk_no = chunk_idx // CHUNK_BOYUTU + 1
        toplam_chunk = (len(islenecek_gorseller) + CHUNK_BOYUTU - 1) // CHUNK_BOYUTU
        print(f"\n{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  Chunk {chunk_no}/{toplam_chunk} ({len(chunk)} gorsel){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")

        chunk_havuzlari = {}

        print(f"\n{Fore.BLUE}[*] [Chunk {chunk_no}] RT-DETRv2-X yukleniyor ve {len(chunk)} gorsel taranıyor...{Style.RESET_ALL}")
        from ultralytics import RTDETR
        rtdetr_yolu = yapilandirma.get("multi_model", {}).get("agirliklar", {}).get("rtdetr", "rtdetr-v2-x.pt")
        rtdetr_model_yolu = PROJE_KOKU / rtdetr_yolu
        if not rtdetr_model_yolu.exists():
            rtdetr_model_yolu = egitilmis_model_yolu_bul()

        if rtdetr_model_yolu is not None:
            try:
                rtdetr_model = RTDETR(str(rtdetr_model_yolu))
            except Exception as hata:
                print(f"{Fore.RED}[-] RT-DETR yuklenemedi: {hata}{Style.RESET_ALL}")
                rtdetr_model = None

            if rtdetr_model is not None:
                for gorsel_yolu in chunk:
                    try:
                        import numpy as np
                        gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
                        gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
                        if gorsel is None:
                            raise ValueError("Gorsel okunamadi")

                        if gorsel_yolu not in chunk_havuzlari:
                            chunk_havuzlari[gorsel_yolu] = _ram_havuzu_olustur()
                        if chunk_havuzlari[gorsel_yolu].get("kalite_telemetrisi") is None:
                            chunk_havuzlari[gorsel_yolu]["kalite_telemetrisi"] = gorsel_kalitesini_analiz_et(
                                gorsel,
                                yapilandirma.get("cikarim", {}).get("tta_adaptif", {}),
                            )

                        _tek_model_tara(
                            RTDETR, rtdetr_model_yolu, "rt-detr-v2-x", gorsel, chunk_havuzlari[gorsel_yolu],
                            yapilandirma.get("multi_model", {}).get("guven_esigi", 0.25),
                            yapilandirma.get("cikarim", {}).get("iou_esigi", 0.7),
                            yapilandirma.get("cikarim", {}).get("sinif_guven_esikleri", {}),
                            yapilandirma.get("siniflar", {}),
                            yapilandirma.get("cikarim", {}).get("sahi_aktif", False),
                            yapilandirma.get("cikarim", {}).get("sahi_dilim_boyutu", 640),
                            yapilandirma.get("cikarim", {}).get("sahi_adaptif", {}),
                            yapilandirma.get("multi_model", {}).get("otomatik_yedekleme_cpu", True),
                            yapilandirma.get("multi_model", {}).get("ram_optimizasyonu", True),
                            tta_ayar=yapilandirma.get("cikarim", {}).get("tta_adaptif", {}),
                            kalite_raporu=chunk_havuzlari[gorsel_yolu].get("kalite_telemetrisi"),
                            tta_zorla=yapilandirma.get("cikarim", {}).get("tta_aktif", False),
                        )
                    except Exception as hata:
                        print(f"{Fore.YELLOW}[!] {gorsel_yolu.name} RT-DETR hatası, atlanıyor: {hata}{Style.RESET_ALL}")
                        hata_alan_gorseller.append(str(gorsel_yolu))

                del rtdetr_model
                _model_bosalt(True)
        else:
            print(f"{Fore.YELLOW}[!] RT-DETR modeli bulunamadi, atlandi.{Style.RESET_ALL}")

        print(f"\n{Fore.BLUE}[*] [Chunk {chunk_no}] YOLOv12x yukleniyor ve {len(chunk)} gorsel taranıyor...{Style.RESET_ALL}")
        from ultralytics import YOLO
        yolo_yolu = yapilandirma.get("multi_model", {}).get("agirliklar", {}).get("yolo", "yolov12x.pt")
        yolo_model_yolu = PROJE_KOKU / yolo_yolu
        if not yolo_model_yolu.exists():
            yolo_model_yolu = egitilmis_model_yolu_bul()

        if yolo_model_yolu is not None:
            try:
                yolo_model = YOLO(str(yolo_model_yolu))
            except Exception as hata:
                print(f"{Fore.RED}[-] YOLO yuklenemedi: {hata}{Style.RESET_ALL}")
                yolo_model = None

            if yolo_model is not None:
                for gorsel_yolu in chunk:
                    try:
                        import numpy as np
                        gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
                        gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
                        if gorsel is None:
                            raise ValueError("Gorsel okunamadi")

                        if gorsel_yolu not in chunk_havuzlari:
                            chunk_havuzlari[gorsel_yolu] = _ram_havuzu_olustur()
                        if chunk_havuzlari[gorsel_yolu].get("kalite_telemetrisi") is None:
                            chunk_havuzlari[gorsel_yolu]["kalite_telemetrisi"] = gorsel_kalitesini_analiz_et(
                                gorsel,
                                yapilandirma.get("cikarim", {}).get("tta_adaptif", {}),
                            )

                        _tek_model_tara(
                            YOLO, yolo_model_yolu, "yolov12x", gorsel, chunk_havuzlari[gorsel_yolu],
                            yapilandirma.get("multi_model", {}).get("guven_esigi", 0.25),
                            yapilandirma.get("cikarim", {}).get("iou_esigi", 0.7),
                            yapilandirma.get("cikarim", {}).get("sinif_guven_esikleri", {}),
                            yapilandirma.get("siniflar", {}),
                            yapilandirma.get("cikarim", {}).get("sahi_aktif", False),
                            yapilandirma.get("cikarim", {}).get("sahi_dilim_boyutu", 640),
                            yapilandirma.get("cikarim", {}).get("sahi_adaptif", {}),
                            yapilandirma.get("multi_model", {}).get("otomatik_yedekleme_cpu", True),
                            yapilandirma.get("multi_model", {}).get("ram_optimizasyonu", True),
                            tta_ayar=yapilandirma.get("cikarim", {}).get("tta_adaptif", {}),
                            kalite_raporu=chunk_havuzlari[gorsel_yolu].get("kalite_telemetrisi"),
                            tta_zorla=yapilandirma.get("cikarim", {}).get("tta_aktif", False),
                        )
                    except Exception as hata:
                        print(f"{Fore.YELLOW}[!] {gorsel_yolu.name} YOLO hatası, atlanıyor: {hata}{Style.RESET_ALL}")
                        hata_alan_gorseller.append(str(gorsel_yolu))

                del yolo_model
                _model_bosalt(True)
        else:
            print(f"{Fore.YELLOW}[!] YOLO modeli bulunamadi, atlandi.{Style.RESET_ALL}")

        print(f"\n{Fore.BLUE}[*] [Chunk {chunk_no}] WBF kutu birlestirme...{Style.RESET_ALL}")
        for gorsel_yolu in chunk:
            if gorsel_yolu not in chunk_havuzlari:
                continue
            try:
                gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
                gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
                if gorsel is None:
                    continue
                h, w = gorsel.shape[:2]
                chunk_havuzlari[gorsel_yolu]["boxes"] = _wbf_kutu_birlestir(
                    chunk_havuzlari[gorsel_yolu], w, h,
                    iou_esigi=yapilandirma.get("multi_model", {}).get("wbf_iou_esigi", 0.55),
                    guven_esigi=yapilandirma.get("multi_model", {}).get("guven_esigi", 0.25),
                )
            except Exception as hata:
                print(f"{Fore.YELLOW}[!] {gorsel_yolu.name} WBF hatası, atlanıyor: {hata}{Style.RESET_ALL}")

        try:
            from ultralytics import SAM
            sam_yolu = yapilandirma.get("multi_model", {}).get("agirliklar", {}).get("sam", "sam2_s.pt")
            sam_model_yolu = PROJE_KOKU / sam_yolu
            if sam_model_yolu.exists():
                print(f"\n{Fore.BLUE}[*] [Chunk {chunk_no}] SAM 2 yukleniyor ve {len(chunk)} gorsel maskeleniyor...{Style.RESET_ALL}")
                sam_model = SAM(str(sam_model_yolu))
                max_sam = yapilandirma.get("cikarim", {}).get("max_sam_boxes", 20)

                for gorsel_yolu in chunk:
                    if gorsel_yolu not in chunk_havuzlari:
                        continue
                    try:
                        gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
                        gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
                        if gorsel is None:
                            continue

                        gocuk_kutulari = [b for b in chunk_havuzlari[gorsel_yolu]["boxes"] if b.get("sinif_adi", "").lower() in ("gocuk", "dent")]
                        mask_kutulari = gocuk_kutulari if gocuk_kutulari else chunk_havuzlari[gorsel_yolu]["boxes"]
                        mask_kutulari = sorted(mask_kutulari, key=lambda b: b.get("guven", 0), reverse=True)[:max_sam]

                        for kutu_bilgisi in mask_kutulari:
                            k = kutu_bilgisi.get("kutucuk", {})
                            try:
                                sam_sonuc = sam_model.predict(source=gorsel, bboxes=[float(k["x1"]), float(k["y1"]), float(k["x2"]), float(k["y2"])], save=False, verbose=False)
                                for s in sam_sonuc:
                                    if s.masks is not None:
                                        for m in s.masks.data:
                                            chunk_havuzlari[gorsel_yolu]["masks"].append({
                                                "sinif_adi": kutu_bilgisi.get("sinif_adi", "Bilinmeyen"),
                                                "kutucuk": {"x1": int(k["x1"]), "y1": int(k["y1"]), "x2": int(k["x2"]), "y2": int(k["y2"])},
                                                "maske_sekli": list(m.cpu().numpy().shape),
                                                "kaynak_model": "sam2_small",
                                            })
                            except Exception:
                                pass
                    except Exception as hata:
                        print(f"{Fore.YELLOW}[!] {gorsel_yolu.name} SAM 2 hatası, atlanıyor: {hata}{Style.RESET_ALL}")

                del sam_model
                _model_bosalt(True)
        except ImportError:
            print(f"{Fore.YELLOW}[!] SAM 2 kutuphanesi yuklu degil, maskeleme atlandi.{Style.RESET_ALL}")

        try:
            from src.inspector_florence import denetle as florence_denetle
            print(f"\n{Fore.BLUE}[*] [Chunk {chunk_no}] Florence-2 ile {len(chunk)} gorsel denetleniyor...{Style.RESET_ALL}")
            for gorsel_yolu in chunk:
                if gorsel_yolu not in chunk_havuzlari:
                    continue
                try:
                    gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
                    gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
                    if gorsel is None:
                        continue
                    chunk_havuzlari[gorsel_yolu] = florence_denetle(chunk_havuzlari[gorsel_yolu], gorsel, yapilandirma=yapilandirma)
                except Exception as hata:
                    print(f"{Fore.YELLOW}[!] {gorsel_yolu.name} Florence-2 hatası, atlanıyor: {hata}{Style.RESET_ALL}")
        except Exception as hata:
            print(f"{Fore.YELLOW}[!] Florence-2 kullanilamadi: {hata}{Style.RESET_ALL}")

        print(f"\n{Fore.BLUE}[*] [Chunk {chunk_no}] Sonuclar kaydediliyor...{Style.RESET_ALL}")
        for gorsel_yolu in chunk:
            if gorsel_yolu not in chunk_havuzlari:
                continue
            havuz = chunk_havuzlari[gorsel_yolu]

            dogrulanmis_tespitler = havuz.get("boxes", [])
            sinif_sayaclari_gorsel = {}
            for tespit in dogrulanmis_tespitler:
                sadi = tespit.get("sinif_adi", "Bilinmeyen")
                sinif_sayaclari_gorsel[sadi] = sinif_sayaclari_gorsel.get(sadi, 0) + 1
                hasar_sayaclari[sadi] = hasar_sayaclari.get(sadi, 0) + 1
                tum_guven_skorlari.append(tespit.get("guven", 0))

            toplam_taranan += 1
            toplam_hasar += len(dogrulanmis_tespitler)

            detayli_sonuclar.append({
                "gorsel_adi": gorsel_yolu.name,
                "gorsel_yolu": str(gorsel_yolu),
                "tespit_sayisi": len(dogrulanmis_tespitler),
                "maske_sayisi": len(havuz.get("masks", [])),
                "hasar_dagilimi": sinif_sayaclari_gorsel,
                "gecen_sure_saniye": 0,
                "kalite_telemetrisi": {
                    **(havuz.get("kalite_telemetrisi") or {}),
                    "tta_tetiklendi": any(
                        telemetri.get("tta_tetiklendi", False)
                        for telemetri in havuz.get("tta_model_telemetrisi", {}).values()
                    ),
                    "uygulanan_varyantlar": sorted({
                        varyant
                        for telemetri in havuz.get("tta_model_telemetrisi", {}).values()
                        for varyant in telemetri.get("uygulanan_varyantlar", [])
                    }),
                    "tta_nedeni": sorted({
                        neden
                        for telemetri in havuz.get("tta_model_telemetrisi", {}).values()
                        for neden in telemetri.get("tta_nedeni", [])
                    }),
                    "tta_ek_sure_ms": round(sum(
                        float(telemetri.get("tta_ek_sure_ms", 0.0))
                        for telemetri in havuz.get("tta_model_telemetrisi", {}).values()
                    ), 4),
                    "model_telemetrisi": havuz.get("tta_model_telemetrisi", {}),
                },
            })

            try:
                import numpy as np
                gorsel_dizisi = np.fromfile(str(gorsel_yolu), dtype=np.uint8)
                gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
                if gorsel is not None:
                    isaretli = gorsel.copy()
                    for tespit in dogrulanmis_tespitler:
                        k = tespit.get("kutucuk", {})
                        x1, y1, x2, y2 = k.get("x1", 0), k.get("y1", 0), k.get("x2", 0), k.get("y2", 0)
                        sid = tespit.get("sinif_id", 0)
                        renk = SINIF_RENKLERI.get(sid, (255, 255, 255))
                        cv2.rectangle(isaretli, (x1, y1), (x2, y2), renk, 3)
                        etiket = f"{tespit.get('sinif_adi', '?')} {tespit.get('guven', 0):.2f}"
                        (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(isaretli, (x1, y1 - th - 10), (x1 + tw, y1), renk, -1)
                        cv2.putText(isaretli, etiket, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    uzanti = gorsel_yolu.suffix.lower()
                    if uzanti not in ['.jpg', '.jpeg', '.png', '.bmp']:
                        uzanti = '.jpg'
                    cikti_yolu = cikti_klasoru / f"{gorsel_yolu.stem}_tespit_{int(time.time())}{uzanti}"
                    basari, buffer = cv2.imencode(uzanti, isaretli)
                    if basari:
                        buffer.tofile(str(cikti_yolu))
            except Exception:
                pass

        _model_bosalt(True)

    gecen_sure = time.time() - baslangic_zamani
    ortalama_guven = sum(tum_guven_skorlari) / len(tum_guven_skorlari) if tum_guven_skorlari else 0.0

    oransal_dagilim = {}
    if toplam_hasar > 0:
        for sinif_adi, sayi in hasar_sayaclari.items():
            oransal_dagilim[sinif_adi] = round((sayi / toplam_hasar) * 100, 2)

    genel_rapor = {
        "coklu_model": True,
        "yatay_tarama": True,
        "toplam_taranan_resim": toplam_taranan,
        "tespit_edilen_toplam_hasar": toplam_hasar,
        "hasar_tipleri_dagilimi": hasar_sayaclari,
        "hasar_tipleri_oransal_dagilim": oransal_dagilim,
        "ortalama_guven_skoru": round(ortalama_guven, 4),
        "toplam_gecen_sure_saniye": round(gecen_sure, 4),
        "hata_alan_gorseller": hata_alan_gorseller,
        "detayli_sonuclar": detayli_sonuclar,
    }

    zaman_damgasi = int(time.time())
    rapor_yolu = cikti_klasoru / f"genel_rapor_{zaman_damgasi}.json"

    with open(rapor_yolu, "w", encoding="utf-8") as dosya:
        json.dump(genel_rapor, dosya, ensure_ascii=False, indent=2)

    print()
    print(f"{Fore.GREEN}[+] Toplu tarama tamamlandi!{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Toplam Taranan Resim  : {toplam_taranan}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Tespit Edilen Hasar  : {toplam_hasar}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Ortalama Guven Skoru : {ortalama_guven:.4f}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Toplam Gecen Sure    : {gecen_sure:.3f} saniye{Style.RESET_ALL}")

    if hasar_sayaclari:
        print(f"    {Fore.WHITE}Hasar Dagilimi       :{Style.RESET_ALL}")
        for sinif, sayi in hasar_sayaclari.items():
            oran = oransal_dagilim.get(sinif, 0)
            print(f"      {Fore.YELLOW}- {sinif}: {sayi} (%{oran}){Style.RESET_ALL}")

    if hata_alan_gorseller:
        print(f"    {Fore.YELLOW}[!] {len(hata_alan_gorseller)} gorsel islenemedi (detaylar raporda){Style.RESET_ALL}")

    print(f"{Fore.GREEN}[+] Genel rapor kaydedildi: {rapor_yolu}{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return genel_rapor


if __name__ == "__main__":
    if len(sys.argv) > 1:
        hasar_tespiti_yap(sys.argv[1])
    else:
        print(f"{Fore.YELLOW}Kullanim: python pipeline.py <gorsel_yolu>{Style.RESET_ALL}")
