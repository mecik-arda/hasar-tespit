import sys
import json
import cv2
import time
import os
import gc
from pathlib import Path
from colorama import Fore, Style, init

from src.utils import (
    PROJE_KOKU, YAPILANDIRMA_YOLU, EGITIM_KOKU, CIKARIM_KOKU, SINIF_RENKLERI,
    yapilandirma_yukle, yapilandirma_kaydet,
    _directml_cihazini_al, _openvino_kullanilabilir_mi,
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
    sinif_guven_esikleri = cikarim_ayari.get("sinif_guven_esikleri", {})
    sahi_aktif = cikarim_ayari.get("sahi_aktif", False)
    sahi_dilim_boyutu = cikarim_ayari.get("sahi_dilim_boyutu", 640)
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

        if sahi_aktif:
            sonuclar = _sahi_tarama(model, okunan_gorsel, guven_esigi=0.10, iou_esigi=iou_esigi, dilim_boyutu=sahi_dilim_boyutu)
        else:
            sonuclar = model.predict(
                source=okunan_gorsel,
                conf=0.10,
                iou=iou_esigi,
                save=False,
                verbose=False,
                augment=tta_aktif,
            )
    except Exception as hata:
        print(f"{Fore.RED}[-] Cikarim sirasinda hata: {hata}{Style.RESET_ALL}")
        return None

    gecen_sure = time.time() - baslangic_zamani

    if not sonuclar or sonuclar[0].orig_img is None:
        print(f"{Fore.RED}[-] Gorsel islenemedi: {gorsel_yolu}{Style.RESET_ALL}")
        return None

    gorsel = sonuclar[0].orig_img.copy()

    tespit_edilen_hasarlar = []
    sinif_sayaclari = {}

    for sonuc in sonuclar:
        kutucuklar = sonuc.boxes
        if kutucuklar is None:
            continue
        for kutu in kutucuklar:
            x1, y1, x2, y2 = kutu.xyxy[0].cpu().numpy().astype(int)
            sinif_id = int(kutu.cls[0].cpu().numpy())
            guven = float(kutu.conf[0].cpu().numpy())

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
                "kutucuk": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
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
    return {"boxes": [], "masks": []}


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
                    sahi_aktif, sahi_dilim_boyutu, otomatik_yedekleme, ram_optimizasyonu):
    """Tek bir modeli yukler, gorseli tarar, kutulari havuza ekler, modeli bosaltir.

    Returns:
        int: Havuza eklenen kutu sayisi.
    """
    eklenen = 0
    try:
        try:
            model = model_sinifi(str(model_yolu))
            dml = _directml_cihazini_al()
            if dml is not None:
                import torch
                if hasattr(model, 'model') and model.model is not None:
                    model.model.to(dml)
        except RuntimeError as hata:
            if "out of memory" in str(hata).lower() and otomatik_yedekleme:
                print(f"{Fore.YELLOW}[!] VRAM dolu, {kaynak_etiketi} CPU'ya kaydiriliyor...{Style.RESET_ALL}")
                model = model_sinifi(str(model_yolu))
            else:
                raise

        if sahi_aktif:
            sonuclar = _sahi_tarama(model, gorsel, guven_esigi=0.10, iou_esigi=iou_esigi, dilim_boyutu=sahi_dilim_boyutu)
        else:
            sonuclar = model.predict(
                source=gorsel, conf=0.10, iou=iou_esigi, save=False, verbose=False,
            )

        for sonuc in sonuclar:
            if sonuc.boxes is None:
                continue
            for kutu in sonuc.boxes:
                x1, y1, x2, y2 = kutu.xyxy[0].cpu().numpy().astype(int)
                sinif_id = int(kutu.cls[0].cpu().numpy())
                guven = float(kutu.conf[0].cpu().numpy())
                guncel_esik = sinif_guven_esikleri.get(sinif_id, guven_esigi)
                if guven < guncel_esik:
                    continue
                sinif_adi = siniflar.get(sinif_id, f"Sinif_{sinif_id}")
                tespitler_havuzu["boxes"].append({
                    "sinif_id": sinif_id,
                    "sinif_adi": sinif_adi,
                    "guven": round(guven, 4),
                    "kutucuk": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "kaynak_model": kaynak_etiketi,
                })
                eklenen += 1

        del model
        _model_bosalt(ram_optimizasyonu)
    except Exception as hata:
        print(f"{Fore.RED}[-] {kaynak_etiketi} taramasi basarisiz: {hata}{Style.RESET_ALL}")

    return eklenen


def _wbf_sinif_adi_bul(sinif_id, tespitler_havuzu):
    """WBF sonrasi sinif ID'sine karsilik gelen sinif adini havuzdaki kutulardan bulur."""
    for kutu in tespitler_havuzu.get("boxes", []):
        if kutu.get("sinif_id") == sinif_id and "sinif_adi" in kutu:
            return kutu["sinif_adi"]
    return f"Sinif_{sinif_id}"


def _wbf_kutu_birlestir(tespitler_havuzu, gorsel_genisligi, gorsel_yuksekligi, iou_esigi=0.55, guven_esigi=0.25):
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

    kutu_listeleri = []
    skor_listeleri = []
    etiket_listeleri = []

    for kaynak, kutular in model_gruplari.items():
        kutu_listesi = []
        skor_listesi = []
        etiket_listesi = []

        for kutu_bilgisi in kutular:
            koordinat = kutu_bilgisi.get("kutucuk", {})
            x1 = koordinat.get("x1", 0)
            y1 = koordinat.get("y1", 0)
            x2 = koordinat.get("x2", 0)
            y2 = koordinat.get("y2", 0)

            normalize_x1 = max(0.0, min(1.0, x1 / float(gorsel_genisligi)))
            normalize_y1 = max(0.0, min(1.0, y1 / float(gorsel_yuksekligi)))
            normalize_x2 = max(0.0, min(1.0, x2 / float(gorsel_genisligi)))
            normalize_y2 = max(0.0, min(1.0, y2 / float(gorsel_yuksekligi)))

            kutu_listesi.append([normalize_x1, normalize_y1, normalize_x2, normalize_y2])
            skor_listesi.append(float(kutu_bilgisi.get("guven", 0.0)))
            etiket_listesi.append(int(kutu_bilgisi.get("sinif_id", 0)))

        kutu_listeleri.append(kutu_listesi)
        skor_listeleri.append(skor_listesi)
        etiket_listeleri.append(etiket_listesi)

    if not kutu_listeleri:
        return tespitler_havuzu["boxes"]

    birlesmis_kutular, birlesmis_skorlar, birlesmis_etiketler = weighted_boxes_fusion(
        kutu_listeleri,
        skor_listeleri,
        etiket_listeleri,
        iou_thr=iou_esigi,
        skip_box_thr=guven_esigi,
    )

    sonuc_kutular = []
    for i, (kutu, skor, etiket) in enumerate(zip(birlesmis_kutular, birlesmis_skorlar, birlesmis_etiketler)):
        x1 = int(kutu[0] * gorsel_genisligi)
        y1 = int(kutu[1] * gorsel_yuksekligi)
        x2 = int(kutu[2] * gorsel_genisligi)
        y2 = int(kutu[3] * gorsel_yuksekligi)

        sonuc_kutular.append({
            "sinif_id": int(etiket),
            "sinif_adi": _wbf_sinif_adi_bul(int(etiket), tespitler_havuzu),
            "guven": round(float(skor), 4),
            "kutucuk": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "kaynak_model": "wbf",
            "wbf_birlestirildi": True,
        })

    return sonuc_kutular


def _sahi_tarama(model, gorsel, guven_esigi=0.10, iou_esigi=0.7, dilim_boyutu=None):
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        return model.predict(
            source=gorsel,
            conf=guven_esigi,
            iou=iou_esigi,
            save=False,
            verbose=False,
        )

    try:
        detection_model = AutoDetectionModel.from_ultralytics(
            model,
            confidence_threshold=guven_esigi,
        )
        if dilim_boyutu is None:
            yukseklik, genislik = gorsel.shape[:2]
            dilim_boyutu = min(genislik, yukseklik, 640)

        sonuclar = get_sliced_prediction(
            gorsel,
            detection_model,
            slice_height=dilim_boyutu,
            slice_width=dilim_boyutu,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=iou_esigi,
            verbose=0,
        )
        return sonuclar
    except ImportError:
        return model.predict(
            source=gorsel,
            conf=guven_esigi,
            iou=iou_esigi,
            save=False,
            verbose=False,
        )
    except Exception as hata:
        print(f"{Fore.YELLOW}[!] SAHI tarama basarisiz, fallback: {hata}{Style.RESET_ALL}")
        return model.predict(
            source=gorsel,
            conf=guven_esigi,
            iou=iou_esigi,
            save=False,
            verbose=False,
        )


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
    baslangic_zamani = time.time()
    agirliklar = multi_model_ayari.get("agirliklar", {})
    kendi_modellerini_yonet = hazir_modeller is None

    rtdetr_yolu = agirliklar.get("rtdetr", "rtdetr-v2-x.pt")
    rtdetr_model_yolu = PROJE_KOKU / rtdetr_yolu
    if not rtdetr_model_yolu.exists():
        rtdetr_model_yolu = egitilmis_model_yolu_bul()

    if rtdetr_model_yolu is not None:
        from ultralytics import RTDETR
        if kendi_modellerini_yonet:
            rtdetr_eklenen = _tek_model_tara(
                RTDETR, rtdetr_model_yolu, "rt-detr-v2-x", gorsel, tespitler_havuzu,
                guven_esigi, iou_esigi, sinif_guven_esikleri, siniflar,
                sahi_aktif, sahi_dilim_boyutu, otomatik_yedekleme, ram_optimizasyonu,
            )
        else:
            rtdetr_eklenen = _tek_model_tara(
                RTDETR, rtdetr_model_yolu, "rt-detr-v2-x", gorsel, tespitler_havuzu,
                guven_esigi, iou_esigi, sinif_guven_esikleri, siniflar,
                sahi_aktif, sahi_dilim_boyutu, otomatik_yedekleme, ram_optimizasyonu,
            )
        print(f"{Fore.GREEN}[+] RT-DETRv2-X Taramasi... [Bitti] ({rtdetr_eklenen} tespit){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[!] RT-DETR modeli bulunamadi, atlandi.{Style.RESET_ALL}")

    yolo_yolu = agirliklar.get("yolo", "yolov12x.pt")
    yolo_model_yolu = PROJE_KOKU / yolo_yolu
    if not yolo_model_yolu.exists():
        yolo_model_yolu = egitilmis_model_yolu_bul()

    if yolo_model_yolu is not None:
        from ultralytics import YOLO
        yolo_eklenen = _tek_model_tara(
            YOLO, yolo_model_yolu, "yolov12x", gorsel, tespitler_havuzu,
            guven_esigi, iou_esigi, sinif_guven_esikleri, siniflar,
            sahi_aktif, sahi_dilim_boyutu, otomatik_yedekleme, ram_optimizasyonu,
        )
        print(f"{Fore.GREEN}[+] YOLOv12x Taramasi... [Bitti] ({yolo_eklenen} tespit){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[!] YOLO modeli bulunamadi, atlandi.{Style.RESET_ALL}")

    gorsel_yuksekligi, gorsel_genisligi = gorsel.shape[:2]
    birlesmis_kutular = _wbf_kutu_birlestir(tespitler_havuzu, gorsel_genisligi, gorsel_yuksekligi, iou_esigi=wbf_iou_esigi, guven_esigi=guven_esigi)
    tespitler_havuzu["boxes"] = birlesmis_kutular

    sam_yolu = agirliklar.get("sam", "sam2_s.pt")
    sam_model_yolu = PROJE_KOKU / sam_yolu

    try:
        from ultralytics import SAM
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

        del sam_model
        _model_bosalt(ram_optimizasyonu)
    except ImportError:
        print(f"{Fore.YELLOW}[!] SAM 2 kutuphanesi yuklu degil, maskeleme atlandi.{Style.RESET_ALL}")
    except Exception as hata:
        print(f"{Fore.RED}[-] SAM 2 maskelemesi basarisiz: {hata}{Style.RESET_ALL}")

    try:
        from src.inspector_florence import denetle as florence_denetle
        tespitler_havuzu = florence_denetle(tespitler_havuzu, gorsel, yapilandirma=yapilandirma)
    except Exception as hata:
        print(f"{Fore.RED}[-] Florence-2 denetimi basarisiz: {hata}{Style.RESET_ALL}")

    gecen_sure = time.time() - baslangic_zamani

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
        "toplam_tespit": len(dogrulanmis_tespitler),
        "toplam_maske": len(tespitler_havuzu.get("masks", [])),
        "hasar_dagilimi": sinif_sayaclari,
        "tespitler": dogrulanmis_tespitler,
        "maskeler": tespitler_havuzu.get("masks", []),
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

                        _tek_model_tara(
                            RTDETR, rtdetr_model_yolu, "rt-detr-v2-x", gorsel, chunk_havuzlari[gorsel_yolu],
                            yapilandirma.get("multi_model", {}).get("guven_esigi", 0.25),
                            yapilandirma.get("cikarim", {}).get("iou_esigi", 0.7),
                            yapilandirma.get("cikarim", {}).get("sinif_guven_esikleri", {}),
                            yapilandirma.get("siniflar", {}),
                            yapilandirma.get("cikarim", {}).get("sahi_aktif", False),
                            yapilandirma.get("cikarim", {}).get("sahi_dilim_boyutu", 640),
                            yapilandirma.get("multi_model", {}).get("otomatik_yedekleme_cpu", True),
                            yapilandirma.get("multi_model", {}).get("ram_optimizasyonu", True),
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

                        _tek_model_tara(
                            YOLO, yolo_model_yolu, "yolov12x", gorsel, chunk_havuzlari[gorsel_yolu],
                            yapilandirma.get("multi_model", {}).get("guven_esigi", 0.25),
                            yapilandirma.get("cikarim", {}).get("iou_esigi", 0.7),
                            yapilandirma.get("cikarim", {}).get("sinif_guven_esikleri", {}),
                            yapilandirma.get("siniflar", {}),
                            yapilandirma.get("cikarim", {}).get("sahi_aktif", False),
                            yapilandirma.get("cikarim", {}).get("sahi_dilim_boyutu", 640),
                            yapilandirma.get("multi_model", {}).get("otomatik_yedekleme_cpu", True),
                            yapilandirma.get("multi_model", {}).get("ram_optimizasyonu", True),
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
