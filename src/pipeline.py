import sys
import json
import yaml
import cv2
import time
from pathlib import Path
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent.parent
YAPILANDIRMA_YOLU = PROJE_KOKU / "config.yaml"
EGITIM_KOKU = PROJE_KOKU / "runs" / "train"
CIKARIM_KOKU = PROJE_KOKU / "runs" / "predict"

SINIF_RENKLERI = {
    0: (0, 0, 255),
    1: (0, 165, 255),
    2: (0, 255, 255),
    3: (0, 128, 0),
    4: (255, 0, 255),
}


def yapilandirma_yukle():
    with open(YAPILANDIRMA_YOLU, "r", encoding="utf-8") as dosya:
        return yaml.safe_load(dosya)


def yapilandirma_kaydet(yapilandirma):
    with open(YAPILANDIRMA_YOLU, "w", encoding="utf-8") as dosya:
        yaml.safe_dump(yapilandirma, dosya, sort_keys=False, default_flow_style=False, allow_unicode=True)


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
    from ultralytics import YOLO

    if yapilandirma is None:
        yapilandirma = yapilandirma_yukle()
    cikarim_ayari = yapilandirma.get("cikarim", {})
    siniflar = yapilandirma.get("siniflar", {})

    guven_esigi = cikarim_ayari.get("guven_eşigi", 0.25)
    iou_esigi = cikarim_ayari.get("iou_esigi", 0.7)
    if cikti_klasoru is None:
        cikti_klasoru = cikarim_ayari.get("cikti_klasoru", "runs/predict")
    cikti_klasoru = Path(cikti_klasoru)
    gorsel_kaydet = cikarim_ayari.get("gorsel_kaydet", True)
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
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print()

        try:
            model = YOLO(str(model_yolu))
        except Exception as hata:
            print(f"{Fore.RED}[-] Model yuklenemedi: {hata}{Style.RESET_ALL}")
            return None
    else:
        model_yolu = getattr(model, "model_path", str(egitilmis_model_yolu_bul()))
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  HADES DETECTOR - Hasar Tespiti{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Cikarim Yapilandirmasi{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Gorsel          : {gorsel_yolu}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Model           : {model_yolu}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Guven Esigi     : {guven_esigi}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}IOU Esigi       : {iou_esigi}{Style.RESET_ALL}")
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

        sonuclar = model.predict(
            source=okunan_gorsel,
            conf=guven_esigi,
            iou=iou_esigi,
            save=False,
            verbose=False,
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
    from ultralytics import YOLO

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
        model = YOLO(str(model_yolu))
    except Exception as hata:
        print(f"{Fore.RED}[-] Model yuklenemedi: {hata}{Style.RESET_ALL}")
        return None

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        hasar_tespiti_yap(sys.argv[1])
    else:
        print(f"{Fore.YELLOW}Kullanim: python pipeline.py <gorsel_yolu>{Style.RESET_ALL}")