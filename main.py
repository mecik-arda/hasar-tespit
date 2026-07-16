import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent
sys.path.insert(0, str(PROJE_KOKU))

SECILI_CIHAZ = None
SECILI_CIHAZ_CIKARIM = None

HADES_LOGO = r"""
  _   _    _    ____  _____ ____    ____   ____    _    _   _ _   _ _____ ____
 | | | |  / \  |  _ \| ____/ ___|  / ___| / ___|  / \  | \ | | \ | | ____|  _ \
 | |_| | / _ \ | | | |  _| \___ \  \___ \| |     / _ \ |  \| |  \| |  _| | |_) |
 |  _  |/ ___ \| |_| | |___ ___) |  ___) | |___ / ___ \| |\  | |\  | |___|  _ <
 |_| |_/_/   \_\____/|_____|____/  |____/ \____/_/   \_\_| \_|_| \_|_____|_| \_\
"""


def ekrani_temizle():
    """Ekrani ANSI escape kodlari ile temizler (cross-platform, shell injection riski yok)."""
    print("\033[2J\033[H", end="")


YARDIM_METINLERI = {
    "ana_menu": """
  {c}ANA MENU YARDIM{rs}
  {c}========================================{rs}
  Secim yapmak icin 0-16 arasi bir rakam girin.

  {w}1-4{rs}   Veri hazirlama (donanim, etiket, artirim, bolme)
  {w}5-8{rs}   Model egitimi, hasar tespiti, rapor, testler
  {w}9-11{rs}  Model secimi, orkestrasyon, cikarim profili
  {w}12-15{rs} Gorsel toplama, kalite kontrol, etiket dogrulama, model bilgisi
  {w}16{rs}    Akıllı Yönlendirici (Gateway) testi
  {w}0{rs}     Cikis
""",
    "egitim": """
  {c}EGITIM YARDIM{rs}
  {c}========================================{rs}
  {w}Epoch{rs}      : Veri setinin kac kez tekrarlanacagi (10-100 arasi)
  {w}Batch Size{rs} : Tek seferde islenecek gorsel sayisi (bellege gore)
  {w}Img Size{rs}   : Gorsel cozunurlugu (320 hizli, 640 kaliteli)
  {w}Cihaz{rs}      : cpu / cuda / colab
  {w}Focal Loss{rs} : Zor ornekleri agirliklandirma (0.0 kapali, 1.5 orta)

  Bos birakmak config.yaml'daki varsayilani kullanir.
""",
    "cikarim": """
  {c}HASAR TESPITI YARDIM{rs}
  {c}========================================{rs}
  {w}Tekli Gorsel{rs} : Belirli bir dosyayi tarar. 'rastgele' yazinca ornek secer.
  {w}Toplu Tarama{rs} : Klasordeki tum gorselleri sirayla tarar, genel rapor cikarir.
  {w}TTA{rs}          : Test Time Augmentation. Gorseli farkli acilarda tekrar tekrar
              tarayip tahminleri birlestirir. Daha dogru ama daha yavas.
""",
    "cihaz_secimi": """
  {c}CIHAZ SECIMI YARDIM{rs}
  {c}========================================{rs}
  {w}GPU{rs}  : En hizli secenek. NVIDIA CUDA veya Intel Arc / AMD GPU.
  {w}CPU{rs}  : Her zaman calisir, GPU yoksa tek secenek.
  {w}NPU{rs}  : Sadece cikarim (inference) icindir, egitimde kullanilmaz.
  {w}Colab{rs}: Google'in ucretsiz GPU'su. Internet gerektirir.
""",
    "genel": """
  {c}HADES YARDIM{rs}
  {c}========================================{rs}
  Herhangi bir giris ekraninda {w}/yardim{rs} yazarak bu ekrani gorebilirsiniz.

  {w}Is Akisi (Tek Model){rs}   : 1 > 9 > 2 > 3 > 4 > 5 > 6
  {w}Is Akisi (Coklu Model){rs} : 1 > 9 > 10 > 11 > 2 > 3 > 4 > 5 > 6
  {w}Klasor{rs}     : hasar-ornek/ (gorseller), data/ (veri seti)
  {w}Cikti{rs}      : hasar-sonucu/ (tespit sonuclari)
  {w}Model{rs}      : runs/train/hades_egitim/weights/best.pt
  {w}Config{rs}     : config.yaml (tum ayarlar)
""",
}


def yardimli_input(istem, yardim_anahtari="genel"):
    while True:
        girdi = input(istem).strip()
        if girdi.lower() in ("/yardim", "/help", "/h", "/?"):
            ekrani_temizle()
            metin = YARDIM_METINLERI.get(yardim_anahtari, YARDIM_METINLERI["genel"])
            print(metin.format(c=Fore.CYAN, w=Fore.WHITE, rs=Style.RESET_ALL, y=Fore.YELLOW))
            continue
        return girdi


def _profil_adi_bul(config):
    multi = config.get("multi_model", {})
    if not multi.get("aktif", False):
        return "Tek Model"
    siralama = multi.get("siralama", [])
    if set(siralama) == {"rt-detr-v2-x", "yolov12x", "sam2_small", "florence-2"}:
        return "Kusursuz (4 Model)"
    elif set(siralama) == {"rt-detr-v2-x", "yolov12x"}:
        return "Hibrit (RT-DETR + YOLO)"
    elif siralama == ["rt-detr-v2-x"]:
        return "Hiz (RT-DETR)"
    else:
        return f"Ozel ({'+'.join(siralama)})"


def basligi_yazdir():
    from src.utils import yapilandirma_yukle
    ekrani_temizle()
    try:
        config = yapilandirma_yukle()
        multi_model_aktif = config.get("multi_model", {}).get("aktif", False)
        if multi_model_aktif:
            model_tur = "Coklu-Model (Orkestrasyon)"
        else:
            model_tur = config.get("model", {}).get("tur", "rtdetr").upper()
        profil = _profil_adi_bul(config)
        model_agirlik = config.get("model", {}).get("agirlik", "?")
    except Exception as e:
        model_tur = "BILINMIYOR"
        profil = "BILINMIYOR"
        model_agirlik = "?"
        print(f"{Fore.YELLOW}[!] Config okunamadi: {e}{Style.RESET_ALL}")

    print(f"{Fore.CYAN}{HADES_LOGO}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  [HADES Hasar Tespiti Sistemi v2.0 - Multi-Model Edition]{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}  Model: {Fore.GREEN}{model_tur} ({model_agirlik}){Style.RESET_ALL}  |  {Fore.WHITE}Profil: {Fore.GREEN}{profil}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()


def menuyu_yazdir():
    print(f"{Fore.YELLOW}  ᛟ [ DONANIM VE EGITIM YAPILANDIRMASI ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[1] {Fore.YELLOW}Donanim Kontrolu (CPU/GPU/NPU Secimi){Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[9] {Fore.YELLOW}Egitilecek Ana Model Secimi (YOLO/RT-DETR + Kapasite){Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}  ᛉ [ VERI HAZIRLIGI VE KALITE ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[2] {Fore.YELLOW}Veri Etiketleme (LabelImg){Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[3] {Fore.YELLOW}Veri Artirimi (Augmentation){Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[4] {Fore.YELLOW}Veri Bolme (Train/Val Split){Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}  ᛏ [ EGITIM VE TEST SURECLERI ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[5] {Fore.YELLOW}Model Egitimini Baslat{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}  ᛤ [ CIKARIM VE COKLU-MODEL ORKESTRASYONU ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[6] {Fore.YELLOW}Hasar Tespiti Yap (Tekil/Toplu Islem){Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}  ᛟ [ RAPORLAMA VE TEST ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[7] {Fore.YELLOW}Egitim Performans Raporu{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[8] {Fore.YELLOW}Sistem Testlerini Calistir (Birim Testler){Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}  ᛉ [ ORKESTRASYON VE PROFIL ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[10] {Fore.YELLOW}Orkestrasyon Yoneticisi (Agirlik ve Model Hub){Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[11] {Fore.YELLOW}Cikarim Profili Secimi (Hizli/Hibrit/Kusursuz/Ozel){Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}  ᛏ [ VERI TOPLAMA VE KONTROL ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[12] {Fore.YELLOW}Gorsel Toplama (Otomatik Veri Indirme){Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[13] {Fore.YELLOW}Veri Kalite Kontrolu{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[14] {Fore.YELLOW}Etiket Dogrulama{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}  ᛤ [ BILGI ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[15] {Fore.YELLOW}Model Bilgileri (Agirlik ve Metrikler){Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}  ᛟ [ AKILLI YONLENDIRICI ]{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[16] {Fore.YELLOW}Akıllı Yönlendirici (Gateway) Testi{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}[0] {Fore.RED}Cikis{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


def donanim_kontrolu_calistir():
    global SECILI_CIHAZ, SECILI_CIHAZ_CIKARIM
    from src.hardware_check import donanim_ozeti_yazdir, cihaz_secimi_yap
    print()
    profil = donanim_ozeti_yazdir()
    print()
    SECILI_CIHAZ = cihaz_secimi_yap(profil, mod="egitim")
    print()
    SECILI_CIHAZ_CIKARIM = cihaz_secimi_yap(profil, mod="cikarim")
    print()


def etiketleme_calistir():
    from src.data_tools import etiketleme_baslat
    print()
    etiketleme_baslat()
    print()


def augmentation_calistir():
    from src.data_tools import augmentation_uygula
    print()
    augmentation_uygula()
    print()


def veri_bolme_calistir():
    from src.data_tools import veri_bol
    print()
    veri_bol()
    print()


def egitim_calistir():
    global SECILI_CIHAZ
    from src.train import egitim_baslat
    print()
    print(f"{Fore.YELLOW}[*] Egitim parametreleri (Bos birakirsiniz varsayilan kullanilir):{Style.RESET_ALL}")
    print()

    varsayilan_cihaz = "auto"
    if SECILI_CIHAZ is not None:
        varsayilan_cihaz = SECILI_CIHAZ.get("cihaz", "auto")
        print(f"{Fore.GREEN}[+] Onceden secilen cihaz: {SECILI_CIHAZ.get('aciklama', varsayilan_cihaz)}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[+] Onceden secilen batch: {SECILI_CIHAZ.get('batch', 'auto')}{Style.RESET_ALL}")
        print()

    epoch_girdi = yardimli_input(f"{Fore.CYAN}    Epoch sayisi [Enter=varsayilan]: {Style.RESET_ALL}", "egitim")
    epoch_sayisi = None
    if epoch_girdi:
        try:
            epoch_sayisi = int(epoch_girdi)
        except ValueError:
            print(f"{Fore.RED}[-] Gecersiz epoch sayisi. Varsayilan kullanilacak.{Style.RESET_ALL}")

    batch_girdi = yardimli_input(f"{Fore.CYAN}    Batch size [Enter=varsayilan]: {Style.RESET_ALL}", "egitim")
    batch_size = None
    if batch_girdi:
        try:
            batch_size = int(batch_girdi)
        except ValueError:
            if batch_girdi.lower() == "auto":
                batch_size = "auto"
            else:
                print(f"{Fore.RED}[-] Gecersiz batch size. Varsayilan kullanilacak.{Style.RESET_ALL}")

    img_girdi = input(f"{Fore.CYAN}    Img size [Enter=varsayilan]: {Style.RESET_ALL}").strip()
    img_size = None
    if img_girdi:
        try:
            img_size = int(img_girdi)
        except ValueError:
            print(f"{Fore.RED}[-] Gecersiz img size. Varsayilan kullanilacak.{Style.RESET_ALL}")

    cihaz_girdi = input(f"{Fore.CYAN}    Cihaz [Enter={varsayilan_cihaz}]: {Style.RESET_ALL}").strip()
    cihaz = None
    if cihaz_girdi:
        cihaz = cihaz_girdi
    elif SECILI_CIHAZ is not None:
        cihaz = SECILI_CIHAZ.get("cihaz", None)
        if batch_size is None:
            batch_size = SECILI_CIHAZ.get("batch", None)

    fl_girdi = input(f"{Fore.CYAN}    Focal Loss gucu [0.0-2.0, Enter=1.5]: {Style.RESET_ALL}").strip()
    fl_gamma = None
    if fl_girdi:
        try:
            fl_gamma = float(fl_girdi)
        except ValueError:
            print(f"{Fore.RED}[-] Gecersiz fl_gamma. Varsayilan kullanilacak.{Style.RESET_ALL}")

    print()
    egitim_baslat(epoch_sayisi=epoch_sayisi, batch_size=batch_size, cihaz=cihaz, img_size=img_size, fl_gamma=fl_gamma)
    print()


def cikarim_calistir():
    global SECILI_CIHAZ_CIKARIM
    import random
    from src.pipeline import (
        hasar_tespiti_yap,
        toplu_hasar_tespiti_yap,
        coklu_model_hasar_tespiti_yap,
        coklu_model_toplu_tespiti_yap,
    )
    from src.utils import yapilandirma_yukle, yapilandirma_kaydet

    cikti_klasoru = PROJE_KOKU / "hasar-sonucu"
    ornek_klasoru = PROJE_KOKU / "hasar-ornek"
    gorsel_uzantilari = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if SECILI_CIHAZ_CIKARIM is not None:
        print(f"{Fore.GREEN}[+] Cikarim cihazi: {SECILI_CIHAZ_CIKARIM.get('aciklama', 'secili degil')}{Style.RESET_ALL}")
        print()

    config = yapilandirma_yukle()
    mevcut_tta = config.get("cikarim", {}).get("tta_aktif", False)
    print(f"{Fore.YELLOW}[*] TTA (Test Time Augmentation): {Fore.GREEN}AKTIF{Style.RESET_ALL}" if mevcut_tta else f"{Fore.YELLOW}[*] TTA (Test Time Augmentation): {Fore.RED}KAPALI{Style.RESET_ALL}")
    tta_girdi = yardimli_input(f"{Fore.CYAN}    TTA aktif edilsin mi? (E/h) [Enter=hayir]: {Style.RESET_ALL}", "cikarim").lower()
    if tta_girdi in ("e", "evet", "y", "yes", "1"):
        config["cikarim"]["tta_aktif"] = True
        yapilandirma_kaydet(config)
        print(f"{Fore.GREEN}[+] TTA aktif edildi.{Style.RESET_ALL}")
    elif tta_girdi in ("h", "hayir", "n", "no", "0"):
        config["cikarim"]["tta_aktif"] = False
        yapilandirma_kaydet(config)
        print(f"{Fore.YELLOW}[+] TTA kapatildi.{Style.RESET_ALL}")
    print()

    multi_model_aktif = config.get("multi_model", {}).get("aktif", False)
    profil_adi = _profil_adi_bul(config)
    if multi_model_aktif:
        print(f"{Fore.YELLOW}[*] Cikarim Profili: {Fore.GREEN}{profil_adi}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[*] Cikarim Profili: {Fore.YELLOW}{profil_adi}{Style.RESET_ALL}")
    print()

    gateway_aktif = False
    gateway_girdi = yardimli_input(
        f"{Fore.CYAN}    Yapay Zeka Agi (CLIP Router) kullanilsin mi? (E/h) [Enter=hayir]: {Style.RESET_ALL}",
        "cikarim",
    ).lower()
    if gateway_girdi in ("e", "evet", "y", "yes", "1"):
        gateway_aktif = True
        print(f"{Fore.GREEN}[+] Akıllı Yönlendirici (CLIP Router) aktif.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}    Görseller önce çöp filtresinden geçecek, sonra kanala yönlendirilecek.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[+] Akıllı Yönlendirici kapali. Standart akış kullanilacak.{Style.RESET_ALL}")
    print()

    print()
    print(f"{Fore.YELLOW}  [HASAR TESPITI MENU]{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}[1] {Fore.YELLOW}Tek Gorsel Isle{Style.RESET_ALL}")
    print(f"      Belirli bir gorselde hasar tespiti yapar.")
    print()
    print(f"  {Fore.WHITE}[2] {Fore.YELLOW}Klasorden Toplu Isle (Batch Processing){Style.RESET_ALL}")
    print(f"      Hasar-ornek klasorunden birden fazla gorseli tarar.")
    print()
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")

    mevcut_gorseller = sorted([f for f in ornek_klasoru.iterdir() if f.suffix.lower() in gorsel_uzantilari]) if ornek_klasoru.exists() else []
    secim = input(f"{Fore.CYAN}  Seciminiz [1-2]: {Style.RESET_ALL}").strip()

    if secim == "1":
        gorsel_yolu = input(f"{Fore.CYAN}    Gorsel yolu (veya 'rastgele'): {Style.RESET_ALL}").strip()

        if not gorsel_yolu:
            print(f"{Fore.RED}[-] Gorsel yolu bos birakilamaz.{Style.RESET_ALL}")
            print()
            return

        if gorsel_yolu.lower() == "rastgele":
            if not mevcut_gorseller:
                print(f"{Fore.RED}[-] Hasar-ornek klasorunde gorsel bulunamadi.{Style.RESET_ALL}")
                print()
                return

            secilen_gorsel = random.choice(mevcut_gorseller)
            gorsel_yolu = str(secilen_gorsel)
            print(f"{Fore.GREEN}[+] Rastgele secilen gorsel: {gorsel_yolu}{Style.RESET_ALL}")
        else:
            gorsel_yolu = gorsel_yolu.strip('"\'')

            try:
                if not os.path.isabs(gorsel_yolu):
                    cozulmus_yol = (PROJE_KOKU / gorsel_yolu).resolve()
                else:
                    cozulmus_yol = Path(gorsel_yolu).resolve()

                if not cozulmus_yol.exists():
                    print(f"{Fore.RED}[-] Dosya bulunamadi: {cozulmus_yol}{Style.RESET_ALL}")
                    print()
                    return

                gorsel_yolu = str(cozulmus_yol)
            except Exception as e:
                print(f"{Fore.RED}[-] Gecersiz dosya yolu: {e}{Style.RESET_ALL}")
                print()
                return

        if gateway_aktif:
            from src.gateway.ai_router import AIRouter
            router = AIRouter()
            print(f"{Fore.CYAN}[*] Akıllı Yönlendirici (CLIP Router) görseli analiz ediyor...{Style.RESET_ALL}")
            router_sonucu = router.process_image(gorsel_yolu)
            print()

            if router_sonucu["status"] == "rejected":
                print(f"{Fore.RED}[✗] Görsel reddedildi: {router_sonucu['sebep']}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}    Hasar tespiti iptal edildi.{Style.RESET_ALL}")
                print()
                return

            if router_sonucu["route_to"] == "YOLO":
                print(f"{Fore.BLUE}[→] YOLO (Hızlı Çözüm Kanalı) seçildi.{Style.RESET_ALL}")
                hasar_tespiti_yap(gorsel_yolu, cikti_klasoru=str(cikti_klasoru))
            else:
                print(f"{Fore.YELLOW}[→] RT-DETR (Kompleks Hasar Kanalı) seçildi.{Style.RESET_ALL}")
                coklu_model_hasar_tespiti_yap(gorsel_yolu, cikti_klasoru=str(cikti_klasoru))
        elif multi_model_aktif:
            coklu_model_hasar_tespiti_yap(gorsel_yolu, cikti_klasoru=str(cikti_klasoru))
        else:
            hasar_tespiti_yap(gorsel_yolu, cikti_klasoru=str(cikti_klasoru))
        print()

    elif secim == "2":
        mevcut_adet = len(mevcut_gorseller)

        print(f"    {Fore.WHITE}Hasar-ornek klasorunde toplam {mevcut_adet} gorsel bulundu.{Style.RESET_ALL}")

        adet_girdi = input(f"{Fore.CYAN}    Taranacak gorsel adedi: {Style.RESET_ALL}").strip()

        try:
            istenen_adet = int(adet_girdi)
        except ValueError:
            print(f"{Fore.RED}[-] Gecersiz adet. Lutfen sayi girin.{Style.RESET_ALL}")
            print()
            return

        if istenen_adet <= 0:
            print(f"{Fore.RED}[-] Adet 0'dan buyuk olmalidir.{Style.RESET_ALL}")
            print()
            return

        if istenen_adet > mevcut_adet:
            print(f"{Fore.YELLOW}[!] UYARI: Istenen adet ({istenen_adet}) mevcut gorsel sayisini ({mevcut_adet}) asiyor!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}    Isleme {mevcut_adet} gorsel ile devam edilecek.{Style.RESET_ALL}")
            istenen_adet = mevcut_adet

        if istenen_adet == 0:
            print(f"{Fore.RED}[-] Taranacak gorsel yok.{Style.RESET_ALL}")
            print()
            return

        if multi_model_aktif:
            coklu_model_toplu_tespiti_yap(
                girdi_klasoru=str(ornek_klasoru),
                cikti_klasoru=str(cikti_klasoru),
                miktar=istenen_adet,
            )
        else:
            toplu_hasar_tespiti_yap(
                girdi_klasoru=str(ornek_klasoru),
                cikti_klasoru=str(cikti_klasoru),
                miktar=istenen_adet,
            )
        print()

    else:
        print(f"{Fore.RED}[-] Gecersiz secim! Lutfen 1 veya 2 girin.{Style.RESET_ALL}")
        print()


def rapor_calistir():
    from src.train import egitim_raporu_goster
    print()
    egitim_raporu_goster()
    print()


def egitim_modeli_secimi_calistir():
    from src.utils import yapilandirma_yukle, yapilandirma_kaydet
    config = yapilandirma_yukle()
    mevcut_tur = config.get("model", {}).get("tur", "yolo")
    mevcut_agirlik = config.get("model", {}).get("agirlik", "yok")

    print()
    print(f"{Fore.YELLOW}  [EGITILECEK ANA MODEL SECIMI]{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}Mevcut: {Fore.GREEN}{mevcut_tur.upper()} ({mevcut_agirlik}){Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}[1] {Fore.YELLOW}YOLO{Style.RESET_ALL}")
    print(f"      Klasik tek asamali nesne tespit modeli. Hizli ve verimli.")
    print(f"      Mevcut surumler: YOLOv8, YOLOv12 (nano'dan x-large'a)")
    print()
    print(f"  {Fore.WHITE}[2] {Fore.CYAN}RT-DETR (Real-Time DEtection TRansformer){Style.RESET_ALL}")
    print(f"      Transformer tabanli ilk gercek zamanli nesne tespit modeli.")
    print(f"      NMS adimina ihtiyac duymaz, daha yuksek mAP sunar.")
    print()
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")

    secim = input(f"{Fore.CYAN}  Model seciminiz [1-2, Enter=iptal]: {Style.RESET_ALL}").strip()

    if secim == "1":
        yeni_tur = "yolo"
    elif secim == "2":
        yeni_tur = "rtdetr"
    else:
        print(f"{Fore.RED}[-] Secim iptal edildi.{Style.RESET_ALL}")
        print()
        return

    print()
    if yeni_tur == "rtdetr":
        print(f"{Fore.YELLOW}  [RT-DETR KAPASITE SECIMI]{Style.RESET_ALL}")
        print()
        print(f"  {Fore.WHITE}[1] {Fore.YELLOW}Large  (l) - Dengeli, yuksek mAP{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}[2] {Fore.YELLOW}X-Large (x) - En akilli, en yavas{Style.RESET_ALL}")
        print()
        boyut_secim = input(f"{Fore.CYAN}  Kapasite secin [1-2]: {Style.RESET_ALL}").strip()
        boyutlar = {"1": "l", "2": "x"}
        if boyut_secim not in boyutlar:
            print(f"{Fore.RED}[-] Gecersiz secim. Iptal edildi.{Style.RESET_ALL}")
            print()
            return
        yeni_agirlik = f"rtdetr-{boyutlar[boyut_secim]}.pt"
    else:
        print(f"{Fore.YELLOW}  [YOLO NESIL SECIMI]{Style.RESET_ALL}")
        print()
        print(f"{Fore.CYAN}  1) YOLOv8{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  2) YOLOv12{Style.RESET_ALL}")
        print()
        surum_secim = input(f"{Fore.CYAN}  Model neslini secin [1-2]: {Style.RESET_ALL}").strip()
        if surum_secim == "1":
            on_ek = "yolov8"
        elif surum_secim == "2":
            on_ek = "yolo12"
        else:
            print(f"{Fore.RED}[-] Gecersiz secim. Iptal edildi.{Style.RESET_ALL}")
            print()
            return

        print()
        print(f"{Fore.YELLOW}  [YOLO KAPASITE SECIMI]{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  1) Nano    (n) - En hizli, en dusuk mAP{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  2) Small   (s) - Hizli, idare eder mAP{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  3) Medium  (m) - Dengeli{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  4) Large   (l) - Yavas, yuksek mAP{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  5) X-Large (x) - En yavas, en akilli{Style.RESET_ALL}")
        print()
        boyut_secim = input(f"{Fore.CYAN}  Kapasite secin [1-5]: {Style.RESET_ALL}").strip()
        boyutlar = {"1": "n", "2": "s", "3": "m", "4": "l", "5": "x"}
        if boyut_secim not in boyutlar:
            print(f"{Fore.RED}[-] Gecersiz secim. Iptal edildi.{Style.RESET_ALL}")
            print()
            return
        yeni_agirlik = f"{on_ek}{boyutlar[boyut_secim]}.pt"

    if "model" not in config:
        config["model"] = {}

    eski_tur = config["model"].get("tur", "")
    eski_agirlik = config["model"].get("agirlik", "")
    config["model"]["tur"] = yeni_tur
    config["model"]["agirlik"] = yeni_agirlik
    yapilandirma_kaydet(config)

    print()
    if eski_tur != yeni_tur:
        print(f"{Fore.GREEN}[+] Model turu: {eski_tur.upper()} -> {yeni_tur.upper()}{Style.RESET_ALL}")
    if eski_agirlik != yeni_agirlik:
        print(f"{Fore.GREEN}[+] Model agirligi: {eski_agirlik} -> {yeni_agirlik}{Style.RESET_ALL}")
    if eski_tur == yeni_tur and eski_agirlik == yeni_agirlik:
        print(f"{Fore.YELLOW}[!] Model zaten {yeni_tur.upper()} ({yeni_agirlik}) olarak ayarli.{Style.RESET_ALL}")
    print()


def orkestrasyon_yoneticisi_calistir():
    from src.utils import yapilandirma_yukle, yapilandirma_kaydet
    config = yapilandirma_yukle()

    if "multi_model" not in config:
        config["multi_model"] = {}
    multi = config["multi_model"]
    if "agirliklar" not in multi:
        multi["agirliklar"] = {}
    if "denetleyici_ayarlari" not in multi:
        multi["denetleyici_ayarlari"] = {}

    agirliklar = multi["agirliklar"]
    denetleyici = multi["denetleyici_ayarlari"]

    print()
    print(f"{Fore.YELLOW}  [ORKESTRASYON YONETICISI - Model Hub]{Style.RESET_ALL}")
    print()
    print(f"  {Fore.CYAN}Mevcut Agirlik Konfigurasyonu:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[1] YOLO Agirligi      : {Fore.GREEN}{agirliklar.get('yolo', 'tanimsiz')}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[2] RT-DETR Agirligi    : {Fore.GREEN}{agirliklar.get('rtdetr', 'tanimsiz')}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[3] SAM Versiyonu      : {Fore.GREEN}{agirliklar.get('sam', 'tanimsiz')}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[4] Florence-2 Modeli  : {Fore.GREEN}{denetleyici.get('model', 'tanimsiz')}{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}[5] {Fore.YELLOW}On Tanimli Profillere Don{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[0] {Fore.RED}Ana Menuye Don{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")

    secim = input(f"{Fore.CYAN}  Duzenlenecek bilesen [1-5, 0=iptal]: {Style.RESET_ALL}").strip()

    if secim == "1":
        mevcut = agirliklar.get("yolo", "yolov12x.pt")
        yeni = input(f"{Fore.CYAN}    YOLO agirlik dosyasi [Enter={mevcut}]: {Style.RESET_ALL}").strip()
        if yeni:
            agirliklar["yolo"] = yeni
            yapilandirma_kaydet(config)
            print(f"{Fore.GREEN}[+] YOLO agirligi guncellendi: {mevcut} -> {yeni}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}[!] Degisiklik yapilmadi.{Style.RESET_ALL}")
    elif secim == "2":
        mevcut = agirliklar.get("rtdetr", "rtdetr-v2-x.pt")
        yeni = input(f"{Fore.CYAN}    RT-DETR agirlik dosyasi [Enter={mevcut}]: {Style.RESET_ALL}").strip()
        if yeni:
            agirliklar["rtdetr"] = yeni
            yapilandirma_kaydet(config)
            print(f"{Fore.GREEN}[+] RT-DETR agirligi guncellendi: {mevcut} -> {yeni}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}[!] Degisiklik yapilmadi.{Style.RESET_ALL}")
    elif secim == "3":
        mevcut = agirliklar.get("sam", "sam2_s.pt")
        print(f"{Fore.CYAN}    SAM surumleri: sam2_s.pt (small), sam2_b.pt (base), sam2_l.pt (large){Style.RESET_ALL}")
        yeni = input(f"{Fore.CYAN}    SAM agirlik dosyasi [Enter={mevcut}]: {Style.RESET_ALL}").strip()
        if yeni:
            agirliklar["sam"] = yeni
            yapilandirma_kaydet(config)
            print(f"{Fore.GREEN}[+] SAM agirligi guncellendi: {mevcut} -> {yeni}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}[!] Degisiklik yapilmadi.{Style.RESET_ALL}")
    elif secim == "4":
        mevcut = denetleyici.get("model", "microsoft/Florence-2-base")
        print(f"{Fore.CYAN}    Florence-2 surumleri: microsoft/Florence-2-base, microsoft/Florence-2-large{Style.RESET_ALL}")
        yeni = input(f"{Fore.CYAN}    Florence-2 model adi [Enter={mevcut}]: {Style.RESET_ALL}").strip()
        if yeni:
            denetleyici["model"] = yeni
            yapilandirma_kaydet(config)
            print(f"{Fore.GREEN}[+] Florence-2 modeli guncellendi: {mevcut} -> {yeni}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}[!] Degisiklik yapilmadi.{Style.RESET_ALL}")
    elif secim == "5":
        agirliklar["yolo"] = "yolov12x.pt"
        agirliklar["rtdetr"] = "rtdetr-v2-x.pt"
        agirliklar["sam"] = "sam2_s.pt"
        denetleyici["model"] = "microsoft/Florence-2-base"
        multi["siralama"] = ["rt-detr-v2-x", "yolov12x", "sam2_small", "florence-2"]
        multi["aktif"] = True
        yapilandirma_kaydet(config)
        print(f"{Fore.GREEN}[+] Tum agirliklar ve profil zinciri (Kusursuz) varsayilana donduruldu.{Style.RESET_ALL}")
    elif secim == "0":
        print(f"{Fore.YELLOW}[!] Islem iptal edildi.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[-] Gecersiz secim.{Style.RESET_ALL}")

    print()


def cikarim_profili_secimi_calistir():
    from src.utils import yapilandirma_yukle, yapilandirma_kaydet
    config = yapilandirma_yukle()

    if "multi_model" not in config:
        config["multi_model"] = {}

    multi = config["multi_model"]
    mevcut_siralama = multi.get("siralama", [])
    mevcut_profil = _profil_adi_bul(config)

    GECERLI_MODELLER = {"rt-detr-v2-x", "yolov12x", "sam2_small", "florence-2"}

    print()
    print(f"{Fore.YELLOW}  [CIKARIM PROFILI SECIMI]{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}Mevcut Profil: {Fore.GREEN}{mevcut_profil}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Mevcut Zincir: {Fore.CYAN}{' -> '.join(mevcut_siralama) if mevcut_siralama else '(tek model)'}{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}[1] {Fore.GREEN}Hiz Profili{Style.RESET_ALL}")
    print(f"      Sadece RT-DETRv2-X calisir. En hizli, en dusuk VRAM.")
    print()
    print(f"  {Fore.WHITE}[2] {Fore.YELLOW}Hibrit Profil{Style.RESET_ALL}")
    print(f"      RT-DETRv2-X + YOLOv12x + WBF birlestirme. Dengeli.")
    print()
    print(f"  {Fore.WHITE}[3] {Fore.RED}Kusursuz Profil{Style.RESET_ALL}")
    print(f"      RT-DETR + YOLO + SAM 2 + Florence-2. En yuksek dogruluk.")
    print()
    print(f"  {Fore.WHITE}[4] {Fore.CYAN}Ozel Profil{Style.RESET_ALL}")
    print(f"      Kendi model zincirini belirle.")
    print()
    print(f"  {Fore.WHITE}[0] {Fore.RED}Ana Menuye Don{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")

    secim = input(f"{Fore.CYAN}  Profil seciminiz [1-4, 0=iptal]: {Style.RESET_ALL}").strip()

    if secim == "1":
        multi["aktif"] = False
        multi["siralama"] = []
        yapilandirma_kaydet(config)
        print(f"{Fore.GREEN}[+] Hiz Profili aktif. Coklu model devre disi, tek model (RT-DETR) ile cikarim yapilacak.{Style.RESET_ALL}")

    elif secim == "2":
        multi["aktif"] = True
        multi["siralama"] = ["rt-detr-v2-x", "yolov12x"]
        yapilandirma_kaydet(config)
        print(f"{Fore.GREEN}[+] Hibrit Profil aktif. RT-DETRv2-X + YOLOv12x + WBF ile cikarim yapilacak.{Style.RESET_ALL}")

    elif secim == "3":
        multi["aktif"] = True
        multi["siralama"] = ["rt-detr-v2-x", "yolov12x", "sam2_small", "florence-2"]
        yapilandirma_kaydet(config)
        print(f"{Fore.GREEN}[+] Kusursuz Profil aktif. 4 model zincirleme cikarim yapilacak.{Style.RESET_ALL}")

    elif secim == "4":
        print()
        print(f"{Fore.CYAN}    Kullanilabilir modeller: rt-detr-v2-x, yolov12x, sam2_small, florence-2{Style.RESET_ALL}")
        print(f"{Fore.CYAN}    Ornek: rt-detr-v2-x, yolov12x, sam2_small, florence-2{Style.RESET_ALL}")
        print()
        ozel_girdi = input(f"{Fore.CYAN}    Model zinciri (virgulle ayirarak): {Style.RESET_ALL}").strip()

        if not ozel_girdi:
            print(f"{Fore.RED}[-] Bos giris. Islem iptal edildi.{Style.RESET_ALL}")
            print()
            return

        ozel_siralama = [m.strip() for m in ozel_girdi.split(",") if m.strip()]
        gecersiz = [m for m in ozel_siralama if m not in GECERLI_MODELLER]

        if gecersiz:
            print(f"{Fore.RED}[-] Gecersiz model adi: {', '.join(gecersiz)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[!] Gecerli modeller: {', '.join(GECERLI_MODELLER)}{Style.RESET_ALL}")
            print()
            return

        if not ozel_siralama:
            print(f"{Fore.RED}[-] Gecerli model bulunamadi. Islem iptal edildi.{Style.RESET_ALL}")
            print()
            return

        multi["aktif"] = True
        multi["siralama"] = ozel_siralama
        yapilandirma_kaydet(config)
        print(f"{Fore.GREEN}[+] Ozel Profil aktif: {' -> '.join(ozel_siralama)}{Style.RESET_ALL}")

    elif secim == "0":
        print(f"{Fore.YELLOW}[!] Islem iptal edildi.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[-] Gecersiz secim.{Style.RESET_ALL}")

    print()


def gorsel_toplama_calistir():
    from src.data_tools import gorsel_indir
    print()
    print(f"{Fore.YELLOW}[*] Hasar siniflari icin otomatik gorsel toplanacak.{Style.RESET_ALL}")
    adet_girdi = input(f"{Fore.CYAN}    Gorsel adedi [Enter=50]: {Style.RESET_ALL}").strip()
    adet = 50
    if adet_girdi:
        try:
            adet = int(adet_girdi)
        except ValueError:
            print(f"{Fore.RED}[-] Gecersiz sayi. Varsayilan (50) kullanilacak.{Style.RESET_ALL}")
    print()
    gorsel_indir(max_sayi=adet)
    print()


def kalite_kontrol_calistir():
    from src.data_tools import veri_kalite_kontrolu
    print()
    veri_kalite_kontrolu()
    print()


def etiket_dogrulama_calistir():
    from src.validator import etiket_validator_calistir
    print()
    etiket_validator_calistir()
    print()


def model_bilgisi_calistir():
    from src.train import model_bilgisi_goster
    print()
    model_bilgisi_goster()
    print()


def testleri_calistir():
    import unittest
    print()
    print(f"{Fore.YELLOW}[*] Tum sistem testleri baslatiliyor. Lutfen bekleyin...{Style.RESET_ALL}")
    test_dizini = PROJE_KOKU / "testler"
    test_paketi = unittest.defaultTestLoader.discover(str(test_dizini), pattern="test_*.py")
    sonuc = unittest.TextTestRunner(verbosity=2).run(test_paketi)
    if sonuc.wasSuccessful():
        print(f"\n{Fore.GREEN}[+] Harika! Tum testler ({sonuc.testsRun}/{sonuc.testsRun}) basariyla gecti.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}[-] Bazi testler basarisiz oldu ({sonuc.testsRun - len(sonuc.failures) - len(sonuc.errors)}/{sonuc.testsRun}). Lutfen yukaridaki loglari inceleyin.{Style.RESET_ALL}")
    print()


def gateway_testi_calistir():
    from src.gateway.ai_router import router_testi_calistir
    print()
    router_testi_calistir()
    print()


def cikis_yap():
    print()
    print(f"{Fore.GREEN}[+] HADES DETECTOR kapatiliyor...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}    Gule gule!{Style.RESET_ALL}")
    print()


def menu_secimi_isle(secim):
    if secim == "1":
        donanim_kontrolu_calistir()
    elif secim == "2":
        etiketleme_calistir()
    elif secim == "3":
        augmentation_calistir()
    elif secim == "4":
        veri_bolme_calistir()
    elif secim == "5":
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        config = yapilandirma_yukle()
        if config.get("multi_model", {}).get("aktif", False):
            print()
            print(f"{Fore.YELLOW}[*] Su an Coklu-Model profili aktif.{Style.RESET_ALL}")
            print(f"{Fore.BLUE}[*] Hangi alt modeli egitmek (Finetune) istiyorsunuz?{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}[1] {Fore.YELLOW}YOLO{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}[2] {Fore.CYAN}RT-DETR{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}[0] {Fore.RED}Iptal{Style.RESET_ALL}")
            print()
            alt_secim = input(f"{Fore.CYAN}  Seciminiz [0-2]: {Style.RESET_ALL}").strip()
            if alt_secim == "1":
                config["model"]["tur"] = "yolo"
                yapilandirma_kaydet(config)
                print(f"{Fore.GREEN}[+] Egitim modeli YOLO olarak guncellendi.{Style.RESET_ALL}")
                print()
                egitim_calistir()
            elif alt_secim == "2":
                config["model"]["tur"] = "rtdetr"
                yapilandirma_kaydet(config)
                print(f"{Fore.GREEN}[+] Egitim modeli RT-DETR olarak guncellendi.{Style.RESET_ALL}")
                print()
                egitim_calistir()
            elif alt_secim == "0":
                print(f"{Fore.YELLOW}[!] Egitim iptal edildi.{Style.RESET_ALL}")
                print()
            else:
                print(f"{Fore.RED}[-] Gecersiz secim. Egitim iptal edildi.{Style.RESET_ALL}")
                print()
            return True
        egitim_calistir()
    elif secim == "6":
        cikarim_calistir()
    elif secim == "7":
        rapor_calistir()
    elif secim == "8":
        testleri_calistir()
    elif secim == "9":
        from src.utils import yapilandirma_yukle
        config = yapilandirma_yukle()
        if config.get("multi_model", {}).get("aktif", False):
            print()
            print(f"{Fore.YELLOW}[!] Coklu-Model orkestrasyonu su anda AKTIF.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[!] Sistemdeki agirliklar Orkestrasyon Yoneticisi [10] uzerinden yonetilmektedir.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[!] Tekil model secimi yapmak istiyorsaniz once [11] Cikarim Profili Secimi menusunden 'Hiz Profili'ni secerek Coklu-Model'i kapatiniz.{Style.RESET_ALL}")
            input(f"\n{Fore.CYAN}[Enter] Ana Menuye Don...{Style.RESET_ALL}")
            print()
            return True
        egitim_modeli_secimi_calistir()
    elif secim == "10":
        orkestrasyon_yoneticisi_calistir()
    elif secim == "11":
        cikarim_profili_secimi_calistir()
    elif secim == "12":
        gorsel_toplama_calistir()
    elif secim == "13":
        kalite_kontrol_calistir()
    elif secim == "14":
        etiket_dogrulama_calistir()
    elif secim == "15":
        model_bilgisi_calistir()
    elif secim == "16":
        gateway_testi_calistir()
    elif secim == "0":
        cikis_yap()
        return False
    else:
        print(f"{Fore.RED}[-] Gecersiz secim! Lutfen 0-16 arasinda bir deger girin.{Style.RESET_ALL}")
        print()
    return True


def ana_dongu():
    calisiyor = True
    while calisiyor:
        try:
            basligi_yazdir()
            menuyu_yazdir()
            secim = yardimli_input(f"\n{Fore.CYAN}  Seciminiz [0-16]: {Style.RESET_ALL}", "ana_menu")
            calisiyor = menu_secimi_isle(secim)
            if calisiyor:
                input(f"\n{Fore.YELLOW}  Devam etmek icin Enter'a basin...{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print()
            cikis_yap()
            break
        except EOFError:
            print()
            cikis_yap()
            break


if __name__ == "__main__":
    ana_dongu()
