import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent
sys.path.insert(0, str(PROJE_KOKU))

SECILI_CIHAZ = None

HADES_LOGO = r"""
  _   _    _    ____  _____ ____    ____   ____    _    _   _ _   _ _____ ____  
 | | | |  / \  |  _ \| ____/ ___|  / ___| / ___|  / \  | \ | | \ | | ____|  _ \ 
 | |_| | / _ \ | | | |  _| \___ \  \___ \| |     / _ \ |  \| |  \| |  _| | |_) |
 |  _  |/ ___ \| |_| | |___ ___) |  ___) | |___ / ___ \| |\  | |\  | |___|  _ < 
 |_| |_/_/   \_\____/|_____|____/  |____/ \____/_/   \_\_| \_|_| \_|_____|_| \_\
"""


def ekrani_temizle():
    os.system("cls" if os.name == "nt" else "clear")


def basligi_yazdir():
    ekrani_temizle()
    print(f"{Fore.CYAN}{HADES_LOGO}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  [Hades Hasar Tespiti Sistemi v1.0.0]{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()


def menuyu_yazdir():
    print(f"{Fore.YELLOW}  [ANA MENU]{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}[1] {Fore.YELLOW}Donanim Kontrolu{Style.RESET_ALL}")
    print(f"      CPU/GPU/NPU kaynaklarini listeler, egitim cihazi secimi yapar.")
    print()
    print(f"  {Fore.WHITE}[2] {Fore.YELLOW}Veri Etiketleme{Style.RESET_ALL}")
    print(f"      hasar-ornek klasorunde LabelImg uygulamasini baslatir.")
    print()
    print(f"  {Fore.WHITE}[3] {Fore.YELLOW}Veri Artirimi (Augmentation){Style.RESET_ALL}")
    print(f"      Etiketlenen gorselleri ayarlara gore cogaltir.")
    print()
    print(f"  {Fore.WHITE}[4] {Fore.YELLOW}Veri Bolme (Train/Val Split){Style.RESET_ALL}")
    print(f"      Verileri train/val (%80-%20) klasorlerine paylastirir.")
    print()
    print(f"  {Fore.WHITE}[5] {Fore.YELLOW}Model Egitimini Baslat{Style.RESET_ALL}")
    print(f"      Transfer ogrenimi ile egitim sureclerini baslatir.")
    print()
    print(f"  {Fore.WHITE}[6] {Fore.YELLOW}Hasar Tespiti Yap (Inference){Style.RESET_ALL}")
    print(f"      Belirtilen gorselde hasarlari tespit eder.")
    print()
    print(f"  {Fore.WHITE}[7] {Fore.YELLOW}Egitim Performans Raporu{Style.RESET_ALL}")
    print(f"      Modellerin dogruluk metriklerini listeler.")
    print()
    print(f"  {Fore.WHITE}[8] {Fore.YELLOW}Sistem Testlerini Calistir{Style.RESET_ALL}")
    print(f"      Uygulamanin tum birim ve entegrasyon testlerini kosar.")
    print()
    print(f"  {Fore.WHITE}[9] {Fore.YELLOW}Model Secimi{Style.RESET_ALL}")
    print(f"      YOLO veya RT-DETR model mimarisini secer.")
    print()
    print(f"  {Fore.WHITE}[10] {Fore.YELLOW}Model Ayarlari{Style.RESET_ALL}")
    print(f"      Secili modelin neslini ve zeka seviyesini yapilandirir.")
    print()
    print(f"  {Fore.WHITE}[0] {Fore.RED}Cikis{Style.RESET_ALL}")
    print(f"      Uygulamayi sonlandirir.")
    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


def donanim_kontrolu_calistir():
    global SECILI_CIHAZ
    from src.hardware_check import donanim_ozeti_yazdir, cihaz_secimi_yap
    print()
    profil = donanim_ozeti_yazdir()
    print()
    SECILI_CIHAZ = cihaz_secimi_yap(profil)
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

    epoch_girdi = input(f"{Fore.CYAN}    Epoch sayisi [Enter=varsayilan]: {Style.RESET_ALL}").strip()
    epoch_sayisi = None
    if epoch_girdi:
        try:
            epoch_sayisi = int(epoch_girdi)
        except ValueError:
            print(f"{Fore.RED}[-] Gecersiz epoch sayisi. Varsayilan kullanilacak.{Style.RESET_ALL}")

    batch_girdi = input(f"{Fore.CYAN}    Batch size [Enter=varsayilan]: {Style.RESET_ALL}").strip()
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

    print()
    egitim_baslat(epoch_sayisi=epoch_sayisi, batch_size=batch_size, cihaz=cihaz, img_size=img_size)
    print()


def cikarim_calistir():
    import random
    from src.pipeline import hasar_tespiti_yap, toplu_hasar_tespiti_yap

    cikti_klasoru = PROJE_KOKU / "hasar-sonucu"
    ornek_klasoru = PROJE_KOKU / "hasar-ornek"
    gorsel_uzantilari = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    print()
    print(f"{Fore.YELLOW}  [HASAR TESPITI MENU]{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}[1] {Fore.YELLOW}Tekli Gorsel{Style.RESET_ALL}")
    print(f"      Belirli bir gorselde hasar tespiti yapar.")
    print()
    print(f"  {Fore.WHITE}[2] {Fore.YELLOW}Toplu Tarama{Style.RESET_ALL}")
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
            gorsel_yolu = gorsel_yolu.strip('"').strip("'")

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


def model_secimi_calistir():
    from src.pipeline import yapilandirma_yukle, yapilandirma_kaydet
    config = yapilandirma_yukle()
    mevcut_tur = config.get("model", {}).get("tur", "yolo")
    mevcut_agirlik = config.get("model", {}).get("agirlik", "yok")

    print()
    print(f"{Fore.YELLOW}  [MODEL SECIMI]{Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}Mevcut Model: {Fore.GREEN}{mevcut_tur.upper()} ({mevcut_agirlik}){Style.RESET_ALL}")
    print()
    print(f"  {Fore.WHITE}[1] {Fore.YELLOW}YOLO{Style.RESET_ALL}")
    print(f"      Klasik tek asamali nesne tespit modeli. Hizli ve verimli.")
    print(f"      Mevcut surumler: YOLOv8, YOLOv12 (nano'dan x-large'a)")
    print()
    print(f"  {Fore.WHITE}[2] {Fore.CYAN}RT-DETR (Real-Time DEtection TRansformer){Style.RESET_ALL}")
    print(f"      Gelistirici: Baidu")
    print(f"      Transformer tabanli ilk gercek zamanli nesne tespit modeli.")
    print(f"      NMS (Non-Maximum Suppression) adimina ihtiyac duymaz.")
    print(f"      Benzer boyuttaki YOLO modellerine gore daha yuksek mAP sunar.")
    print(f"      Ultralytics kutuphanesi tarafindan yerlesik desteklenir.")
    print()
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")

    secim = input(f"{Fore.CYAN}  Model seciminiz [1-2, Enter=iptal]: {Style.RESET_ALL}").strip()

    if secim == "1":
        yeni_tur = "yolo"
        varsayilan_agirlik = "yolo12n.pt"
    elif secim == "2":
        yeni_tur = "rtdetr"
        varsayilan_agirlik = "rtdetr-l.pt"
    else:
        print(f"{Fore.RED}[-] Secim iptal edildi.{Style.RESET_ALL}")
        print()
        return

    if mevcut_tur == yeni_tur:
        print(f"\n{Fore.YELLOW}[!] Model zaten {yeni_tur.upper()} olarak ayarli.{Style.RESET_ALL}\n")
        return

    if "model" not in config:
        config["model"] = {}
    config["model"]["tur"] = yeni_tur
    config["model"]["agirlik"] = varsayilan_agirlik
    yapilandirma_kaydet(config)

    print()
    print(f"{Fore.GREEN}[+] Model turu guncellendi: {mevcut_tur.upper()} -> {yeni_tur.upper()}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}[+] Varsayilan agirlik atandi: {varsayilan_agirlik}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[!] Model boyutunu [10] Model Ayarlari'ndan degistirebilirsiniz.{Style.RESET_ALL}")
    print()


def ayarlar_calistir():
    from src.pipeline import yapilandirma_yukle, yapilandirma_kaydet
    config = yapilandirma_yukle()
    model_tur = config.get("model", {}).get("tur", "yolo")

    print()
    if model_tur == "rtdetr":
        print(f"{Fore.YELLOW}  [RT-DETR MODEL AYARLARI]{Style.RESET_ALL}")
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

        boyut_eki = boyutlar[boyut_secim]
        yeni_agirlik = f"rtdetr-{boyut_eki}.pt"
    else:
        print(f"{Fore.YELLOW}  [YOLO MODEL AYARLARI]{Style.RESET_ALL}")
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
        print(f"{Fore.YELLOW}  [ZEKA / BOYUT SEVIYESI]{Style.RESET_ALL}")
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

        boyut_eki = boyutlar[boyut_secim]
        yeni_agirlik = f"{on_ek}{boyut_eki}.pt"

    if "model" not in config:
        config["model"] = {}

    eski_agirlik = config["model"].get("agirlik", "yok")
    if eski_agirlik == yeni_agirlik:
        print(f"\n{Fore.YELLOW}[!] Model zaten {eski_agirlik} olarak ayarli.{Style.RESET_ALL}\n")
        return

    config["model"]["agirlik"] = yeni_agirlik
    yapilandirma_kaydet(config)

    print()
    print(f"{Fore.GREEN}[+] Model agirligi guncellendi: {eski_agirlik} -> {yeni_agirlik}{Style.RESET_ALL}")
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
        egitim_calistir()
    elif secim == "6":
        cikarim_calistir()
    elif secim == "7":
        rapor_calistir()
    elif secim == "8":
        testleri_calistir()
    elif secim == "9":
        model_secimi_calistir()
    elif secim == "10":
        ayarlar_calistir()
    elif secim == "0":
        cikis_yap()
        return False
    else:
        print(f"{Fore.RED}[-] Gecersiz secim! Lutfen 0-10 arasinda bir deger girin.{Style.RESET_ALL}")
        print()
    return True


def ana_dongu():
    calisiyor = True
    while calisiyor:
        try:
            basligi_yazdir()
            menuyu_yazdir()
            secim = input(f"\n{Fore.CYAN}  Seciminiz [0-10]: {Style.RESET_ALL}").strip()
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