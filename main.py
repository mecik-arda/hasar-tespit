import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent
sys.path.insert(0, str(PROJE_KOKU))

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
    print(f"      Sistemdeki CPU/GPU kaynaklarini kontrol eder.")
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
    print(f"  {Fore.WHITE}[9] {Fore.RED}Cikis{Style.RESET_ALL}")
    print(f"      Uygulamayi sonlandirir.")
    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


def donanim_kontrolu_calistir():
    from src.hardware_check import donanim_ozeti_yazdir
    print()
    donanim_ozeti_yazdir()
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
    from src.train import egitim_baslat
    print()
    print(f"{Fore.YELLOW}[*] Egitim parametreleri (Bos birakirsiniz varsayilan kullanilir):{Style.RESET_ALL}")
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

    cihaz_girdi = input(f"{Fore.CYAN}    Cihaz [Enter=auto]: {Style.RESET_ALL}").strip()
    cihaz = None
    if cihaz_girdi:
        cihaz = cihaz_girdi

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


def testleri_calistir():
    import unittest
    print()
    print(f"{Fore.YELLOW}[*] Tum sistem testleri baslatiliyor. Lutfen bekleyin...{Style.RESET_ALL}")
    test_dizini = PROJE_KOKU / "testler"
    test_paketi = unittest.defaultTestLoader.discover(str(test_dizini), pattern="test_*.py")
    sonuc = unittest.TextTestRunner(verbosity=2).run(test_paketi)
    if sonuc.wasSuccessful():
        print(f"\n{Fore.GREEN}[+] Harika! Tum testler (17/17) basariyla gecti.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}[-] Bazi testler basarisiz oldu. Lutfen yukaridaki loglari inceleyin.{Style.RESET_ALL}")
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
        cikis_yap()
        return False
    else:
        print(f"{Fore.RED}[-] Gecersiz secim! Lutfen 1-9 arasinda bir deger girin.{Style.RESET_ALL}")
        print()
    return True


def ana_dongu():
    calisiyor = True
    while calisiyor:
        try:
            basligi_yazdir()
            menuyu_yazdir()
            secim = input(f"\n{Fore.CYAN}  Seciminiz [1-9]: {Style.RESET_ALL}").strip()
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