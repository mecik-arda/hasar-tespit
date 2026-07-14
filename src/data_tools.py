import sys
import shutil
import random
import subprocess
import yaml
import cv2
from pathlib import Path
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent.parent
YAPILANDIRMA_YOLU = PROJE_KOKU / "config.yaml"


def yapilandirma_yukle():
    with open(YAPILANDIRMA_YOLU, "r", encoding="utf-8") as dosya:
        return yaml.safe_load(dosya)


def etiketleme_baslat():
    yapilandirma = yapilandirma_yukle()
    etiket_klasoru = PROJE_KOKU / yapilandirma["veri"]["etiket_klasoru"]
    sinif_dosyasi = PROJE_KOKU / "data" / "siniflar.txt"

    if not etiket_klasoru.exists():
        print(f"{Fore.RED}[-] Etiketleme klasoru bulunamadi: {etiket_klasoru}{Style.RESET_ALL}")
        return False

    sinif_dosyasi.parent.mkdir(parents=True, exist_ok=True)
    siniflar = yapilandirma.get("siniflar", {})
    with open(sinif_dosyasi, "w", encoding="utf-8") as dosya:
        for anahtar in sorted(siniflar.keys()):
            dosya.write(f"{siniflar[anahtar]}\n")

    try:
        import labelImg as labelimg_pkg
        pkg_klasoru = Path(labelimg_pkg.__path__[0])
        predefined_dosyasi = pkg_klasoru / "data" / "predefined_classes.txt"
        if predefined_dosyasi.exists():
            with open(predefined_dosyasi, "w", encoding="utf-8") as f:
                f.write("")
            print(f"{Fore.GREEN}[+] LabelImg on tanimli siniflari temizlendi.{Style.RESET_ALL}")
    except Exception:
        pass

    print(f"{Fore.BLUE}[*] LabelImg baslatiliyor...{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Klasor : {etiket_klasoru}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Siniflar: {sinif_dosyasi}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] Etiketleme formati: YOLO{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] LabelImg kapana kadar bekleniyor...{Style.RESET_ALL}")

    try:
        subprocess.call([
            sys.executable, "-m", "labelImg",
            str(etiket_klasoru),
            str(sinif_dosyasi),
        ])
        print(f"{Fore.GREEN}[+] Etiketleme tamamlandi.{Style.RESET_ALL}")
        return True
    except FileNotFoundError:
        print(f"{Fore.RED}[-] LabelImg yuklu degil. 'pip install labelImg' ile yukleyin.{Style.RESET_ALL}")
        return False
    except Exception as hata:
        print(f"{Fore.RED}[-] LabelImg baslatilirken hata: {hata}{Style.RESET_ALL}")
        return False


def etiket_dosyasini_oku(etiket_yolu, genislik, yukseklik):
    kutucuklar = []
    if not etiket_yolu.exists():
        return kutucuklar
    with open(etiket_yolu, "r", encoding="utf-8") as dosya:
        for satir in dosya:
            parcalar = satir.strip().split()
            if len(parcalar) == 5:
                sinif_id = int(parcalar[0])
                x_merkez = float(parcalar[1])
                y_merkez = float(parcalar[2])
                gen = float(parcalar[3])
                yuk = float(parcalar[4])
                x1 = int((x_merkez - gen / 2) * genislik)
                y1 = int((y_merkez - yuk / 2) * yukseklik)
                x2 = int((x_merkez + gen / 2) * genislik)
                y2 = int((y_merkez + yuk / 2) * yukseklik)
                x1 = max(0, min(x1, genislik))
                y1 = max(0, min(y1, yukseklik))
                x2 = max(0, min(x2, genislik))
                y2 = max(0, min(y2, yukseklik))
                kutucuklar.append({
                    "sinif": sinif_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
    return kutucuklar


def kutucukleri_yolo_donustur(kutucuklar, genislik, yukseklik):
    satirlar = []
    for kutu in kutucuklar:
        x_merkez = ((kutu["x1"] + kutu["x2"]) / 2) / genislik
        y_merkez = ((kutu["y1"] + kutu["y2"]) / 2) / yukseklik
        gen = (kutu["x2"] - kutu["x1"]) / genislik
        yuk = (kutu["y2"] - kutu["y1"]) / yukseklik
        x_merkez = max(0.0, min(1.0, x_merkez))
        y_merkez = max(0.0, min(1.0, y_merkez))
        gen = max(0.0, min(1.0, gen))
        yuk = max(0.0, min(1.0, yuk))
        satirlar.append(f"{kutu['sinif']} {x_merkez:.6f} {y_merkez:.6f} {gen:.6f} {yuk:.6f}")
    return satirlar


def augmentation_uygula():
    from albumentations import (
        Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
        GaussNoise, GaussianBlur, BboxParams
    )

    yapilandirma = yapilandirma_yukle()
    aug_ayar = yapilandirma.get("augmentation", {})

    if not aug_ayar.get("aktif", False):
        print(f"{Fore.YELLOW}[*] Augmentation kapali. config.yaml dosyasinden aktif edebilirsiniz.{Style.RESET_ALL}")
        return False

    etiket_klasoru = PROJE_KOKU / yapilandirma["veri"]["etiket_klasoru"]
    cikti_klasoru = etiket_klasoru / "augmented"
    cikti_klasoru.mkdir(parents=True, exist_ok=True)

    carpma_katsayisi = aug_ayar.get("carpma_katsayisi", 3)
    donderme_acisi = aug_ayar.get("donderme_acisi", 15)
    parlaklik_limit = aug_ayar.get("parlaklik_limit", 0.3)
    kontrast_limit = aug_ayar.get("kontrast_limit", 0.3)
    yatay_cevirme = aug_ayar.get("yatay_cevirme", True)
    dikey_cevirme = aug_ayar.get("dikey_cevirme", False)
    gauss_gurultu = aug_ayar.get("gauss_gurultu", True)
    bulaniklastirma = aug_ayar.get("bulaniklastirma", True)

    donusum_listesi = []
    if yatay_cevirme:
        donusum_listesi.append(HorizontalFlip(p=0.5))
    if dikey_cevirme:
        donusum_listesi.append(VerticalFlip(p=0.5))
    donusum_listesi.append(Rotate(limit=donderme_acisi, p=0.5))
    donusum_listesi.append(RandomBrightnessContrast(brightness_limit=parlaklik_limit, contrast_limit=kontrast_limit, p=0.5))
    if gauss_gurultu:
        donusum_listesi.append(GaussNoise(p=0.3))
    if bulaniklastirma:
        donusum_listesi.append(GaussianBlur(p=0.3))

    donusum = Compose(
        donusum_listesi,
        bbox_params=BboxParams(format="pascal_voc", label_fields=["siniflar"]),
    )

    gorsel_uzantilari = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    gorseller = []
    for uzanti in gorsel_uzantilari:
        gorseller.extend(etiket_klasoru.glob(f"*{uzanti}"))
        gorseller.extend(etiket_klasoru.glob(f"*{uzanti.upper()}"))

    gorseller = [g for g in gorseller if "augmented" not in str(g)]

    if not gorseller:
        print(f"{Fore.RED}[-] Etiket klasorunde gorsel bulunamadi: {etiket_klasoru}{Style.RESET_ALL}")
        return False

    print(f"{Fore.BLUE}[*] Augmentation basliyor...{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Gorsel sayisi: {len(gorseller)}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Carpma katsayisi: {carpma_katsayisi}{Style.RESET_ALL}")

    toplam_uretilen = 0
    for gorsel_yolu in gorseller:
        gorsel = cv2.imread(str(gorsel_yolu))
        if gorsel is None:
            continue
        yukseklik, genislik = gorsel.shape[:2]

        etiket_yolu = gorsel_yolu.with_suffix(".txt")
        kutucuklar = etiket_dosyasini_oku(etiket_yolu, genislik, yukseklik)

        bboxes = [[k["x1"], k["y1"], k["x2"], k["y2"]] for k in kutucuklar]
        siniflar_listesi = [k["sinif"] for k in kutucuklar]

        temel_ad = gorsel_yolu.stem

        for i in range(carpma_katsayisi):
            if bboxes:
                sonuc = donusum(image=gorsel, bboxes=bboxes, siniflar=siniflar_listesi)
                artirilmis_gorsel = sonuc["image"]
                artirilmis_bboxes = sonuc["bboxes"]
                artirilmis_siniflar = sonuc["siniflar"]
            else:
                sonuc = donusum(image=gorsel, bboxes=[], siniflar=[])
                artirilmis_gorsel = sonuc["image"]
                artirilmis_bboxes = []
                artirilmis_siniflar = []

            cikti_gorsel_adi = f"{temel_ad}_aug_{i + 1}{gorsel_yolu.suffix}"
            cikti_etiket_adi = f"{temel_ad}_aug_{i + 1}.txt"

            cv2.imwrite(str(cikti_klasoru / cikti_gorsel_adi), artirilmis_gorsel)

            yeni_kutucuklar = []
            yeni_yukseklik, yeni_genislik = artirilmis_gorsel.shape[:2]
            for idx, bbox in enumerate(artirilmis_bboxes):
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(int(x1), yeni_genislik))
                y1 = max(0, min(int(y1), yeni_yukseklik))
                x2 = max(0, min(int(x2), yeni_genislik))
                y2 = max(0, min(int(y2), yeni_yukseklik))
                if x2 > x1 and y2 > y1:
                    yeni_kutucuklar.append({
                        "sinif": artirilmis_siniflar[idx],
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })

            yolo_satirlar = kutucukleri_yolo_donustur(yeni_kutucuklar, yeni_genislik, yeni_yukseklik)
            with open(cikti_klasoru / cikti_etiket_adi, "w", encoding="utf-8") as dosya:
                dosya.write("\n".join(yolo_satirlar))

            toplam_uretilen += 1

    print(f"{Fore.GREEN}[+] Augmentation tamamlandi. {toplam_uretilen} yeni gorsel uretildi.{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Cikti klasoru: {cikti_klasoru}{Style.RESET_ALL}")
    return True


def veri_bol():
    yapilandirma = yapilandirma_yukle()
    etiket_klasoru = PROJE_KOKU / yapilandirma["veri"]["etiket_klasoru"]
    veri_klasoru = PROJE_KOKU / yapilandirma["veri"]["cikti_klasoru"]
    train_orani = yapilandirma["veri"].get("train_orani", 0.8)

    train_gorsel_klasoru = veri_klasoru / "images" / "train"
    val_gorsel_klasoru = veri_klasoru / "images" / "val"
    train_etiket_klasoru = veri_klasoru / "labels" / "train"
    val_etiket_klasoru = veri_klasoru / "labels" / "val"

    for klasor in [train_gorsel_klasoru, val_gorsel_klasoru, train_etiket_klasoru, val_etiket_klasoru]:
        klasor.mkdir(parents=True, exist_ok=True)

    gorsel_uzantilari = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    gorseller = []
    for uzanti in gorsel_uzantilari:
        gorseller.extend(etiket_klasoru.glob(f"*{uzanti}"))
        gorseller.extend(etiket_klasoru.glob(f"*{uzanti.upper()}"))

    augmented_klasoru = etiket_klasoru / "augmented"
    if augmented_klasoru.exists():
        for uzanti in gorsel_uzantilari:
            gorseller.extend(augmented_klasoru.glob(f"*{uzanti}"))
            gorseller.extend(augmented_klasoru.glob(f"*{uzanti.upper()}"))

    gorseller = list(set(gorseller))

    if not gorseller:
        print(f"{Fore.RED}[-] Bolunecek gorsel bulunamadi.{Style.RESET_ALL}")
        return False

    gecerli_gorseller = []
    for gorsel_yolu in gorseller:
        etiket_yolu = gorsel_yolu.with_suffix(".txt")
        if etiket_yolu.exists():
            gecerli_gorseller.append(gorsel_yolu)

    if not gecerli_gorseller:
        print(f"{Fore.RED}[-] Etiket dosyasi bulunamadi. Once etiketleme yapin.{Style.RESET_ALL}")
        return False

    random.shuffle(gecerli_gorseller)
    ayrac_noktasi = int(len(gecerli_gorseller) * train_orani)
    train_gorselleri = gecerli_gorseller[:ayrac_noktasi]
    val_gorselleri = gecerli_gorseller[ayrac_noktasi:]

    print(f"{Fore.BLUE}[*] Veri bolme islemi basliyor...{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Toplam gorsel: {len(gecerli_gorseller)}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Train: {len(train_gorselleri)} | Val: {len(val_gorselleri)}{Style.RESET_ALL}")

    for gorsel_yolu in train_gorselleri:
        etiket_yolu = gorsel_yolu.with_suffix(".txt")
        shutil.move(str(gorsel_yolu), str(train_gorsel_klasoru / gorsel_yolu.name))
        if etiket_yolu.exists():
            shutil.move(str(etiket_yolu), str(train_etiket_klasoru / etiket_yolu.name))

    for gorsel_yolu in val_gorselleri:
        etiket_yolu = gorsel_yolu.with_suffix(".txt")
        shutil.move(str(gorsel_yolu), str(val_gorsel_klasoru / gorsel_yolu.name))
        if etiket_yolu.exists():
            shutil.move(str(etiket_yolu), str(val_etiket_klasoru / etiket_yolu.name))

    dataset_yaml_yolu = veri_klasoru / "dataset.yaml"
    siniflar = yapilandirma.get("siniflar", {})
    yaml_icerik = {
        "path": str(veri_klasoru.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(siniflar),
        "names": {int(k): v for k, v in siniflar.items()},
    }
    import yaml as yaml_mod
    with open(dataset_yaml_yolu, "w", encoding="utf-8") as dosya:
        yaml_mod.dump(yaml_icerik, dosya, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"{Fore.GREEN}[+] Veri bolme tamamlandi.{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Train klasoru: {train_gorsel_klasoru}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Val klasoru  : {val_gorsel_klasoru}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}    Dataset YAML : {dataset_yaml_yolu}{Style.RESET_ALL}")
    return True


def veri_kalite_kontrolu(klasor=None):
    hashlib_modulu = __import__("hashlib")

    if klasor is None:
        yapilandirma = yapilandirma_yukle()
        klasor = PROJE_KOKU / yapilandirma["veri"]["etiket_klasoru"]

    klasor = Path(klasor)
    if not klasor.exists():
        print(f"{Fore.RED}[-] Klasor bulunamadi: {klasor}{Style.RESET_ALL}")
        return None

    gorsel_uzantilari = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    gorseller = sorted([f for f in klasor.iterdir() if f.suffix.lower() in gorsel_uzantilari])

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Veri Kalite Kontrolu{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}[*] Kontrol edilen klasor: {klasor}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] Toplam gorsel: {len(gorseller)}{Style.RESET_ALL}")
    print()

    bozuk = []
    dusuk_cozunurluk = []
    supheli = []
    saglam = []
    gorulen_hashler = {}

    for i, gorsel_yolu in enumerate(gorseller, 1):
        if i % 20 == 0 or i == len(gorseller):
            print(f"{Fore.BLUE}[*] ({i}/{len(gorseller)}) Isleniyor...{Style.RESET_ALL}")

        dosya_boyutu = gorsel_yolu.stat().st_size
        if dosya_boyutu < 1024:
            bozuk.append({"dosya": gorsel_yolu.name, "sebep": f"Dosya cok kucuk: {dosya_boyutu} byte"})
            continue

        gorsel = cv2.imread(str(gorsel_yolu))
        if gorsel is None:
            bozuk.append({"dosya": gorsel_yolu.name, "sebep": "cv2.imread None dondu (bozuk dosya)"})
            continue

        yukseklik, genislik = gorsel.shape[:2]
        if genislik < 100 or yukseklik < 100:
            dusuk_cozunurluk.append({"dosya": gorsel_yolu.name, "sebep": f"Cozunurluk: {genislik}x{yukseklik}"})
            continue

        piksel_std = gorsel.std()
        if piksel_std < 5:
            supheli.append({"dosya": gorsel_yolu.name, "sebep": f"Tekduze gorsel (std={piksel_std:.1f})"})
            continue

        with open(gorsel_yolu, "rb") as f:
            dosya_hash = hashlib_modulu.md5(f.read()).hexdigest()
        if dosya_hash in gorulen_hashler:
            supheli.append({
                "dosya": gorsel_yolu.name,
                "sebep": f"Yinelenen gorsel (ayni: {gorulen_hashler[dosya_hash]})",
            })
            continue
        gorulen_hashler[dosya_hash] = gorsel_yolu.name

        saglam.append(gorsel_yolu.name)

    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}[+] Saglam: {len(saglam)} gorsel{Style.RESET_ALL}")
    print(f"{Fore.RED}[-] Bozuk: {len(bozuk)} gorsel{Style.RESET_ALL}")
    for b in bozuk:
        print(f"    {Fore.WHITE}{b['dosya']} -> {b['sebep']}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[!] Dusuk cozunurluk: {len(dusuk_cozunurluk)} gorsel{Style.RESET_ALL}")
    for d in dusuk_cozunurluk:
        print(f"    {Fore.WHITE}{d['dosya']} -> {d['sebep']}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}[?] Supheli: {len(supheli)} gorsel{Style.RESET_ALL}")
    for s in supheli:
        print(f"    {Fore.WHITE}{s['dosya']} -> {s['sebep']}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return {"saglam": saglam, "bozuk": bozuk, "dusuk_cozunurluk": dusuk_cozunurluk, "supheli": supheli}


def gorsel_indir(sinif_adi=None, max_sayi=50, hedef_klasor=None):
    try:
        from icrawler.builtin import GoogleImageCrawler
    except ImportError:
        print(f"{Fore.RED}[-] icrawler yuklu degil. 'pip install icrawler' ile yukleyin.{Style.RESET_ALL}")
        return None

    yapilandirma = yapilandirma_yukle()
    siniflar = yapilandirma.get("siniflar", {})

    if sinif_adi is None:
        arama_terimleri = yapilandirma.get("veri", {}).get("arama_terimleri", {})
        if not arama_terimleri:
            arama_terimleri = {ad: f"araba {ad.lower()}" for ad in siniflar.values()}
    else:
        arama_terimleri = {sinif_adi: sinif_adi}

    if hedef_klasor is None:
        etiket_klasoru = PROJE_KOKU / yapilandirma["veri"]["etiket_klasoru"]
    else:
        etiket_klasoru = Path(hedef_klasor)

    etiket_klasoru.mkdir(parents=True, exist_ok=True)

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Otomatik Gorsel Toplama{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}[*] Hedef klasor: {etiket_klasoru}{Style.RESET_ALL}")
    print()

    toplam_indirilen = 0
    for ad, sorgu in arama_terimleri.items():
        sinif_klasoru = etiket_klasoru / ad.replace(" ", "_")
        sinif_klasoru.mkdir(parents=True, exist_ok=True)

        print(f"{Fore.BLUE}[*] '{sorgu}' araniyor (hedef: {max_sayi} gorsel)...{Style.RESET_ALL}")
        try:
            crawler = GoogleImageCrawler(
                storage={"root_dir": str(sinif_klasoru)},
                downloader_threads=4,
            )
            crawler.crawl(
                keyword=sorgu,
                max_num=max_sayi,
                filters={"type": "photo", "size": "large"},
            )
            indirilen = len(list(sinif_klasoru.glob("*")))
            toplam_indirilen += indirilen
            print(f"    {Fore.GREEN}[+] '{ad}': {indirilen} gorsel indirildi -> {sinif_klasoru.name}{Style.RESET_ALL}")
        except Exception as hata:
            print(f"    {Fore.RED}[-] '{ad}' indirilirken hata: {hata}{Style.RESET_ALL}")

    print()
    print(f"{Fore.GREEN}[+] Toplam {toplam_indirilen} gorsel indirildi.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[!] Indirilen gorselleri etiketlemeden once kalite kontrolu yapmaniz onerilir (Menu 12).{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return toplam_indirilen


if __name__ == "__main__":
    secim = sys.argv[1] if len(sys.argv) > 1 else ""
    if secim == "etiketle":
        etiketleme_baslat()
    elif secim == "augment":
        augmentation_uygula()
    elif secim == "bol":
        veri_bol()
    elif secim == "kalite":
        veri_kalite_kontrolu()
    elif secim == "indir":
        gorsel_indir()
    else:
        print(f"{Fore.YELLOW}Kullanim: python data_tools.py [etiketle|augment|bol|kalite|indir]{Style.RESET_ALL}")