import sys
import yaml
from pathlib import Path
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent.parent
YAPILANDIRMA_YOLU = PROJE_KOKU / "config.yaml"
VERI_KOKU = PROJE_KOKU / "data"
EGITIM_KOKU = PROJE_KOKU / "runs" / "train"


def yapilandirma_yukle():
    with open(YAPILANDIRMA_YOLU, "r", encoding="utf-8") as dosya:
        return yaml.safe_load(dosya)


def egitim_baslat(epoch_sayisi=None, batch_size=None, cihaz=None, img_size=None):
    from ultralytics import YOLO, RTDETR
    from src.hardware_check import donanim_profili_olustur

    yapilandirma = yapilandirma_yukle()
    model_ayari = yapilandirma.get("model", {})
    egitim_ayari = yapilandirma.get("egitim", {})

    agirlik = model_ayari.get("agirlik", "yolo12n.pt")
    hedef_epoch = epoch_sayisi or model_ayari.get("epoch_sayisi", 100)
    hedef_img_size = img_size or model_ayari.get("img_size", 640)

    if not isinstance(hedef_epoch, int) or hedef_epoch <= 0:
        print(f"{Fore.YELLOW}[!] Uyari: Gecersiz epoch degeri tespit edildi. Varsayilan kullaniliyor: 100{Style.RESET_ALL}")
        hedef_epoch = 100

    if not isinstance(hedef_img_size, int) or hedef_img_size <= 0:
        print(f"{Fore.YELLOW}[!] Uyari: Gecersiz gorsel boyutu tespit edildi. Varsayilan kullaniliyor: 640{Style.RESET_ALL}")
        hedef_img_size = 640

    donanim_profili = donanim_profili_olustur()
    onerilen_batch = donanim_profili["onerilen_batch"]
    onerilen_cihaz = donanim_profili["hedef_cihaz"]

    hedef_batch = batch_size or model_ayari.get("batch_size", onerilen_batch)
    if hedef_batch == "auto":
        hedef_batch = onerilen_batch
    elif not isinstance(hedef_batch, int) or hedef_batch <= 0:
        print(f"{Fore.YELLOW}[!] Uyari: Gecersiz batch degeri tespit edildi. Varsayilan kullaniliyor: {onerilen_batch}{Style.RESET_ALL}")
        hedef_batch = onerilen_batch

    hedef_cihaz = cihaz or model_ayari.get("cihaz", "auto")
    if hedef_cihaz == "auto":
        hedef_cihaz = onerilen_cihaz

    veri_seti_yolu = VERI_KOKU / "dataset.yaml"
    if not veri_seti_yolu.exists():
        print(f"{Fore.RED}[-] Veri seti yapilandirmasi bulunamadi: {veri_seti_yolu}{Style.RESET_ALL}")
        return False

    train_klasoru = VERI_KOKU / "images" / "train"
    if not train_klasoru.exists() or not any(train_klasoru.iterdir()):
        print(f"{Fore.RED}[-] Egitim verisi bulunamadi. Once veri bolme islemini yapin.{Style.RESET_ALL}")
        return False

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Model Egitimi{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] Egitim Yapilandirmasi{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Model          : {agirlik}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Epoch          : {hedef_epoch}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Batch Size     : {hedef_batch}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Img Size       : {hedef_img_size}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Cihaz          : {hedef_cihaz}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Veri Seti      : {veri_seti_yolu}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    transfer_ogrenimi = egitim_ayari.get("transfer_ogrenimi", True)
    model_tur = yapilandirma.get("model", {}).get("tur", "yolo")

    if transfer_ogrenimi:
        print(f"{Fore.BLUE}[*] Transfer ogrenimi aktif. On egitimli agirliklar kullaniliyor.{Style.RESET_ALL}")

    ModelSinifi = RTDETR if model_tur == "rtdetr" else YOLO
    try:
        model = ModelSinifi(agirlik)
    except Exception as hata:
        print(f"{Fore.RED}[-] Model yuklenemedi: {hata}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Internet baglantinizi ve model adini kontrol edin.{Style.RESET_ALL}")
        return False

    optimizer = egitim_ayari.get("optimizer", "auto")
    lr0 = egitim_ayari.get("lr0", 0.01)
    lrf = egitim_ayari.get("lrf", 0.01)
    momentum = egitim_ayari.get("momentum", 0.937)
    weight_decay = egitim_ayari.get("weight_decay", 0.0005)
    warmup_epochs = egitim_ayari.get("warmup_epochs", 3)
    warmup_momentum = egitim_ayari.get("warmup_momentum", 0.8)
    warmup_bias_lr = egitim_ayari.get("warmup_bias_lr", 0.1)

    print(f"{Fore.BLUE}[*] Egitim basliyor...{Style.RESET_ALL}")
    print()

    try:
        sonuclar = model.train(
            data=str(veri_seti_yolu),
            epochs=hedef_epoch,
            batch=hedef_batch,
            imgsz=hedef_img_size,
            device=hedef_cihaz,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            project=str(EGITIM_KOKU),
            name="hades_egitim",
            exist_ok=True,
            pretrained=transfer_ogrenimi,
            verbose=True,
        )
        print()
        print(f"{Fore.GREEN}[+] Egitim tamamlandi!{Style.RESET_ALL}")
        print(f"{Fore.WHITE}    Sonuclar: {EGITIM_KOKU / 'hades_egitim'}{Style.RESET_ALL}")
        return True
    except Exception as hata:
        print(f"{Fore.RED}[-] Egitim sirasinda hata: {hata}{Style.RESET_ALL}")
        return False


def egitim_raporu_goster():
    egitim_klasoru = EGITIM_KOKU / "hades_egitim"
    if not egitim_klasoru.exists():
        print(f"{Fore.RED}[-] Egitim sonuclari bulunamadi. Once model egitimi yapin.{Style.RESET_ALL}")
        return False

    sonuclar_yolu = egitim_klasoru / "results.csv"
    args_yolu = egitim_klasoru / "args.yaml"

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Egitim Performans Raporu{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    if args_yolu.exists():
        with open(args_yolu, "r", encoding="utf-8") as dosya:
            args = yaml.safe_load(dosya)
        print(f"{Fore.YELLOW}[*] Egitim Yapilandirmasi{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Model          : {args.get('model', 'Bilinmiyor')}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Epoch          : {args.get('epochs', 'Bilinmiyor')}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Batch Size     : {args.get('batch', 'Bilinmiyor')}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Img Size       : {args.get('imgsz', 'Bilinmiyor')}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Cihaz          : {args.get('device', 'Bilinmiyor')}{Style.RESET_ALL}")
        print()

    if sonuclar_yolu.exists():
        import csv
        with open(sonuclar_yolu, "r", encoding="utf-8") as dosya:
            okuyucu = csv.DictReader(dosya)
            satirlar = list(okuyucu)

        if not satirlar:
            print(f"{Fore.RED}[-] Sonuc dosyasi bos.{Style.RESET_ALL}")
            return False

        son_satir = satirlar[-1]
        en_iyi_satir = son_satir

        print(f"{Fore.YELLOW}[*] Son Epoch Sonuclari{Style.RESET_ALL}")
        metrik_anahtarlari = [
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        ]

        for anahtar in metrik_anahtarlari:
            if anahtar in son_satir:
                deger = son_satir[anahtar].strip()
                print(f"    {Fore.WHITE}{anahtar:30s}: {deger}{Style.RESET_ALL}")

        print()

        en_iyi_map = 0
        for satir in satirlar:
            map_anahtari = "metrics/mAP50(B)"
            if map_anahtari in satir:
                try:
                    map_degeri = float(satir[map_anahtari].strip())
                    if map_degeri > en_iyi_map:
                        en_iyi_map = map_degeri
                        en_iyi_satir = satir
                except (ValueError, KeyError):
                    pass

        print(f"{Fore.YELLOW}[*] En Iyi Performans{Style.RESET_ALL}")
        for anahtar in metrik_anahtarlari:
            if anahtar in en_iyi_satir:
                deger = en_iyi_satir[anahtar].strip()
                print(f"    {Fore.GREEN}{anahtar:30s}: {deger}{Style.RESET_ALL}")

        print()
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Toplam Epoch Sayisi: {len(satirlar)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[-] Sonuc dosyasi bulunamadi: {sonuclar_yolu}{Style.RESET_ALL}")
        return False

    return True


if __name__ == "__main__":
    secim = sys.argv[1] if len(sys.argv) > 1 else ""
    if secim == "egit":
        egitim_baslat()
    elif secim == "rapor":
        egitim_raporu_goster()
    else:
        print(f"{Fore.YELLOW}Kullanim: python train.py [egit|rapor]{Style.RESET_ALL}")