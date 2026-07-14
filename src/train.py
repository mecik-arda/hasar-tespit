import sys
import yaml
import os
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


def _directml_cihazini_al():
    """DirectML GPU cihazini dondurur. Kullanilamiyorsa None."""
    try:
        import torch_directml
        dml = torch_directml.device()
        return dml
    except (ImportError, Exception):
        return None


def _intel_cpu_optimizasyonu_uygula():
    """Intel CPU'larda OpenMP thread sayisini fiziksel cekirdege ayarlar."""
    try:
        import psutil
        fiziksel_cekirdek = psutil.cpu_count(logical=False)
        if fiziksel_cekirdek:
            os.environ["OMP_NUM_THREADS"] = str(fiziksel_cekirdek)
            os.environ["MKL_NUM_THREADS"] = str(fiziksel_cekirdek)
            return fiziksel_cekirdek
    except ImportError:
        pass
    return None


def _directml_ortamini_hazirla(hedef_cihaz):
    """DirectML secilmisse kullaniciyi bilgilendir. Egitim CPU'da calisacak.
    DirectML su an yalnizca cikarim (inference) icin kullanilabilir.
    Ultralytics YOLO/RT-DETR egitimi DirectML'i desteklemez.

    Returns:
        tuple: (ultralytics_device_str, directml_device_or_none, bilgi_mesaji)
    """
    if hedef_cihaz != "directml":
        return hedef_cihaz, None, None

    dml = _directml_cihazini_al()
    gpu_adi = "DirectML GPU"
    if dml is not None:
        try:
            import torch_directml
            gpu_adi = torch_directml.device_name(0)
        except Exception:
            pass

    print(f"{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  DirectML GPU Kullanimi Hakkinda{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Tespit edilen GPU: {Fore.CYAN}{gpu_adi}{Style.RESET_ALL}")
    print()
    if dml is not None:
        print(f"  {Fore.GREEN}[+] DirectML GPU, cikarim (inference) islemlerinde kullanilacak.{Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}[!] DirectML paketi yuklu degil. Kurmak icin:{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}    pip install torch_directml{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}[!] Ultralytics YOLO/RT-DETR egitimi DirectML ile calismaz.{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}[!] Egitim CPU uzerinde gerceklesecek.{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}[*] Onerilen alternatifler:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}    - Google Colab (Menu'de [3] secenegi): Ucretsiz NVIDIA T4 GPU{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}    - NVIDIA GPU: CUDA ile tam GPU egitimi{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}[*] Cikarim (Menu [6]) icin DirectML veya OpenVINO GPU kullanilir.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}")
    print()

    # Intel CPU optimizasyonu
    cekirdek = _intel_cpu_optimizasyonu_uygula()
    if cekirdek:
        print(f"{Fore.GREEN}[+] Intel CPU optimizasyonu: {cekirdek} fiziksel cekirdek kullanilacak.{Style.RESET_ALL}")
        print()

    msg = f"DirectML GPU ({gpu_adi}) - Egitim CPU, Cikarim GPU"
    return "cpu", dml, msg


def egitim_baslat(epoch_sayisi=None, batch_size=None, cihaz=None, img_size=None, fl_gamma=None):
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

    # --- DirectML tespiti ve aktivasyonu ---
    ultralytics_cihaz, dml_cihaz, dml_mesaji = _directml_ortamini_hazirla(hedef_cihaz)
    if dml_mesaji:
        print(f"{Fore.GREEN}[+] {dml_mesaji}{Style.RESET_ALL}")
        hedef_cihaz_str = "directml"
    else:
        hedef_cihaz_str = ultralytics_cihaz

    if hedef_cihaz_str == "colab":
        print(f"{Fore.CYAN}[+] Google Colab ile egitim secildi.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Adimlar:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}    1. notebooks/hades_colab_egitim.ipynb dosyasini Google Drive'a yukleyin{Style.RESET_ALL}")
        print(f"{Fore.WHITE}    2. Google Colab'da acin (colab.research.google.com){Style.RESET_ALL}")
        print(f"{Fore.WHITE}    3. Runtime > Change runtime type > T4 GPU secin{Style.RESET_ALL}")
        print(f"{Fore.WHITE}    4. Veri setinizi Drive'a yukleyip hucreleri sirayla calistirin{Style.RESET_ALL}")
        print(f"{Fore.WHITE}    5. Egitim bitince best.pt dosyasini indirip runs/train/hades_egitim/weights/ klasorune koyun{Style.RESET_ALL}")
        print()
        print(f"{Fore.GREEN}[+] Colab linki: https://colab.research.google.com/{Style.RESET_ALL}")
        return True

    veri_seti_yolu = VERI_KOKU / "dataset.yaml"
    if not veri_seti_yolu.exists():
        print(f"{Fore.RED}[-] Veri seti yapilandirmasi bulunamadi: {veri_seti_yolu}{Style.RESET_ALL}")
        return False

    train_klasoru = VERI_KOKU / "images" / "train"
    if not train_klasoru.exists() or not any(train_klasoru.iterdir()):
        train_klasoru = VERI_KOKU / "train" / "images"
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
    print(f"    {Fore.WHITE}Cihaz          : {hedef_cihaz_str}{Style.RESET_ALL}")
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
    fl_gamma = fl_gamma if fl_gamma is not None else egitim_ayari.get("fl_gamma", 0.0)

    print(f"{Fore.BLUE}[*] Egitim basliyor...{Style.RESET_ALL}")
    print()

    try:
        sonuclar = model.train(
            data=str(veri_seti_yolu),
            epochs=hedef_epoch,
            batch=hedef_batch,
            imgsz=hedef_img_size,
            device=ultralytics_cihaz,
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

        # Otomatik OpenVINO / ONNX export
        _egitim_sonrasi_export()

        return True
    except Exception as hata:
        print(f"{Fore.RED}[-] Egitim sirasinda hata: {hata}{Style.RESET_ALL}")
        return False


def _egitim_sonrasi_export():
    """Egitim tamamlandiktan sonra, sistemdeki GPU/NPU'ya uygun
    optimize edilmis formatta model export'u yapar.

    Oncelik sirasi:
      1. Intel Arc / Intel NPU varsa → OpenVINO
      2. NVIDIA CUDA varsa → TensorRT (deneysel)
      3. AMD / diger GPU varsa → ONNX
      4. Sadece CPU → ONNX
    """
    from src.hardware_check import donanim_profili_olustur, directml_bilgisi_al

    best_pt = EGITIM_KOKU / "hades_egitim" / "weights" / "best.pt"
    if not best_pt.exists():
        return  # Egitim basarisiz, export yapilamaz

    profil = donanim_profili_olustur()
    cuda = profil.get("cuda", {})
    intel_arc = profil.get("intel_arc_gpu", [])
    npu = profil.get("npu", [])
    dml = directml_bilgisi_al()

    export_formati = None
    export_nedeni = ""

    # Intel GPU/NPU sistemi → OpenVINO en iyi secim
    if intel_arc or npu:
        export_formati = "openvino"
        export_nedeni = "Intel Arc GPU / NPU tespit edildi"
    elif cuda.get("durum"):
        export_formati = "engine"
        export_nedeni = "NVIDIA CUDA GPU tespit edildi"
    elif dml.get("durum"):
        export_formati = "onnx"
        export_nedeni = "DirectML uyumlu GPU tespit edildi"
    else:
        export_formati = "onnx"
        export_nedeni = "Genel amacli (CPU)"

    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Otomatik Model Export{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Neden: {export_nedeni}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Format: {export_formati.upper()}{Style.RESET_ALL}")
    print()

    try:
        from ultralytics import YOLO, RTDETR
        model = YOLO(str(best_pt))
        model.export(format=export_formati)
        print(f"{Fore.GREEN}[+] Model {export_formati.upper()} formatina export edildi.{Style.RESET_ALL}")
        ov_klasor = best_pt.parent / "best_openvino_model"
        if export_formati == "openvino" and ov_klasor.exists():
            print(f"{Fore.GREEN}[+] OpenVINO modeli hazir: {ov_klasor}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[*] Menu [6] Hasar Tespiti artik GPU uzerinde calisacak.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}[!] Otomatik export basarisiz: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[!] Manuel export: python src/export.py {export_formati}{Style.RESET_ALL}")

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


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


def model_bilgisi_goster():
    import datetime
    egitim_klasoru = EGITIM_KOKU / "hades_egitim"
    if not egitim_klasoru.exists():
        print(f"{Fore.YELLOW}[!] Henuz egitim yapilmamis.{Style.RESET_ALL}")
        return False

    args_yolu = egitim_klasoru / "args.yaml"
    sonuclar_yolu = egitim_klasoru / "results.csv"
    best_pt = egitim_klasoru / "weights" / "best.pt"

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Model Bilgileri{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    if best_pt.exists():
        boyut_mb = best_pt.stat().st_size / (1024 * 1024)
        tarih = datetime.datetime.fromtimestamp(best_pt.stat().st_mtime)
        print(f"{Fore.GREEN}[+] Son egitim: {tarih.strftime('%d.%m.%Y %H:%M')}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}    Model dosyasi: best.pt ({boyut_mb:.1f} MB){Style.RESET_ALL}")

    if args_yolu.exists():
        with open(args_yolu, "r", encoding="utf-8") as dosya:
            args = yaml.safe_load(dosya)
        print()
        print(f"{Fore.YELLOW}[*] Model Yapilandirmasi{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Model          : {args.get('model', 'Bilinmiyor')}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Epoch          : {args.get('epochs', 'Bilinmiyor')}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Batch Size     : {args.get('batch', 'Bilinmiyor')}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Img Size       : {args.get('imgsz', 'Bilinmiyor')}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Cihaz          : {args.get('device', 'Bilinmiyor')}{Style.RESET_ALL}")

    if sonuclar_yolu.exists():
        import csv
        with open(sonuclar_yolu, "r", encoding="utf-8") as dosya:
            okuyucu = csv.DictReader(dosya)
            satirlar = list(okuyucu)

        if satirlar:
            en_iyi_satir = satirlar[0]
            en_iyi_map = 0
            for satir in satirlar:
                try:
                    map_degeri = float(satir.get("metrics/mAP50(B)", "0").strip())
                    if map_degeri > en_iyi_map:
                        en_iyi_map = map_degeri
                        en_iyi_satir = satir
                except (ValueError, KeyError):
                    pass

            son_satir = satirlar[-1]

            metrik_anahtarlari = [
                ("metrics/precision(B)", "Precision"),
                ("metrics/recall(B)", "Recall"),
                ("metrics/mAP50(B)", "mAP50"),
                ("metrics/mAP50-95(B)", "mAP50-95"),
            ]

            print()
            print(f"{Fore.YELLOW}[*] En Iyi Dogruluk (En Yuksek mAP50){Style.RESET_ALL}")
            for anahtar, etiket in metrik_anahtarlari:
                if anahtar in en_iyi_satir:
                    deger = float(en_iyi_satir[anahtar].strip())
                    print(f"    {Fore.GREEN}{etiket:15s}: {deger:.4f}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}Toplam Epoch   : {len(satirlar)}{Style.RESET_ALL}")

            print()
            print(f"{Fore.YELLOW}[*] Son Epoch Sonuclari{Style.RESET_ALL}")
            for anahtar, etiket in metrik_anahtarlari:
                if anahtar in son_satir:
                    deger = float(son_satir[anahtar].strip())
                    print(f"    {Fore.WHITE}{etiket:15s}: {deger:.4f}{Style.RESET_ALL}")

    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    return True


if __name__ == "__main__":
    secim = sys.argv[1] if len(sys.argv) > 1 else ""
    if secim == "egit":
        egitim_baslat()
    elif secim == "rapor":
        egitim_raporu_goster()
    elif secim == "bilgi":
        model_bilgisi_goster()
    else:
        print(f"{Fore.YELLOW}Kullanim: python train.py [egit|rapor|bilgi]{Style.RESET_ALL}")