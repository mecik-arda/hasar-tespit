import sys
from pathlib import Path
from colorama import Fore, Style, init

from src.utils import PROJE_KOKU, EGITIM_KOKU, yapilandirma_yukle

init()


def model_dışa_aktar(format="onnx"):
    from ultralytics import YOLO
    from src.hardware_check import donanim_profili_olustur

    egitim_klasoru = EGITIM_KOKU / "hades_egitim"
    en_iyi_agirlik = egitim_klasoru / "weights" / "best.pt"

    if not en_iyi_agirlik.exists():
        print(f"{Fore.RED}[-] Egitilmis model bulunamadi: {en_iyi_agirlik}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Once model egitimi yapin.{Style.RESET_ALL}")
        return False

    donanim_profili = donanim_profili_olustur()
    hedef_cihaz = donanim_profili["hedef_cihaz"]

    gecerli_formatlar = ["onnx", "engine", "openvino", "coreml", "tflite", "pb", "torchscript"]

    if format not in gecerli_formatlar:
        print(f"{Fore.RED}[-] Gecersiz format: {format}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Gecerli formatlar: {', '.join(gecerli_formatlar)}{Style.RESET_ALL}")
        return False

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Model Disa Aktarimi{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] Aktarim Yapilandirmasi{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Kaynak Model   : {en_iyi_agirlik}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Hedef Format    : {format}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Cihaz           : {hedef_cihaz}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    try:
        model = YOLO(str(en_iyi_agirlik))
    except Exception as hata:
        print(f"{Fore.RED}[-] Model yuklenemedi: {hata}{Style.RESET_ALL}")
        return False

    try:
        print(f"{Fore.BLUE}[*] Disa aktarim basliyor...{Style.RESET_ALL}")
        model.export(format=format)
        print()
        print(f"{Fore.GREEN}[+] Disa aktarim tamamlandi!{Style.RESET_ALL}")
        cikti_yolu = en_iyi_agirlik.with_suffix(f".{format}")
        if format == "engine":
            cikti_yolu = en_iyi_agirlik.with_suffix(".engine")
        elif format == "torchscript":
            cikti_yolu = egitim_klasoru / "weights" / "best.torchscript"
        print(f"{Fore.WHITE}    Cikti: {cikti_yolu}{Style.RESET_ALL}")
        return True
    except Exception as hata:
        print(f"{Fore.RED}[-] Disa aktarim sirasinda hata: {hata}{Style.RESET_ALL}")
        return False


def optimize_edilmis_model_olustur():
    from src.hardware_check import donanim_profili_olustur

    donanim_profili = donanim_profili_olustur()
    hedef_cihaz = donanim_profili["hedef_cihaz"]

    if hedef_cihaz == "cuda":
        print(f"{Fore.BLUE}[*] NVIDIA GPU tespit edildi. TensorRT formatina aktariliy...{Style.RESET_ALL}")
        return model_dışa_aktar(format="engine")
    elif hedef_cihaz == "cpu":
        import platform
        isletim_sistemi = platform.system()
        if isletim_sistemi == "Windows":
            print(f"{Fore.BLUE}[*] Intel/AMD CPU tespit edildi. OpenVINO formatina aktariliy...{Style.RESET_ALL}")
            return model_dışa_aktar(format="openvino")
        else:
            print(f"{Fore.BLUE}[*] CPU tespit edildi. ONNX formatina aktariliy...{Style.RESET_ALL}")
            return model_dışa_aktar(format="onnx")
    else:
        print(f"{Fore.BLUE}[*] ONNX formatina aktariliy...{Style.RESET_ALL}")
        return model_dışa_aktar(format="onnx")


if __name__ == "__main__":
    secim = sys.argv[1] if len(sys.argv) > 1 else ""
    if secim == "onnx":
        model_dışa_aktar(format="onnx")
    elif secim == "engine":
        model_dışa_aktar(format="engine")
    elif secim == "openvino":
        model_dışa_aktar(format="openvino")
    elif secim == "optimize":
        optimize_edilmis_model_olustur()
    else:
        print(f"{Fore.YELLOW}Kullanim: python export.py [onnx|engine|openvino|optimize]{Style.RESET_ALL}")