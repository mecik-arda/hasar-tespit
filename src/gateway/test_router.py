"""Akıllı Yönlendirici (AI Router) manuel test scripti.

Bu script, farklı tiplerde görselleri AIRouter'a göndererek sistemin
çöp filtresi ve kanal yönlendirmesinin nasıl çalıştığını simüle eder.

Çalıştırmak için:
    python src/gateway/test_router.py
"""

import os
import sys
import random
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJE_KOKU = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from colorama import Fore, Style, init
from src.gateway.ai_router import AIRouter

init()


def test_router_calistir():
    """AIRouter'ı farklı görsel tipleriyle test eder ve logları ekrana basar.

    Test, hasar-ornek klasöründeki görselleri kullanır. Eğer klasörde görsel
    yoksa, bilgilendirme mesajı gösterilir.
    """
    ornek_klasoru = PROJE_KOKU / "hasar-ornek"
    gorsel_uzantilari = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES - Akıllı Yönlendirici Manuel Testi{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    if not ornek_klasoru.exists():
        print(f"{Fore.RED}[-] hasar-ornek klasörü bulunamadı: {ornek_klasoru}{Style.RESET_ALL}")
        print()
        return False

    mevcut_gorseller = sorted(
        [f for f in ornek_klasoru.iterdir() if f.suffix.lower() in gorsel_uzantilari]
    )

    if not mevcut_gorseller:
        print(f"{Fore.RED}[-] hasar-ornek klasöründe test edilecek görsel bulunamadı.{Style.RESET_ALL}")
        print()
        return False

    test_edilecek_gorseller = random.sample(mevcut_gorseller, min(4, len(mevcut_gorseller)))

    print(f"{Fore.YELLOW}[*] Test edilecek {len(test_edilecek_gorseller)} görsel seçildi:{Style.RESET_ALL}")
    for gorsel in test_edilecek_gorseller:
        print(f"    {Fore.WHITE}- {gorsel.name}{Style.RESET_ALL}")
    print()
    print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
    print()

    router = AIRouter()

    kabul_edilen = 0
    reddedilen = 0
    yolo_yonlenen = 0
    rtdetr_yonlenen = 0

    for i, gorsel_yolu in enumerate(test_edilecek_gorseller, 1):
        print(f"{Fore.YELLOW}[TEST {i}/{len(test_edilecek_gorseller)}] {gorsel_yolu.name}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─' * 40}{Style.RESET_ALL}")

        sonuc = router.process_image(gorsel_yolu)

        if sonuc["status"] == "accepted":
            kabul_edilen += 1
            if sonuc["route_to"] == "YOLO":
                yolo_yonlenen += 1
            elif sonuc["route_to"] == "RT-DETR":
                rtdetr_yonlenen += 1
            print(f"    {Fore.GREEN}[✓] KABUL EDİLDİ{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}Kanal     : {Fore.YELLOW}{sonuc['route_to']}{Style.RESET_ALL}")
        else:
            reddedilen += 1
            print(f"    {Fore.RED}[✗] REDDEDİLDİ{Style.RESET_ALL}")

        print(f"    {Fore.WHITE}Güven     : %{sonuc['confidence']*100:.1f}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}CLIP Aktif: {'Evet' if sonuc['clip_aktif'] else 'Hayır (yedek mod)'}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Sebep     : {sonuc['sebep']}{Style.RESET_ALL}")
        print()

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  TEST ÖZETİ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Toplam Test    : {len(test_edilecek_gorseller)}{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Kabul Edilen   : {kabul_edilen}{Style.RESET_ALL}")
    print(f"  {Fore.RED}Reddedilen     : {reddedilen}{Style.RESET_ALL}")
    print(f"  {Fore.BLUE}YOLO Kanalı    : {yolo_yonlenen}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}RT-DETR Kanalı : {rtdetr_yonlenen}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    return True


if __name__ == "__main__":
    test_router_calistir()