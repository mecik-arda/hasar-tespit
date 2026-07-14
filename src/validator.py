import sys
import os
from pathlib import Path
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent.parent
YAPILANDIRMA_YOLU = PROJE_KOKU / "config.yaml"


def yapilandirma_yukle():
    import yaml
    with open(YAPILANDIRMA_YOLU, "r", encoding="utf-8") as dosya:
        return yaml.safe_load(dosya)


def etiket_format_kontrolu(etiket_klasoru):
    """Her .txt dosyasindaki satirlarin gecerli YOLO formatinda olup olmadigini kontrol eder."""
    hatali = []
    txt_dosyalari = sorted(etiket_klasoru.glob("*.txt"))
    for txt_yolu in txt_dosyalari:
        with open(txt_yolu, "r", encoding="utf-8") as dosya:
            satirlar = dosya.readlines()
        for satir_no, satir in enumerate(satirlar, 1):
            satir = satir.strip()
            if not satir:
                continue
            parcalar = satir.split()
            if len(parcalar) != 5:
                hatali.append({
                    "dosya": str(txt_yolu.name),
                    "satir": satir_no,
                    "sebep": f"5 deger bekleniyor, {len(parcalar)} bulundu",
                })
                continue
            try:
                degerler = [float(p) for p in parcalar]
            except ValueError:
                hatali.append({
                    "dosya": str(txt_yolu.name),
                    "satir": satir_no,
                    "sebep": "Sayisal olmayan deger",
                })
    return hatali


def etiket_sinir_kontrolu(etiket_klasoru):
    """Tum degerlerin 0.0-1.0 araliginda olup olmadigini kontrol eder."""
    hatali = []
    txt_dosyalari = sorted(etiket_klasoru.glob("*.txt"))
    for txt_yolu in txt_dosyalari:
        with open(txt_yolu, "r", encoding="utf-8") as dosya:
            satirlar = dosya.readlines()
        for satir_no, satir in enumerate(satirlar, 1):
            satir = satir.strip()
            if not satir:
                continue
            parcalar = satir.split()
            if len(parcalar) != 5:
                continue
            try:
                sinif_id = int(parcalar[0])
                x, y, w, h = float(parcalar[1]), float(parcalar[2]), float(parcalar[3]), float(parcalar[4])
            except ValueError:
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                hatali.append({
                    "dosya": str(txt_yolu.name),
                    "satir": satir_no,
                    "sebep": f"Degerler 0.0-1.0 disinda: x={x}, y={y}, w={w}, h={h}",
                })
    return hatali


def etiket_sinif_kontrolu(etiket_klasoru, sinif_sayisi):
    """Sinif ID'lerinin gecerli aralikta olup olmadigini kontrol eder."""
    hatali = []
    txt_dosyalari = sorted(etiket_klasoru.glob("*.txt"))
    for txt_yolu in txt_dosyalari:
        with open(txt_yolu, "r", encoding="utf-8") as dosya:
            satirlar = dosya.readlines()
        for satir_no, satir in enumerate(satirlar, 1):
            satir = satir.strip()
            if not satir:
                continue
            parcalar = satir.split()
            if len(parcalar) != 5:
                continue
            try:
                sinif_id = int(parcalar[0])
            except ValueError:
                continue
            if sinif_id < 0 or sinif_id >= sinif_sayisi:
                hatali.append({
                    "dosya": str(txt_yolu.name),
                    "satir": satir_no,
                    "sebep": f"Gecersiz sinif ID: {sinif_id} (0-{sinif_sayisi - 1} bekleniyor)",
                })
    return hatali


def etiket_boyut_kontrolu(etiket_klasoru, min_boyut=0.01, max_boyut=0.95):
    """Bounding box boyutlarinin cok kucuk veya cok buyuk olup olmadigini kontrol eder."""
    hatali = []
    txt_dosyalari = sorted(etiket_klasoru.glob("*.txt"))
    for txt_yolu in txt_dosyalari:
        with open(txt_yolu, "r", encoding="utf-8") as dosya:
            satirlar = dosya.readlines()
        for satir_no, satir in enumerate(satirlar, 1):
            satir = satir.strip()
            if not satir:
                continue
            parcalar = satir.split()
            if len(parcalar) != 5:
                continue
            try:
                w, h = float(parcalar[3]), float(parcalar[4])
            except ValueError:
                continue
            if w < min_boyut or h < min_boyut:
                hatali.append({
                    "dosya": str(txt_yolu.name),
                    "satir": satir_no,
                    "sebep": f"Kutu cok kucuk: w={w:.4f}, h={h:.4f} (min: {min_boyut})",
                })
            if w > max_boyut or h > max_boyut:
                hatali.append({
                    "dosya": str(txt_yolu.name),
                    "satir": satir_no,
                    "sebep": f"Kutu cok buyuk: w={w:.4f}, h={h:.4f} (max: {max_boyut})",
                })
    return hatali


def etiket_overlap_kontrolu(etiket_klasoru, esik=0.8):
    """Ayni gorseldeki kutularin %80'den fazla ortusup ortusmedigini kontrol eder."""
    hatali = []
    txt_dosyalari = sorted(etiket_klasoru.glob("*.txt"))
    for txt_yolu in txt_dosyalari:
        kutular = []
        with open(txt_yolu, "r", encoding="utf-8") as dosya:
            satirlar = dosya.readlines()
        for satir in satirlar:
            satir = satir.strip()
            if not satir:
                continue
            parcalar = satir.split()
            if len(parcalar) != 5:
                continue
            try:
                sinif_id = int(parcalar[0])
                x, y, w, h = float(parcalar[1]), float(parcalar[2]), float(parcalar[3]), float(parcalar[4])
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
                kutular.append((sinif_id, x1, y1, x2, y2))
            except ValueError:
                continue
        for i in range(len(kutular)):
            for j in range(i + 1, len(kutular)):
                s1, ax1, ay1, ax2, ay2 = kutular[i]
                s2, bx1, by1, bx2, by2 = kutular[j]
                kesisim_x1 = max(ax1, bx1)
                kesisim_y1 = max(ay1, by1)
                kesisim_x2 = min(ax2, bx2)
                kesisim_y2 = min(ay2, by2)
                if kesisim_x1 >= kesisim_x2 or kesisim_y1 >= kesisim_y2:
                    continue
                kesisim_alani = (kesisim_x2 - kesisim_x1) * (kesisim_y2 - kesisim_y1)
                alan_a = (ax2 - ax1) * (ay2 - ay1)
                alan_b = (bx2 - bx1) * (by2 - by1)
                kucuk_alan = min(alan_a, alan_b)
                if kucuk_alan > 0 and (kesisim_alani / kucuk_alan) > esik:
                    hatali.append({
                        "dosya": str(txt_yolu.name),
                        "satir": f"sinif {s1} <-> sinif {s2}",
                        "sebep": f"Kutular %{kesisim_alani / kucuk_alan * 100:.1f} oraninda ortusuyor",
                    })
    return hatali


def etiket_eslesme_kontrolu(gorsel_klasoru, etiket_klasoru):
    """Her gorselin etiketi, her etiketin gorseli var mi kontrol eder."""
    gorsel_uzantilari = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    etiketsiz = []
    gorselsiz = []

    gorseller = {f.stem: f for f in gorsel_klasoru.iterdir() if f.suffix.lower() in gorsel_uzantilari}
    etiketler = {f.stem: f for f in etiket_klasoru.iterdir() if f.suffix == ".txt"}

    for ad in gorseller:
        if ad not in etiketler:
            etiketsiz.append(f"{ad} (gorsel var, etiket yok)")

    for ad in etiketler:
        if ad not in gorseller:
            gorselsiz.append(f"{ad} (etiket var, gorsel yok)")

    return {"etiketsiz": etiketsiz, "gorselsiz": gorselsiz}


def etiket_dagilim_raporu(etiket_klasoru, siniflar):
    """Her siniftan kacar etiket oldugunu hesaplar."""
    sayac = {siniflar.get(i, f"Sinif_{i}"): 0 for i in range(len(siniflar))}
    toplam = 0

    txt_dosyalari = sorted(etiket_klasoru.glob("*.txt"))
    for txt_yolu in txt_dosyalari:
        with open(txt_yolu, "r", encoding="utf-8") as dosya:
            for satir in dosya:
                satir = satir.strip()
                if not satir:
                    continue
                parcalar = satir.split()
                if len(parcalar) != 5:
                    continue
                try:
                    sinif_id = int(parcalar[0])
                    sinif_adi = siniflar.get(sinif_id, f"Sinif_{sinif_id}")
                    sayac[sinif_adi] = sayac.get(sinif_adi, 0) + 1
                    toplam += 1
                except ValueError:
                    continue
    return sayac, toplam


def etiket_validator_calistir(klasor=None):
    """Ana validator fonksiyonu. Belirtilen klasordeki etiketleri tum kontrollerden gecirir."""
    yapilandirma = yapilandirma_yukle()
    siniflar = yapilandirma.get("siniflar", {})

    if klasor is None:
        hedef = PROJE_KOKU / "hasar-ornek"
    else:
        hedef = Path(klasor)

    etiket_klasoru = hedef
    gorsel_klasoru = hedef

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Etiket Tutarlilik Denetimi{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()
    print(f"{Fore.YELLOW}[*] Kontrol edilen klasor: {hedef}{Style.RESET_ALL}")
    print()

    tum_hatalar = []

    print(f"{Fore.YELLOW}[1/7] Format kontrolu...{Style.RESET_ALL}")
    hatalar = etiket_format_kontrolu(etiket_klasoru)
    tum_hatalar.extend(hatalar)
    if hatalar:
        print(f"    {Fore.RED}[-] {len(hatalar)} format hatasi bulundu{Style.RESET_ALL}")
    else:
        print(f"    {Fore.GREEN}[+] Tum etiketler gecerli YOLO formatinda{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}[2/7] Sinir kontrolu (0.0-1.0)...{Style.RESET_ALL}")
    hatalar = etiket_sinir_kontrolu(etiket_klasoru)
    tum_hatalar.extend(hatalar)
    if hatalar:
        print(f"    {Fore.RED}[-] {len(hatalar)} sinir disi deger bulundu{Style.RESET_ALL}")
    else:
        print(f"    {Fore.GREEN}[+] Tum degerler 0.0-1.0 araliginda{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}[3/7] Sinif ID kontrolu...{Style.RESET_ALL}")
    hatalar = etiket_sinif_kontrolu(etiket_klasoru, len(siniflar))
    tum_hatalar.extend(hatalar)
    if hatalar:
        print(f"    {Fore.RED}[-] {len(hatalar)} gecersiz sinif ID'si bulundu{Style.RESET_ALL}")
    else:
        print(f"    {Fore.GREEN}[+] Tum sinif ID'leri gecerli ({len(siniflar)} sinif){Style.RESET_ALL}")

    print(f"{Fore.YELLOW}[4/7] Kutu boyut kontrolu...{Style.RESET_ALL}")
    hatalar = etiket_boyut_kontrolu(etiket_klasoru)
    tum_hatalar.extend(hatalar)
    if hatalar:
        print(f"    {Fore.RED}[-] {len(hatalar)} anormal boyutlu kutu bulundu{Style.RESET_ALL}")
    else:
        print(f"    {Fore.GREEN}[+] Tum kutu boyutlari normal{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}[5/7] Overlap kontrolu (>%80)...{Style.RESET_ALL}")
    hatalar = etiket_overlap_kontrolu(etiket_klasoru)
    tum_hatalar.extend(hatalar)
    if hatalar:
        print(f"    {Fore.RED}[-] {len(hatalar)} asiri ortusen kutu bulundu{Style.RESET_ALL}")
    else:
        print(f"    {Fore.GREEN}[+] Asiri ortusen kutu yok{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}[6/7] Eslesme kontrolu...{Style.RESET_ALL}")
    eslesme = etiket_eslesme_kontrolu(gorsel_klasoru, etiket_klasoru)
    if eslesme["etiketsiz"]:
        print(f"    {Fore.RED}[-] {len(eslesme['etiketsiz'])} etiketsiz gorsel bulundu{Style.RESET_ALL}")
    if eslesme["gorselsiz"]:
        print(f"    {Fore.RED}[-] {len(eslesme['gorselsiz'])} gorselsiz etiket bulundu{Style.RESET_ALL}")
    if not eslesme["etiketsiz"] and not eslesme["gorselsiz"]:
        print(f"    {Fore.GREEN}[+] Tum gorsel-etiket eslesmeleri tam{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}[7/7] Sinif dagilimi...{Style.RESET_ALL}")
    dagilim, toplam = etiket_dagilim_raporu(etiket_klasoru, siniflar)
    for sinif_adi, sayi in dagilim.items():
        oran = f"%{sayi / toplam * 100:.1f}" if toplam > 0 else "%0"
        bar = "#" * max(1, int(sayi / max(1, toplam) * 30))
        renk = Fore.GREEN if sayi > 0 else Fore.RED
        print(f"    {renk}{sinif_adi:15s}: {sayi:4d} ({oran})  {bar}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}{'TOPLAM':15s}: {toplam:4d}{Style.RESET_ALL}")

    print()
    if tum_hatalar:
        print(f"{Fore.RED}[-] Toplam {len(tum_hatalar)} sorun bulundu.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Detayli hatalar:{Style.RESET_ALL}")
        for h in tum_hatalar[:20]:
            print(f"    {Fore.WHITE}{h['dosya']}:{h.get('satir', '?')}  -> {h['sebep']}{Style.RESET_ALL}")
        if len(tum_hatalar) > 20:
            print(f"    {Fore.YELLOW}... ve {len(tum_hatalar) - 20} daha fazla{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}[+] Harika! Hicbir sorun bulunamadi. Etiketler temiz.{Style.RESET_ALL}")

    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return {"hatalar": tum_hatalar, "eslesme": eslesme, "dagilim": dagilim, "toplam": toplam}


if __name__ == "__main__":
    etiket_validator_calistir()
