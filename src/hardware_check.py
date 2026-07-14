import psutil
import cpuinfo
import torch
import subprocess
import platform
from colorama import Fore, Style, init

init()


def cpu_bilgisi_al():
    bilgi = cpuinfo.get_cpu_info()
    cpu_adi = bilgi.get("brand_raw", "Bilinmeyen CPU")
    cekirdek_sayisi = psutil.cpu_count(logical=False)
    mantiksal_cekirdek = psutil.cpu_count(logical=True)
    frekans = psutil.cpu_freq()
    frekans_mhz = frekans.current if frekans else 0
    return {
        "ad": cpu_adi,
        "cekirdek": cekirdek_sayisi,
        "mantiksal_cekirdek": mantiksal_cekirdek,
        "frekans_mhz": frekans_mhz,
    }


def ram_bilgisi_al():
    ram = psutil.virtual_memory()
    toplam_gb = ram.total / (1024 ** 3)
    kullanilan_gb = ram.used / (1024 ** 3)
    yuzde = ram.percent
    return {
        "toplam_gb": toplam_gb,
        "kullanilan_gb": kullanilan_gb,
        "yuzde": yuzde,
    }


def nvidia_gpu_bilgisi_al():
    try:
        cikti = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).decode("utf-8").strip()
        satirlar = cikti.split("\n")
        gpu_listesi = []
        for satir in satirlar:
            parcalar = [p.strip() for p in satir.split(",")]
            if len(parcalar) >= 3:
                gpu_listesi.append({
                    "ad": parcalar[0],
                    "vram_mb": int(parcalar[1]),
                    "surucu": parcalar[2],
                })
        return gpu_listesi
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []


def torch_cuda_bilgisi_al():
    cuda_durumu = torch.cuda.is_available()
    if not cuda_durumu:
        return {"durum": False, "sayac": 0, "cihazlar": []}
    sayac = torch.cuda.device_count()
    cihazlar = []
    for i in range(sayac):
        cihazlar.append({
            "ad": torch.cuda.get_device_name(i),
            "vram_gb": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),
        })
    return {"durum": True, "sayac": sayac, "cihazlar": cihazlar}


def wmic_gpu_bilgisi_al(marka_filtreleri):
    if platform.system() != "Windows":
        return []
    try:
        cikti = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM", "/format:csv"],
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).decode("utf-8", errors="ignore").strip()
        gpu_listesi = []
        satirlar = cikti.split("\n")
        for satir in satirlar[1:]:
            parcalar = satir.split(",")
            if len(parcalar) >= 3:
                ad = parcalar[2].strip()
                if ad and any(marka in ad for marka in marka_filtreleri):
                    try:
                        vram_mb = int(parcalar[1]) // (1024 * 1024)
                    except (ValueError, IndexError):
                        vram_mb = 0
                    gpu_listesi.append({"ad": ad, "vram_mb": vram_mb})
        return gpu_listesi
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def amd_gpu_bilgisi_al():
    return wmic_gpu_bilgisi_al(["AMD", "Radeon"])


def intel_gpu_bilgisi_al():
    return wmic_gpu_bilgisi_al(["Intel", "Arc", "Iris", "UHD"])


def donanim_profili_olustur():
    cpu = cpu_bilgisi_al()
    ram = ram_bilgisi_al()
    nvidia = nvidia_gpu_bilgisi_al()
    cuda = torch_cuda_bilgisi_al()
    amd = amd_gpu_bilgisi_al()
    intel = intel_gpu_bilgisi_al()

    if cuda["durum"] and cuda["sayac"] > 0:
        hedef_cihaz = "cuda"
        onerilen_batch = 16
    elif nvidia:
        hedef_cihaz = "cuda"
        onerilen_batch = 16
    else:
        hedef_cihaz = "cpu"
        onerilen_batch = 4

    return {
        "cpu": cpu,
        "ram": ram,
        "nvidia_gpu": nvidia,
        "cuda": cuda,
        "amd_gpu": amd,
        "intel_gpu": intel,
        "hedef_cihaz": hedef_cihaz,
        "onerilen_batch": onerilen_batch,
    }


def donanim_ozeti_yazdir():
    profil = donanim_profili_olustur()
    cpu = profil["cpu"]
    ram = profil["ram"]

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES DETECTOR - Donanim Analizi{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    print(f"{Fore.YELLOW}[*] CPU Bilgileri{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Model          : {cpu['ad']}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Cekirdek       : {cpu['cekirdek']} Fiziksel / {cpu['mantiksal_cekirdek']} Mantiksal{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Frekans        : {cpu['frekans_mhz']:.0f} MHz{Style.RESET_ALL}")
    print()

    print(f"{Fore.YELLOW}[*] RAM Bilgileri{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Toplam         : {ram['toplam_gb']:.2f} GB{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Kullanilan     : {ram['kullanilan_gb']:.2f} GB (%{ram['yuzde']:.1f}){Style.RESET_ALL}")
    print()

    if profil["cuda"]["durum"]:
        print(f"{Fore.GREEN}[+] CUDA Destegi: Aktif{Style.RESET_ALL}")
        for i, cihaz in enumerate(profil["cuda"]["cihazlar"]):
            print(f"    {Fore.WHITE}GPU {i}          : {cihaz['ad']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}VRAM           : {cihaz['vram_gb']:.2f} GB{Style.RESET_ALL}")
        print()
    elif profil["nvidia_gpu"]:
        print(f"{Fore.GREEN}[+] NVIDIA GPU Tespit Edildi{Style.RESET_ALL}")
        for gpu in profil["nvidia_gpu"]:
            print(f"    {Fore.WHITE}Model          : {gpu['ad']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}VRAM           : {gpu['vram_mb']} MB{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}Surucu         : {gpu['surucu']}{Style.RESET_ALL}")
        print()
    else:
        print(f"{Fore.RED}[-] NVIDIA/CUDA GPU bulunamadi.{Style.RESET_ALL}")
        print()

    if profil["amd_gpu"]:
        print(f"{Fore.BLUE}[*] AMD GPU Tespit Edildi{Style.RESET_ALL}")
        for gpu in profil["amd_gpu"]:
            print(f"    {Fore.WHITE}Model          : {gpu['ad']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}VRAM           : {gpu['vram_mb']} MB{Style.RESET_ALL}")
        print()

    if profil["intel_gpu"]:
        print(f"{Fore.BLUE}[*] Intel GPU Tespit Edildi{Style.RESET_ALL}")
        for gpu in profil["intel_gpu"]:
            print(f"    {Fore.WHITE}Model          : {gpu['ad']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}VRAM           : {gpu['vram_mb']} MB{Style.RESET_ALL}")
        print()

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] Egitim Onerisi{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Hedef Cihaz    : {profil['hedef_cihaz'].upper()}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Onerilen Batch : {profil['onerilen_batch']}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return profil


if __name__ == "__main__":
    donanim_ozeti_yazdir()