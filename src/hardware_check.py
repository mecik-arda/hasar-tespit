import psutil
import cpuinfo
import torch
import subprocess
import platform
from colorama import Fore, Style, init

init()

PROJE_KOKU = __import__("pathlib").Path(__file__).parent.parent


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
                    "tur": "NVIDIA",
                    "tip": "Harici",
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


def wmic_gpu_bilgisi_al(marka_filtreleri, tur_etiketi):
    """Belirtilen marka filtrelerine uyan GPU'ları PowerShell ile tespit eder (WMIC fallback'li)."""
    if platform.system() != "Windows":
        return []
    gpu_listesi = []
    ham_cikti = None
    powershell_formati = True

    powershell_komutu = (
        'Get-CimInstance Win32_VideoController | '
        'Select-Object Name, AdapterRAM, DriverVersion | '
        'ConvertTo-Csv -NoTypeInformation'
    )
    try:
        ham_cikti = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", powershell_komutu],
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).decode("utf-8", errors="ignore").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if not ham_cikti:
        powershell_formati = False
        try:
            ham_cikti = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM,DriverVersion", "/format:csv"],
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ).decode("utf-8", errors="ignore").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    satirlar = ham_cikti.split("\n")
    for satir in satirlar[1:]:
        satir = satir.strip()
        if not satir or satir.startswith("#TYPE"):
            continue
        parcalar = [p.strip().strip('"') for p in satir.split(",")]

        if powershell_formati:
            if len(parcalar) < 3:
                continue
            ad = parcalar[0]
            vram_ham = parcalar[1]
            surucu = parcalar[2]
        else:
            if len(parcalar) < 4:
                continue
            ad = parcalar[1]
            vram_ham = parcalar[2]
            surucu = parcalar[3] if len(parcalar) > 3 else ""

        if ad and any(marka.lower() in ad.lower() for marka in marka_filtreleri):
            try:
                vram_mb = int(vram_ham) // (1024 * 1024) if vram_ham.strip() else 0
            except (ValueError, IndexError):
                vram_mb = 0
            surucu = surucu if surucu else "Bilinmiyor"
            gpu_tipi = "Harici" if vram_mb > 0 else "Entegre"
            gpu_listesi.append({
                "ad": ad,
                "vram_mb": vram_mb,
                "surucu": surucu,
                "tur": tur_etiketi,
                "tip": gpu_tipi,
            })
    return gpu_listesi


def amd_gpu_bilgisi_al():
    return wmic_gpu_bilgisi_al(["AMD", "Radeon"], "AMD")


def intel_gpu_bilgisi_al():
    """Tüm Intel GPU'larını tespit eder (Arc, Iris Xe, UHD vb.)."""
    return wmic_gpu_bilgisi_al(["Intel", "Arc", "Iris", "UHD"], "Intel")


def intel_arc_gpu_bilgisi_al():
    """Yalnızca Intel Arc (eğitim yapabilen) GPU'larını tespit eder."""
    return wmic_gpu_bilgisi_al(["Intel(R) Arc", "Intel Arc"], "Intel Arc")


def npu_bilgisi_al():
    """Intel AI Boost, AMD Ryzen AI gibi NPU'ları tespit eder."""
    npu_listesi = []

    if platform.system() != "Windows":
        return npu_listesi

    npu_anahtar_kelimeleri = [
        "Intel(R) AI Boost",
        "Intel AI Boost",
        "Neural Processing Unit",
        "Intel(R) Neural",
        "AMD IPU",
        "AMD Neural",
        "Qualcomm Neural",
        "Qualcomm Hexagon",
    ]

    npu_haric_kelimeleri = [
        "Microsoft Input",
        "HID",
        "Keyboard",
        "Mouse",
        "Touchpad",
        "Audio",
        "Speaker",
        "Camera",
        "Bluetooth",
        "Wi-Fi",
        "Storage",
        "Disk",
        "Battery",
        "ACPI",
        "USB",
    ]

    ham_cikti = None
    powershell_formati = True

    where_kosulu = " -or ".join(
        [f'$_.FriendlyName -like "*{k}*"' for k in npu_anahtar_kelimeleri]
    )
    powershell_komutu = (
        f'Get-PnpDevice | Where-Object {{ {where_kosulu} }} | '
        'Select-Object FriendlyName, Status | '
        'ConvertTo-Csv -NoTypeInformation'
    )
    try:
        ham_cikti = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", powershell_komutu],
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).decode("utf-8", errors="ignore").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if not ham_cikti:
        powershell_formati = False
        try:
            ham_cikti = subprocess.check_output(
                ["wmic", "path", "Win32_PnPEntity", "get", "Name,Status", "/format:csv"],
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ).decode("utf-8", errors="ignore").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return npu_listesi

    gorulen = set()
    satirlar = ham_cikti.split("\n")
    for satir in satirlar[1:]:
        satir = satir.strip()
        if not satir or satir.startswith("#TYPE"):
            continue
        parcalar = [p.strip().strip('"') for p in satir.split(",")]

        if powershell_formati:
            if len(parcalar) < 2:
                continue
            ad = parcalar[0]
            durum = parcalar[1]
        else:
            if len(parcalar) < 3:
                continue
            ad = parcalar[2]
            durum = parcalar[1]

        if ad and ad not in gorulen:
            haric_tut = any(haric.lower() in ad.lower() for haric in npu_haric_kelimeleri)
            if haric_tut:
                continue
            for anahtar in npu_anahtar_kelimeleri:
                if anahtar.lower() in ad.lower():
                    gorulen.add(ad)
                    if "intel" in ad.lower():
                        tur = "Intel NPU"
                    elif "amd" in ad.lower():
                        tur = "AMD NPU"
                    elif "qualcomm" in ad.lower() or "hexagon" in ad.lower():
                        tur = "Qualcomm NPU"
                    else:
                        tur = "NPU"
                    npu_listesi.append({
                        "ad": ad,
                        "durum": durum,
                        "tur": tur,
                    })
                    break

    return npu_listesi


def tum_gpu_bilgisi_al():
    """Sistemdeki tüm GPU'ları tek bir listede toplar."""
    tumu = []
    tumu.extend(nvidia_gpu_bilgisi_al())
    tumu.extend(amd_gpu_bilgisi_al())
    tumu.extend(intel_gpu_bilgisi_al())
    return tumu


def directml_bilgisi_al():
    """DirectML (Intel Arc / AMD / NVIDIA GPU) kullanilabilir mi?
    DirectML, DirectX 12 uzerinden GPU hizlandirmasi saglar.
    Intel Arc GPU'larda XPU'nun calismadigi durumlarda en iyi alternatiftir.
    """
    try:
        import torch_directml
        sayac = torch_directml.device_count()
        cihazlar = []
        for i in range(sayac):
            try:
                ad = torch_directml.device_name(i)
            except Exception:
                ad = f"DirectML GPU {i}"
            cihazlar.append({"ad": ad})
        return {
            "durum": True,
            "sayac": sayac,
            "cihazlar": cihazlar,
        }
    except ImportError:
        return {"durum": False, "sayac": 0, "cihazlar": []}
    except Exception:
        return {"durum": False, "sayac": 0, "cihazlar": []}


def egitim_yapabilir_gpu_var_mi(profil):
    """CUDA veya Intel Arc gibi eğitim yapabilecek GPU var mı kontrol eder."""
    cuda = profil.get("cuda", {})
    if cuda.get("durum") and cuda.get("sayac", 0) > 0:
        return True

    nvidia = profil.get("nvidia_gpu", [])
    if nvidia:
        return True

    intel_arc = profil.get("intel_arc_gpu", [])
    if intel_arc:
        return True

    amd_gpu = profil.get("amd_gpu", [])
    for gpu in amd_gpu:
        if "radeon" in gpu.get("ad", "").lower():
            vram = gpu.get("vram_mb", 0)
            if vram >= 2048:
                return True

    return False


def donanim_profili_olustur():
    cpu = cpu_bilgisi_al()
    ram = ram_bilgisi_al()
    nvidia = nvidia_gpu_bilgisi_al()
    cuda = torch_cuda_bilgisi_al()
    amd = amd_gpu_bilgisi_al()
    intel_gpu = intel_gpu_bilgisi_al()
    intel_arc = intel_arc_gpu_bilgisi_al()
    npu = npu_bilgisi_al()
    tum_gpu = tum_gpu_bilgisi_al()
    dml = directml_bilgisi_al()

    if cuda["durum"] and cuda["sayac"] > 0:
        hedef_cihaz = "cuda"
        onerilen_batch = 16
        cihaz_aciklamasi = "NVIDIA CUDA GPU"
    elif nvidia:
        hedef_cihaz = "cuda"
        onerilen_batch = 16
        cihaz_aciklamasi = "NVIDIA GPU (nvidia-smi)"
    elif dml["durum"] and dml["sayac"] > 0:
        dml_adi = dml["cihazlar"][0]["ad"] if dml["cihazlar"] else "DirectML GPU"
        hedef_cihaz = "directml"
        onerilen_batch = 16
        cihaz_aciklamasi = f"DirectML GPU - {dml_adi} (Intel Arc / AMD / NVIDIA)"
    elif intel_arc:
        hedef_cihaz = "cpu"
        onerilen_batch = 8
        cihaz_aciklamasi = "Intel Arc GPU (DirectML yuklu degil - 'pip install torch_directml' ile kurun)"
    elif amd and any("radeon" in g.get("ad", "").lower() and g.get("vram_mb", 0) >= 2048 for g in amd):
        hedef_cihaz = "cpu"
        onerilen_batch = 8
        cihaz_aciklamasi = "AMD Radeon GPU (DirectML / ROCm onerilir)"
    else:
        hedef_cihaz = "cpu"
        onerilen_batch = 4
        cihaz_aciklamasi = "Yalnizca CPU mevcut"

    return {
        "cpu": cpu,
        "ram": ram,
        "nvidia_gpu": nvidia,
        "cuda": cuda,
        "amd_gpu": amd,
        "intel_gpu": intel_gpu,
        "intel_arc_gpu": intel_arc,
        "npu": npu,
        "tum_gpu": tum_gpu,
        "directml": dml,
        "hedef_cihaz": hedef_cihaz,
        "onerilen_batch": onerilen_batch,
        "cihaz_aciklamasi": cihaz_aciklamasi,
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

    tum_gpu = profil.get("tum_gpu", [])
    if tum_gpu:
        print(f"{Fore.YELLOW}[*] GPU Bilgileri ({len(tum_gpu)} adet){Style.RESET_ALL}")
        for i, gpu in enumerate(tum_gpu):
            tur = gpu.get("tur", "GPU")
            tip = gpu.get("tip", "")
            vram_mb = gpu.get("vram_mb", 0)
            if vram_mb > 0:
                vram_str = f"{vram_mb} MB" if vram_mb < 1024 else f"{vram_mb / 1024:.2f} GB"
            else:
                vram_str = "Paylasimli (sistem RAM'i)"
            surucu = gpu.get("surucu", "")
            print(f"    {Fore.GREEN}GPU {i}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}Model          : {gpu['ad']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}Tur            : {tur} ({tip}){Style.RESET_ALL}")
            print(f"    {Fore.WHITE}VRAM           : {vram_str}{Style.RESET_ALL}")
            if surucu and surucu != "Bilinmiyor":
                print(f"    {Fore.WHITE}Surucu         : {surucu}{Style.RESET_ALL}")
            print()
    else:
        print(f"{Fore.RED}[-] GPU bulunamadi.{Style.RESET_ALL}")
        print()

    cuda = profil["cuda"]
    if cuda["durum"]:
        print(f"{Fore.GREEN}[+] CUDA Destegi: Aktif ({cuda['sayac']} cihaz){Style.RESET_ALL}")
        for i, cihaz in enumerate(cuda["cihazlar"]):
            print(f"    {Fore.WHITE}GPU {i}          : {cihaz['ad']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}VRAM           : {cihaz['vram_gb']:.2f} GB{Style.RESET_ALL}")
        print()

    dml = profil.get("directml", {})
    if dml.get("durum"):
        print(f"{Fore.GREEN}[+] DirectML Destegi: Aktif ({dml['sayac']} cihaz){Style.RESET_ALL}")
        for i, cihaz in enumerate(dml["cihazlar"]):
            print(f"    {Fore.WHITE}GPU {i}          : {cihaz['ad']}{Style.RESET_ALL}")
        print(f"    {Fore.CYAN}DirectML, DirectX 12 uzerinden Intel/AMD/NVIDIA GPU hizlandirmasi saglar.{Style.RESET_ALL}")
        print()

    npu = profil.get("npu", [])
    if npu:
        print(f"{Fore.MAGENTA}[*] NPU (Yapay Zeka Islemsici) - {len(npu)} adet{Style.RESET_ALL}")
        for i, n in enumerate(npu):
            durum_icon = f"{Fore.GREEN}AKTIF{Style.RESET_ALL}" if "OK" in n.get("durum", "") else f"{Fore.YELLOW}BILINMIYOR{Style.RESET_ALL}"
            print(f"    {Fore.WHITE}NPU {i}        : {n['ad']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}Tur            : {n['tur']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}Durum          : {durum_icon}{Style.RESET_ALL}")
            print(f"    {Fore.YELLOW}Not            : NPU yalnizca cikarim (inference) icindir, egitimde kullanilmaz.{Style.RESET_ALL}")
        print()

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[*] Egitim Onerisi{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Hedef Cihaz    : {profil['hedef_cihaz'].upper()}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Onerilen Batch : {profil['onerilen_batch']}{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Aciklama       : {profil['cihaz_aciklamasi']}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    return profil


def cihaz_secimi_yap(profil=None, mod="egitim"):
    """Donanım analizinden sonra kullanıcıya cihaz seçtirir.

    Args:
        profil: donanim_profili_olustur() çıktısı
        mod: "egitim" (NPU yasak) veya "cikarim" (NPU serbest)

    Returns:
        dict: {"cihaz": "cuda"|"cpu"|"npu", "batch": int, "aciklama": str}
    """
    if profil is None:
        profil = donanim_profili_olustur()

    npu_var = len(profil.get("npu", [])) > 0

    print()
    if mod == "cikarim":
        print(f"{Fore.YELLOW}  [CIKARIM CIHAZI SECIMI]{Style.RESET_ALL}")
        print()
        print(f"{Fore.CYAN}  Cikarim icin kullanilabilir islemciler:{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}  [EGITIM CIHAZI SECIMI]{Style.RESET_ALL}")
        print()
        print(f"{Fore.CYAN}  Egitim icin kullanilabilir islemciler:{Style.RESET_ALL}")
    print()

    secenekler = []
    secenek_no = 1

    cuda = profil.get("cuda", {})
    nvidia = profil.get("nvidia_gpu", [])
    intel_arc = profil.get("intel_arc_gpu", [])
    amd_gpu = profil.get("amd_gpu", [])

    if cuda.get("durum") and cuda.get("sayac", 0) > 0:
        for i, cihaz in enumerate(cuda["cihazlar"]):
            print(f"  {Fore.WHITE}[{secenek_no}] {Fore.GREEN}NVIDIA CUDA GPU{Style.RESET_ALL}")
            print(f"      {cihaz['ad']} ({cihaz['vram_gb']:.1f} GB VRAM)")
            print(f"      {Fore.CYAN}En hizli secenek - CUDA hizlandirmasi tam destek{Style.RESET_ALL}")
            print()
            secenekler.append({
                "no": secenek_no, "cihaz": "cuda", "batch": 16,
                "aciklama": f"NVIDIA CUDA - {cihaz['ad']}",
            })
            secenek_no += 1
    elif nvidia:
        for gpu in nvidia:
            vram_str = f"{gpu['vram_mb']} MB" if gpu['vram_mb'] < 1024 else f"{gpu['vram_mb'] / 1024:.1f} GB"
            print(f"  {Fore.WHITE}[{secenek_no}] {Fore.GREEN}NVIDIA GPU{Style.RESET_ALL}")
            print(f"      {gpu['ad']} ({vram_str} VRAM)")
            print(f"      {Fore.YELLOW}CUDA destegi PyTorch kurulumunuza bagli{Style.RESET_ALL}")
            print()
            secenekler.append({
                "no": secenek_no, "cihaz": "cuda", "batch": 16,
                "aciklama": f"NVIDIA GPU - {gpu['ad']}",
            })
            secenek_no += 1

    dml = profil.get("directml", {})
    if dml.get("durum") and dml.get("sayac", 0) > 0:
        for i, cihaz in enumerate(dml["cihazlar"]):
            if mod == "cikarim":
                print(f"  {Fore.WHITE}[{secenek_no}] {Fore.CYAN}DirectML GPU (Onerilen - Cikarim){Style.RESET_ALL}")
                print(f"      {cihaz['ad']}")
                print(f"      {Fore.GREEN}DirectX 12 GPU cikarim hizlandirmasi - En hizli secenek{Style.RESET_ALL}")
            else:
                print(f"  {Fore.WHITE}[{secenek_no}] {Fore.CYAN}DirectML GPU{Style.RESET_ALL}")
                print(f"      {cihaz['ad']}")
                print(f"      {Fore.YELLOW}Not: DirectML su an yalnizca cikarim (inference) icindir.{Style.RESET_ALL}")
                print(f"      {Fore.YELLOW}Egitim CPU'da calisacak. En hizli egitim icin Google Colab [son secenek] onerilir.{Style.RESET_ALL}")
            print()
            secenekler.append({
                "no": secenek_no, "cihaz": "directml", "batch": 16,
                "aciklama": f"DirectML GPU - {cihaz['ad']} (Cikarim GPU, Egitim CPU)",
            })
            secenek_no += 1

    for gpu in intel_arc:
        vram_str = f"{gpu['vram_mb']} MB" if gpu['vram_mb'] < 1024 else f"{gpu['vram_mb'] / 1024:.1f} GB"
        gpu_adi = gpu['ad']
        if mod == "cikarim":
            print(f"  {Fore.WHITE}[{secenek_no}] {Fore.BLUE}Intel Arc GPU (XPU){Style.RESET_ALL}")
            print(f"      {gpu_adi} ({vram_str} VRAM)")
            print(f"      {Fore.YELLOW}XPU calismiyorsa DirectML secenegini kullanin{Style.RESET_ALL}")
        else:
            print(f"  {Fore.WHITE}[{secenek_no}] {Fore.BLUE}Intel Arc GPU (XPU){Style.RESET_ALL}")
            print(f"      {gpu_adi} ({vram_str} VRAM)")
            print(f"      {Fore.YELLOW}XPU calismiyorsa DirectML secenegini kullanin{Style.RESET_ALL}")
        print()
        secenekler.append({
            "no": secenek_no, "cihaz": "cpu", "batch": 8,
            "aciklama": f"Intel Arc GPU - {gpu_adi}",
        })
        secenek_no += 1

    for gpu in amd_gpu:
        vram_mb = gpu.get("vram_mb", 0)
        if vram_mb >= 2048:
            vram_str = f"{vram_mb} MB" if vram_mb < 1024 else f"{vram_mb / 1024:.1f} GB"
            print(f"  {Fore.WHITE}[{secenek_no}] {Fore.RED}AMD Radeon GPU{Style.RESET_ALL}")
            print(f"      {gpu['ad']} ({vram_str} VRAM)")
            print(f"      {Fore.CYAN}DirectML / ROCm ile {'cikarim' if mod == 'cikarim' else 'egitim'} yapilabilir{Style.RESET_ALL}")
            print()
            secenekler.append({
                "no": secenek_no, "cihaz": "cpu", "batch": 8,
                "aciklama": f"AMD Radeon GPU - {gpu['ad']}",
            })
            secenek_no += 1

    cpu = profil.get("cpu", {})
    print(f"  {Fore.WHITE}[{secenek_no}] {Fore.YELLOW}CPU (Islemci){Style.RESET_ALL}")
    print(f"      {cpu.get('ad', 'Bilinmeyen CPU')} - {cpu.get('cekirdek', '?')} cekirdek")
    print(f"      {Fore.CYAN}Her zaman calisir, en uyumlu secenek{Style.RESET_ALL}")
    print()
    secenekler.append({
        "no": secenek_no, "cihaz": "cpu", "batch": 4,
        "aciklama": f"CPU - {cpu.get('ad', 'Bilinmeyen')}",
    })
    secenek_no += 1

    npu = profil.get("npu", [])
    if mod == "cikarim" and npu:
        for n in npu:
            if "intel" in n.get("tur", "").lower():
                npu_format = "OpenVINO (.xml + .bin)"
                npu_not = "Intel AI Boost NPU - OpenVINO export sonrasi kullanilabilir"
            elif "amd" in n.get("tur", "").lower():
                npu_format = "ONNX Runtime + VitisAI (.onnx INT8)"
                npu_not = "AMD Ryzen AI NPU - ONNX INT8 quantize sonrasi kullanilabilir"
            else:
                npu_format = "ONNX / TFLite"
                npu_not = f"{n['tur']} - Model export sonrasi kullanilabilir"
            print(f"  {Fore.WHITE}[{secenek_no}] {Fore.MAGENTA}{n['tur']}{Style.RESET_ALL}")
            print(f"      {n['ad']}")
            print(f"      {Fore.CYAN}{npu_not}{Style.RESET_ALL}")
            print(f"      {Fore.YELLOW}Format: {npu_format} (once model.export() yapin){Style.RESET_ALL}")
            print()
            secenekler.append({
                "no": secenek_no, "cihaz": "cpu", "batch": 4,
                "aciklama": f"NPU Cikarim - {n['tur']}",
            })
            secenek_no += 1
    elif mod == "egitim" and npu:
        print(f"  {Fore.MAGENTA}[!] NPU tespit edildi ancak egitimde kullanilmaz.{Style.RESET_ALL}")
        for n in npu:
            print(f"      {n['tur']}: {n['ad']} (sadece cikarim - menu [6])")
        print()

    if mod == "egitim":
        print(f"  {Fore.WHITE}[{secenek_no}] {Fore.CYAN}Google Colab (Ucretsiz GPU){Style.RESET_ALL}")
        print(f"      NVIDIA T4 GPU")
        print(f"      {Fore.CYAN}notebooks/hades_colab_egitim.ipynb dosyasini Colab'a yukleyin{Style.RESET_ALL}")
    else:
        print(f"  {Fore.WHITE}[{secenek_no}] {Fore.CYAN}Google Colab (Ucretsiz GPU){Style.RESET_ALL}")
        print(f"      NVIDIA T4 GPU")
        print(f"      {Fore.CYAN}Cikarim icin Colab notebook'u kullanilabilir{Style.RESET_ALL}")
    print()
    secenekler.append({
        "no": secenek_no, "cihaz": "colab", "batch": 16,
        "aciklama": "Google Colab - Ucretsiz NVIDIA T4 GPU",
    })
    secenek_no += 1

    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")
    while True:
        try:
            secim = input(f"{Fore.CYAN}  Cihaz seciminiz [1-{len(secenekler)}, Enter=1]: {Style.RESET_ALL}").strip()
            if secim == "":
                secilen = secenekler[0]
                break
            secim_no = int(secim)
            if 1 <= secim_no <= len(secenekler):
                secilen = secenekler[secim_no - 1]
                break
            print(f"{Fore.RED}  Gecersiz secim. Lutfen 1-{len(secenekler)} arasinda bir deger girin.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}  Gecersiz secim. Lutfen bir sayi girin.{Style.RESET_ALL}")
        except (EOFError, KeyboardInterrupt):
            secilen = secenekler[0]
            print()
            break

    print()
    if secilen["cihaz"] == "colab":
        print(f"{Fore.CYAN}[+] Secilen cihaz: {secilen['aciklama']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[!] Google Colab'da egitim icin:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}    1. notebooks/hades_colab_egitim.ipynb dosyasini acin{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}    2. Google Drive'a yukleyip Colab'da calistirin{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}    3. Runtime > Change runtime type > T4 GPU secin{Style.RESET_ALL}")
    elif secilen["cihaz"] == "directml":
        print(f"{Fore.GREEN}[+] Secilen cihaz: {secilen['aciklama']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[+] Batch size    : {secilen['batch']}{Style.RESET_ALL}")
        if mod == "egitim":
            print(f"{Fore.YELLOW}[!] DirectML GPU tespit edildi ancak egitim CPU'da calisacak.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[!] Egitim sonrasi otomatik OpenVINO/ONNX export yapilir.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[*] Cikarim (Menu [6]) GPU uzerinde calisacak.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[*] En hizli egitim: Google Colab (menu secenegi [son secenek]).{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}[+] DirectML GPU cikarimda kullanilacak.{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}[+] Secilen cihaz: {secilen['aciklama']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[+] Batch size    : {secilen['batch']}{Style.RESET_ALL}")
    print()

    return secilen


if __name__ == "__main__":
    profil = donanim_ozeti_yazdir()
    cihaz_secimi_yap(profil)
