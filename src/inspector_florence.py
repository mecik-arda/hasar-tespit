import gc
import numpy as np
from pathlib import Path
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent.parent
YAPILANDIRMA_YOLU = PROJE_KOKU / "config.yaml"

_FLORENCE_MODEL = None
_FLORENCE_PROCESSOR = None
_FLORENCE_CIHAZ = None
_KONTROL_EDILDI = False
_MEVCUT = False

CAPRAZ_SORGULAR = {
    "Cizik": "<DETAILED_CAPTION> Is this a scratch on the car paint or just a light reflection or shadow?",
    "Gocuk": "<DETAILED_CAPTION> Is this a dent on the car body or a reflection distortion?",
    "Cam Kirigi": "<DETAILED_CAPTION> Is the glass cracked or is this a reflection on the window?",
    "Pas": "<DETAILED_CAPTION> Is this rust corrosion or just dirt or mud on the surface?",
    "Kus Pisligi": "<DETAILED_CAPTION> Is this bird droppings or a dirt spot?",
    "Far Kirigi": "<DETAILED_CAPTION> Is the headlight cracked or is this a light reflection?",
    "Patlak Lastik": "<DETAILED_CAPTION> Is the tire flat or deflated or is this a shadow under the wheel?",
}


def _florence_kullanilabilir_mi():
    global _KONTROL_EDILDI, _MEVCUT
    if _KONTROL_EDILDI:
        return _MEVCUT
    _KONTROL_EDILDI = True
    try:
        import transformers
        _MEVCUT = True
        return True
    except ImportError:
        _MEVCUT = False
        return False


def _cihaz_sec(otomatik_yedekleme_cpu=True):
    import torch
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml
        return "directml"
    except (ImportError, Exception):
        pass
    if not otomatik_yedekleme_cpu:
        raise RuntimeError("GPU bulunamadi ve CPU yedekleme kapali.")
    return "cpu"


def _florence_modeli_yukle(model_adi, otomatik_yedekleme_cpu=True):
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR, _FLORENCE_CIHAZ

    if _FLORENCE_MODEL is not None and _FLORENCE_PROCESSOR is not None:
        return _FLORENCE_MODEL, _FLORENCE_PROCESSOR, _FLORENCE_CIHAZ

    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    cihaz = _cihaz_sec(otomatik_yedekleme_cpu)

    try:
        if cihaz == "cuda":
            _FLORENCE_CIHAZ = "cuda"
            islemci = AutoProcessor.from_pretrained(model_adi, trust_remote_code=True)
            print(f"{Fore.YELLOW}[!] Guvenlik: Florence-2 modeli uzaktan kod calistirabilir. Resmi Microsoft reposundan (microsoft/Florence-2) indirildiginden emin olun.{Style.RESET_ALL}")
            model = AutoModelForCausalLM.from_pretrained(
                model_adi,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to("cuda")
        elif cihaz == "directml":
            _FLORENCE_CIHAZ = "directml"
            import torch_directml
            dml = torch_directml.device()
            islemci = AutoProcessor.from_pretrained(model_adi, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_adi,
                trust_remote_code=True,
            ).to(dml)
        else:
            _FLORENCE_CIHAZ = "cpu"
            islemci = AutoProcessor.from_pretrained(model_adi, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_adi,
                trust_remote_code=True,
            )
    except RuntimeError as hata:
        if "out of memory" in str(hata).lower() and otomatik_yedekleme_cpu:
            print(f"{Fore.YELLOW}[!] VRAM dolu, Florence-2 CPU'ya kaydiriliyor...{Style.RESET_ALL}")
            _FLORENCE_CIHAZ = "cpu"
            islemci = AutoProcessor.from_pretrained(model_adi, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_adi,
                trust_remote_code=True,
            )
        else:
            raise

    _FLORENCE_MODEL = model
    _FLORENCE_PROCESSOR = islemci
    return model, islemci, _FLORENCE_CIHAZ


def _florence_modelini_bosalt():
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR
    _FLORENCE_MODEL = None
    _FLORENCE_PROCESSOR = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, Exception):
        pass


def _bolge_kirp(gorsel, kutu):
    x1, y1, x2, y2 = kutu
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(gorsel.shape[1], int(x2))
    y2 = min(gorsel.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return gorsel[y1:y2, x1:x2].copy()


def _florence_sorgula(model, islemci, cihaz, gorsel_dizisi, gorev="<OD>"):
    import torch
    from PIL import Image as PILImage

    if isinstance(gorsel_dizisi, np.ndarray):
        pil_gorsel = PILImage.fromarray(bgr_to_rgb(gorsel_dizisi))
    else:
        pil_gorsel = gorsel_dizisi

    girdiler = islemci(text=gorev, images=pil_gorsel, return_tensors="pt")

    if cihaz == "cuda":
        girdiler = {k: v.to("cuda") for k, v in girdiler.items()}
    elif cihaz == "directml":
        import torch_directml
        dml = torch_directml.device()
        girdiler = {k: v.to(dml) for k, v in girdiler.items()}

    with torch.no_grad():
        ciktilar = model.generate(
            input_ids=girdiler["input_ids"],
            pixel_values=girdiler.get("pixel_values"),
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

    sonuc_metni = islemci.batch_decode(ciktilar, skip_special_tokens=False)[0]
    ayrilmis = islemci.post_process_generation(
        sonuc_metni, task=gorev, image_size=(pil_gorsel.width, pil_gorsel.height)
    )
    return ayrilmis


def bgr_to_rgb(gorsel_dizisi):
    if len(gorsel_dizisi.shape) == 3 and gorsel_dizisi.shape[2] == 3:
        return gorsel_dizisi[:, :, ::-1].copy()
    return gorsel_dizisi


def _hasar_siniflandir(metin, ekstra_siniflar=None):
    import re
    metin_kucuk = metin.lower()

    eslesmeler = {
        "scratch": "Cizik",
        "dent": "Gocuk",
        "gouck": "Gocuk",
        "crack": "Cam Kirigi",
        "glass": "Cam Kirigi",
        "rust": "Pas",
        "bird": "Kus Pisligi",
        "headlight": "Far Kirigi",
        "tire": "Patlak Lastik",
        "cizik": "Cizik",
        "gocuk": "Gocuk",
        "cam": "Cam Kirigi",
        "kirik": "Cam Kirigi",
        "pas": "Pas",
        "far": "Far Kirigi",
        "lastik": "Patlak Lastik",
    }

    if ekstra_siniflar:
        for ekstra in ekstra_siniflar:
            ekstra_kucuk = ekstra.lower()
            if "patlak" in ekstra_kucuk or "flat" in ekstra_kucuk or "tire" in ekstra_kucuk:
                eslesmeler[ekstra_kucuk] = "Patlak Lastik"

    if any(kw in metin_kucuk for kw in ["reflection", "shadow", "dirt", "mud"]):
        return "Bilinmeyen"

    for anahtar, deger in eslesmeler.items():
        if re.search(r'\b' + re.escape(anahtar) + r'\b', metin_kucuk):
            return deger

    return "Bilinmeyen"


def denetle(tespitler_havuzu, gorsel, yapilandirma=None):
    if not _florence_kullanilabilir_mi():
        print(f"{Fore.RED}[-] Florence-2 kutuphanesi yuklu degil. 'pip install transformers' calistirin.{Style.RESET_ALL}")
        return tespitler_havuzu

    if yapilandirma is None:
        import yaml
        with open(YAPILANDIRMA_YOLU, "r", encoding="utf-8") as dosya:
            yapilandirma = yaml.safe_load(dosya)

    multi_model_ayari = yapilandirma.get("multi_model", {})
    denetleyici_ayari = multi_model_ayari.get("denetleyici_ayarlari", {})
    model_adi = denetleyici_ayari.get("model", "microsoft/Florence-2-base")
    gorev = denetleyici_ayari.get("gorev", "<OD>")
    ekstra_siniflar = denetleyici_ayari.get("ekstra_siniflar", [])
    otomatik_yedekleme = multi_model_ayari.get("otomatik_yedekleme_cpu", True)

    print(f"{Fore.BLUE}[*] Florence-2 Denetimi Yapiliyor...{Style.RESET_ALL}")

    try:
        model, islemci, cihaz = _florence_modeli_yukle(model_adi, otomatik_yedekleme)
        print(f"    {Fore.WHITE}Backend         : {Fore.GREEN}{cihaz}{Style.RESET_ALL}")
    except Exception as hata:
        print(f"{Fore.RED}[-] Florence-2 yuklenemedi: {hata}{Style.RESET_ALL}")
        return tespitler_havuzu

    dogrulanmis_tespitler = []

    try:
        for i, kutu_bilgisi in enumerate(tespitler_havuzu.get("boxes", [])):
            kutu = kutu_bilgisi.get("kutucuk", None) or kutu_bilgisi.get("bbox", None)
            if kutu is None:
                continue

            if "kutucuk" in kutu_bilgisi:
                koordinat = kutu_bilgisi["kutucuk"]
                x1 = koordinat.get("x1", 0)
                y1 = koordinat.get("y1", 0)
                x2 = koordinat.get("x2", 0)
                y2 = koordinat.get("y2", 0)
            else:
                x1, y1, x2, y2 = kutu

            crop = _bolge_kirp(gorsel, (x1, y1, x2, y2))
            if crop is None:
                continue

            try:
                sinif_adi = kutu_bilgisi.get("sinif_adi", "")
                capraz_gorev = CAPRAZ_SORGULAR.get(sinif_adi, gorev)
                sonuc = _florence_sorgula(model, islemci, cihaz, crop, gorev=capraz_gorev)

                tespit_metni = ""
                if capraz_gorev in sonuc:
                    bolum = sonuc[capraz_gorev]
                    if isinstance(bolum, dict) and "bboxes" in bolum:
                        etiketler = bolum.get("labels", [])
                        tespit_metni = " ".join(etiketler) if etiketler else ""
                    elif isinstance(bolum, str):
                        tespit_metni = bolum

                dogrulanmis_sinif = _hasar_siniflandir(tespit_metni, ekstra_siniflar)
                orijinal_sinif = kutu_bilgisi.get("sinif_adi", "Bilinmeyen")

                if dogrulanmis_sinif != "Bilinmeyen":
                    nihai_sinif = dogrulanmis_sinif
                else:
                    nihai_sinif = orijinal_sinif

                dogrulanmis_tespit = dict(kutu_bilgisi)
                dogrulanmis_tespit["sinif_adi"] = nihai_sinif
                dogrulanmis_tespit["florence_dogrulama"] = tespit_metni.strip()
                dogrulanmis_tespit["orijinal_sinif"] = orijinal_sinif
                dogrulanmis_tespit["denetlendi"] = True
                dogrulanmis_tespitler.append(dogrulanmis_tespit)

            except Exception as hata:
                print(f"{Fore.YELLOW}[!] Kutu {i} denetlenemedi: {hata}{Style.RESET_ALL}")
                dogrulanmis_tespit = dict(kutu_bilgisi)
                dogrulanmis_tespit["denetlendi"] = False
                dogrulanmis_tespitler.append(dogrulanmis_tespit)

        for maske_bilgisi in tespitler_havuzu.get("masks", []):
            kutu = maske_bilgisi.get("kutucuk", None) or maske_bilgisi.get("bbox", None)
            if kutu is None:
                continue

            if "kutucuk" in maske_bilgisi:
                koordinat = maske_bilgisi["kutucuk"]
                x1 = koordinat.get("x1", 0)
                y1 = koordinat.get("y1", 0)
                x2 = koordinat.get("x2", 0)
                y2 = koordinat.get("y2", 0)
            else:
                x1, y1, x2, y2 = kutu

            crop = _bolge_kirp(gorsel, (x1, y1, x2, y2))
            if crop is None:
                continue

            try:
                maske_sinif = maske_bilgisi.get("sinif_adi", "")
                capraz_gorev_maske = CAPRAZ_SORGULAR.get(maske_sinif, gorev)
                sonuc = _florence_sorgula(model, islemci, cihaz, crop, gorev=capraz_gorev_maske)
                tespit_metni = sonuc.get(capraz_gorev_maske, "") if isinstance(sonuc, dict) else str(sonuc)
                dogrulanmis_sinif = _hasar_siniflandir(tespit_metni, ekstra_siniflar)
                orijinal_maske_sinif = maske_bilgisi.get("sinif_adi", "Bilinmeyen")

                if dogrulanmis_sinif != "Bilinmeyen":
                    nihai_maske_sinif = dogrulanmis_sinif
                else:
                    nihai_maske_sinif = orijinal_maske_sinif

                dogrulanmis_maske = dict(maske_bilgisi)
                dogrulanmis_maske["sinif_adi"] = nihai_maske_sinif
                dogrulanmis_maske["florence_dogrulama"] = tespit_metni.strip()
                dogrulanmis_maske["orijinal_sinif"] = orijinal_maske_sinif
                dogrulanmis_maske["denetlendi"] = True
                dogrulanmis_tespitler.append(dogrulanmis_maske)

            except Exception as hata:
                print(f"{Fore.YELLOW}[!] Maske denetlenemedi: {hata}{Style.RESET_ALL}")
                dogrulanmis_maske = dict(maske_bilgisi)
                dogrulanmis_maske["denetlendi"] = False
                dogrulanmis_tespitler.append(dogrulanmis_maske)

    finally:
        if multi_model_ayari.get("ram_optimizasyonu", True):
            _florence_modelini_bosalt()
            print(f"{Fore.GREEN}[+] Florence-2 bellekten teslim edildi.{Style.RESET_ALL}")

    print(f"{Fore.GREEN}[+] Florence-2 Denetimi Yapiliyor... [Bitti]{Style.RESET_ALL}")
    print(f"    {Fore.WHITE}Dogrulanmis Tespit: {len(dogrulanmis_tespitler)}{Style.RESET_ALL}")

    return {
        "boxes": dogrulanmis_tespitler,
        "masks": tespitler_havuzu.get("masks", []),
    }


if __name__ == "__main__":
    print(f"{Fore.YELLOW}Florence-2 Denetleyici Modulu{Style.RESET_ALL}")
    print(f"Bu modul pipeline.py tarafindan cagrilir.")