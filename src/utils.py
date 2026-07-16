"""HADES Scanner - Ortak yardımcı fonksiyonlar ve sabitler.

Bu modül, proje genelinde tekrarlanan yapılandırma yükleme, DirectML cihaz tespiti,
OpenVINO kontrolü gibi işlevleri tek bir yerden sunarak DRY prensibine uygunluğu sağlar.
"""

import yaml
from pathlib import Path
from colorama import Fore, Style, init

init()

PROJE_KOKU = Path(__file__).parent.parent
YAPILANDIRMA_YOLU = PROJE_KOKU / "config.yaml"
EGITIM_KOKU = PROJE_KOKU / "runs" / "train"
CIKARIM_KOKU = PROJE_KOKU / "runs" / "predict"
VERI_KOKU = PROJE_KOKU / "data"

# Her sınıf için atanmış renkler (BGR formatı - OpenCV uyumlu)
SINIF_RENKLERI = {
    0: (0, 0, 255),
    1: (0, 165, 255),
    2: (0, 255, 255),
    3: (0, 128, 0),
    4: (255, 0, 255),
    5: (255, 0, 0),
    6: (128, 128, 128),
}

_DML_CIHAZ = None
_DML_KONTROL_EDILDI = False
_OPENVINO_KONTROL_EDILDI = False
_OPENVINO_VAR = False
_CONFIG_CACHE = None


def yapilandirma_yukle():
    """config.yaml dosyasını okur ve dict olarak döndürür. Sonucu bellekte cache'ler."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    with open(YAPILANDIRMA_YOLU, "r", encoding="utf-8") as dosya:
        _CONFIG_CACHE = yaml.safe_load(dosya)
    return _CONFIG_CACHE


def yapilandirma_kaydet(yapilandirma):
    """Yapılandırma dict'ini config.yaml dosyasına yazar, cache'i günceller."""
    global _CONFIG_CACHE
    with open(YAPILANDIRMA_YOLU, "w", encoding="utf-8") as dosya:
        yaml.safe_dump(yapilandirma, dosya, sort_keys=False, default_flow_style=False, allow_unicode=True)
    _CONFIG_CACHE = yapilandirma


def _directml_cihazini_al():
    """DirectML GPU cihazini dondurur. Kullanilamiyorsa None.
    Sonucu cache'ler, tekrar tekrar import etmez.
    """
    global _DML_CIHAZ, _DML_KONTROL_EDILDI
    if _DML_KONTROL_EDILDI:
        return _DML_CIHAZ
    _DML_KONTROL_EDILDI = True
    try:
        import torch_directml
        _DML_CIHAZ = torch_directml.device()
        return _DML_CIHAZ
    except ImportError:
        return None


def _openvino_kullanilabilir_mi():
    """OpenVINO paketinin yuklu olup olmadigini kontrol eder (tek sefer)."""
    global _OPENVINO_KONTROL_EDILDI, _OPENVINO_VAR
    if _OPENVINO_KONTROL_EDILDI:
        return _OPENVINO_VAR
    _OPENVINO_KONTROL_EDILDI = True
    try:
        import openvino
        _OPENVINO_VAR = True
        return True
    except ImportError:
        return False
