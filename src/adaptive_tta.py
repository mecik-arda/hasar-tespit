import time

import cv2
import numpy as np


def _sayisal_deger(ayarlar, anahtar, varsayilan, asgari=None, azami=None):
    try:
        deger = float(ayarlar.get(anahtar, varsayilan))
    except (TypeError, ValueError):
        deger = float(varsayilan)
    if asgari is not None:
        deger = max(float(asgari), deger)
    if azami is not None:
        deger = min(float(azami), deger)
    return deger


def _analiz_gorselini_normalize_et(gorsel, uzun_kenar):
    yukseklik, genislik = gorsel.shape[:2]
    hedef = max(32, int(uzun_kenar))
    oran = hedef / float(max(yukseklik, genislik))
    yeni_genislik = max(1, int(round(genislik * oran)))
    yeni_yukseklik = max(1, int(round(yukseklik * oran)))
    yeniden_boyutlandirilmis = cv2.resize(gorsel, (yeni_genislik, yeni_yukseklik), interpolation=cv2.INTER_AREA)
    ust = (hedef - yeni_yukseklik) // 2
    alt = hedef - yeni_yukseklik - ust
    sol = (hedef - yeni_genislik) // 2
    sag = hedef - yeni_genislik - sol
    letterbox = cv2.copyMakeBorder(yeniden_boyutlandirilmis, ust, alt, sol, sag, cv2.BORDER_REPLICATE)
    icerik = letterbox[ust:ust + yeni_yukseklik, sol:sol + yeni_genislik]
    return icerik, [int(letterbox.shape[1]), int(letterbox.shape[0])]


def gorsel_kalitesini_analiz_et(gorsel, ayarlar=None):
    ayarlar = ayarlar or {}
    baslangic = time.perf_counter()
    analiz_gorseli, analiz_boyutu = _analiz_gorselini_normalize_et(
        gorsel,
        _sayisal_deger(ayarlar, "analiz_uzun_kenar", 640, 32),
    )
    gri = cv2.cvtColor(analiz_gorseli, cv2.COLOR_BGR2GRAY)
    gri_float = gri.astype(np.float32)
    parlaklik_ortalamasi = float(np.mean(gri_float))
    parlaklik_p10, parlaklik_p50, parlaklik_p90 = [float(deger) for deger in np.percentile(gri_float, [10, 50, 90])]
    siyaha_kirpma_orani = float(np.mean(gri <= int(_sayisal_deger(ayarlar, "siyah_piksel_esigi", 15, 0, 255))))
    beyaza_kirpma_orani = float(np.mean(gri >= int(_sayisal_deger(ayarlar, "beyaz_piksel_esigi", 245, 0, 255))))
    kontrast_standart_sapmasi = float(np.std(gri_float))
    laplacian_varyansi = float(cv2.Laplacian(gri, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gri_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gri_float, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad_enerjisi = float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y))
    kenar_alt = int(_sayisal_deger(ayarlar, "canny_alt_esik", 60, 0, 255))
    kenar_ust = int(_sayisal_deger(ayarlar, "canny_ust_esik", 180, kenar_alt, 255))
    kenar_yogunlugu = float(np.mean(cv2.Canny(gri, kenar_alt, kenar_ust) > 0))
    laplacian_orani = min(1.0, laplacian_varyansi / _sayisal_deger(ayarlar, "laplacian_referansi", 180.0, 0.001))
    tenengrad_orani = min(1.0, tenengrad_enerjisi / _sayisal_deger(ayarlar, "tenengrad_referansi", 1800.0, 0.001))
    kenar_orani = min(1.0, kenar_yogunlugu / _sayisal_deger(ayarlar, "kenar_yogunlugu_referansi", 0.08, 0.000001))
    agirlik_toplami = 0.45 + 0.35 + 0.20
    netlik_skoru = (laplacian_orani * 0.45 + tenengrad_orani * 0.35 + kenar_orani * 0.20) / agirlik_toplami
    yeterli_gorsel_bilgisi = (
        kontrast_standart_sapmasi >= _sayisal_deger(ayarlar, "minimum_netlik_analiz_kontrasti", 8.0, 0)
        or kenar_yogunlugu >= _sayisal_deger(ayarlar, "minimum_netlik_analiz_kenar_orani", 0.005, 0, 1)
    )
    karanlik = (
        parlaklik_p50 < _sayisal_deger(ayarlar, "karanlik_medyan_esigi", 55.0, 0, 255)
        or siyaha_kirpma_orani > _sayisal_deger(ayarlar, "siyaha_kirpma_orani_esigi", 0.20, 0, 1)
    )
    asiri_parlak = (
        parlaklik_p50 > _sayisal_deger(ayarlar, "parlak_medyan_esigi", 205.0, 0, 255)
        or beyaza_kirpma_orani > _sayisal_deger(ayarlar, "beyaza_kirpma_orani_esigi", 0.12, 0, 1)
    )
    bulanik = yeterli_gorsel_bilgisi and netlik_skoru < _sayisal_deger(ayarlar, "netlik_skoru_esigi", 0.42, 0, 1)
    agir_bulanik = yeterli_gorsel_bilgisi and netlik_skoru < _sayisal_deger(ayarlar, "agir_bulaniklik_esigi", 0.18, 0, 1)
    dusuk_bilgi = not yeterli_gorsel_bilgisi
    nedenler = []
    if karanlik:
        nedenler.append("karanlik")
    if asiri_parlak:
        nedenler.append("asiri_parlak")
    if bulanik:
        nedenler.append("bulanik")
    kalite_profili = "normal" if not nedenler else "+".join(nedenler)
    return {
        "kalite_profili": kalite_profili,
        "parlaklik_skoru": round(parlaklik_ortalamasi / 255.0, 6),
        "parlaklik_ortalamasi": round(parlaklik_ortalamasi, 4),
        "parlaklik_yuzdelikleri": {
            "p10": round(parlaklik_p10, 4),
            "p50": round(parlaklik_p50, 4),
            "p90": round(parlaklik_p90, 4),
        },
        "siyaha_kirpma_orani": round(siyaha_kirpma_orani, 6),
        "parlama_orani": round(beyaza_kirpma_orani, 6),
        "kontrast_standart_sapmasi": round(kontrast_standart_sapmasi, 4),
        "bulaniklik_skoru": round(1.0 - netlik_skoru, 6),
        "netlik_skoru": round(netlik_skoru, 6),
        "laplacian_varyansi": round(laplacian_varyansi, 4),
        "tenengrad_enerjisi": round(tenengrad_enerjisi, 4),
        "kenar_yogunlugu": round(kenar_yogunlugu, 6),
        "analiz_boyutu": analiz_boyutu,
        "karanlik": karanlik,
        "asiri_parlak": asiri_parlak,
        "bulanik": bulanik,
        "dusuk_gorsel_bilgisi": dusuk_bilgi,
        "tta_tetiklendi": bool(nedenler) and bool(ayarlar.get("aktif", False)),
        "tta_nedeni": nedenler,
        "uygulanan_varyantlar": [],
        "sinirda_guvenilirlik": bool(agir_bulanik or dusuk_bilgi),
        "kalite_analiz_suresi_ms": round((time.perf_counter() - baslangic) * 1000.0, 4),
        "tta_ek_sure_ms": 0.0,
    }


def _gamma_duzelt(gorsel, gamma):
    gamma = max(0.05, float(gamma))
    tablo = np.array([255.0 * ((deger / 255.0) ** gamma) for deger in range(256)], dtype=np.uint8)
    return cv2.LUT(gorsel, tablo)


def _lab_clahe_uygula(gorsel, clip_limiti, izgara_boyutu):
    lab = cv2.cvtColor(gorsel, cv2.COLOR_BGR2LAB)
    parlaklik, a_kanali, b_kanali = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limiti), tileGridSize=(int(izgara_boyutu), int(izgara_boyutu)))
    duzeltilmis = clahe.apply(parlaklik)
    return cv2.cvtColor(cv2.merge((duzeltilmis, a_kanali, b_kanali)), cv2.COLOR_LAB2BGR)


def tta_varyantlarini_olustur(gorsel, kalite_raporu, ayarlar=None, zorla=False):
    ayarlar = ayarlar or {}
    azami_varyant = max(1, int(_sayisal_deger(ayarlar, "azami_varyant", 3, 1, 3)))
    varyantlar = [{"ad": "orijinal", "gorsel": gorsel, "donusum": "orijinal", "olcek": 1.0}]
    etkin = bool(ayarlar.get("aktif", False)) and bool(kalite_raporu.get("tta_tetiklendi", False))
    if not etkin and not zorla:
        return varyantlar
    if kalite_raporu.get("karanlik"):
        varyantlar.append({
            "ad": "gamma",
            "gorsel": _gamma_duzelt(gorsel, _sayisal_deger(ayarlar, "gamma", 0.7, 0.05, 1.0)),
            "donusum": "orijinal",
            "olcek": 1.0,
        })
        varyantlar.append({
            "ad": "lab_clahe",
            "gorsel": _lab_clahe_uygula(
                gorsel,
                _sayisal_deger(ayarlar, "clahe_clip_limiti", 2.0, 0.1, 10.0),
                _sayisal_deger(ayarlar, "clahe_izgara_boyutu", 8, 2, 32),
            ),
            "donusum": "orijinal",
            "olcek": 1.0,
        })
    elif kalite_raporu.get("asiri_parlak"):
        varyantlar.append({
            "ad": "parlaklik_gamma",
            "gorsel": _gamma_duzelt(gorsel, _sayisal_deger(ayarlar, "parlaklik_gamma", 1.25, 1.0, 3.0)),
            "donusum": "orijinal",
            "olcek": 1.0,
        })
    if kalite_raporu.get("bulanik") or (zorla and len(varyantlar) == 1):
        olcek = _sayisal_deger(ayarlar, "yuksek_olcek", 1.25, 1.0, 2.0)
        yukseklik, genislik = gorsel.shape[:2]
        buyutulmus = cv2.resize(
            gorsel,
            (max(1, int(round(genislik * olcek))), max(1, int(round(yukseklik * olcek)))),
            interpolation=cv2.INTER_LINEAR,
        )
        varyantlar.append({"ad": "yuksek_olcek", "gorsel": buyutulmus, "donusum": "olcek", "olcek": olcek})
        varyantlar.append({"ad": "yatay_cevirme", "gorsel": cv2.flip(gorsel, 1), "donusum": "yatay_cevirme", "olcek": 1.0})
    elif len(varyantlar) < azami_varyant and bool(ayarlar.get("yatay_cevirme", True)):
        varyantlar.append({"ad": "yatay_cevirme", "gorsel": cv2.flip(gorsel, 1), "donusum": "yatay_cevirme", "olcek": 1.0})
    benzersiz = []
    kullanilan_adlar = set()
    for varyant in varyantlar:
        if varyant["ad"] in kullanilan_adlar:
            continue
        kullanilan_adlar.add(varyant["ad"])
        benzersiz.append(varyant)
    return benzersiz[:azami_varyant]


def tta_tahminini_orijinale_tasi(tahmin, varyant, orijinal_genislik, orijinal_yukseklik):
    sonuc = dict(tahmin)
    kutu = dict(tahmin["kutucuk"])
    if varyant["donusum"] == "olcek":
        olcek = max(0.000001, float(varyant["olcek"]))
        kutu = {anahtar: int(round(deger / olcek)) for anahtar, deger in kutu.items()}
    elif varyant["donusum"] == "yatay_cevirme":
        onceki_x1 = kutu["x1"]
        onceki_x2 = kutu["x2"]
        kutu["x1"] = int(orijinal_genislik - onceki_x2)
        kutu["x2"] = int(orijinal_genislik - onceki_x1)
    kutu["x1"] = max(0, min(int(orijinal_genislik), int(kutu["x1"])))
    kutu["x2"] = max(0, min(int(orijinal_genislik), int(kutu["x2"])))
    kutu["y1"] = max(0, min(int(orijinal_yukseklik), int(kutu["y1"])))
    kutu["y2"] = max(0, min(int(orijinal_yukseklik), int(kutu["y2"])))
    sonuc["kutucuk"] = kutu
    sonuc["tta_varyanti"] = varyant["ad"]
    sonuc["adaptif_tta"] = varyant["ad"] != "orijinal"
    return sonuc
