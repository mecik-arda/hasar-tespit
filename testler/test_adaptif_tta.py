import unittest
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

adaptive_tta_modulu = import_module("src.adaptive_tta")
advanced_benchmarks_modulu = import_module("src.advanced_benchmarks")
pipeline_modulu = import_module("src.pipeline")
_analiz_gorselini_normalize_et = adaptive_tta_modulu._analiz_gorselini_normalize_et
_gamma_duzelt = adaptive_tta_modulu._gamma_duzelt
gorsel_kalitesini_analiz_et = adaptive_tta_modulu.gorsel_kalitesini_analiz_et
tta_tahminini_orijinale_tasi = adaptive_tta_modulu.tta_tahminini_orijinale_tasi
tta_varyantlarini_olustur = adaptive_tta_modulu.tta_varyantlarini_olustur
bootstrap_map50_farkini_hesapla = advanced_benchmarks_modulu.bootstrap_map50_farkini_hesapla
tta_kalibrasyon_benchmark_calistir = advanced_benchmarks_modulu.tta_kalibrasyon_benchmark_calistir
_adaptif_tta_tarama = pipeline_modulu._adaptif_tta_tarama
hasar_tespiti_yap = pipeline_modulu.hasar_tespiti_yap


class SahteDizi:
    def __init__(self, deger):
        self.deger = np.asarray(deger)

    def __getitem__(self, indeks):
        return SahteDizi(self.deger[indeks])

    def cpu(self):
        return self

    def numpy(self):
        return self.deger


class SahteKutu:
    def __init__(self):
        self.xyxy = SahteDizi([[10, 10, 80, 80]])
        self.cls = SahteDizi([0])
        self.conf = SahteDizi([0.9])


class SahteModel:
    def __init__(self):
        self.cagrilar = []

    def predict(self, **kwargs):
        self.cagrilar.append(kwargs)
        return [type("Sonuc", (), {"boxes": [SahteKutu()]})()]


class KaliteAnalizcisiTesti(unittest.TestCase):
    def setUp(self):
        self.ayar = {
            "aktif": True,
            "analiz_uzun_kenar": 640,
            "karanlik_medyan_esigi": 55,
            "parlak_medyan_esigi": 205,
            "siyaha_kirpma_orani_esigi": 0.2,
            "beyaza_kirpma_orani_esigi": 0.12,
            "netlik_skoru_esigi": 0.42,
            "agir_bulaniklik_esigi": 0.18,
            "azami_varyant": 3,
            "gamma": 0.7,
            "yuksek_olcek": 1.25,
        }

    def test_aspect_ratio_korunur_ve_letterbox_boyutu_sabittir(self):
        gorsel = np.zeros((360, 640, 3), dtype=np.uint8)
        icerik, letterbox_boyutu = _analiz_gorselini_normalize_et(gorsel, 640)
        self.assertEqual(icerik.shape[:2], (360, 640))
        self.assertEqual(letterbox_boyutu, [640, 640])

    def test_karanlik_gorsel_tta_tetikler(self):
        gorsel = np.full((240, 320, 3), 10, dtype=np.uint8)
        rapor = gorsel_kalitesini_analiz_et(gorsel, self.ayar)
        self.assertTrue(rapor["karanlik"])
        self.assertTrue(rapor["tta_tetiklendi"])
        self.assertIn("karanlik", rapor["tta_nedeni"])

    def test_asiri_parlak_gorsel_algilanir(self):
        gorsel = np.full((240, 320, 3), 250, dtype=np.uint8)
        rapor = gorsel_kalitesini_analiz_et(gorsel, self.ayar)
        self.assertTrue(rapor["asiri_parlak"])
        self.assertGreater(rapor["parlama_orani"], 0.9)

    def test_duz_kaporta_bulanik_yerine_dusuk_bilgi_olarak_isaretlenir(self):
        gorsel = np.full((240, 320, 3), 128, dtype=np.uint8)
        rapor = gorsel_kalitesini_analiz_et(gorsel, self.ayar)
        self.assertFalse(rapor["bulanik"])
        self.assertTrue(rapor["dusuk_gorsel_bilgisi"])
        self.assertTrue(rapor["sinirda_guvenilirlik"])

    def test_gamma_07_karanlik_gorseli_acar(self):
        gorsel = np.full((32, 32, 3), 40, dtype=np.uint8)
        duzeltilmis = _gamma_duzelt(gorsel, 0.7)
        self.assertGreater(float(duzeltilmis.mean()), float(gorsel.mean()))

    def test_varyant_sayisi_uc_ile_sinirlanir(self):
        gorsel = np.full((240, 320, 3), 10, dtype=np.uint8)
        rapor = gorsel_kalitesini_analiz_et(gorsel, self.ayar)
        varyantlar = tta_varyantlarini_olustur(gorsel, rapor, self.ayar)
        self.assertLessEqual(len(varyantlar), 3)
        self.assertEqual(varyantlar[0]["ad"], "orijinal")
        self.assertIn("gamma", [varyant["ad"] for varyant in varyantlar])

    def test_yatay_cevrilen_kutu_orijinale_tasinir(self):
        tahmin = {"sinif_id": 0, "guven": 0.9, "kutucuk": {"x1": 10, "y1": 20, "x2": 40, "y2": 60}}
        varyant = {"ad": "yatay_cevirme", "donusum": "yatay_cevirme", "olcek": 1.0}
        sonuc = tta_tahminini_orijinale_tasi(tahmin, varyant, 100, 80)
        self.assertEqual(sonuc["kutucuk"], {"x1": 60, "y1": 20, "x2": 90, "y2": 60})

    def test_yuksek_olcekli_kutu_orijinale_tasinir(self):
        tahmin = {"sinif_id": 0, "guven": 0.9, "kutucuk": {"x1": 25, "y1": 25, "x2": 100, "y2": 75}}
        varyant = {"ad": "yuksek_olcek", "donusum": "olcek", "olcek": 1.25}
        sonuc = tta_tahminini_orijinale_tasi(tahmin, varyant, 100, 80)
        self.assertEqual(sonuc["kutucuk"], {"x1": 20, "y1": 20, "x2": 80, "y2": 60})

    def test_model_ici_fusyon_tta_dallarini_tek_kumeye_indirir(self):
        gorsel = np.full((120, 160, 3), 10, dtype=np.uint8)
        rapor = gorsel_kalitesini_analiz_et(gorsel, self.ayar)
        model = SahteModel()
        tahminler, telemetri = _adaptif_tta_tarama(model, gorsel, rapor, self.ayar)
        self.assertEqual(len(model.cagrilar), 3)
        self.assertTrue(all(cagri["augment"] is False for cagri in model.cagrilar))
        self.assertEqual(len(tahminler), 1)
        self.assertTrue(telemetri["tta_tetiklendi"])
        self.assertEqual(telemetri["uygulanan_varyantlar"], ["orijinal", "gamma", "lab_clahe"])

    def test_tekli_cikarim_json_telemetrisini_uretir(self):
        gorsel = np.full((120, 160, 3), 10, dtype=np.uint8)
        model = SahteModel()
        yapilandirma = {
            "siniflar": {0: "Cizik"},
            "cikarim": {
                "guven_esigi": 0.25,
                "iou_esigi": 0.7,
                "gorsel_kaydet": False,
                "json_kaydet": False,
                "tta_aktif": False,
                "tta_adaptif": self.ayar,
                "sahi_aktif": False,
                "sinif_guven_esikleri": {0: 0.25},
            },
        }
        with tempfile.TemporaryDirectory() as gecici_klasor:
            gorsel_yolu = Path(gecici_klasor) / "karanlik.jpg"
            cv2.imencode(".jpg", gorsel)[1].tofile(str(gorsel_yolu))
            sonuc = hasar_tespiti_yap(gorsel_yolu, model=model, yapilandirma=yapilandirma)
        self.assertTrue(sonuc["kalite_telemetrisi"]["tta_tetiklendi"])
        self.assertEqual(sonuc["kalite_telemetrisi"]["uygulanan_varyantlar"], ["orijinal", "gamma", "lab_clahe"])
        self.assertEqual(sonuc["toplam_tespit"], 1)
        self.assertIn("adaptif_tta", sonuc["tespitler"][0])


class BootstrapKalibrasyonTesti(unittest.TestCase):
    def test_eslesmis_bootstrap_pozitif_map_farkini_dogrular(self):
        gercekler = []
        tta_tahminleri = []
        for indeks in range(6):
            gorsel_id = str(indeks)
            gercekler.append({"gorsel_id": gorsel_id, "sinif_id": 0, "kutucuk": [10, 10, 50, 50]})
            tta_tahminleri.append({"gorsel_id": gorsel_id, "sinif_id": 0, "guven": 0.9, "kutucuk": [10, 10, 50, 50]})
        sonuc = bootstrap_map50_farkini_hesapla([], tta_tahminleri, gercekler, {0: "Cizik"}, tekrar=40, guven_duzeyi=0.95)
        self.assertGreater(sonuc["ortalama"], 0.9)
        self.assertGreater(sonuc["alt_sinir"], 0.0)
        self.assertEqual(sonuc["tekrar"], 40)

    def test_kalibrasyon_istatistiksel_ve_gecikme_kriterlerini_uygular(self):
        kayitlar = []
        gercekler = []
        for indeks in range(4):
            gorsel_id = str(indeks)
            etiket = {"gorsel_id": gorsel_id, "sinif_id": 0, "kutucuk": [10, 10, 50, 50]}
            kayitlar.append({"gorsel_id": gorsel_id, "gorsel": np.full((64, 64, 3), 128, dtype=np.uint8)})
            gercekler.append(etiket)

        def temel_uretici(gorsel, gorsel_id):
            return []

        def tta_uretici(gorsel, gorsel_id):
            return [{"gorsel_id": gorsel_id, "sinif_id": 0, "guven": 0.9, "kutucuk": [10, 10, 50, 50]}]

        yapilandirma = {
            "siniflar": {0: "Cizik"},
            "cikarim": {
                "tta_adaptif": {
                    "kalibrasyon": {
                        "minimum_map50_artisi": 0.02,
                        "guven_duzeyi": 0.95,
                        "bootstrap_tekrari": 10,
                        "azami_gecikme_artisi_yuzdesi": 1000000,
                    }
                }
            },
        }
        with patch.object(advanced_benchmarks_modulu, "_etiketli_veriyi_hazirla", return_value=(kayitlar, gercekler, "test")):
            rapor = tta_kalibrasyon_benchmark_calistir(
                miktar=4,
                siddetler=(1,),
                temel_tahmin_uretici=temel_uretici,
                tta_tahmin_uretici=tta_uretici,
                yapilandirma=yapilandirma,
                rapor_uret=False,
            )
        self.assertEqual(rapor["durum"], "Tamamlandı")
        for bozulma in ("karanlik", "parlama", "hareket_bulanikligi"):
            self.assertTrue(rapor["sonuclar"][bozulma]["1"]["tta_etkinlestirme_onerisi"])


if __name__ == "__main__":
    unittest.main()
