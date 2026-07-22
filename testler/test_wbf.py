import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.pipeline import _wbf_kutu_birlestir, _ram_havuzu_olustur, _wbf_model_agirliklarini_hesapla


class WbfTesti(unittest.TestCase):
    def setUp(self):
        self.havuz = _ram_havuzu_olustur()

    def test_bos_havuz_birlestirme(self):
        sonuc = _wbf_kutu_birlestir(self.havuz, 10000, 10000, iou_esigi=0.55, guven_esigi=0.25)
        self.assertEqual(len(sonuc), 0)

    def test_dinamik_agirlik_en_yuksek_metrigi_azami_degere_tasir(self):
        yapilandirma = {
            "multi_model": {
                "wbf_dinamik_agirliklandirma": {
                    "aktif": True,
                    "asgari_agirlik": 1.0,
                    "azami_agirlik": 2.5,
                    "duyarlilik": 4.0,
                    "model_metrikleri": {
                        "rt-detr-v2-x": {"genel": 0.965},
                        "yolov12x": {"genel": 0.937},
                    },
                }
            }
        }
        agirliklar = _wbf_model_agirliklarini_hesapla("Pas", ["rt-detr-v2-x", "yolov12x"], yapilandirma)
        self.assertEqual(agirliklar[0], 2.5)
        self.assertGreater(agirliklar[0], agirliklar[1])
        self.assertGreater(agirliklar[1], 1.0)

    def test_sinif_metrigi_genel_metrikten_onceliklidir(self):
        yapilandirma = {
            "multi_model": {
                "wbf_dinamik_agirliklandirma": {
                    "aktif": True,
                    "model_metrikleri": {
                        "rt-detr-v2-x": {"genel": 0.965, "siniflar": {"Cizik": 0.8}},
                        "yolov12x": {"genel": 0.937, "siniflar": {"Cizik": 0.95}},
                    },
                }
            }
        }
        agirliklar = _wbf_model_agirliklarini_hesapla("Cizik", ["rt-detr-v2-x", "yolov12x"], yapilandirma)
        self.assertLess(agirliklar[0], agirliklar[1])
        self.assertEqual(agirliklar[1], 2.5)

    def test_yuzde_bicimli_metrikler_normalize_edilir(self):
        yapilandirma = {
            "multi_model": {
                "wbf_dinamik_agirliklandirma": {
                    "aktif": True,
                    "model_metrikleri": {
                        "rt-detr-v2-x": {"genel": 96.5},
                        "yolov12x": {"genel": 93.7},
                    },
                }
            }
        }
        agirliklar = _wbf_model_agirliklarini_hesapla("Gocuk", ["rt-detr-v2-x", "yolov12x"], yapilandirma)
        self.assertEqual(agirliklar[0], 2.5)
        self.assertGreater(agirliklar[1], 1.0)

    def test_metrik_yoksa_sabit_sinif_agirliklari_kullanilir(self):
        yapilandirma = {
            "multi_model": {
                "wbf_dinamik_agirliklandirma": {"aktif": True, "model_metrikleri": {}},
                "wbf_sinif_agirliklari": {
                    "Cizik": {"rt-detr-v2-x": 1.0, "yolov12x": 2.0}
                },
            }
        }
        agirliklar = _wbf_model_agirliklarini_hesapla("Cizik", ["rt-detr-v2-x", "yolov12x"], yapilandirma)
        self.assertEqual(agirliklar, [1.0, 2.0])

    def test_cakismayan_kutular_korunur(self):
        self.havuz["boxes"] = [
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.9,
                "kutucuk": {"x1": 10, "y1": 10, "x2": 100, "y2": 100},
                "kaynak_model": "rt-detr-v2-x",
            },
            {
                "sinif_id": 1,
                "sinif_adi": "Gocuk",
                "guven": 0.85,
                "kutucuk": {"x1": 5000, "y1": 5000, "x2": 6000, "y2": 6000},
                "kaynak_model": "yolov12x",
            },
        ]
        sonuc = _wbf_kutu_birlestir(self.havuz, 10000, 10000, iou_esigi=0.55, guven_esigi=0.25)
        self.assertEqual(len(sonuc), 2)

    def test_cakisan_kutular_birlesir(self):
        self.havuz["boxes"] = [
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.9,
                "kutucuk": {"x1": 100, "y1": 100, "x2": 300, "y2": 300},
                "kaynak_model": "rt-detr-v2-x",
            },
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.85,
                "kutucuk": {"x1": 105, "y1": 105, "x2": 295, "y2": 295},
                "kaynak_model": "yolov12x",
            },
        ]
        sonuc = _wbf_kutu_birlestir(self.havuz, 10000, 10000, iou_esigi=0.55, guven_esigi=0.25)
        self.assertEqual(len(sonuc), 1)
        self.assertTrue(sonuc[0].get("wbf_birlestirildi", False))
        self.assertIn("wbf_model_agirliklari", sonuc[0])

    def test_dusuk_guven_skoru_eliminir(self):
        self.havuz["boxes"] = [
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.9,
                "kutucuk": {"x1": 100, "y1": 100, "x2": 300, "y2": 300},
                "kaynak_model": "rt-detr-v2-x",
            },
            {
                "sinif_id": 1,
                "sinif_adi": "Gocuk",
                "guven": 0.05,
                "kutucuk": {"x1": 5000, "y1": 5000, "x2": 6000, "y2": 6000},
                "kaynak_model": "yolov12x",
            },
        ]
        sonuc = _wbf_kutu_birlestir(self.havuz, 10000, 10000, iou_esigi=0.55, guven_esigi=0.25)
        self.assertEqual(len(sonuc), 1)
        self.assertEqual(sonuc[0]["sinif_adi"], "Cizik")

    def test_iou_esigi_alti_korunur(self):
        self.havuz["boxes"] = [
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.9,
                "kutucuk": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
                "kaynak_model": "rt-detr-v2-x",
            },
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.85,
                "kutucuk": {"x1": 5000, "y1": 5000, "x2": 6000, "y2": 6000},
                "kaynak_model": "yolov12x",
            },
        ]
        sonuc = _wbf_kutu_birlestir(self.havuz, 10000, 10000, iou_esigi=0.55, guven_esigi=0.25)
        self.assertEqual(len(sonuc), 2)

    def test_coklu_model_gruplama(self):
        self.havuz["boxes"] = [
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.9,
                "kutucuk": {"x1": 100, "y1": 100, "x2": 300, "y2": 300},
                "kaynak_model": "rt-detr-v2-x",
            },
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.8,
                "kutucuk": {"x1": 100, "y1": 100, "x2": 300, "y2": 300},
                "kaynak_model": "yolov12x",
            },
            {
                "sinif_id": 1,
                "sinif_adi": "Gocuk",
                "guven": 0.7,
                "kutucuk": {"x1": 5000, "y1": 5000, "x2": 6000, "y2": 6000},
                "kaynak_model": "rt-detr-v2-x",
            },
        ]
        sonuc = _wbf_kutu_birlestir(self.havuz, 10000, 10000, iou_esigi=0.55, guven_esigi=0.25)
        self.assertEqual(len(sonuc), 2)

    def test_birlesmis_kutu_koordinat_gecerli(self):
        self.havuz["boxes"] = [
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.9,
                "kutucuk": {"x1": 100, "y1": 100, "x2": 300, "y2": 300},
                "kaynak_model": "rt-detr-v2-x",
            },
            {
                "sinif_id": 0,
                "sinif_adi": "Cizik",
                "guven": 0.85,
                "kutucuk": {"x1": 110, "y1": 110, "x2": 290, "y2": 290},
                "kaynak_model": "yolov12x",
            },
        ]
        sonuc = _wbf_kutu_birlestir(self.havuz, 10000, 10000, iou_esigi=0.55, guven_esigi=0.25)
        self.assertEqual(len(sonuc), 1)
        kutu = sonuc[0]["kutucuk"]
        self.assertIn("x1", kutu)
        self.assertIn("y1", kutu)
        self.assertIn("x2", kutu)
        self.assertIn("y2", kutu)
        self.assertLessEqual(kutu["x1"], kutu["x2"])
        self.assertLessEqual(kutu["y1"], kutu["y2"])


if __name__ == "__main__":
    unittest.main()
