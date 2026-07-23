import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.pipeline import (
    _ram_havuzu_olustur,
    _wbf_kutu_birlestir,
    _wbf_model_agirliklarini_hesapla,
    yapilandirma_yukle,
)


class WbfTesti(unittest.TestCase):
    def setUp(self):
        self.havuz = _ram_havuzu_olustur()

    def _iki_model_tahminlerini_olustur(self, sinif_id, sinif_adi):
        return {
            "boxes": [
                {
                    "sinif_id": sinif_id,
                    "sinif_adi": sinif_adi,
                    "guven": 0.6,
                    "kutucuk": {"x1": 100, "y1": 100, "x2": 500, "y2": 500},
                    "kaynak_model": "rt-detr-v2-x",
                },
                {
                    "sinif_id": sinif_id,
                    "sinif_adi": sinif_adi,
                    "guven": 0.9,
                    "kutucuk": {"x1": 150, "y1": 150, "x2": 550, "y2": 550},
                    "kaynak_model": "yolov12x",
                },
            ]
        }

    def _esit_agirlikli_birlestir(self, sinif_id, sinif_adi):
        return _wbf_kutu_birlestir(
            self._iki_model_tahminlerini_olustur(sinif_id, sinif_adi),
            1000,
            1000,
            iou_esigi=0.55,
            guven_esigi=0.25,
            yapilandirma={"multi_model": {"wbf_dinamik_agirliklandirma": {"aktif": False}}},
        )[0]

    def _config_agirlikli_birlestir(self, sinif_id, sinif_adi):
        return _wbf_kutu_birlestir(
            self._iki_model_tahminlerini_olustur(sinif_id, sinif_adi),
            1000,
            1000,
            iou_esigi=0.55,
            guven_esigi=0.25,
            yapilandirma=yapilandirma_yukle(),
        )[0]

    def test_config_sinif_bazli_wbf_agirlik_sozlesmesi(self):
        agirliklar = yapilandirma_yukle()["multi_model"]["wbf_sinif_agirliklari"]
        self.assertEqual(
            agirliklar,
            {
                "Cizik": {"yolov12x": 2.0, "rt-detr-v2-x": 1.0},
                "Gocuk": {"rt-detr-v2-x": 2.0, "yolov12x": 1.0},
            },
        )

    def test_cizik_ikiye_bir_agirligi_skoru_ve_kutuyu_yoloya_yaklastirir(self):
        esit_sonuc = self._esit_agirlikli_birlestir(0, "Cizik")
        agirlikli_sonuc = self._config_agirlikli_birlestir(0, "Cizik")
        self.assertEqual(esit_sonuc["guven"], 0.75)
        self.assertEqual(esit_sonuc["kutucuk"], {"x1": 130, "y1": 130, "x2": 530, "y2": 530})
        self.assertEqual(agirlikli_sonuc["guven"], 0.8)
        self.assertEqual(agirlikli_sonuc["kutucuk"], {"x1": 137, "y1": 137, "x2": 537, "y2": 537})
        self.assertEqual(
            agirlikli_sonuc["wbf_model_agirliklari"],
            {"rt-detr-v2-x": 1.0, "yolov12x": 2.0},
        )
        self.assertGreater(agirlikli_sonuc["guven"], esit_sonuc["guven"])
        self.assertGreater(agirlikli_sonuc["kutucuk"]["x1"], esit_sonuc["kutucuk"]["x1"])

    def test_gocuk_ikiye_bir_agirligi_skoru_ve_kutuyu_rtdetre_yaklastirir(self):
        esit_sonuc = self._esit_agirlikli_birlestir(1, "Gocuk")
        agirlikli_sonuc = self._config_agirlikli_birlestir(1, "Gocuk")
        self.assertEqual(esit_sonuc["guven"], 0.75)
        self.assertEqual(esit_sonuc["kutucuk"], {"x1": 130, "y1": 130, "x2": 530, "y2": 530})
        self.assertEqual(agirlikli_sonuc["guven"], 0.7)
        self.assertEqual(agirlikli_sonuc["kutucuk"], {"x1": 121, "y1": 121, "x2": 521, "y2": 521})
        self.assertEqual(
            agirlikli_sonuc["wbf_model_agirliklari"],
            {"rt-detr-v2-x": 2.0, "yolov12x": 1.0},
        )
        self.assertLess(agirlikli_sonuc["guven"], esit_sonuc["guven"])
        self.assertLess(agirlikli_sonuc["kutucuk"]["x1"], esit_sonuc["kutucuk"]["x1"])

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
