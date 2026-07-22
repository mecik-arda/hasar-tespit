import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.benchmark import (
    _gorsel_orijinal_mi,
    bellek_olcu_al,
    dogruluk_metriklerini_hesapla,
    kutu_iou_hesapla,
    rapor_kaydet,
)


class BellekVeFiltreTesti(unittest.TestCase):
    def test_bellek_olcumu_gerekli_alanlari_dondurur(self):
        sonuc = bellek_olcu_al()
        self.assertIn("toplam_ram_mb", sonuc)
        self.assertIn("surec_ram_mb", sonuc)
        self.assertIn("cuda_vram", sonuc)
        self.assertGreater(sonuc["toplam_ram_mb"], 0)

    def test_artirilmis_gorseller_filtrelenir(self):
        self.assertFalse(_gorsel_orijinal_mi(Path("veri") / "augmented" / "arac.jpg"))
        self.assertFalse(_gorsel_orijinal_mi(Path("veri") / "arac_augbright.jpg"))
        self.assertTrue(_gorsel_orijinal_mi(Path("veri") / "arac.jpg"))


class DogrulukMetrikleriTesti(unittest.TestCase):
    def test_ayni_kutularin_iou_degeri_birdir(self):
        self.assertAlmostEqual(kutu_iou_hesapla([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_mukemmel_tahmin_tam_skor_uretir(self):
        gercekler = [
            {"gorsel_id": "a", "sinif_id": 0, "kutucuk": [0, 0, 100, 100]},
            {"gorsel_id": "b", "sinif_id": 1, "kutucuk": [20, 20, 80, 80]},
        ]
        tahminler = [
            {"gorsel_id": "a", "sinif_id": 0, "guven": 0.95, "kutucuk": [0, 0, 100, 100]},
            {"gorsel_id": "b", "sinif_id": 1, "guven": 0.90, "kutucuk": [20, 20, 80, 80]},
        ]
        sonuc = dogruluk_metriklerini_hesapla(tahminler, gercekler, {0: "Cizik", 1: "Gocuk"})
        self.assertEqual(sonuc["mAP50"], 1.0)
        self.assertEqual(sonuc["mAP50_95"], 1.0)
        self.assertEqual(sonuc["precision"], 1.0)
        self.assertEqual(sonuc["recall"], 1.0)
        self.assertEqual(sonuc["tp"], 2)
        self.assertEqual(sonuc["fp"], 0)
        self.assertEqual(sonuc["fn"], 0)

    def test_sinif_uyusmazligi_fp_ve_fn_uretir(self):
        gercekler = [{"gorsel_id": "a", "sinif_id": 0, "kutucuk": [0, 0, 100, 100]}]
        tahminler = [{"gorsel_id": "a", "sinif_id": 1, "guven": 0.95, "kutucuk": [0, 0, 100, 100]}]
        sonuc = dogruluk_metriklerini_hesapla(tahminler, gercekler)
        self.assertEqual(sonuc["tp"], 0)
        self.assertEqual(sonuc["fp"], 1)
        self.assertEqual(sonuc["fn"], 1)
        self.assertEqual(sonuc["mAP50"], 0.0)


class RaporlamaTesti(unittest.TestCase):
    def test_json_ve_markdown_raporlari_olusturulur(self):
        rapor = {
            "zaman_damgasi": "2026-07-22T12:00:00+03:00",
            "calistirma": {"hedef": "gercek"},
            "ortam": {"python_surumu": "3.11"},
            "bellek_baslangic": {"surec_ram_mb": 100},
        }
        with tempfile.TemporaryDirectory() as gecici_klasor:
            yollar = rapor_kaydet(rapor, gecici_klasor)
            json_yolu = Path(yollar["json"])
            markdown_yolu = Path(yollar["markdown"])
            self.assertTrue(json_yolu.exists())
            self.assertTrue(markdown_yolu.exists())
            self.assertIn("rapor_dosyalari", json_yolu.read_text(encoding="utf-8"))
            self.assertIn("HADES Hyper Benchmark Raporu", markdown_yolu.read_text(encoding="utf-8"))


class PipelineZamanlamaTesti(unittest.TestCase):
    def test_coklu_model_ciktisi_asama_surelerini_icerir(self):
        yapilandirma = {
            "cikarim": {"gorsel_kaydet": False, "json_kaydet": False},
            "siniflar": {0: "Cizik"},
            "multi_model": {
                "aktif": True,
                "siralama": ["rt-detr-v2-x", "yolov12x", "sam2_small", "florence-2"],
                "agirliklar": {},
                "ram_optimizasyonu": False,
                "otomatik_yedekleme_cpu": True,
            },
        }
        sahte_sam = SimpleNamespace(predict=lambda **parametreler: [])
        sahte_ultralytics = SimpleNamespace(RTDETR=object, YOLO=object, SAM=object)
        with tempfile.TemporaryDirectory() as gecici_klasor:
            gorsel_yolu = Path(gecici_klasor) / "ornek.jpg"
            cv2.imwrite(str(gorsel_yolu), np.zeros((32, 32, 3), dtype=np.uint8))
            with patch.dict(sys.modules, {"ultralytics": sahte_ultralytics}):
                with patch("src.pipeline._tek_model_tara", return_value=0):
                    with patch("src.pipeline._wbf_kutu_birlestir", return_value=[]):
                        with patch("src.inspector_florence.denetle", side_effect=lambda havuz, gorsel, yapilandirma: havuz):
                            from src.pipeline import coklu_model_hasar_tespiti_yap
                            sonuc = coklu_model_hasar_tespiti_yap(
                                gorsel_yolu,
                                json_kaydet=False,
                                yapilandirma=yapilandirma,
                                hazir_modeller={"rtdetr": object(), "yolo": object(), "sam": sahte_sam},
                            )
        self.assertIn("asama_sureleri", sonuc)
        self.assertEqual(
            set(sonuc["asama_sureleri"]),
            {"rtdetr_saniye", "yolo_saniye", "wbf_saniye", "sam_saniye", "florence_saniye"},
        )


if __name__ == "__main__":
    unittest.main()
