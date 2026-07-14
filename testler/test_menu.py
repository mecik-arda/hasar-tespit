import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from io import StringIO

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

import main


class MenuAralikTesti(unittest.TestCase):
    @patch("main.donanim_kontrolu_calistir")
    @patch("main.etiketleme_calistir")
    @patch("main.augmentation_calistir")
    @patch("main.veri_bolme_calistir")
    @patch("main.egitim_calistir")
    @patch("main.cikarim_calistir")
    @patch("main.rapor_calistir")
    @patch("main.testleri_calistir")
    @patch("main.model_secimi_calistir")
    @patch("main.ayarlar_calistir")
    @patch("main.gorsel_toplama_calistir")
    @patch("main.kalite_kontrol_calistir")
    @patch("main.etiket_dogrulama_calistir")
    @patch("main.model_bilgisi_calistir")
    @patch("main.cikis_yap")
    def test_menu_araligi_0_10(
        self, m0, m14, m13, m12, m11, m10, m9, m8, m7, m6, m5, m4, m3, m2, m1,
    ):
        self.assertTrue(main.menu_secimi_isle("1"))
        m1.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("2"))
        m2.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("3"))
        m3.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("4"))
        m4.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("5"))
        m5.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("6"))
        m6.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("7"))
        m7.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("8"))
        m8.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("9"))
        m9.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("10"))
        m10.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("11"))
        m11.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("12"))
        m12.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("13"))
        m13.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("14"))
        m14.assert_called_once()

    @patch("main.cikis_yap")
    def test_menu_cikis_false_doner(self, mock_cikis):
        self.assertFalse(main.menu_secimi_isle("0"))
        mock_cikis.assert_called_once()

    @patch("sys.stdout", new_callable=StringIO)
    def test_gecersiz_secim_uyari_verir(self, mock_stdout):
        main.menu_secimi_isle("99")
        cikti = mock_stdout.getvalue()
        self.assertIn("Gecersiz secim", cikti)
        self.assertIn("0-14", cikti)


class ModelSecimiTesti(unittest.TestCase):
    def setUp(self):
        from src.pipeline import yapilandirma_yukle, yapilandirma_kaydet
        self.orijinal_config = yapilandirma_yukle()

    def tearDown(self):
        from src.pipeline import yapilandirma_kaydet
        yapilandirma_kaydet(self.orijinal_config)

    @patch("builtins.input", return_value="1")
    def test_model_secimi_yolo(self, mock_input):
        main.model_secimi_calistir()
        from src.pipeline import yapilandirma_yukle
        config = yapilandirma_yukle()
        self.assertEqual(config["model"]["tur"], "yolo")

    @patch("builtins.input", return_value="2")
    def test_model_secimi_rtdetr(self, mock_input):
        main.model_secimi_calistir()
        from src.pipeline import yapilandirma_yukle
        config = yapilandirma_yukle()
        self.assertEqual(config["model"]["tur"], "rtdetr")
        self.assertIn("rtdetr", config["model"]["agirlik"])

    @patch("builtins.input", return_value="99")
    def test_model_secimi_gecersiz_iptal(self, mock_input):
        from src.pipeline import yapilandirma_yukle
        onceki = yapilandirma_yukle()
        main.model_secimi_calistir()
        sonraki = yapilandirma_yukle()
        self.assertEqual(onceki["model"]["tur"], sonraki["model"]["tur"])

    @patch("builtins.input", side_effect=["1", "1"])
    def test_ayarlar_yolo_nano(self, mock_input):
        from src.pipeline import yapilandirma_yukle, yapilandirma_kaydet
        config = yapilandirma_yukle()
        config["model"]["tur"] = "yolo"
        yapilandirma_kaydet(config)
        main.ayarlar_calistir()
        config = yapilandirma_yukle()
        self.assertIn("yolov8n.pt", config["model"]["agirlik"])

    @patch("builtins.input", side_effect=["1"])
    def test_ayarlar_rtdetr_large(self, mock_input):
        from src.pipeline import yapilandirma_yukle, yapilandirma_kaydet
        config = yapilandirma_yukle()
        config["model"]["tur"] = "rtdetr"
        yapilandirma_kaydet(config)
        main.ayarlar_calistir()
        config = yapilandirma_yukle()
        self.assertEqual(config["model"]["agirlik"], "rtdetr-l.pt")


if __name__ == "__main__":
    unittest.main()
