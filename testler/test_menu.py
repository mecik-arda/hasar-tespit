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
    @patch("src.utils.yapilandirma_yukle", return_value={"multi_model": {"aktif": False}, "model": {"tur": "rtdetr", "agirlik": "rtdetr-x.pt"}, "siniflar": {}, "cikarim": {}})
    @patch("main.benchmark_calistir")
    @patch("main.gateway_testi_calistir")
    @patch("main.model_bilgisi_calistir")
    @patch("main.etiket_dogrulama_calistir")
    @patch("main.kalite_kontrol_calistir")
    @patch("main.gorsel_toplama_calistir")
    @patch("main.cikarim_profili_secimi_calistir")
    @patch("main.orkestrasyon_yoneticisi_calistir")
    @patch("main.egitim_modeli_secimi_calistir")
    @patch("main.testleri_calistir")
    @patch("main.rapor_calistir")
    @patch("main.cikarim_calistir")
    @patch("main.egitim_calistir")
    @patch("main.veri_bolme_calistir")
    @patch("main.augmentation_calistir")
    @patch("main.etiketleme_calistir")
    @patch("main.donanim_kontrolu_calistir")
    @patch("main.cikis_yap")
    def test_menu_araligi_0_17(
        self,
        m_cikis, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, _mock_config,
    ):
        self.assertFalse(main.menu_secimi_isle("0"))
        m_cikis.assert_called_once()
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
        self.assertTrue(main.menu_secimi_isle("15"))
        m15.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("16"))
        m16.assert_called_once()
        self.assertTrue(main.menu_secimi_isle("17"))
        m17.assert_called_once()

    @patch("main.cikis_yap")
    def test_menu_cikis_false_doner(self, mock_cikis):
        self.assertFalse(main.menu_secimi_isle("0"))
        mock_cikis.assert_called_once()

    @patch("sys.stdout", new_callable=StringIO)
    def test_gecersiz_secim_uyari_verir(self, mock_stdout):
        main.menu_secimi_isle("99")
        cikti = mock_stdout.getvalue()
        self.assertIn("Gecersiz secim", cikti)
        self.assertIn("0-17", cikti)

    @patch("main._gelismis_benchmark_sonucunu_yazdir")
    @patch("src.advanced_benchmarks.dayaniklilik_benchmark_calistir", return_value={"durum": "Tamamlandı"})
    @patch("builtins.input", side_effect=["2", "10"])
    def test_benchmark_menusu_dayaniklilik_suitini_yonlendirir(self, _mock_input, mock_dayaniklilik, mock_yazdir):
        main.benchmark_calistir()
        mock_dayaniklilik.assert_called_once_with(miktar=10)
        mock_yazdir.assert_called_once_with({"durum": "Tamamlandı"})


class ModelSecimiTesti(unittest.TestCase):
    def setUp(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        self.orijinal_config = yapilandirma_yukle()

    def tearDown(self):
        from src.utils import yapilandirma_kaydet
        yapilandirma_kaydet(self.orijinal_config)

    @patch("builtins.input", return_value="1")
    def test_egitim_modeli_secimi_yolo(self, mock_input):
        main.egitim_modeli_secimi_calistir()
        from src.utils import yapilandirma_yukle
        config = yapilandirma_yukle()
        self.assertEqual(config["model"]["tur"], "yolo")

    @patch("builtins.input", return_value="2")
    def test_egitim_modeli_secimi_rtdetr(self, mock_input):
        main.egitim_modeli_secimi_calistir()
        from src.utils import yapilandirma_yukle
        config = yapilandirma_yukle()
        self.assertEqual(config["model"]["tur"], "rtdetr")
        self.assertIn("rtdetr", config["model"]["agirlik"])

    @patch("builtins.input", return_value="99")
    def test_egitim_modeli_secimi_gecersiz_iptal(self, mock_input):
        from src.utils import yapilandirma_yukle
        onceki = yapilandirma_yukle()
        main.egitim_modeli_secimi_calistir()
        sonraki = yapilandirma_yukle()
        self.assertEqual(onceki["model"]["tur"], sonraki["model"]["tur"])

    @patch("builtins.input", side_effect=["1", "1", "1"])
    def test_egitim_modeli_secimi_yolo_v8_nano(self, mock_input):
        main.egitim_modeli_secimi_calistir()
        from src.utils import yapilandirma_yukle
        config = yapilandirma_yukle()
        self.assertIn("yolov8n.pt", config["model"]["agirlik"])

    @patch("builtins.input", side_effect=["2", "1"])
    def test_egitim_modeli_secimi_rtdetr_large(self, mock_input):
        main.egitim_modeli_secimi_calistir()
        from src.utils import yapilandirma_yukle
        config = yapilandirma_yukle()
        self.assertEqual(config["model"]["agirlik"], "rtdetr-l.pt")


class MenuOrkestrasyonTesti(unittest.TestCase):
    def setUp(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        self.orijinal_config = yapilandirma_yukle()

    def tearDown(self):
        from src.utils import yapilandirma_kaydet
        yapilandirma_kaydet(self.orijinal_config)

    @patch("builtins.input", return_value="")
    @patch("main.egitim_modeli_secimi_calistir")
    @patch("src.utils.yapilandirma_yukle", return_value={"multi_model": {"aktif": True}, "model": {"tur": "rtdetr", "agirlik": "rtdetr-x.pt"}, "siniflar": {}, "cikarim": {}})
    def test_menu_9_coklu_model_aktifken_engellenir(self, mock_config, mock_egitim_secim, mock_input):
        sonuc = main.menu_secimi_isle("9")
        self.assertTrue(sonuc)
        mock_egitim_secim.assert_not_called()

    @patch("main.egitim_calistir")
    @patch("src.utils.yapilandirma_kaydet")
    @patch("builtins.input", return_value="1")
    @patch("src.utils.yapilandirma_yukle", return_value={"multi_model": {"aktif": True}, "model": {"tur": "rtdetr", "agirlik": "rtdetr-x.pt"}, "siniflar": {}, "cikarim": {}})
    def test_menu_5_coklu_model_alt_model_yolo(self, mock_config, mock_input, mock_kaydet, mock_egitim):
        sonuc = main.menu_secimi_isle("5")
        self.assertTrue(sonuc)
        mock_kaydet.assert_called_once()
        self.assertEqual(mock_kaydet.call_args[0][0]["model"]["tur"], "yolo")
        mock_egitim.assert_called_once()

    @patch("main.egitim_calistir")
    @patch("src.utils.yapilandirma_kaydet")
    @patch("builtins.input", return_value="2")
    @patch("src.utils.yapilandirma_yukle", return_value={"multi_model": {"aktif": True}, "model": {"tur": "yolo", "agirlik": "yolo12n.pt"}, "siniflar": {}, "cikarim": {}})
    def test_menu_5_coklu_model_alt_model_rtdetr(self, mock_config, mock_input, mock_kaydet, mock_egitim):
        sonuc = main.menu_secimi_isle("5")
        self.assertTrue(sonuc)
        mock_kaydet.assert_called_once()
        self.assertEqual(mock_kaydet.call_args[0][0]["model"]["tur"], "rtdetr")
        mock_egitim.assert_called_once()

    @patch("main.egitim_calistir")
    @patch("src.utils.yapilandirma_kaydet")
    @patch("builtins.input", return_value="0")
    @patch("src.utils.yapilandirma_yukle", return_value={"multi_model": {"aktif": True}, "model": {"tur": "rtdetr", "agirlik": "rtdetr-x.pt"}, "siniflar": {}, "cikarim": {}})
    def test_menu_5_coklu_model_iptal(self, mock_config, mock_input, mock_kaydet, mock_egitim):
        sonuc = main.menu_secimi_isle("5")
        self.assertTrue(sonuc)
        mock_kaydet.assert_not_called()
        mock_egitim.assert_not_called()

    @patch("main.egitim_calistir")
    @patch("src.utils.yapilandirma_kaydet")
    @patch("builtins.input", return_value="99")
    @patch("src.utils.yapilandirma_yukle", return_value={"multi_model": {"aktif": True}, "model": {"tur": "rtdetr", "agirlik": "rtdetr-x.pt"}, "siniflar": {}, "cikarim": {}})
    def test_menu_5_coklu_model_gecersiz_secim(self, mock_config, mock_input, mock_kaydet, mock_egitim):
        sonuc = main.menu_secimi_isle("5")
        self.assertTrue(sonuc)
        mock_kaydet.assert_not_called()
        mock_egitim.assert_not_called()


if __name__ == "__main__":
    unittest.main()
