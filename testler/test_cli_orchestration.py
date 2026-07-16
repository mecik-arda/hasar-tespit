import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

ORIJINAL_CONFIG_YOLU = PROJE_KOKU / "config.yaml"


class CLIIOrchestrationTesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._yedek_klasor = tempfile.mkdtemp(prefix="hades_test_")
        cls._yedek_config = Path(cls._yedek_klasor) / "config_backup.yaml"
        shutil.copy2(str(ORIJINAL_CONFIG_YOLU), str(cls._yedek_config))

    @classmethod
    def tearDownClass(cls):
        shutil.copy2(str(cls._yedek_config), str(ORIJINAL_CONFIG_YOLU))
        shutil.rmtree(cls._yedek_klasor, ignore_errors=True)

    def setUp(self):
        import yaml
        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            self.orijinal_config = yaml.safe_load(f)

    def tearDown(self):
        import yaml
        from src.utils import yapilandirma_kaydet
        yapilandirma_kaydet(self.orijinal_config)

    def test_egitim_model_birlestirme_yolo_to_rtdetr(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "model" not in config:
            config["model"] = {}
        config["model"]["tur"] = "rtdetr"
        config["model"]["agirlik"] = "rtdetr-x.pt"
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertEqual(kaydedilen["model"]["tur"], "rtdetr")
        self.assertEqual(kaydedilen["model"]["agirlik"], "rtdetr-x.pt")

    def test_egitim_model_birlestirme_boyut_degisimi(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "model" not in config:
            config["model"] = {}
        config["model"]["tur"] = "yolo"
        config["model"]["agirlik"] = "yolo12x.pt"
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertEqual(kaydedilen["model"]["tur"], "yolo")
        self.assertEqual(kaydedilen["model"]["agirlik"], "yolo12x.pt")

    def test_egitim_model_izolasyon_multi_model_etkilenmez(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        orijinal_multi_siralama = config.get("multi_model", {}).get("siralama", []).copy()
        orijinal_multi_aktif = config.get("multi_model", {}).get("aktif", False)
        orijinal_multi_agirliklar = config.get("multi_model", {}).get("agirliklar", {}).copy()

        if "model" not in config:
            config["model"] = {}
        config["model"]["tur"] = "rtdetr"
        config["model"]["agirlik"] = "rtdetr-l.pt"
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertEqual(kaydedilen["model"]["tur"], "rtdetr")
        self.assertEqual(kaydedilen["multi_model"]["siralama"], orijinal_multi_siralama)
        self.assertEqual(kaydedilen["multi_model"]["aktif"], orijinal_multi_aktif)
        self.assertEqual(kaydedilen["multi_model"]["agirliklar"], orijinal_multi_agirliklar)

    def test_orkestrasyon_yoneticisi_yolo_agirlik_guncelleme(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "multi_model" not in config:
            config["multi_model"] = {}
        if "agirliklar" not in config["multi_model"]:
            config["multi_model"]["agirliklar"] = {}
        config["multi_model"]["agirliklar"]["yolo"] = "yolov12x_test.pt"
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertEqual(kaydedilen["multi_model"]["agirliklar"]["yolo"], "yolov12x_test.pt")

    def test_orkestrasyon_yoneticisi_sam_agirlik_guncelleme(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "multi_model" not in config:
            config["multi_model"] = {}
        if "agirliklar" not in config["multi_model"]:
            config["multi_model"]["agirliklar"] = {}
        config["multi_model"]["agirliklar"]["sam"] = "sam2_b_test.pt"
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertEqual(kaydedilen["multi_model"]["agirliklar"]["sam"], "sam2_b_test.pt")

    def test_orkestrasyon_yoneticisi_florence_model_guncelleme(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "multi_model" not in config:
            config["multi_model"] = {}
        if "denetleyici_ayarlari" not in config["multi_model"]:
            config["multi_model"]["denetleyici_ayarlari"] = {}
        config["multi_model"]["denetleyici_ayarlari"]["model"] = "microsoft/Florence-2-large-test"
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertEqual(kaydedilen["multi_model"]["denetleyici_ayarlari"]["model"], "microsoft/Florence-2-large-test")

    def test_orkestrasyon_yoneticisi_izolasyon_model_etkilenmez(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        orijinal_model = config.get("model", {}).copy()

        if "multi_model" not in config:
            config["multi_model"] = {}
        if "agirliklar" not in config["multi_model"]:
            config["multi_model"]["agirliklar"] = {}
        config["multi_model"]["agirliklar"]["rtdetr"] = "rtdetr-v2-x_test.pt"
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertEqual(kaydedilen["model"], orijinal_model)

    def test_cikarim_profili_hiz(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "multi_model" not in config:
            config["multi_model"] = {}
        config["multi_model"]["aktif"] = False
        config["multi_model"]["siralama"] = []
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertFalse(kaydedilen["multi_model"]["aktif"])
        self.assertEqual(kaydedilen["multi_model"]["siralama"], [])

    def test_cikarim_profili_hibrit(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "multi_model" not in config:
            config["multi_model"] = {}
        config["multi_model"]["aktif"] = True
        config["multi_model"]["siralama"] = ["rt-detr-v2-x", "yolov12x"]
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertTrue(kaydedilen["multi_model"]["aktif"])
        self.assertEqual(kaydedilen["multi_model"]["siralama"], ["rt-detr-v2-x", "yolov12x"])

    def test_cikarim_profili_kusursuz(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "multi_model" not in config:
            config["multi_model"] = {}
        config["multi_model"]["aktif"] = True
        config["multi_model"]["siralama"] = ["rt-detr-v2-x", "yolov12x", "sam2_small", "florence-2"]
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertTrue(kaydedilen["multi_model"]["aktif"])
        self.assertEqual(len(kaydedilen["multi_model"]["siralama"]), 4)

    def test_cikarim_profili_ozel(self):
        from src.utils import yapilandirma_yukle, yapilandirma_kaydet
        import yaml

        config = yapilandirma_yukle()
        if "multi_model" not in config:
            config["multi_model"] = {}
        config["multi_model"]["aktif"] = True
        config["multi_model"]["siralama"] = ["rt-detr-v2-x", "sam2_small"]
        yapilandirma_kaydet(config)

        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            kaydedilen = yaml.safe_load(f)

        self.assertTrue(kaydedilen["multi_model"]["aktif"])
        self.assertEqual(kaydedilen["multi_model"]["siralama"], ["rt-detr-v2-x", "sam2_small"])

    def test_profil_adi_bul_tek_model(self):
        from main import _profil_adi_bul
        config = {"multi_model": {"aktif": False, "siralama": []}}
        self.assertEqual(_profil_adi_bul(config), "Tek Model")

    def test_profil_adi_bul_kusursuz(self):
        from main import _profil_adi_bul
        config = {
            "multi_model": {
                "aktif": True,
                "siralama": ["rt-detr-v2-x", "yolov12x", "sam2_small", "florence-2"],
            }
        }
        self.assertEqual(_profil_adi_bul(config), "Kusursuz (4 Model)")

    def test_profil_adi_bul_hibrit(self):
        from main import _profil_adi_bul
        config = {
            "multi_model": {
                "aktif": True,
                "siralama": ["rt-detr-v2-x", "yolov12x"],
            }
        }
        self.assertEqual(_profil_adi_bul(config), "Hibrit (RT-DETR + YOLO)")

    def test_profil_adi_bul_hiz(self):
        from main import _profil_adi_bul
        config = {
            "multi_model": {
                "aktif": True,
                "siralama": ["rt-detr-v2-x"],
            }
        }
        self.assertEqual(_profil_adi_bul(config), "Hiz (RT-DETR)")

    def test_profil_adi_bul_ozel(self):
        from main import _profil_adi_bul
        config = {
            "multi_model": {
                "aktif": True,
                "siralama": ["rt-detr-v2-x", "florence-2"],
            }
        }
        sonuc = _profil_adi_bul(config)
        self.assertIn("Ozel", sonuc)


if __name__ == "__main__":
    unittest.main()
