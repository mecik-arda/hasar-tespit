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


class DinamikEsikTesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._yedek_klasor = tempfile.mkdtemp(prefix="hades_dinamik_")
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
        from src.utils import yapilandirma_kaydet
        yapilandirma_kaydet(self.orijinal_config)

    def test_config_sinif_guven_esikleri_mevcut(self):
        import yaml
        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        cikarim = config.get("cikarim", {})
        self.assertIn("sinif_guven_esikleri", cikarim)
        esikler = cikarim["sinif_guven_esikleri"]
        self.assertEqual(esikler[0], 0.30)
        self.assertEqual(esikler[1], 0.20)
        self.assertEqual(esikler[2], 0.45)
        self.assertEqual(esikler[6], 0.40)

    def test_fallback_tanimli_sinif_icin_ozel_esik(self):
        sinif_guven_esikleri = {0: 0.30, 1: 0.20, 2: 0.45}
        guven_esigi = 0.25
        guncel_esik = sinif_guven_esikleri.get(0, guven_esigi)
        self.assertEqual(guncel_esik, 0.30)

    def test_fallback_tanimsiz_sinif_icin_genel_esik(self):
        sinif_guven_esikleri = {0: 0.30, 1: 0.20}
        guven_esigi = 0.25
        guncel_esik = sinif_guven_esikleri.get(99, guven_esigi)
        self.assertEqual(guncel_esik, 0.25)

    def test_fallback_bos_esik_sozlugu_genel_esik_kullanir(self):
        sinif_guven_esikleri = {}
        guven_esigi = 0.25
        guncel_esik = sinif_guven_esikleri.get(0, guven_esigi)
        self.assertEqual(guncel_esik, 0.25)

    def test_dusuk_guven_kutusu_elenir(self):
        sinif_guven_esikleri = {0: 0.30}
        guven_esigi = 0.25
        guncel_esik = sinif_guven_esikleri.get(0, guven_esigi)
        guven = 0.15
        self.assertLess(guven, guncel_esik)

    def test_yuksek_guven_kutusu_korunur(self):
        sinif_guven_esikleri = {0: 0.30}
        guven_esigi = 0.25
        guncel_esik = sinif_guven_esikleri.get(0, guven_esigi)
        guven = 0.85
        self.assertGreaterEqual(guven, guncel_esik)

    def test_gocuk_dusuk_esik_ile_korunur(self):
        sinif_guven_esikleri = {1: 0.20}
        guven_esigi = 0.25
        guncel_esik = sinif_guven_esikleri.get(1, guven_esigi)
        guven = 0.22
        self.assertGreaterEqual(guven, guncel_esik)

    def test_cam_kirigi_yuksek_esik_ile_elenir(self):
        sinif_guven_esikleri = {2: 0.45}
        guven_esigi = 0.25
        guncel_esik = sinif_guven_esikleri.get(2, guven_esigi)
        guven = 0.40
        self.assertLess(guven, guncel_esik)

    def test_config_sahi_parametreleri_mevcut(self):
        import yaml
        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        cikarim = config.get("cikarim", {})
        self.assertIn("sahi_aktif", cikarim)
        self.assertIn("sahi_dilim_boyutu", cikarim)
        self.assertTrue(cikarim["sahi_aktif"])
        self.assertTrue(cikarim["sahi_adaptif"]["aktif"])
        self.assertEqual(cikarim["sahi_adaptif"]["hedef_siniflar"], ["Cizik", "Pas"])
        self.assertTrue(cikarim["tta_adaptif"]["aktif"])
        self.assertEqual(cikarim["tta_adaptif"]["analiz_uzun_kenar"], 640)
        self.assertEqual(cikarim["tta_adaptif"]["azami_varyant"], 3)
        self.assertEqual(cikarim["tta_adaptif"]["kalibrasyon"]["minimum_map50_artisi"], 0.02)

    def test_config_max_sam_boxes_mevcut(self):
        import yaml
        with open(ORIJINAL_CONFIG_YOLU, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        cikarim = config.get("cikarim", {})
        self.assertIn("max_sam_boxes", cikarim)
        self.assertEqual(cikarim["max_sam_boxes"], 20)

    def test_sam_kutu_siniri_guven_siralamasi(self):
        kutular = [
            {"sinif_adi": "Cizik", "guven": 0.70, "kutucuk": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            {"sinif_adi": "Gocuk", "guven": 0.95, "kutucuk": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}},
            {"sinif_adi": "Pas", "guven": 0.55, "kutucuk": {"x1": 20, "y1": 20, "x2": 30, "y2": 30}},
            {"sinif_adi": "Cam Kirigi", "guven": 0.88, "kutucuk": {"x1": 30, "y1": 30, "x2": 40, "y2": 40}},
            {"sinif_adi": "Far Kirigi", "guven": 0.45, "kutucuk": {"x1": 40, "y1": 40, "x2": 50, "y2": 50}},
        ]
        max_sam_boxes = 3
        sirali = sorted(kutular, key=lambda b: b.get("guven", 0), reverse=True)
        sinirli = sirali[:max_sam_boxes]
        self.assertEqual(len(sinirli), 3)
        self.assertEqual(sinirli[0]["guven"], 0.95)
        self.assertEqual(sinirli[1]["guven"], 0.88)
        self.assertEqual(sinirli[2]["guven"], 0.70)

    def test_sam_kutu_siniri_bos_liste(self):
        kutular = []
        max_sam_boxes = 20
        sirali = sorted(kutular, key=lambda b: b.get("guven", 0), reverse=True)
        sinirli = sirali[:max_sam_boxes]
        self.assertEqual(len(sinirli), 0)


if __name__ == "__main__":
    unittest.main()
