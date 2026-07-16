import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))


class CaprazSorgulamaTesti(unittest.TestCase):
    def test_negatif_reflection_oncesi_elenir(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("this is a reflection, not a scratch")
        self.assertEqual(sonuc, "Bilinmeyen")

    def test_negatif_shadow_oncesi_elenir(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("this is just a shadow under the car")
        self.assertEqual(sonuc, "Bilinmeyen")

    def test_negatif_dirt_oncesi_elenir(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("this is dirt on the surface, not rust")
        self.assertEqual(sonuc, "Bilinmeyen")

    def test_negatif_mud_oncesi_elenir(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("mud splatter on the car door")
        self.assertEqual(sonuc, "Bilinmeyen")

    def test_pozitif_scratch_negatif_yoksa_korunur(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("deep scratch on the car hood")
        self.assertEqual(sonuc, "Cizik")

    def test_pozitif_dent_negatif_yoksa_korunur(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("large dent on the car door")
        self.assertEqual(sonuc, "Gocuk")

    def test_pozitif_crack_negatif_yoksa_korunur(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("crack on the windshield glass")
        self.assertEqual(sonuc, "Cam Kirigi")

    def test_pozitif_rust_negatif_yoksa_korunur(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("rust on the metal surface")
        self.assertEqual(sonuc, "Pas")

    def test_negatif_once_pozitif_sonra_bile_elenir(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("light reflection on the glass, not a crack")
        self.assertEqual(sonuc, "Bilinmeyen")

    def test_capraz_sorgu_sozlugu_tum_siniflari_kapsar(self):
        from src.inspector_florence import CAPRAZ_SORGULAR
        beklenen_siniflar = [
            "Cizik", "Gocuk", "Cam Kirigi", "Pas",
            "Kus Pisligi", "Far Kirigi", "Patlak Lastik",
        ]
        for sinif in beklenen_siniflar:
            self.assertIn(sinif, CAPRAZ_SORGULAR)
            self.assertIn("<DETAILED_CAPTION>", CAPRAZ_SORGULAR[sinif])

    def test_capraz_sorgu_bilinmeyen_sinif_fallback(self):
        from src.inspector_florence import CAPRAZ_SORGULAR
        sonuc = CAPRAZ_SORGULAR.get("BilinmeyenSinif", "<OD>")
        self.assertEqual(sonuc, "<OD>")

    def test_negatif_kelimeler_tam_eslesme_gerektirmez(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("shadows from the tree branches")
        self.assertEqual(sonuc, "Bilinmeyen")

    def test_reflection_ve_scratch_ayni_metinde_negatif_kazanir(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir(
            "This appears to be a light reflection rather than an actual scratch on the paint"
        )
        self.assertEqual(sonuc, "Bilinmeyen")


if __name__ == "__main__":
    unittest.main()
