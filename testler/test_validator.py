import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
import shutil
import cv2
import numpy as np
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.validator import (
    etiket_format_kontrolu,
    etiket_sinir_kontrolu,
    etiket_sinif_kontrolu,
    etiket_boyut_kontrolu,
    etiket_overlap_kontrolu,
    etiket_eslesme_kontrolu,
    etiket_dagilim_raporu,
    etiket_validator_calistir,
)


class ValidatorTesti(unittest.TestCase):
    def setUp(self):
        self.gecici_klasor = PROJE_KOKU / "test_validator_gecici"
        self.gecici_klasor.mkdir(exist_ok=True)
        self.etiket_klasoru = self.gecici_klasor

    def tearDown(self):
        if self.gecici_klasor.exists():
            shutil.rmtree(self.gecici_klasor)

    def _etiket_yaz(self, ad, icerik):
        with open(self.etiket_klasoru / f"{ad}.txt", "w", encoding="utf-8") as f:
            f.write(icerik)

    def _gorsel_yaz(self, ad, genislik=640, yukseklik=640):
        resim = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)
        _, kodlanmis = cv2.imencode('.jpg', resim)
        kodlanmis.tofile(str(self.etiket_klasoru / f"{ad}.jpg"))

    def test_format_kontrolu_gecerli(self):
        self._etiket_yaz("test1", "0 0.5 0.5 0.2 0.2\n")
        hatalar = etiket_format_kontrolu(self.etiket_klasoru)
        self.assertEqual(len(hatalar), 0)

    def test_format_kontrolu_eksik_deger(self):
        self._etiket_yaz("hatali", "0 0.5 0.5\n")
        hatalar = etiket_format_kontrolu(self.etiket_klasoru)
        self.assertGreater(len(hatalar), 0)

    def test_sinir_kontrolu_gecerli(self):
        self._etiket_yaz("test1", "0 0.5 0.5 0.2 0.2\n")
        hatalar = etiket_sinir_kontrolu(self.etiket_klasoru)
        self.assertEqual(len(hatalar), 0)

    def test_sinir_kontrolu_tasma(self):
        self._etiket_yaz("tas", "0 1.5 0.5 0.2 0.2\n")
        hatalar = etiket_sinir_kontrolu(self.etiket_klasoru)
        self.assertGreater(len(hatalar), 0)

    def test_sinif_kontrolu_gecerli(self):
        self._etiket_yaz("test1", "0 0.5 0.5 0.2 0.2\n3 0.3 0.3 0.1 0.1\n")
        hatalar = etiket_sinif_kontrolu(self.etiket_klasoru, 5)
        self.assertEqual(len(hatalar), 0)

    def test_sinif_kontrolu_gecersiz(self):
        self._etiket_yaz("hatali", "9 0.5 0.5 0.2 0.2\n")
        hatalar = etiket_sinif_kontrolu(self.etiket_klasoru, 5)
        self.assertGreater(len(hatalar), 0)

    def test_boyut_kontrolu_cok_kucuk(self):
        self._etiket_yaz("kucuk", "0 0.5 0.5 0.001 0.001\n")
        hatalar = etiket_boyut_kontrolu(self.etiket_klasoru)
        self.assertGreater(len(hatalar), 0)

    def test_boyut_kontrolu_cok_buyuk(self):
        self._etiket_yaz("buyuk", "0 0.5 0.5 0.99 0.99\n")
        hatalar = etiket_boyut_kontrolu(self.etiket_klasoru)
        self.assertGreater(len(hatalar), 0)

    def test_overlap_kontrolu_gecerli(self):
        self._etiket_yaz("ayrik", "0 0.2 0.2 0.1 0.1\n1 0.6 0.6 0.1 0.1\n")
        hatalar = etiket_overlap_kontrolu(self.etiket_klasoru)
        self.assertEqual(len(hatalar), 0)

    def test_overlap_kontrolu_cakisan(self):
        self._etiket_yaz("cakisan", "0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.5 0.5\n")
        hatalar = etiket_overlap_kontrolu(self.etiket_klasoru)
        self.assertGreater(len(hatalar), 0)

    def test_eslesme_kontrolu_etiketsiz(self):
        self._gorsel_yaz("etiketsiz_resim")
        self._etiket_yaz("etiketli_resim", "0 0.5 0.5 0.2 0.2\n")
        self._gorsel_yaz("etiketli_resim")
        sonuc = etiket_eslesme_kontrolu(self.etiket_klasoru, self.etiket_klasoru)
        self.assertGreater(len(sonuc["etiketsiz"]), 0)

    def test_dagilim_raporu(self):
        siniflar = {0: "Cizik", 1: "Gocuk", 2: "Cam Kirigi"}
        self._etiket_yaz("test1", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
        self._etiket_yaz("test2", "0 0.6 0.6 0.1 0.1\n")
        dagilim, toplam = etiket_dagilim_raporu(self.etiket_klasoru, siniflar)
        self.assertEqual(toplam, 3)
        self.assertEqual(dagilim["Cizik"], 2)
        self.assertEqual(dagilim["Gocuk"], 1)

    def test_validator_calistir_hatasiz(self):
        self._gorsel_yaz("temiz1")
        self._gorsel_yaz("temiz2")
        self._etiket_yaz("temiz1", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
        self._etiket_yaz("temiz2", "2 0.6 0.6 0.15 0.15\n")
        sonuc = etiket_validator_calistir(klasor=str(self.etiket_klasoru))
        self.assertIsNotNone(sonuc)
        self.assertEqual(sonuc["toplam"], 3)
        self.assertEqual(len(sonuc["hatalar"]), 0)


if __name__ == "__main__":
    unittest.main()
