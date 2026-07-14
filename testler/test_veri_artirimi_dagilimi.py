import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
import shutil
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src import data_tools

class VeriArtirimiTesti(unittest.TestCase):
    def setUp(self):
        self.gecici_klasor = PROJE_KOKU / "test_artirim"
        self.gecici_klasor.mkdir(exist_ok=True)
        
        self.gorsel_yolu = self.gecici_klasor / "ornek.jpg"
        resim = np.zeros((640, 640, 3), dtype=np.uint8)
        _, kodlanmis = cv2.imencode('.jpg', resim)
        kodlanmis.tofile(str(self.gorsel_yolu))
        
        self.etiket_yolu = self.gecici_klasor / "ornek.txt"
        with open(self.etiket_yolu, "w") as dosya:
            dosya.write("0 0.5 0.5 0.8 0.8\n")

    def tearDown(self):
        if self.gecici_klasor.exists():
            shutil.rmtree(self.gecici_klasor)

    @patch("src.data_tools.yapilandirma_yukle")
    def test_bounding_box_sinirlari(self, mock_yapi):
        mock_yapi.return_value = {
            "veri": {"etiket_klasoru": "test_artirim"},
            "augmentation": {
                "aktif": True,
                "carpma_katsayisi": 1,
                "donderme_acisi": 15,
                "parlaklik_limit": 0.3,
                "kontrast_limit": 0.3,
                "yatay_cevirme": True,
                "gauss_gurultu": True,
                "bulaniklastirma": True
            }
        }
        
        data_tools.augmentation_uygula()
        
        yeni_etiketler = list((self.gecici_klasor / "augmented").glob("*.txt"))
        self.assertGreater(len(yeni_etiketler), 0)
        
        for etiket_yolu in yeni_etiketler:
            with open(etiket_yolu, "r") as dosya:
                satirlar = dosya.readlines()
                for satir in satirlar:
                    parcalar = satir.strip().split()
                    if len(parcalar) == 5:
                        sinif_id, x_merkez, y_merkez, genislik, yukseklik = map(float, parcalar)
                        self.assertGreaterEqual(x_merkez, 0.0)
                        self.assertLessEqual(x_merkez, 1.0)
                        self.assertGreaterEqual(y_merkez, 0.0)
                        self.assertLessEqual(y_merkez, 1.0)

if __name__ == "__main__":
    unittest.main()
