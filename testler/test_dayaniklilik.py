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

from src.pipeline import hasar_tespiti_yap

class DayaniklilikTesti(unittest.TestCase):
    def setUp(self):
        self.gecici_klasor = PROJE_KOKU / "test_gecici"
        self.gecici_klasor.mkdir(exist_ok=True)
        
        self.orijinal_gorsel_yolu = self.gecici_klasor / "orijinal.jpg"
        resim = np.zeros((640, 640, 3), dtype=np.uint8)
        _, kodlanmis = cv2.imencode('.jpg', resim)
        kodlanmis.tofile(str(self.orijinal_gorsel_yolu))

    def tearDown(self):
        if self.gecici_klasor.exists():
            shutil.rmtree(self.gecici_klasor)

    def test_karanlik_gorsel_dayanikliligi(self):
        gorsel_dizisi = np.fromfile(str(self.orijinal_gorsel_yolu), dtype=np.uint8)
        gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
        
        karanlik_gorsel = cv2.convertScaleAbs(gorsel, alpha=0.3, beta=0)
        
        karanlik_yol = self.gecici_klasor / "karanlik.jpg"
        _, kodlanmis = cv2.imencode('.jpg', karanlik_gorsel)
        kodlanmis.tofile(str(karanlik_yol))
        
        sonuc = hasar_tespiti_yap(str(karanlik_yol))
        
        self.assertTrue(self.gecici_klasor.exists())
        self.assertTrue(karanlik_yol.exists())

    def test_gurultulu_gorsel_dayanikliligi(self):
        gorsel_dizisi = np.fromfile(str(self.orijinal_gorsel_yolu), dtype=np.uint8)
        gorsel = cv2.imdecode(gorsel_dizisi, cv2.IMREAD_COLOR)
        
        gurultu = np.random.normal(0, 25, gorsel.shape).astype(np.uint8)
        gurultulu_gorsel = cv2.add(gorsel, gurultu)
        
        gurultulu_yol = self.gecici_klasor / "gurultulu.jpg"
        _, kodlanmis = cv2.imencode('.jpg', gurultulu_gorsel)
        kodlanmis.tofile(str(gurultulu_yol))
        
        sonuc = hasar_tespiti_yap(str(gurultulu_yol))
        
        self.assertTrue(gurultulu_yol.exists())

if __name__ == "__main__":
    unittest.main()
