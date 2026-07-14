import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
import threading
import shutil
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src import pipeline

class StresTesti(unittest.TestCase):
    def setUp(self):
        self.gecici_klasor = PROJE_KOKU / "test_stres"
        self.gecici_klasor.mkdir(exist_ok=True)
        
        self.gorsel_yolu = self.gecici_klasor / "stres.jpg"
        resim = np.zeros((640, 640, 3), dtype=np.uint8)
        _, kodlanmis = cv2.imencode('.jpg', resim)
        kodlanmis.tofile(str(self.gorsel_yolu))

    def tearDown(self):
        if self.gecici_klasor.exists():
            shutil.rmtree(self.gecici_klasor)

    def is_parcacigi(self, sonuclar_listesi, index):
        sonuc = pipeline.hasar_tespiti_yap(str(self.gorsel_yolu))
        sonuclar_listesi[index] = sonuc

    @patch("src.pipeline.egitilmis_model_yolu_bul")
    def test_es_zamanlilik(self, mock_model_yolu):
        mock_model_yolu.return_value = PROJE_KOKU / "yolov8n.pt"
        
        is_parcaciklari = []
        parcacik_sayisi = 5
        sonuclar = [None] * parcacik_sayisi
        
        for i in range(parcacik_sayisi):
            parcacik = threading.Thread(target=self.is_parcacigi, args=(sonuclar, i))
            is_parcaciklari.append(parcacik)
            parcacik.start()
            
        for parcacik in is_parcaciklari:
            parcacik.join()
            
        for sonuc in sonuclar:
            self.assertIsNotNone(sonuc)
            self.assertIn("gorsel_yolu", sonuc)

if __name__ == "__main__":
    unittest.main()
