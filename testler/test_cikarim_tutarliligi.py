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

from src import pipeline

class TutarlilikTesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gecici_klasor = PROJE_KOKU / "test_tutarlilik"
        cls.gecici_klasor.mkdir(exist_ok=True)
        
        cls.gorsel_yolu = cls.gecici_klasor / "tutarlilik.jpg"
        resim = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(resim, (100, 100), (500, 500), (255, 255, 255), -1)
        _, kodlanmis = cv2.imencode('.jpg', resim)
        kodlanmis.tofile(str(cls.gorsel_yolu))
        
        config = pipeline.yapilandirma_yukle()
        model_agirligi = config.get("model", {}).get("agirlik", "yolov12n.pt")
        model_tur = config.get("model", {}).get("tur", "yolo")
        cls.pt_yolu = PROJE_KOKU / model_agirligi
        cls.onnx_yolu = PROJE_KOKU / model_agirligi.replace(".pt", ".onnx")

        if model_tur == "rtdetr":
            from ultralytics import RTDETR as ModelSinifi
        else:
            from ultralytics import YOLO as ModelSinifi

        if cls.pt_yolu.exists() and not cls.onnx_yolu.exists():
            model = ModelSinifi(str(cls.pt_yolu))
            model.export(format="onnx")

    @classmethod
    def tearDownClass(cls):
        if cls.gecici_klasor.exists():
            shutil.rmtree(cls.gecici_klasor)

    @patch("src.pipeline.egitilmis_model_yolu_bul")
    def test_model_ciktisi_tutarliligi(self, mock_model_yolu):
        mock_model_yolu.return_value = self.pt_yolu
        sonuc_pt = pipeline.hasar_tespiti_yap(str(self.gorsel_yolu))
        self.assertIsNotNone(sonuc_pt)
        
        if self.onnx_yolu.exists():
            mock_model_yolu.return_value = self.onnx_yolu
            sonuc_onnx = pipeline.hasar_tespiti_yap(str(self.gorsel_yolu))
            self.assertIsNotNone(sonuc_onnx)

if __name__ == "__main__":
    unittest.main()
