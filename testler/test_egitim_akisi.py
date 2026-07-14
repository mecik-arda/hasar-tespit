import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src import train

class EgitimAkisiTesti(unittest.TestCase):
    def setUp(self):
        self.gecici_klasor = PROJE_KOKU / "test_egitim_sandbox"
        self.gecici_klasor.mkdir(exist_ok=True)
        
        self.veri_koku = self.gecici_klasor / "data"
        self.egitim_koku = self.gecici_klasor / "runs" / "train"
        
        self.train_images = self.veri_koku / "images" / "train"
        self.train_labels = self.veri_koku / "labels" / "train"
        self.train_images.mkdir(parents=True, exist_ok=True)
        self.train_labels.mkdir(parents=True, exist_ok=True)
        
        self.val_images = self.veri_koku / "images" / "val"
        self.val_labels = self.veri_koku / "labels" / "val"
        self.val_images.mkdir(parents=True, exist_ok=True)
        self.val_labels.mkdir(parents=True, exist_ok=True)
        
        gorsel_dizisi = np.zeros((640, 640, 3), dtype=np.uint8)
        _, kodlanmis = cv2.imencode('.jpg', gorsel_dizisi)
        kodlanmis.tofile(str(self.train_images / "sanal.jpg"))
        kodlanmis.tofile(str(self.val_images / "sanal.jpg"))
        
        with open(self.train_labels / "sanal.txt", "w") as dosya:
            dosya.write("0 0.5 0.5 0.2 0.2\n")
        with open(self.val_labels / "sanal.txt", "w") as dosya:
            dosya.write("0 0.5 0.5 0.2 0.2\n")
            
        dataset_yaml_yolu = self.veri_koku / "dataset.yaml"
        yaml_icerik = {
            "path": str(self.veri_koku.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "Cizik", 1: "Gocuk", 2: "Cam Kirigi"}
        }
        with open(dataset_yaml_yolu, "w", encoding="utf-8") as dosya:
            yaml.dump(yaml_icerik, dosya)

    def tearDown(self):
        if self.gecici_klasor.exists():
            shutil.rmtree(self.gecici_klasor)

    @patch("src.train.yapilandirma_yukle")
    def test_egitim_dongusu_ve_agirlik_olusumu(self, mock_yapi):
        mock_yapi.return_value = {
            "model": {"tur": "yolo", "agirlik": "yolov8n.pt", "epoch_sayisi": 1, "batch_size": 1, "img_size": 640, "cihaz": "cpu"},
            "egitim": {"transfer_ogrenimi": True, "optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005, "warmup_epochs": 3, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1},
        }
        eski_veri_koku = train.VERI_KOKU
        eski_egitim_koku = train.EGITIM_KOKU

        train.VERI_KOKU = self.veri_koku
        train.EGITIM_KOKU = self.egitim_koku

        try:
            sonuc = train.egitim_baslat(epoch_sayisi=1, batch_size=1, cihaz="cpu")
            self.assertTrue(sonuc)

            beklenen_agirlik = self.egitim_koku / "hades_egitim" / "weights" / "best.pt"
            self.assertTrue(beklenen_agirlik.exists())
        finally:
            train.VERI_KOKU = eski_veri_koku
            train.EGITIM_KOKU = eski_egitim_koku

if __name__ == "__main__":
    unittest.main()
