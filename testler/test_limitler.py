import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from io import StringIO

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src import train

class LimitlerTesti(unittest.TestCase):
    @patch("src.train.yapilandirma_yukle")
    @patch("sys.stdout", new_callable=StringIO)
    def test_negatif_epoch(self, mock_stdout, mock_yapi):
        mock_yapi.return_value = {
            "model": {"tur": "yolo", "agirlik": "yolov8n.pt", "epoch_sayisi": 100, "batch_size": "auto", "img_size": 640, "cihaz": "auto"},
            "egitim": {},
        }
        eski_veri_koku = train.VERI_KOKU

        train.VERI_KOKU = PROJE_KOKU / "test_olmayan_veri"
        try:
            train.egitim_baslat(epoch_sayisi=-5, batch_size=-10, cihaz="cpu", img_size=-10)
        except Exception:
            pass
        finally:
            train.VERI_KOKU = eski_veri_koku

        cikti = mock_stdout.getvalue()
        self.assertIn("Uyari", cikti)
        self.assertIn("Gecersiz epoch", cikti)
        self.assertIn("Gecersiz batch", cikti)
        self.assertIn("Gecersiz gorsel", cikti)

if __name__ == "__main__":
    unittest.main()
