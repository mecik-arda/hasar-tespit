import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
import time
from pathlib import Path
from unittest.mock import patch

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.export import optimize_edilmis_model_olustur


class PerformansTesti(unittest.TestCase):
    @patch("src.export.model_dışa_aktar")
    @patch("src.hardware_check.donanim_profili_olustur")
    def test_optimize_model_sureleri(self, donanim_mock, disa_aktar_mock):
        donanim_mock.return_value = {"hedef_cihaz": "cpu"}
        disa_aktar_mock.return_value = True

        baslangic_zamani = time.time()
        sonuc = optimize_edilmis_model_olustur()
        gecen_sure = time.time() - baslangic_zamani

        self.assertTrue(sonuc, "Optimizasyon basarili olmali")
        self.assertLess(gecen_sure, 60, f"Optimizasyon cok uzun surdu: {gecen_sure:.1f}s")


if __name__ == "__main__":
    unittest.main()
