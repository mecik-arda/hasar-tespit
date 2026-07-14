import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.data_tools import yapilandirma_yukle

class VeriAraclariTesti(unittest.TestCase):
    def test_yapilandirma_dosya_dogrulamasi(self):
        yapilandirma = yapilandirma_yukle()
        self.assertIsNotNone(yapilandirma)
        self.assertIn("proje", yapilandirma)
        self.assertIn("veri", yapilandirma)
        self.assertIn("siniflar", yapilandirma)

    def test_veri_sinif_sayisi(self):
        yapilandirma = yapilandirma_yukle()
        siniflar = yapilandirma["siniflar"]
        self.assertEqual(len(siniflar), 5)

if __name__ == "__main__":
    unittest.main()
