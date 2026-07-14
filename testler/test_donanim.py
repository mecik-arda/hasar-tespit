import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.hardware_check import donanim_profili_olustur

class DonanimTesti(unittest.TestCase):
    def test_donanim_profili_yapisi(self):
        profil = donanim_profili_olustur()
        self.assertIn("cpu", profil)
        self.assertIn("ram", profil)
        self.assertIn("hedef_cihaz", profil)
        self.assertIn("onerilen_batch", profil)

    def test_cpu_degerleri(self):
        profil = donanim_profili_olustur()
        cpu = profil["cpu"]
        self.assertIsNotNone(cpu["ad"])
        self.assertGreater(cpu["cekirdek"], 0)

    def test_ram_degerleri(self):
        profil = donanim_profili_olustur()
        ram = profil["ram"]
        self.assertGreater(ram["toplam_gb"], 0)

if __name__ == "__main__":
    unittest.main()
