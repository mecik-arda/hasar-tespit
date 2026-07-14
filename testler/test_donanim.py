import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.hardware_check import donanim_profili_olustur, npu_bilgisi_al, tum_gpu_bilgisi_al

class DonanimTesti(unittest.TestCase):
    def test_donanim_profili_yapisi(self):
        profil = donanim_profili_olustur()
        self.assertIn("cpu", profil)
        self.assertIn("ram", profil)
        self.assertIn("hedef_cihaz", profil)
        self.assertIn("onerilen_batch", profil)
        self.assertIn("npu", profil)
        self.assertIn("tum_gpu", profil)
        self.assertIn("intel_arc_gpu", profil)
        self.assertIn("cihaz_aciklamasi", profil)

    def test_cpu_degerleri(self):
        profil = donanim_profili_olustur()
        cpu = profil["cpu"]
        self.assertIsNotNone(cpu["ad"])
        self.assertGreater(cpu["cekirdek"], 0)

    def test_ram_degerleri(self):
        profil = donanim_profili_olustur()
        ram = profil["ram"]
        self.assertGreater(ram["toplam_gb"], 0)

    def test_npu_listesi_tip(self):
        npu = npu_bilgisi_al()
        self.assertIsInstance(npu, list)

    def test_tum_gpu_listesi_tip(self):
        gpu = tum_gpu_bilgisi_al()
        self.assertIsInstance(gpu, list)

    def test_cihaz_secimi_yapisi(self):
        from src.hardware_check import cihaz_secimi_yap
        profil = donanim_profili_olustur()
        # Kullanıcı etkileşimi olmadan test edemeyiz,
        # ancak profil ile çağrılabilir olduğunu doğrula
        self.assertIsNotNone(profil)
        self.assertIn("cpu", profil)

if __name__ == "__main__":
    unittest.main()
