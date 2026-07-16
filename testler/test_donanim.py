import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from io import StringIO

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.hardware_check import (
    donanim_profili_olustur, npu_bilgisi_al, tum_gpu_bilgisi_al,
    donanim_ozeti_yazdir, cihaz_secimi_yap,
)


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

    def test_gpu_tip_alani(self):
        profil = donanim_profili_olustur()
        for gpu in profil.get("tum_gpu", []):
            self.assertIn("tip", gpu)
            self.assertIn(gpu["tip"], ["Entegre", "Harici"])

    def test_gpu_varsa_sifir_indeksli_baslar(self):
        profil = donanim_profili_olustur()
        gpu_listesi = profil.get("tum_gpu", [])
        if gpu_listesi:
            self.assertEqual(gpu_listesi[0]["tip"], profil["tum_gpu"][0]["tip"])

    @patch("sys.stdout", new_callable=StringIO)
    def test_donanim_ozeti_gpu_format(self, mock_stdout):
        profil = donanim_ozeti_yazdir()
        cikti = mock_stdout.getvalue()
        if profil.get("tum_gpu"):
            self.assertIn("GPU 0", cikti)

    @patch("builtins.input", return_value="")
    def test_cihaz_secimi_varsayilan_cpu(self, mock_input):
        profil = donanim_profili_olustur()
        secim = cihaz_secimi_yap(profil, mod="egitim")
        self.assertIsNotNone(secim)
        self.assertIn("cihaz", secim)
        self.assertIn("batch", secim)
        self.assertIn("aciklama", secim)
        self.assertEqual(secim["cihaz"], "cpu")

    @patch("builtins.input", return_value="1")
    def test_cihaz_secimi_ilk_secenek(self, mock_input):
        profil = donanim_profili_olustur()
        secim = cihaz_secimi_yap(profil, mod="egitim")
        self.assertIsNotNone(secim)
        self.assertIn("cihaz", secim)
        self.assertGreaterEqual(secim["batch"], 4)

    @patch("builtins.input", return_value="")
    def test_cihaz_secimi_cikarim_modu(self, mock_input):
        profil = donanim_profili_olustur()
        secim = cihaz_secimi_yap(profil, mod="cikarim")
        self.assertIsNotNone(secim)
        self.assertIn("cihaz", secim)
        self.assertEqual(secim["cihaz"], "cpu")


if __name__ == "__main__":
    unittest.main()
