import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
import shutil
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.pipeline import hasar_tespiti_yap

class GecersizGirdiTesti(unittest.TestCase):
    def setUp(self):
        self.gecici_klasor = PROJE_KOKU / "test_hatali_girdiler"
        self.gecici_klasor.mkdir(exist_ok=True)
        
        self.bos_dosya = self.gecici_klasor / "bos_resim.jpg"
        with open(self.bos_dosya, "wb") as dosya:
            pass
            
        self.yanlis_format_dosya = self.gecici_klasor / "sahte.pdf"
        with open(self.yanlis_format_dosya, "w") as dosya:
            dosya.write("Bu bir pdf dosyasi gibi davranan metin belgesidir.")

    def tearDown(self):
        if self.gecici_klasor.exists():
            shutil.rmtree(self.gecici_klasor)

    def test_bos_gorsel_yolu(self):
        sonuc = hasar_tespiti_yap("")
        self.assertIsNone(sonuc)

    def test_olmayan_dosya(self):
        sonuc = hasar_tespiti_yap(str(self.gecici_klasor / "yokboylebirdosya.jpg"))
        self.assertIsNone(sonuc)

    def test_bos_dosya_icerigi(self):
        sonuc = hasar_tespiti_yap(str(self.bos_dosya))
        self.assertIsNone(sonuc)

    def test_yanlis_format_icerik(self):
        sonuc = hasar_tespiti_yap(str(self.yanlis_format_dosya))
        self.assertIsNone(sonuc)

if __name__ == "__main__":
    unittest.main()
