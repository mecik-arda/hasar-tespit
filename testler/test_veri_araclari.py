import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import sys
import unittest
from importlib import import_module
from pathlib import Path

import yaml

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

SINIF_ID_MAP = import_module("_cardd_donustur").SINIF_ID_MAP
CAPRAZ_SORGULAR = import_module("src.inspector_florence").CAPRAZ_SORGULAR
utils_modulu = import_module("src.utils")
SINIF_RENKLERI = utils_modulu.SINIF_RENKLERI
yapilandirma_yukle = utils_modulu.yapilandirma_yukle


BEKLENEN_SINIFLAR = {
    0: "Cizik",
    1: "Gocuk",
    2: "Cam Kirigi",
    3: "Pas",
    4: "Kus Pisligi",
    5: "Far Kirigi",
    6: "Patlak Lastik",
}

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
        self.assertEqual(siniflar, BEKLENEN_SINIFLAR)

    def test_yapilandirma_sinif_bazli_alanlari_yedi_sinifi_kapsar(self):
        yapilandirma = yapilandirma_yukle()
        self.assertEqual(set(yapilandirma["veri"]["arama_terimleri"]), set(BEKLENEN_SINIFLAR.values()))
        self.assertEqual(set(yapilandirma["cikarim"]["sinif_guven_esikleri"]), set(BEKLENEN_SINIFLAR))

    def test_dataset_yaml_yedi_sinifi_tanimlar(self):
        dataset_yolu = PROJE_KOKU / "data" / "dataset.yaml"
        dataset = yaml.safe_load(dataset_yolu.read_text(encoding="utf-8"))
        self.assertEqual(dataset["nc"], 7)
        self.assertEqual(dataset["names"], BEKLENEN_SINIFLAR)

    def test_calisma_zamani_sinif_haritalari_tamdir(self):
        self.assertEqual(set(SINIF_RENKLERI), set(BEKLENEN_SINIFLAR))
        self.assertEqual(set(CAPRAZ_SORGULAR), set(BEKLENEN_SINIFLAR.values()))
        self.assertEqual({anahtar.replace("_", " "): deger for anahtar, deger in SINIF_ID_MAP.items()}, {deger: anahtar for anahtar, deger in BEKLENEN_SINIFLAR.items()})

    def test_egitim_notebooklari_yedi_sinifi_tanimlar(self):
        notebook_yollari = [
            PROJE_KOKU / "notebooks" / "hades_colab_egitim.ipynb",
            PROJE_KOKU / "notebooks" / "florence2_colab_egitim.ipynb",
        ]
        for notebook_yolu in notebook_yollari:
            notebook = json.loads(notebook_yolu.read_text(encoding="utf-8"))
            kaynak = "\n".join("".join(hucre.get("source", [])) for hucre in notebook["cells"])
            for sinif_id, sinif_adi in BEKLENEN_SINIFLAR.items():
                self.assertIn(f'{sinif_id}: "{sinif_adi}"', kaynak)
        hades_kaynagi = "\n".join("".join(hucre.get("source", [])) for hucre in json.loads(notebook_yollari[0].read_text(encoding="utf-8"))["cells"])
        self.assertIn('"nc": 7', hades_kaynagi)
        self.assertNotIn('"nc": 5', hades_kaynagi)

    def test_readme_yedi_sinifi_belgeler(self):
        readme = (PROJE_KOKU / "README.md").read_text(encoding="utf-8")
        self.assertIn("Hasar sınıfının numarası (0-6)", readme)
        fine_tune_bolumu = readme.split("### Florence-2 Fine-Tune İş Akışı", 1)[1].split("---", 1)[0]
        for sinif_adi in BEKLENEN_SINIFLAR.values():
            self.assertIn(f"`{sinif_adi}`", fine_tune_bolumu)

    def test_model_tur_gecerliligi(self):
        yapilandirma = yapilandirma_yukle()
        model = yapilandirma.get("model", {})
        self.assertIn("tur", model)
        self.assertIn(model["tur"], ["yolo", "rtdetr"])
        self.assertIn("agirlik", model)

if __name__ == "__main__":
    unittest.main()
