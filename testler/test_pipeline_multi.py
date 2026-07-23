import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import sys
import unittest
from pathlib import Path

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

from src.pipeline import (
    _ram_havuzu_olustur,
    _model_bosalt,
    _wbf_kutu_birlestir,
    yapilandirma_yukle,
)


class PipelineMultiModelTesti(unittest.TestCase):
    def setUp(self):
        self.yapilandirma = yapilandirma_yukle()

    def test_ram_havuzu_olustur_yapisi(self):
        havuz = _ram_havuzu_olustur()
        self.assertIsInstance(havuz, dict)
        self.assertIn("boxes", havuz)
        self.assertIn("masks", havuz)
        self.assertEqual(len(havuz["boxes"]), 0)
        self.assertEqual(len(havuz["masks"]), 0)

    def test_ram_havuzu_veri_tasima(self):
        havuz = _ram_havuzu_olustur()
        havuz["boxes"].append({
            "sinif_id": 0,
            "sinif_adi": "Cizik",
            "guven": 0.9,
            "kutucuk": {"x1": 10, "y1": 10, "x2": 100, "y2": 100},
            "kaynak_model": "rt-detr-v2-x",
        })
        havuz["masks"].append({
            "sinif_adi": "Gocuk",
            "kutucuk": {"x1": 10, "y1": 10, "x2": 100, "y2": 100},
            "maske_sekli": [100, 100],
            "kaynak_model": "sam2_small",
        })
        self.assertEqual(len(havuz["boxes"]), 1)
        self.assertEqual(len(havuz["masks"]), 1)
        self.assertEqual(havuz["boxes"][0]["kaynak_model"], "rt-detr-v2-x")
        self.assertEqual(havuz["masks"][0]["kaynak_model"], "sam2_small")

    def test_model_bosalt_calisir(self):
        class SahteModel:
            def __init__(self):
                self.veri = [1, 2, 3]

        model = SahteModel()
        try:
            _model_bosalt(ram_optimizasyonu=True)
        except Exception as hata:
            self.fail(f"_model_bosalt beklenmeyen hata firlatti: {hata}")

    def test_model_bosalt_ram_optimizasyonu_kapali(self):
        class SahteModel:
            pass

        model = SahteModel()
        try:
            _model_bosalt(ram_optimizasyonu=False)
        except Exception as hata:
            self.fail(f"_model_bosalt beklenmeyen hata firlatti: {hata}")

    def test_config_multi_model_bolumu_mevcut(self):
        multi_model = self.yapilandirma.get("multi_model", {})
        self.assertIn("aktif", multi_model)
        self.assertIn("siralama", multi_model)
        self.assertIn("agirliklar", multi_model)

    def test_config_multi_model_siralama_gecerli(self):
        siralama = self.yapilandirma.get("multi_model", {}).get("siralama", [])
        self.assertIsInstance(siralama, list)
        beklenen_modeller = {"rt-detr-v2-x", "yolov12x", "sam2_small", "florence-2"}
        for model in siralama:
            self.assertIn(model, beklenen_modeller)

    def test_config_ram_optimizasyonu_gecerli(self):
        ram_opt = self.yapilandirma.get("multi_model", {}).get("ram_optimizasyonu")
        self.assertIsInstance(ram_opt, bool)

    def test_config_otomatik_yedekleme_cpu_gecerli(self):
        yedekleme = self.yapilandirma.get("multi_model", {}).get("otomatik_yedekleme_cpu")
        self.assertIsInstance(yedekleme, bool)

    def test_config_denetleyici_ayarlari_gecerli(self):
        denetleyici = self.yapilandirma.get("multi_model", {}).get("denetleyici_ayarlari", {})
        self.assertIn("model", denetleyici)
        self.assertIn("gorev", denetleyici)

    def test_config_florence_lora_adaptorunu_kullanir(self):
        denetleyici = self.yapilandirma["multi_model"]["denetleyici_ayarlari"]
        model_yolu = denetleyici["model"]
        self.assertEqual(model_yolu, "models/florence_hades_lora")
        self.assertTrue(denetleyici["dogrudan_sinif_ciktisi"])
        adapter_yolu = PROJE_KOKU / model_yolu
        adapter_ayari = json.loads((adapter_yolu / "adapter_config.json").read_text(encoding="utf-8"))
        self.assertTrue((adapter_yolu / "adapter_model.safetensors").is_file())
        self.assertEqual(adapter_ayari["base_model_name_or_path"], "microsoft/Florence-2-base")
        self.assertEqual(adapter_ayari["peft_type"], "LORA")

    def test_florence_lora_kaynaklari_cozulur(self):
        from src.inspector_florence import _florence_model_kaynaklarini_bul
        islemci, taban_model, adapter = _florence_model_kaynaklarini_bul("models/florence_hades_lora")
        self.assertEqual(Path(islemci), PROJE_KOKU / "models" / "florence_hades_lora")
        self.assertEqual(taban_model, "microsoft/Florence-2-base")
        self.assertEqual(Path(adapter), PROJE_KOKU / "models" / "florence_hades_lora")

    def test_config_agirliklar_gecerli(self):
        agirliklar = self.yapilandirma.get("multi_model", {}).get("agirliklar", {})
        self.assertIn("rtdetr", agirliklar)
        self.assertIn("yolo", agirliklar)
        self.assertIn("sam", agirliklar)

    def test_config_dinamik_wbf_gecerli(self):
        ayar = self.yapilandirma.get("multi_model", {}).get("wbf_dinamik_agirliklandirma", {})
        self.assertTrue(ayar.get("aktif"))
        self.assertEqual(ayar.get("metrik_adi"), "mAP50")
        self.assertEqual(ayar.get("azami_agirlik"), 2.5)
        self.assertIn("rt-detr-v2-x", ayar.get("model_metrikleri", {}))
        self.assertIn("yolov12x", ayar.get("model_metrikleri", {}))

    def test_coklu_model_hasar_tespiti_fonksiyonu_mevcut(self):
        from src.pipeline import coklu_model_hasar_tespiti_yap
        self.assertTrue(callable(coklu_model_hasar_tespiti_yap))

    def test_coklu_model_toplu_tespiti_fonksiyonu_mevcut(self):
        from src.pipeline import coklu_model_toplu_tespiti_yap
        self.assertTrue(callable(coklu_model_toplu_tespiti_yap))

    def test_sahi_tarama_fonksiyonu_mevcut(self):
        from src.pipeline import _sahi_tarama
        self.assertTrue(callable(_sahi_tarama))

    def test_inspector_florence_modulu_mevcut(self):
        from src.inspector_florence import denetle
        self.assertTrue(callable(denetle))

    def test_inspector_florence_siniflandirma_fonksiyonu_mevcut(self):
        from src.inspector_florence import _hasar_siniflandir
        self.assertTrue(callable(_hasar_siniflandir))

    def test_inspector_florence_siniflandir_dent(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("dent on the car door")
        self.assertEqual(sonuc, "Gocuk")

    def test_inspector_florence_siniflandir_scratch(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("deep scratch on the hood")
        self.assertEqual(sonuc, "Cizik")

    def test_inspector_florence_siniflandir_kus_pisligi(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("Kus Pisligi")
        self.assertEqual(sonuc, "Kus Pisligi")

    def test_inspector_florence_dogrudan_sinif_ciktisini_kabul_eder(self):
        from src.inspector_florence import _dogrudan_hasar_siniflandir
        self.assertEqual(_dogrudan_hasar_siniflandir("  Cam Kirigi  "), "Cam Kirigi")

    def test_inspector_florence_aciklamayi_dogrudan_sinif_saymaz(self):
        from src.inspector_florence import _dogrudan_hasar_siniflandir
        sonuc = _dogrudan_hasar_siniflandir("The red car has a black tire in the background.")
        self.assertEqual(sonuc, "Bilinmeyen")

    def test_inspector_florence_siniflandir_bilinmeyen(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("a beautiful sunset")
        self.assertEqual(sonuc, "Bilinmeyen")

    def test_inspector_florence_siniflandir_ekstra_sinif(self):
        from src.inspector_florence import _hasar_siniflandir
        sonuc = _hasar_siniflandir("flat tire on the wheel", ["lastik patlagi", "flat tire"])
        self.assertEqual(sonuc, "Patlak Lastik")

    def test_inspector_florence_bolge_kirp(self):
        import numpy as np
        from src.inspector_florence import _bolge_kirp
        gorsel = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = _bolge_kirp(gorsel, (10, 10, 50, 50))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[0], 40)
        self.assertEqual(crop.shape[1], 40)

    def test_inspector_florence_bolge_kirp_gecersiz(self):
        import numpy as np
        from src.inspector_florence import _bolge_kirp
        gorsel = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = _bolge_kirp(gorsel, (90, 90, 10, 10))
        self.assertIsNone(crop)


if __name__ == "__main__":
    unittest.main()
