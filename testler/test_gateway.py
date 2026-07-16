"""Akıllı Yonlendirici (AI Router) birim testleri.

Bu testler, CLIP modelini gercekten indirmeden AIRouter sinifinin
cop filtresi ve kanal yonlendirme mantigini dogrular.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))


class AIRouterCopFiltresiTesti(unittest.TestCase):
    """CLIP cop filtresi (Asama 1) testleri."""

    @patch.object(Path, "exists", return_value=True)
    def test_gorsel_bulunamadi_reddedilir(self, mock_exists):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        with patch.object(Path, "exists", return_value=False):
            sonuc = router.process_image("/gecersiz/yol/resim.jpg")
        self.assertEqual(sonuc["status"], "rejected")
        self.assertIsNone(sonuc["route_to"])
        self.assertFalse(sonuc["clip_aktif"])

    @patch.object(Path, "exists", return_value=True)
    def test_clip_yuklenemezse_yedek_mod_devreye_girer(self, mock_exists):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        router._clip_modeli_yukle = MagicMock(return_value=False)
        with patch("src.gateway.ai_router.AIRouter._yedek_yonlendirme") as mock_yedek:
            mock_yedek.return_value = {
                "status": "accepted",
                "route_to": "YOLO",
                "confidence": 0.50,
                "sebep": "Yedek mod",
                "clip_aktif": False,
            }
            sonuc = router.process_image("/test/resim.jpg")
            mock_yedek.assert_called_once()
            self.assertEqual(sonuc["status"], "accepted")
            self.assertFalse(sonuc["clip_aktif"])

    def test_cop_filtresi_cop_olarak_isaretler(self):
        from src.gateway.ai_router import AIRouter
        import numpy as np
        router = AIRouter()
        cop_logitleri = np.array([5.0, 3.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with patch.object(router, "_cop_baraji_al", return_value=0.70):
            cop_mu, guven = router._cop_filtresi_hesapla(cop_logitleri)
        self.assertTrue(cop_mu)

    def test_cop_filtresi_araba_olarak_isaretler(self):
        from src.gateway.ai_router import AIRouter
        import numpy as np
        router = AIRouter()
        cop_logitleri = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        with patch.object(router, "_cop_baraji_al", return_value=0.70):
            cop_mu, guven = router._cop_filtresi_hesapla(cop_logitleri)
        self.assertFalse(cop_mu)


class AIRouterKanalYonlendirmeTesti(unittest.TestCase):
    """CLIP kanal yonlendirme (Asama 2) testleri."""

    def test_genis_aci_rt_detr_kanalina_yonlendirir(self):
        from src.gateway.ai_router import AIRouter
        import numpy as np
        router = AIRouter()
        kanal_logitleri = np.array([5.0, 4.0, 3.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        kanal, guven = router._kanal_yonlendir_hesapla(kanal_logitleri)
        self.assertEqual(kanal, "RT-DETR")

    def test_yakin_cekim_yolo_kanalina_yonlendirir(self):
        from src.gateway.ai_router import AIRouter
        import numpy as np
        router = AIRouter()
        kanal_logitleri = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 3.0, 2.0, 0.0])
        kanal, guven = router._kanal_yonlendir_hesapla(kanal_logitleri)
        self.assertEqual(kanal, "YOLO")


class AIRouterTekForwardPassTesti(unittest.TestCase):
    """Tek forward pass optimizasyonunun dogrulugunu test eder."""

    def test_logitlerden_skorlara_toplam_bir(self):
        from src.gateway.ai_router import AIRouter
        import numpy as np
        router = AIRouter()
        sahte_logitler = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        skorlar = router._logitlerden_skorlara(sahte_logitler)
        self.assertAlmostEqual(sum(skorlar), 1.0, places=5)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in skorlar))


class AIRouterYedekYonlendirmeTesti(unittest.TestCase):
    """CLIP yuklenemediginde yedek yonlendirme testleri."""

    def test_yedek_genis_gorsel_rt_detr_yonlendirir(self):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        router._clip_modeli_yukle = MagicMock(return_value=False)
        sahte_gorsel = MagicMock()
        sahte_gorsel.size = (1920, 1080)
        with patch("PIL.Image.open", return_value=sahte_gorsel):
            with patch.object(Path, "exists", return_value=True):
                sonuc = router.process_image("/test/resim.jpg")
        self.assertEqual(sonuc["status"], "accepted")
        self.assertEqual(sonuc["route_to"], "RT-DETR")
        self.assertFalse(sonuc["clip_aktif"])

    def test_yedek_kare_gorsel_yolo_yonlendirir(self):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        router._clip_modeli_yukle = MagicMock(return_value=False)
        sahte_gorsel = MagicMock()
        sahte_gorsel.size = (640, 640)
        with patch("PIL.Image.open", return_value=sahte_gorsel):
            with patch.object(Path, "exists", return_value=True):
                sonuc = router.process_image("/test/resim.jpg")
        self.assertEqual(sonuc["status"], "accepted")
        self.assertEqual(sonuc["route_to"], "YOLO")
        self.assertFalse(sonuc["clip_aktif"])


class AIRouterConfigTesti(unittest.TestCase):
    """AIRouter'in config.yaml'dan parametre okuma testleri."""

    def test_config_ai_router_bloku_mevcut(self):
        import yaml
        config_yolu = PROJE_KOKU / "config.yaml"
        with open(config_yolu, "r", encoding="utf-8") as dosya:
            config = yaml.safe_load(dosya)
        self.assertIn("ai_router", config)
        self.assertIn("model_adi", config["ai_router"])
        self.assertIn("cop_baraji", config["ai_router"])

    def test_router_model_adi_configden_okunur(self):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        model_adi = router._model_adi_al()
        self.assertEqual(model_adi, "openai/clip-vit-base-patch32")

    def test_router_cop_baraji_configden_okunur(self):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        cop_baraji = router._cop_baraji_al()
        self.assertAlmostEqual(cop_baraji, 0.70)

    def test_router_manuel_parametre_oncelikli(self):
        from src.gateway.ai_router import AIRouter
        router = AIRouter(model_adi="ozel-model", cop_baraji=0.85)
        self.assertEqual(router._model_adi_al(), "ozel-model")
        self.assertAlmostEqual(router._cop_baraji_al(), 0.85)

    def test_config_cache_calisiyor(self):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        self.assertIsNone(router._config_cache)
        config1 = router._yapilandirma_yukle()
        self.assertIsNotNone(router._config_cache)
        config2 = router._yapilandirma_yukle()
        self.assertIs(config1, config2)


class AIRouterPathValidationTesti(unittest.TestCase):
    """Gorsel yolu dogrulama ve path traversal korumasi testleri."""

    def test_gecersiz_yol_none_doner(self):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        sonuc = router._gorsel_yolu_dogrula("/boyle/bir/dosya/yok.jpg")
        self.assertIsNone(sonuc)

    def test_zararli_yol_none_doner(self):
        from src.gateway.ai_router import AIRouter
        router = AIRouter()
        sonuc = router._gorsel_yolu_dogrula("../../../etc/passwd")
        self.assertIsNone(sonuc)


class AIRouterProcessImageTesti(unittest.TestCase):
    """process_image ana fonksiyonunun uctan uca testleri."""

    @patch.object(Path, "exists", return_value=True)
    def test_cop_gorsel_reddedilir(self, mock_exists):
        from src.gateway.ai_router import AIRouter
        import numpy as np
        router = AIRouter()
        router._model_yuklendi = True
        router._clip_model = MagicMock()
        router._clip_islemci = MagicMock()
        cop_logitleri = np.array([5.0, 3.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        kanal_logitleri = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with patch.object(router, "_gorseli_hazirla", return_value=MagicMock()):
            with patch.object(router, "_clip_tum_logitlari_hesapla", return_value=(cop_logitleri, kanal_logitleri)):
                with patch.object(router, "_cop_baraji_al", return_value=0.70):
                    sonuc = router.process_image("/test/cop.jpg")
        self.assertEqual(sonuc["status"], "rejected")
        self.assertIsNone(sonuc["route_to"])
        self.assertTrue(sonuc["clip_aktif"])

    @patch.object(Path, "exists", return_value=True)
    def test_temiz_gorsel_rt_detr_kanalina_yonlendirilir(self, mock_exists):
        from src.gateway.ai_router import AIRouter
        import numpy as np
        router = AIRouter()
        router._model_yuklendi = True
        router._clip_model = MagicMock()
        router._clip_islemci = MagicMock()
        cop_logitleri = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        kanal_logitleri = np.array([5.0, 4.0, 3.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with patch.object(router, "_gorseli_hazirla", return_value=MagicMock()):
            with patch.object(router, "_clip_tum_logitlari_hesapla", return_value=(cop_logitleri, kanal_logitleri)):
                with patch.object(router, "_cop_baraji_al", return_value=0.70):
                    sonuc = router.process_image("/test/arac.jpg")
        self.assertEqual(sonuc["status"], "accepted")
        self.assertEqual(sonuc["route_to"], "RT-DETR")
        self.assertTrue(sonuc["clip_aktif"])

    @patch.object(Path, "exists", return_value=True)
    def test_temiz_gorsel_yolo_kanalina_yonlendirilir(self, mock_exists):
        from src.gateway.ai_router import AIRouter
        import numpy as np
        router = AIRouter()
        router._model_yuklendi = True
        router._clip_model = MagicMock()
        router._clip_islemci = MagicMock()
        cop_logitleri = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        kanal_logitleri = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 3.0, 2.0, 0.0])
        with patch.object(router, "_gorseli_hazirla", return_value=MagicMock()):
            with patch.object(router, "_clip_tum_logitlari_hesapla", return_value=(cop_logitleri, kanal_logitleri)):
                with patch.object(router, "_cop_baraji_al", return_value=0.70):
                    sonuc = router.process_image("/test/lastik.jpg")
        self.assertEqual(sonuc["status"], "accepted")
        self.assertEqual(sonuc["route_to"], "YOLO")
        self.assertTrue(sonuc["clip_aktif"])


if __name__ == "__main__":
    unittest.main()
