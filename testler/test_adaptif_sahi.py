import sys
import unittest
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJE_KOKU = Path(__file__).parent.parent
sys.path.insert(0, str(PROJE_KOKU))

pipeline_modulu = import_module("src.pipeline")
_adaptif_sahi_dilim_boyutunu_hesapla = pipeline_modulu._adaptif_sahi_dilim_boyutunu_hesapla
_sahi_hedef_sinif_idlerini_al = pipeline_modulu._sahi_hedef_sinif_idlerini_al
_sahi_tarama = pipeline_modulu._sahi_tarama
_sinif_bazli_tahmin_birlestir = pipeline_modulu._sinif_bazli_tahmin_birlestir


class SahteDizi:
    def __init__(self, deger):
        self.deger = np.asarray(deger)

    def __getitem__(self, indeks):
        return SahteDizi(self.deger[indeks])

    def cpu(self):
        return self

    def numpy(self):
        return self.deger


class SahteUltralyticsKutusu:
    def __init__(self, koordinatlar, sinif_id, guven):
        self.xyxy = SahteDizi([koordinatlar])
        self.cls = SahteDizi([sinif_id])
        self.conf = SahteDizi([guven])


class SahteModel:
    def __init__(self, kutular):
        self.kutular = kutular
        self.cagri_sayisi = 0

    def predict(self, **kwargs):
        self.cagri_sayisi += 1
        return [SimpleNamespace(boxes=self.kutular)]


class SahteSahiKutusu:
    def __init__(self, koordinatlar):
        self.koordinatlar = koordinatlar

    def to_xyxy(self):
        return self.koordinatlar


class SahteSahiTahmini:
    def __init__(self, koordinatlar, sinif_id, guven):
        self.bbox = SahteSahiKutusu(koordinatlar)
        self.category = SimpleNamespace(id=sinif_id)
        self.score = SimpleNamespace(value=guven)


class AdaptifSahiTesti(unittest.TestCase):
    def setUp(self):
        self.ayar = {
            "aktif": True,
            "hedef_siniflar": ["Cizik", "Pas"],
            "minimum_uzun_kenar": 1024,
            "dilim_orani": 0.5,
            "asgari_dilim_boyutu": 384,
            "azami_dilim_boyutu": 768,
            "bindirme_orani": 0.2,
            "birlestirme_iou_esigi": 0.5,
        }

    def test_dilim_boyutu_gorsel_cozunurlugune_gore_hesaplanir(self):
        gorsel = np.zeros((1080, 1920, 3), dtype=np.uint8)
        sonuc = _adaptif_sahi_dilim_boyutunu_hesapla(gorsel, 640, self.ayar)
        self.assertEqual(sonuc, 544)

    def test_kucuk_gorselde_dilimleme_atlanir(self):
        gorsel = np.zeros((800, 800, 3), dtype=np.uint8)
        sonuc = _adaptif_sahi_dilim_boyutunu_hesapla(gorsel, 640, self.ayar)
        self.assertIsNone(sonuc)

    def test_tta_tam_gorsel_tahmini_sahi_tarafindan_yeniden_uretilmez(self):
        gorsel = np.zeros((800, 800, 3), dtype=np.uint8)
        model = SahteModel([])
        tam_gorsel = [{
            "sinif_id": 0,
            "guven": 0.9,
            "kutucuk": {"x1": 10, "y1": 10, "x2": 80, "y2": 80},
            "adaptif_tta": True,
        }]
        sonuc = _sahi_tarama(model, gorsel, 0.1, 0.7, 640, self.ayar, {0: "Cizik"}, tam_gorsel)
        self.assertEqual(model.cagri_sayisi, 0)
        self.assertEqual(sonuc, tam_gorsel)

    def test_hedef_siniflar_yedi_sinif_haritasindan_cozulur(self):
        siniflar = {0: "Cizik", 1: "Gocuk", 2: "Cam Kirigi", 3: "Pas", 4: "Kus Pisligi", 5: "Far Kirigi", 6: "Patlak Lastik"}
        sonuc = _sahi_hedef_sinif_idlerini_al(siniflar, self.ayar)
        self.assertEqual(sonuc, {0, 3})

    def test_ayni_siniftaki_cakisan_tahminlerden_guvenli_olan_kalir(self):
        tahminler = [
            {"sinif_id": 0, "guven": 0.8, "kutucuk": {"x1": 10, "y1": 10, "x2": 100, "y2": 100}},
            {"sinif_id": 0, "guven": 0.95, "kutucuk": {"x1": 12, "y1": 12, "x2": 98, "y2": 98}},
            {"sinif_id": 1, "guven": 0.7, "kutucuk": {"x1": 10, "y1": 10, "x2": 100, "y2": 100}},
        ]
        sonuc = _sinif_bazli_tahmin_birlestir(tahminler, 0.5)
        self.assertEqual(len(sonuc), 2)
        self.assertEqual([tahmin for tahmin in sonuc if tahmin["sinif_id"] == 0][0]["guven"], 0.95)

    def test_tam_gorsel_ve_hedefli_sahi_sonuclari_birlesir(self):
        model = SahteModel([
            SahteUltralyticsKutusu([100, 100, 200, 200], 0, 0.8),
            SahteUltralyticsKutusu([300, 300, 600, 600], 1, 0.9),
        ])
        nesne_tahminleri = [
            SahteSahiTahmini([102, 102, 198, 198], 0, 0.95),
            SahteSahiTahmini([310, 310, 590, 590], 1, 0.99),
        ]
        sahi_modulu = ModuleType("sahi")
        predict_modulu = ModuleType("sahi.predict")
        dilim_cagrilari = []

        class SahteAutoDetectionModel:
            cagri_sayisi = 0

            @staticmethod
            def from_pretrained(**kwargs):
                SahteAutoDetectionModel.cagri_sayisi += 1
                return SimpleNamespace(**kwargs)

        sahi_modulu.AutoDetectionModel = SahteAutoDetectionModel
        def dilimli_tahmin(*args, **kwargs):
            dilim_cagrilari.append(kwargs)
            return SimpleNamespace(object_prediction_list=nesne_tahminleri)

        predict_modulu.get_sliced_prediction = dilimli_tahmin
        gorsel = np.zeros((1080, 1920, 3), dtype=np.uint8)
        siniflar = {0: "Cizik", 1: "Gocuk", 3: "Pas"}
        with patch.dict(sys.modules, {"sahi": sahi_modulu, "sahi.predict": predict_modulu}):
            sonuc = _sahi_tarama(model, gorsel, 0.1, 0.7, 640, self.ayar, siniflar)
            _sahi_tarama(model, gorsel, 0.1, 0.7, 640, self.ayar, siniflar)
        self.assertEqual(model.cagri_sayisi, 2)
        self.assertEqual(SahteAutoDetectionModel.cagri_sayisi, 1)
        self.assertEqual(dilim_cagrilari[0]["exclude_classes_by_id"], [1])
        self.assertEqual(len(sonuc), 2)
        cizik = [tahmin for tahmin in sonuc if tahmin["sinif_id"] == 0][0]
        gocuk = [tahmin for tahmin in sonuc if tahmin["sinif_id"] == 1][0]
        self.assertTrue(cizik["adaptif_sahi"])
        self.assertEqual(cizik["sahi_dilim_boyutu"], 544)
        self.assertFalse(gocuk["adaptif_sahi"])


if __name__ == "__main__":
    unittest.main()
