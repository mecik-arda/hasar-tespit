import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.advanced_benchmarks import (
    BOZULMA_TURLERI,
    bozulma_uygula,
    eszamanlilik_stres_testi_calistir,
    gelismis_rapor_kaydet,
    karisiklik_matrisi_hesapla,
    vlm_dogrulama_benchmark_calistir,
    vlm_skorlarini_hesapla,
    wbf_grid_search_calistir,
    wbf_onerisini_yapilandirmaya_uygula,
)


class BozulmaFiltresiTesti(unittest.TestCase):
    def setUp(self):
        yatay = np.tile(np.arange(96, dtype=np.uint8), (64, 1))
        self.gorsel = np.dstack((yatay, np.flip(yatay, axis=1), np.full_like(yatay, 128)))

    def test_tum_bozulmalar_bicimi_korur_ve_gorseli_degistirir(self):
        for bozulma_turu in BOZULMA_TURLERI:
            sonuc = bozulma_uygula(self.gorsel, bozulma_turu, 2, tohum=17)
            self.assertEqual(sonuc.shape, self.gorsel.shape)
            self.assertEqual(sonuc.dtype, np.uint8)
            self.assertFalse(np.array_equal(sonuc, self.gorsel))

    def test_rastgele_bozulmalar_sabit_tohumla_aynidir(self):
        birinci = bozulma_uygula(self.gorsel, "gauss_gurultusu", 3, tohum=91)
        ikinci = bozulma_uygula(self.gorsel, "gauss_gurultusu", 3, tohum=91)
        self.assertTrue(np.array_equal(birinci, ikinci))

    def test_gecersiz_bozulma_reddedilir(self):
        with self.assertRaises(ValueError):
            bozulma_uygula(self.gorsel, "gecersiz", 1)


class WbfGridSearchTesti(unittest.TestCase):
    def test_dedektor_onbellegi_bir_kez_olusturulur_ve_99_kombinasyon_taranir(self):
        kayitlar = []
        gercekler = []
        for indeks in range(3):
            gorsel_id = f"gorsel-{indeks}"
            kayitlar.append({
                "gorsel_yolu": Path(f"{gorsel_id}.jpg"),
                "gorsel_id": gorsel_id,
                "gorsel": np.zeros((32, 32, 3), dtype=np.uint8),
                "genislik": 32,
                "yukseklik": 32,
                "etiketler": [],
            })
            gercekler.append({"gorsel_id": gorsel_id, "sinif_id": 0, "kutucuk": [1, 1, 20, 20]})
        cagrilar = []

        def ham_uretici(gorsel_yolu):
            cagrilar.append(gorsel_yolu)
            return []

        def degerlendir(onbellek, etiketler, yapilandirma, iou_esigi, guven_esigi):
            return {
                "iou_esigi": iou_esigi,
                "guven_esigi": guven_esigi,
                "mAP50": iou_esigi,
                "mAP50_95": iou_esigi,
                "precision": guven_esigi,
                "recall": 1.0,
                "f1": guven_esigi,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }

        with patch("src.advanced_benchmarks._etiketli_veriyi_hazirla", return_value=(kayitlar, gercekler, "test")):
            with patch("src.advanced_benchmarks._wbf_parametrelerini_degerlendir", side_effect=degerlendir) as metrik:
                rapor = wbf_grid_search_calistir(
                    miktar=3,
                    ince_ayar=False,
                    ham_tespit_uretici=ham_uretici,
                    yapilandirma={"siniflar": {0: "Cizik"}},
                    rapor_uret=False,
                )
        self.assertEqual(len(cagrilar), 3)
        self.assertEqual(metrik.call_count, 99)
        self.assertEqual(rapor["kaba_kombinasyon_sayisi"], 99)
        self.assertFalse(rapor["onerilen_parametreler"]["config_yaml_degistirildi"])

    def test_onaylanan_oneri_yapilandirmaya_uygulanir(self):
        rapor = {
            "onerilen_parametreler": {
                "wbf_iou_esigi": 0.61,
                "guven_esigi": 0.24,
                "config_yaml_degistirildi": False,
            }
        }
        yapilandirma = {"multi_model": {"wbf_iou_esigi": 0.55, "guven_esigi": 0.25}}
        with patch("src.advanced_benchmarks.yapilandirma_yukle", return_value=yapilandirma):
            with patch("src.advanced_benchmarks.yapilandirma_kaydet") as kaydet:
                sonuc = wbf_onerisini_yapilandirmaya_uygula(rapor)
        kaydedilen = kaydet.call_args.args[0]
        self.assertEqual(kaydedilen["multi_model"]["wbf_iou_esigi"], 0.61)
        self.assertEqual(kaydedilen["multi_model"]["guven_esigi"], 0.24)
        self.assertTrue(rapor["onerilen_parametreler"]["config_yaml_degistirildi"])
        self.assertEqual(sonuc["onceki_degerler"]["wbf_iou_esigi"], 0.55)


class KarisiklikMatrisiTesti(unittest.TestCase):
    def test_sinif_karisikligi_fp_ve_fn_arka_planla_hesaplanir(self):
        gercekler = [
            {"gorsel_id": "a", "sinif_id": 0, "kutucuk": [0, 0, 20, 20]},
            {"gorsel_id": "a", "sinif_id": 1, "kutucuk": [30, 30, 50, 50]},
            {"gorsel_id": "a", "sinif_id": 1, "kutucuk": [60, 60, 80, 80]},
        ]
        tahminler = [
            {"gorsel_id": "a", "sinif_id": 1, "guven": 0.9, "kutucuk": [0, 0, 20, 20]},
            {"gorsel_id": "a", "sinif_id": 1, "guven": 0.8, "kutucuk": [30, 30, 50, 50]},
            {"gorsel_id": "a", "sinif_id": 0, "guven": 0.7, "kutucuk": [85, 85, 95, 95]},
        ]
        sonuc = karisiklik_matrisi_hesapla(tahminler, gercekler, {0: "Cizik", 1: "Gocuk"})
        self.assertEqual(sonuc["etiketler"], ["Cizik", "Gocuk", "Arka Plan"])
        self.assertEqual(sonuc["matris"][0][1], 1)
        self.assertEqual(sonuc["matris"][1][1], 1)
        self.assertEqual(sonuc["matris"][1][2], 1)
        self.assertEqual(sonuc["matris"][2][0], 1)
        self.assertEqual(sonuc["sinif_bazli"]["Cizik"]["fp"], 1)
        self.assertEqual(sonuc["sinif_bazli"]["Cizik"]["fn"], 1)


class EszamanlilikTesti(unittest.TestCase):
    def test_mock_adapter_tum_isleri_tamamlar(self):
        kilit = threading.Lock()
        aktif = 0
        tepe_aktif = 0

        def is_fonksiyonu(gorsel_yolu):
            nonlocal aktif, tepe_aktif
            with kilit:
                aktif += 1
                tepe_aktif = max(tepe_aktif, aktif)
            time.sleep(0.01)
            with kilit:
                aktif -= 1
            return {"gorsel_yolu": str(gorsel_yolu)}

        guvenli = {"guvenli": True, "nedenler": [], "bos_ram_mb": 10000, "tahmini_gereken_ram_mb": 100, "vram": None}
        with patch("src.advanced_benchmarks._stres_on_kontrol", return_value=guvenli):
            rapor = eszamanlilik_stres_testi_calistir(
                gorseller=[Path("ornek.jpg")],
                isci_seviyeleri=(3,),
                is_fonksiyonu=is_fonksiyonu,
                is_sayisi_katsayisi=2,
                tahmini_isci_ram_mb=1,
                rapor_uret=False,
            )
        sonuc = rapor["seviyeler"]["3"]
        self.assertEqual(sonuc["toplam_is"], 6)
        self.assertEqual(sonuc["basarili_is"], 6)
        self.assertEqual(sonuc["hata_sayisi"], 0)
        self.assertGreaterEqual(tepe_aktif, 2)


class FlorenceKilitTesti(unittest.TestCase):
    def test_florence_denetimleri_seri_calistirilir(self):
        from src import inspector_florence
        kilit = threading.Lock()
        aktif = 0
        tepe_aktif = 0

        def sahte_denetim(tespitler_havuzu, gorsel, yapilandirma=None):
            nonlocal aktif, tepe_aktif
            with kilit:
                aktif += 1
                tepe_aktif = max(tepe_aktif, aktif)
            time.sleep(0.01)
            with kilit:
                aktif -= 1
            return tespitler_havuzu

        with patch("src.inspector_florence._denetle_kilitsiz", side_effect=sahte_denetim):
            is_parcaciklari = [
                threading.Thread(target=inspector_florence.denetle, args=({"boxes": [], "masks": []}, np.zeros((8, 8, 3), dtype=np.uint8)))
                for _ in range(4)
            ]
            for is_parcacigi in is_parcaciklari:
                is_parcacigi.start()
            for is_parcacigi in is_parcaciklari:
                is_parcacigi.join()
        self.assertEqual(tepe_aktif, 1)


class VlmTesti(unittest.TestCase):
    def test_dogruluk_ve_halusinasyon_orani_hesaplanir(self):
        pozitifler = [
            {"gercek": "Cizik", "tahmin": "Cizik"},
            {"gercek": "Gocuk", "tahmin": "Cizik"},
            {"gercek": "Pas", "tahmin": "Bilinmeyen"},
        ]
        negatifler = [
            {"gercek": "Hasarsiz Arka Plan", "tahmin": "Bilinmeyen"},
            {"gercek": "Hasarsiz Arka Plan", "tahmin": "Cizik"},
        ]
        sonuc = vlm_skorlarini_hesapla(pozitifler, negatifler)
        self.assertAlmostEqual(sonuc["dogruluk"], 1 / 3, places=6)
        self.assertEqual(sonuc["yanlis_siniflandirma"], 1)
        self.assertEqual(sonuc["bilinmeyen_sayisi"], 1)
        self.assertEqual(sonuc["halusinasyon_orani"], 0.5)

    def test_vlm_benchmark_artirilmis_veri_kullanmadan_pozitif_ve_negatif_ornek_uretir(self):
        gorsel = np.full((100, 100, 3), 120, dtype=np.uint8)
        kayit = {
            "gorsel_yolu": Path("orijinal.jpg"),
            "gorsel_id": "orijinal",
            "gorsel": gorsel,
            "genislik": 100,
            "yukseklik": 100,
            "etiketler": [{"gorsel_id": "orijinal", "sinif_id": 0, "kutucuk": [5, 5, 25, 25]}],
        }

        def sorgula(crop, negatif=False):
            return "Bilinmeyen" if negatif else "Cizik"

        with patch("src.advanced_benchmarks._etiketli_veriyi_hazirla", return_value=([kayit], kayit["etiketler"], "test")):
            rapor = vlm_dogrulama_benchmark_calistir(
                ornek_sayisi=10,
                negatif_ornek_sayisi=1,
                vlm_sorgulayici=sorgula,
                yapilandirma={"siniflar": {0: "Cizik"}},
                rapor_uret=False,
            )
        self.assertEqual(rapor["skorlar"]["pozitif_ornek_sayisi"], 1)
        self.assertEqual(rapor["skorlar"]["dogruluk"], 1.0)
        self.assertEqual(rapor["skorlar"]["negatif_ornek_sayisi"], 1)
        self.assertEqual(rapor["skorlar"]["halusinasyon_orani"], 0.0)
        self.assertFalse(rapor["artirilmis_gorseller_dahil"])


class RaporTesti(unittest.TestCase):
    def test_json_ve_markdown_raporu_olusturulur(self):
        with tempfile.TemporaryDirectory() as gecici_klasor:
            yollar = gelismis_rapor_kaydet({"durum": "Tamamlandı", "deger": np.float32(0.5)}, gecici_klasor, "test")
            json_yolu = Path(yollar["json"])
            markdown_yolu = Path(yollar["markdown"])
            self.assertTrue(json_yolu.exists())
            self.assertTrue(markdown_yolu.exists())
            self.assertIn("HADES Gelişmiş Benchmark Raporu", markdown_yolu.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
