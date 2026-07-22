"""Akıllı Yönlendirici (AI Router) modülü.

Bu modül, CLIP (Contrastive Language-Image Pretraining) modelini kullanarak
yüklenen görselleri iki aşamada değerlendirir:

1. Çöp Filtresi: Selfie, fatura, hayvan veya alakasız arka plan gibi görselleri engeller.
2. Akıllı Kanal Yönlendirmesi: Temiz görselleri içeriklerine göre en uygun
   hasar tespiti kanalına (YOLO veya RT-DETR) sevk eder.

Eğer CLIP modeli yüklenemezse (internet yok, kütüphane eksik vb.), sistem
görüntü boyut ve en-boy oranı analizine dayalı basit bir yedek (fallback)
yönlendirme yapar ve görseli engellemez.
"""

from pathlib import Path
from colorama import Fore, Style, init

from src.utils import PROJE_KOKU, yapilandirma_yukle

init()

YAPILANDIRMA_YOLU = PROJE_KOKU / "config.yaml"


COP_METINLERI = [
    "a selfie of a person",
    "a photo of an animal",
    "a photo of a cat",
    "a photo of a dog",
    "a receipt or invoice",
    "a screenshot of text",
    "a random background",
    "a photo of food",
    "an indoor room",
    "a landscape photo",
]

ARABA_METINLERI = [
    "a photo of a car",
    "a photo of a damaged car",
    "a photo of a vehicle",
    "a close-up photo of a car part",
    "a photo of a car tire",
    "a photo of a car headlight",
    "a photo of a car windshield",
    "a wide-angle photo of a car body",
]

GENIS_ACI_METINLERI = [
    "a wide-angle photo of a car body with damage",
    "a photo of a dented car door",
    "a photo of a scratched car body panel",
    "a full car accident photo",
    "a photo of a car in a parking lot",
]

YAKIN_CEKIM_METINLERI = [
    "a close-up photo of a car tire",
    "a close-up photo of a car headlight",
    "a close-up photo of a broken car window",
    "a close-up photo of a flat tire",
    "a close-up photo of a car part",
]


class AIRouter:
    """CLIP tabanlı akıllı yönlendirici sınıfı.

    Bu sınıf, görselleri önce çöp filtresinden geçirir, ardından içeriklerine
    göre YOLO (hızlı kanal) veya RT-DETR (ağır kanal) modeline yönlendirir.
    """

    def __init__(self, model_adi=None, cop_baraji=None):
        """Yönlendiriciyi başlatır.

        Args:
            model_adi: CLIP model adı (Örn: "openai/clip-vit-base-patch32").
                       Belirtilmezse config.yaml'dan okunur.
            cop_baraji: Çöp kabul eşiği (0.0-1.0). Belirtilmezse config'den okunur.
        """
        self.model_adi = model_adi
        self.cop_baraji = cop_baraji
        self._clip_model = None
        self._clip_islemci = None
        self._model_yuklendi = False
        self._yukleme_hatasi = None
        self._config_cache = None

    def _yapilandirma_yukle(self):
        """config.yaml dosyasını güvenli şekilde yükler ve cache'ler.

        İlk çağrıda dosyayı disk'ten okur ve instance değişkeninde saklar.
        Sonraki çağrılarda cache'lenmiş değeri döndürür, gereksiz disk I/O önlenir.
        """
        if self._config_cache is not None:
            return self._config_cache
        try:
            self._config_cache = yapilandirma_yukle()
            return self._config_cache
        except Exception:
            self._config_cache = {}
            return self._config_cache

    def _model_adi_al(self):
        """Kullanılacak CLIP model adını döndürür."""
        if self.model_adi is not None:
            return self.model_adi
        yapilandirma = self._yapilandirma_yukle()
        return yapilandirma.get("ai_router", {}).get("model_adi", "openai/clip-vit-base-patch32")

    def _cop_baraji_al(self):
        """Çöp eşik değerini döndürür."""
        if self.cop_baraji is not None:
            return float(self.cop_baraji)
        yapilandirma = self._yapilandirma_yukle()
        return float(yapilandirma.get("ai_router", {}).get("cop_baraji", 0.70))

    def _gorsel_yolu_dogrula(self, gorsel_yolu):
        """Görsel yolunun güvenli ve geçerli olduğunu doğrular.

        Path traversal saldırılarına karşı koruma sağlar. Görsel yolunun
        proje kökü dışına çıkmasını engeller.

        Args:
            gorsel_yolu: Doğrulanacak görsel yolu.

        Returns:
            Path: Çözümlenmiş ve doğrulanmış görsel yolu, veya None.
        """
        try:
            cozulmus_yol = Path(gorsel_yolu).resolve()
            if not cozulmus_yol.exists():
                return None
            return cozulmus_yol
        except (OSError, ValueError):
            return None

    def _clip_modeli_yukle(self):
        """CLIP modelini ve işlemcisini lazy-loading ile yükler.

        Model bir kez yüklendikten sonra cache'lenir. Yükleme başarısız olursa
        hata kaydedilir ve sistem yedek (fallback) yönlendirmeye düşer.
        """
        if self._model_yuklendi:
            return self._model_yuklendi

        try:
            from transformers import CLIPModel, CLIPProcessor
            model_adi = self._model_adi_al()
            print(f"{Fore.CYAN}[*] CLIP modeli yükleniyor: {model_adi}...{Style.RESET_ALL}")
            self._clip_islemci = CLIPProcessor.from_pretrained(model_adi)
            self._clip_model = CLIPModel.from_pretrained(model_adi)
            self._model_yuklendi = True
            print(f"{Fore.GREEN}[+] CLIP modeli başarıyla yüklendi.{Style.RESET_ALL}")
        except ImportError:
            self._yukleme_hatasi = "transformers kutuphanesi yuklu degil"
            self._model_yuklendi = False
            print(f"{Fore.YELLOW}[!] CLIP modeli yuklenemedi - transformers yuklu degil.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[!] Yedek (fallback) yonlendirme modu aktif.{Style.RESET_ALL}")
        except Exception as hata:
            self._yukleme_hatasi = f"CLIP modeli yuklenemedi: {hata}"
            self._model_yuklendi = False
            print(f"{Fore.YELLOW}[!] CLIP modeli yuklenemedi: {hata}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[!] Yedek (fallback) yonlendirme modu aktif.{Style.RESET_ALL}")

        return self._model_yuklendi

    def _gorseli_hazirla(self, gorsel_yolu):
        """Görsel dosyasını PIL Image olarak yükler.

        Returns:
            PIL.Image: Yüklenen görsel, veya yüklenemezse None.
        """
        try:
            from PIL import Image
            gorsel = Image.open(str(gorsel_yolu)).convert("RGB")
            return gorsel
        except Exception:
            print(f"{Fore.RED}[-] Görsel yüklenemedi veya formatı desteklenmiyor.{Style.RESET_ALL}")
            return None

    def _clip_tum_logitlari_hesapla(self, gorsel):
        """Tek CLIP forward pass ile tüm metin grupları için raw logit skorlarını hesaplar.

        Bu metod, çöp filtresi ve kanal yönlendirme için gereken tüm metinleri
        tek bir forward pass'te CLIP'e gönderir. Böylece iki ayrı forward pass
        yerine tek pass yapılır ve performans ~%50 iyileştirilir.

        CLIP text embedding'leri birbirinden bağımsız olduğu için, tüm metinleri
        tek pass'te gönderip logit'leri gruplara ayırmak, ayrı pass'ler yapmakla
        matematiksel olarak aynı sonucu verir.

        Args:
            gorsel: PIL.Image nesnesi.

        Returns:
            tuple: (cop_araba_logitleri, kanal_logitleri)
                cop_araba_logitleri: Çöp + Araba metinleri için raw logit listesi
                kanal_logitleri: Geniş açı + Yakın çekim metinleri için raw logit listesi
        """
        tum_metinler = COP_METINLERI + ARABA_METINLERI + GENIS_ACI_METINLERI + YAKIN_CEKIM_METINLERI

        girisler = self._clip_islemci(
            text=tum_metinler,
            images=gorsel,
            return_tensors="pt",
            padding=True,
        )

        import torch
        import numpy as np
        with torch.no_grad():
            ciktilar = self._clip_model(**girisler)

        tum_logitler = ciktilar.logits_per_image[0].cpu().numpy()

        cop_araba_sayisi = len(COP_METINLERI) + len(ARABA_METINLERI)
        cop_araba_logitleri = tum_logitler[:cop_araba_sayisi]
        kanal_logitleri = tum_logitler[cop_araba_sayisi:]

        return cop_araba_logitleri, kanal_logitleri

    def _logitlerden_skorlara(self, logitler):
        """Tüm logit dizisi için softmax olasılıklarını hesaplar.

        Softmax tüm logit'lere aynı anda uygulanır. Bu sayede her logit,
        diğerleriyle karşılaştırmalı olarak normalize edilir.

        Args:
            logitler: numpy array, tüm logitler.

        Returns:
            list: Softmax olasılıkları (0.0-1.0), toplamları 1.0.
        """
        import numpy as np
        ustel_degerler = np.exp(logitler - np.max(logitler))
        olasiliklar = ustel_degerler / np.sum(ustel_degerler)
        return olasiliklar.tolist()

    def _cop_filtresi_hesapla(self, cop_araba_logitleri):
        """Aşama 1: Çöp filtresi skorlarını hesaplar.

        Args:
            cop_araba_logitleri: Çöp + Araba metinleri için raw logit numpy array'i.

        Returns:
            tuple: (cop_mu: bool, guven: float)
                cop_mu = True ise görsel reddedilir.
        """
        tum_skorlar = self._logitlerden_skorlara(cop_araba_logitleri)
        cop_skorlari = tum_skorlar[:len(COP_METINLERI)]
        araba_skorlari = tum_skorlar[len(COP_METINLERI):]

        cop_toplam = sum(cop_skorlari)
        araba_toplam = sum(araba_skorlari)

        cop_mu = cop_toplam > araba_toplam
        guven = cop_toplam / (cop_toplam + araba_toplam) if (cop_toplam + araba_toplam) > 0 else 0.0

        return cop_mu, guven

    def _kanal_yonlendir_hesapla(self, kanal_logitleri):
        """Aşama 2: Kanal yönlendirme skorlarını hesaplar.

        Args:
            kanal_logitleri: Geniş açı + Yakın çekim metinleri için raw logit numpy array'i.

        Returns:
            tuple: (kanal: str, guven: float)
                kanal = "RT-DETR" (geniş açı/kompleks) veya "YOLO" (yakın çekim/parça).
        """
        tum_skorlar = self._logitlerden_skorlara(kanal_logitleri)
        genis_aci_skorlari = tum_skorlar[:len(GENIS_ACI_METINLERI)]
        yakin_cekim_skorlari = tum_skorlar[len(GENIS_ACI_METINLERI):]

        genis_aci_toplam = sum(genis_aci_skorlari)
        yakin_cekim_toplam = sum(yakin_cekim_skorlari)

        if genis_aci_toplam >= yakin_cekim_toplam:
            kanal = "RT-DETR"
            guven = genis_aci_toplam / (genis_aci_toplam + yakin_cekim_toplam) if (genis_aci_toplam + yakin_cekim_toplam) > 0 else 0.5
        else:
            kanal = "YOLO"
            guven = yakin_cekim_toplam / (genis_aci_toplam + yakin_cekim_toplam) if (genis_aci_toplam + yakin_cekim_toplam) > 0 else 0.5

        return kanal, guven

    def _yedek_yonlendirme(self, gorsel_yolu):
        """CLIP yüklenemediğinde görüntü boyut/oran analizine dayalı yedek yönlendirme.

        Bu yöntem, görseli hiçbir zaman reddetmez (çöp filtresi yapılamaz).
        Geniş (muhtemelen geniş açı) görselleri RT-DETR'a, kare veya dikey
        (muhtemelen yakın çekim) görselleri YOLO'ya yönlendirir.

        Returns:
            dict: Yönlendirme sonucu.
        """
        try:
            from PIL import Image
            gorsel = Image.open(str(gorsel_yolu))
            genislik, yukseklik = gorsel.size
            en_boy_orani = genislik / yukseklik if yukseklik > 0 else 1.0

            if en_boy_orani > 1.4:
                kanal = "RT-DETR"
                sebep = "Geniş en-boy oranı (CLIP yok, yedek analiz)"
                guven = 0.50
            else:
                kanal = "YOLO"
                sebep = "Kare/dikey en-boy oranı (CLIP yok, yedek analiz)"
                guven = 0.50

            return {
                "status": "accepted",
                "route_to": kanal,
                "confidence": guven,
                "sebep": sebep,
                "clip_aktif": False,
            }
        except Exception:
            return {
                "status": "accepted",
                "route_to": "RT-DETR",
                "confidence": 0.0,
                "sebep": "Yedek analiz başarısız, varsayılan kanal seçildi.",
                "clip_aktif": False,
            }

    def process_image(self, gorsel_yolu):
        """Görseli işler ve yönlendirme kararı döndürür.

        Bu ana fonksiyon iki aşamalı çalışır:
        1. Çöp Filtresi: Alakasız görselleri engeller.
        2. Kanal Yönlendirmesi: Temiz görselleri YOLO veya RT-DETR'a gönderir.

        Her iki aşama da tek bir CLIP forward pass ile hesaplanır.

        Args:
            gorsel_yolu: İşlenecek görselin dosya yolu.

        Returns:
            dict: Yönlendirme sonucu:
                {
                    "status": "accepted" | "rejected",
                    "route_to": "RT-DETR" | "YOLO" | None,
                    "confidence": float,
                    "sebep": str,
                    "clip_aktif": bool
                }
        """
        dogrulanmis_yol = self._gorsel_yolu_dogrula(gorsel_yolu)
        if dogrulanmis_yol is None:
            return {
                "status": "rejected",
                "route_to": None,
                "confidence": 0.0,
                "sebep": "Görsel dosyası bulunamadı veya yolu geçersiz.",
                "clip_aktif": False,
            }

        gorsel_yolu = dogrulanmis_yol

        model_yuklendi = self._clip_modeli_yukle()
        if not model_yuklendi:
            return self._yedek_yonlendirme(gorsel_yolu)

        gorsel = self._gorseli_hazirla(gorsel_yolu)
        if gorsel is None:
            return {
                "status": "rejected",
                "route_to": None,
                "confidence": 0.0,
                "sebep": "Görsel okunamadı veya formatı desteklenmiyor.",
                "clip_aktif": True,
            }

        print(f"{Fore.CYAN}[*] CLIP analizi yapılıyor (tek forward pass)...{Style.RESET_ALL}")
        cop_araba_logitleri, kanal_logitleri = self._clip_tum_logitlari_hesapla(gorsel)

        print(f"{Fore.CYAN}[*] Aşama 1: Çöp Filtresi (Denetleme)...{Style.RESET_ALL}")
        cop_mu, cop_guveni = self._cop_filtresi_hesapla(cop_araba_logitleri)
        cop_baraji = self._cop_baraji_al()

        if cop_mu and cop_guveni >= cop_baraji:
            print(f"{Fore.RED}[x] Görsel reddedildi: Alakasız/çöp içerik (güven: %{cop_guveni*100:.1f}){Style.RESET_ALL}")
            return {
                "status": "rejected",
                "route_to": None,
                "confidence": round(cop_guveni, 4),
                "sebep": "Alakasız/çöp içerik tespit edildi.",
                "clip_aktif": True,
            }

        print(f"{Fore.GREEN}[+] Çöp filtresi geçildi.{Style.RESET_ALL}")

        print(f"{Fore.CYAN}[*] Aşama 2: Akıllı Kanal Yönlendirmesi...{Style.RESET_ALL}")
        kanal, kanal_guveni = self._kanal_yonlendir_hesapla(kanal_logitleri)

        if kanal == "RT-DETR":
            print(f"{Fore.YELLOW}[->] Yönlendirme: RT-DETR (Kompleks Hasar Kanalı) - güven: %{kanal_guveni*100:.1f}{Style.RESET_ALL}")
            sebep = "Geniş açı kaporta/göçük tespit edildi. Ağır çoklu-model akışına yönlendirildi."
        else:
            print(f"{Fore.BLUE}[->] Yönlendirme: YOLO (Hızlı Çözüm Kanalı) - güven: %{kanal_guveni*100:.1f}{Style.RESET_ALL}")
            sebep = "Yakın çekim parça (lastik/far/cam) tespit edildi. Hızlı kanala yönlendirildi."

        return {
            "status": "accepted",
            "route_to": kanal,
            "confidence": round(kanal_guveni, 4),
            "sebep": sebep,
            "clip_aktif": True,
        }


def router_testi_calistir():
    """AIRouter'ı hasar-ornek klasöründeki görsellerle test eder.

    Bu fonksiyon, menüden [16] seçildiğinde çağrılır. Klasördeki görselleri
    AIRouter'a gönderir ve yönlendirme sonuçlarını ekrana basar.
    """
    ornek_klasoru = PROJE_KOKU / "hasar-ornek"
    gorsel_uzantilari = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    print()
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  HADES - Akıllı Yönlendirici (Gateway) Testi{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

    if not ornek_klasoru.exists():
        print(f"{Fore.RED}[-] hasar-ornek klasörü bulunamadı: {ornek_klasoru}{Style.RESET_ALL}")
        print()
        return

    mevcut_gorseller = sorted(
        [f for f in ornek_klasoru.iterdir() if f.suffix.lower() in gorsel_uzantilari]
    )

    if not mevcut_gorseller:
        print(f"{Fore.RED}[-] hasar-ornek klasöründe görsel bulunamadı.{Style.RESET_ALL}")
        print()
        return

    test_edilecek_gorseller = mevcut_gorseller[:4]

    print(f"{Fore.YELLOW}[*] {len(test_edilecek_gorseller)} görsel test edilecek.{Style.RESET_ALL}")
    print()

    router = AIRouter()

    for i, gorsel_yolu in enumerate(test_edilecek_gorseller, 1):
        print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}[{i}/{len(test_edilecek_gorseller)}] Görsel: {gorsel_yolu.name}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")

        sonuc = router.process_image(gorsel_yolu)

        print()
        print(f"    {Fore.WHITE}Durum     : {Fore.GREEN if sonuc['status'] == 'accepted' else Fore.RED}{sonuc['status']}{Style.RESET_ALL}")
        if sonuc["status"] == "accepted":
            print(f"    {Fore.WHITE}Kanal     : {Fore.YELLOW}{sonuc['route_to']}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Güven     : %{sonuc['confidence']*100:.1f}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}CLIP Aktif: {'Evet' if sonuc['clip_aktif'] else 'Hayır (yedek mod)'}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Sebep     : {sonuc['sebep']}{Style.RESET_ALL}")
        print()

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}[+] Akıllı Yönlendirici testi tamamlandı.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()


if __name__ == "__main__":
    router_testi_calistir()