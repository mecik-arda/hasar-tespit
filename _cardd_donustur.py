import zipfile
import json
import shutil
import os
from pathlib import Path

PROJE_KOKU = Path(__file__).parent
HEDEF_KLASOR = PROJE_KOKU / "hasar-ornek-labelli"
ZIP_DOSYASI = PROJE_KOKU / "_cardd_dataset.zip"
CIKARMA_KLASORU = PROJE_KOKU / "_cardd_extracted"

SINIF_ESLEME = {
    1: "Cizik",
    2: "Cizik",
    3: "Gocuk",
    4: "Gocuk",
    5: "Cam_Kirigi",
    6: "Far_Kirigi",
    7: "Gocuk",
    8: "Cizik",
    9: "Patlak_Lastik",
    10: "Cam_Kirigi",
}

SINIF_ID_MAP = {
    "Cizik": 0,
    "Gocuk": 1,
    "Cam_Kirigi": 2,
    "Pas": 3,
    "Kus_Pisligi": 4,
    "Far_Kirigi": 5,
    "Patlak_Lastik": 6,
}

ZIP_SLIP_GUVENLI_SUFFIX = ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".json", ".txt"


def _zip_guvenli_cikar(zf, hedef_klasor):
    """Zip dosyasini Zip Slip saldirisina karsi korumali sekilde cikarir.

    Her uyenin cikarilacagi yolu cozumleyip hedef klasorun disina cikmadigini
    dogrular. Ayrica sadece guvenli dosya turlerine izin verir.
    """
    hedef_klasor = Path(hedef_klasor).resolve()
    for uye in zf.infolist():
        cikis_yolu = (hedef_klasor / uye.filename).resolve()
        if not str(cikis_yolu).startswith(str(hedef_klasor)):
            print(f"    [!] Zip Slip engellendi: {uye.filename}")
            continue
        if uye.is_dir():
            cikis_yolu.mkdir(parents=True, exist_ok=True)
            continue
        if cikis_yolu.suffix.lower() not in ZIP_SLIP_GUVENLI_SUFFIX:
            continue
        cikis_yolu.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(uye) as kaynak, open(cikis_yolu, "wb") as hedef:
            hedef.write(kaynak.read())


def zip_dosyasini_bul():
    """Indirilen zip dosyasini veya varsa zaten cikarilmis klasoru bul."""
    if ZIP_DOSYASI.exists():
        return ZIP_DOSYASI

    for f in PROJE_KOKU.glob("*.zip"):
        if "cardd" in f.name.lower() or "car_dd" in f.name.lower():
            return f

    yolo_klasoru = PROJE_KOKU / "YoloForCarDefect-1"
    if yolo_klasoru.exists():
        for f in yolo_klasoru.glob("*.zip"):
            return f
        for f in yolo_klasoru.glob("**/annotations.json"):
            return None

    return None


def coco_yolo_donustur(annotations_path, images_dir, output_dir):
    """COCO formatindaki annotations.json dosyasini YOLO .txt etiketlerine donusturur.

    Args:
        annotations_path: COCO annotations.json dosya yolu
        images_dir: Goruntulerin bulundugu klasor
        output_dir: YOLO .txt dosyalarinin yazilacagi klasor
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    image_map = {}
    for img in coco.get('images', []):
        image_map[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height'],
        }

    category_map = {}
    for cat in coco.get('categories', []):
        category_map[cat['id']] = cat['name']

    label_map = {}

    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        cat_id = ann['category_id']
        bbox = ann['bbox']

        if img_id not in image_map:
            continue

        img_info = image_map[img_id]
        img_w = img_info['width']
        img_h = img_info['height']

        x_tl, y_tl, bw, bh = bbox

        x_center = (x_tl + bw / 2) / img_w
        y_center = (y_tl + bh / 2) / img_h
        norm_w = bw / img_w
        norm_h = bh / img_h

        hades_adi = SINIF_ESLEME.get(cat_id, None)
        if hades_adi is None:
            continue

        hades_id = SINIF_ID_MAP.get(hades_adi, -1)
        if hades_id < 0:
            continue

        img_name = image_map[img_id]['file_name']
        if img_name not in label_map:
            label_map[img_name] = []
        label_map[img_name].append((hades_id, x_center, y_center, norm_w, norm_h))

    uretilen = 0

    for img_name, labels in label_map.items():
        txt_name = Path(img_name).stem + '.txt'
        txt_path = output_dir / txt_name

        with open(txt_path, 'w', encoding='utf-8') as f:
            for cls_id, xc, yc, w, h in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        uretilen += 1

    return uretilen, label_map


def _sinif_adi_bul(cls_id):
    """Sınıf ID'sinden HADES sınıf adını döndürür."""
    for ad, sid in SINIF_ID_MAP.items():
        if sid == cls_id:
            return ad.replace('_', ' ')
    return f"Sinif_{cls_id}"


def _sinif_dagilimi_hesapla(label_map):
    """Tüm etiketler üzerinden sınıf başına kutu sayısını hesaplar."""
    sinif_sayilari = {}
    for labels in label_map.values():
        for cls_id, _, _, _, _ in labels:
            sinif_sayilari[cls_id] = sinif_sayilari.get(cls_id, 0) + 1
    return sinif_sayilari


def _klasorden_gorsel_kopyala(images_dir, hedef_klasor, tum_label_map):
    """Sadece etiketi olan görselleri hedef klasöre kopyalar."""
    kopyalanan = 0
    for kok, _, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue
            if f not in tum_label_map:
                continue
            kaynak = Path(kok) / f
            hedef = hedef_klasor / f
            if not hedef.exists():
                shutil.copy2(kaynak, hedef)
                kopyalanan += 1
    return kopyalanan


def main():
    print("=" * 60)
    print("  CarDD -> HADES YOLO Format Donusturucu")
    print("=" * 60)
    print()

    zip_path = zip_dosyasini_bul()

    if zip_path and zip_path.exists():
        print(f"[*] Zip dosyasi bulundu: {zip_path.name}")
        print(f"[*] Boyut: {zip_path.stat().st_size / (1024**2):.1f} MB")
        print()

        if not CIKARMA_KLASORU.exists():
            print("[*] Zip dosyasi cikariliyor...")
            CIKARMA_KLASORU.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                _zip_guvenli_cikar(zf, CIKARMA_KLASORU)
            print(f"[+] Cikarildi: {CIKARMA_KLASORU}")
        else:
            print(f"[*] Zaten cikarilmis: {CIKARMA_KLASORU}")
    else:
        yolo_klasoru = PROJE_KOKU / "YoloForCarDefect-1"
        if yolo_klasoru.exists():
            print(f"[*] YoloForCarDefect-1 klasoru bulundu, zip gerekmiyor.")
            CIKARMA_KLASORU.mkdir(parents=True, exist_ok=True)
        else:
            print("[!] CarDD dataset bulunamadi.")
            print("[!] Once gdown ile indirin:")
            print("    gdown https://drive.google.com/uc?id=1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx")
            return

    print()

    carrd_kok = None
    for kok in [
        CIKARMA_KLASORU / "CarDD_release" / "CarDD_COCO",
        CIKARMA_KLASORU / "CarDD_COCO",
    ]:
        if kok.exists():
            carrd_kok = kok
            break

    if carrd_kok is None:
        print("[!] CarDD_COCO klasoru bulunamadi.")
        for root, dirs, files in os.walk(CIKARMA_KLASORU):
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                print(f"    Bulunan JSON'lar: {root} -> {json_files}")
            img_dirs = [d for d in dirs if 'image' in d.lower()]
            if img_dirs:
                print(f"    Bulunan resim klasorleri: {root} -> {img_dirs}")
        return

    annotations_dir = carrd_kok / "annotations"
    images_dir = carrd_kok / "jpeg" if (carrd_kok / "jpeg").exists() else carrd_kok

    print(f"[*] CarDD kok dizini: {carrd_kok}")
    print(f"[*] Annotations: {annotations_dir}")
    print(f"[*] Images: {images_dir}")

    json_dosyalari = sorted(annotations_dir.glob("*.json"))
    print(f"[*] {len(json_dosyalari)} JSON dosyasi bulundu: {[j.name for j in json_dosyalari]}")
    print()

    toplam_etiket = 0
    tum_label_map = {}

    for json_path in json_dosyalari:
        split_adi = json_path.stem.replace("instances_", "")
        print(f"[*] Isleniyor: {json_path.name} ({split_adi})")
        uretilen, label_map = coco_yolo_donustur(json_path, images_dir, HEDEF_KLASOR)
        toplam_etiket += uretilen
        tum_label_map.update(label_map)
        print(f"    [+] {uretilen} etiket dosyasi olusturuldu.")

    print()
    print(f"[+] Toplam {toplam_etiket} YOLO etiket dosyasi olusturuldu.")
    print()

    print("[*] Goruntuler kopyalaniyor...")
    toplam_kopyalanan = _klasorden_gorsel_kopyala(images_dir, HEDEF_KLASOR, tum_label_map)
    print(f"[+] {toplam_kopyalanan} gorsel kopyalandi (sadece etiketli olanlar).")
    print()

    print("=" * 60)
    print("  ISLEM TAMAMLANDI")
    print("=" * 60)
    print(f"  Hedef klasor: {HEDEF_KLASOR}")
    print(f"  Etiket dosyasi: {toplam_etiket} adet .txt")
    print(f"  Gorsel: {toplam_kopyalanan} adet")
    print()

    sinif_sayilari = _sinif_dagilimi_hesapla(tum_label_map)

    print("  Sinif dagilimi:")
    for cls_id in sorted(sinif_sayilari.keys()):
        ad = _sinif_adi_bul(cls_id)
        print(f"    [{cls_id}] {ad}: {sinif_sayilari[cls_id]} adet")
    print("=" * 60)

    print()
    print("[*] config.yaml guncelleme onerisi:")
    print("    Asagidaki siniflari config.yaml'daki 'siniflar' bolumune ekleyin:")
    for cls_id in sorted(sinif_sayilari.keys()):
        if cls_id > 4:
            print(f"      {cls_id}: {_sinif_adi_bul(cls_id)}")
    print()


if __name__ == "__main__":
    main()
