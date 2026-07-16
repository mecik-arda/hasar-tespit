import os
import sys
from roboflow import Roboflow


def roboflow_indir():
    """Roboflow'dan CarDD veri setini indirir.

    API anahtarini ROBOFLOW_API_KEY cevre degiskeninden okur.
    En son versiyonu otomatik tespit edip indirir.
    """
    API_ANAHTARI = os.environ.get("ROBOFLOW_API_KEY")
    if not API_ANAHTARI:
        print("HATA: ROBOFLOW_API_KEY cevre degiskeni tanimli degil.")
        print("  Ayarlamak icin:")
        print('    set ROBOFLOW_API_KEY=sizin_api_anahtariniz   (Windows CMD)')
        print('    $env:ROBOFLOW_API_KEY="sizin_api_anahtariniz"  (PowerShell)')
        print('    export ROBOFLOW_API_KEY=sizin_api_anahtariniz  (Linux/macOS)')
        sys.exit(1)

    rf = Roboflow(api_key=API_ANAHTARI)

    print("Dataset araniyor...")
    project = rf.project("cardefeatmodelscomparison/yoloforcardefect")
    print(f"Proje: {project.name}")

    versiyonlar = sorted(project.versions(), key=lambda v: v.version, reverse=True)
    print("Versiyonlar:")
    for v in versiyonlar:
        print(f"  v{v.version}: {v.name}")

    print()
    for version in versiyonlar:
        print(f"v{version.version} deneniyor...")
        try:
            dataset = version.download("yolov8")
            print(f"Indirme tamamlandi: {dataset.location}")
            return
        except Exception as e:
            print(f"v{version.version} hata: {e}")
            continue

    print("HATA: Hicbir versiyon indirilemedi.")
    sys.exit(1)


if __name__ == "__main__":
    roboflow_indir()
