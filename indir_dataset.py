import sys
from roboflow import Roboflow

rf = Roboflow(api_key="3pUF1YYnnLsn34ORapb9")

print("Dataset araniyor...")
project = rf.project("cardefeatmodelscomparison/yoloforcardefect")
print(f"Proje: {project.name}")
print("Versiyonlar:")
for v in project.versions():
    print(f"  v{v.version}: {v.name}")

print()
print("En son versiyon deneniyor...")
try:
    version = project.version(1)
    dataset = version.download("yolov8")
    print(f"Indirme tamamlandi: {dataset.location}")
except Exception as e:
    print(f"v1 hata: {e}")
    try:
        version = project.version(2)
        dataset = version.download("yolov8")
        print(f"Indirme tamamlandi: {dataset.location}")
    except Exception as e2:
        print(f"v2 hata: {e2}")
