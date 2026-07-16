import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

PROJE_KOKU = Path(__file__).parent
sys.path.insert(0, str(PROJE_KOKU))

from src.train import egitim_baslat

cardd_veri_koku = PROJE_KOKU / "YoloForCarDefect-1"

egitim_baslat(
    epoch_sayisi=10,
    batch_size=16,
    cihaz="cpu",
    img_size=320,
    veri_koku=str(cardd_veri_koku),
)
