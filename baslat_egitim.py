import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

PROJE_KOKU = Path(__file__).parent
sys.path.insert(0, str(PROJE_KOKU))

import src.train as train_mod

original_veri_koku = train_mod.VERI_KOKU
train_mod.VERI_KOKU = PROJE_KOKU / "YoloForCarDefect-1"

from src.train import egitim_baslat
egitim_baslat(epoch_sayisi=10, batch_size=16, cihaz="cpu", img_size=320)

train_mod.VERI_KOKU = original_veri_koku
