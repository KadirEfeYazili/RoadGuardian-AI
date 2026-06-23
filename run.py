"""
RoadGuardian-CV - TEK DOSYA CALISTIRICI

Bu dosyayi calistirinca trafik videosu; arac takibi + plaka okuma (OCR) +
USTTE HOLOGRAM plaka paneli ile birlikte otomatik baslar. Baska bir sey
yapmana gerek yok.

Calistirmak (proje kok dizininden):
    venv\\Scripts\\python run.py

Istege bagli:
    venv\\Scripts\\python run.py --save output\\anpr_demo.mp4   (videoyu kaydet)
    venv\\Scripts\\python run.py --video data\\test_traffic.mp4  (baska video)
    venv\\Scripts\\python run.py --no-show                       (penceresiz)

Cikis: pencere acikken 'q' tusu.
"""

import sys
from pathlib import Path

# Proje kokunu sys.path'e ekle ki tum moduller (core, traffic_module) bulunabilsin.
sys.path.append(str(Path(__file__).resolve().parent))

from traffic_module.run_anpr import main  # noqa: E402

if __name__ == "__main__":
    main()
