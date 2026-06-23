"""
RoadGuardian-CV - Kontrol Paneli Baslaticisi

Video secme/yukleme + ulke/secenek ayarlari icin masaustu paneli acar.
Buradan istenilen video secilip ANPR + hologram islemi başlatılır.

Calistirmak (proje kok dizininden):
    venv\\Scripts\\python run_ui.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
