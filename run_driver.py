"""
RoadGuardian-CV - SÜRÜCÜ MODÜLÜ TEK DOSYA ÇALIŞTIRICI

Bu dosyayı çalıştırınca sürücü uyku/dikkat sensörü başlar: webcam ya da videodan
yüz okunur, göz/esneme/baş eğikliğinden uyku durumu hesaplanır ve ALARM'da
görsel + sesli uyarı verilir.

Çalıştırmak (proje kök dizininden):
    venv\\Scripts\\python run_driver.py --cam 0

İsteğe bağlı:
    venv\\Scripts\\python run_driver.py --video data\\test_driver.mp4
    venv\\Scripts\\python run_driver.py --cam 0 --save output\\driver_demo.mp4
    venv\\Scripts\\python run_driver.py --cam 0 --log

Çıkış: pencere açıkken 'q' tuşu.
"""

import sys
from pathlib import Path

# Tüm modüller (core, driver_module) bulabilsin diye Proje kökü sys.path'e eklendi
sys.path.append(str(Path(__file__).resolve().parent))

from driver_module.run_driver import main  # noqa: E402

if __name__ == "__main__":
    main()
