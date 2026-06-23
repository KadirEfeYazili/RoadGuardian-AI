"""
RoadGuardian-CV - Basit Canli Goruntuleyici (Trafik Takip)

Bu betik, trafik videosunu YOLO takibiyle birlikte CANLI bir pencerede oynatir.
Test ederken sonucu aninda gormek icin tasarlanmistir.

Calistirmak (proje kok dizininden):
    venv\\Scripts\\python traffic_module\\run_tracker.py

Tuslar:
    q  ->  cikis
"""

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from traffic_module.tracker import TrafficTracker  # noqa: E402

# Pencere yuksek cozunurluklu videolarda ekrana sigsin diye bu genislige
# kuculterek gosterilir (orijinal isleme/kayit tam cozunurlukte kalir).
DISPLAY_MAX_WIDTH = 1280

WINDOW_NAME = "RoadGuardian-CV | Trafik Takibi  (cikis: q)"


def _fit_to_screen(frame):
    """Goruntuyu en-boy oranini koruyarak ekrana sigacak sekilde kuculur."""
    h, w = frame.shape[:2]
    if w <= DISPLAY_MAX_WIDTH:
        return frame
    scale = DISPLAY_MAX_WIDTH / w
    return cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(h * scale)))


def main():
    tracker = TrafficTracker()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # show=False: pencere cizimini biz yonetiyoruz (kuculterek gostermek icin).
    # track_video, orig_img uzerine box+ID'yi zaten cizdigi icin tekrar
    # annotate etmiyoruz; dogrudan sonucu kuculterek gosteriyoruz.
    for result in tracker.track_video(show=False):
        cv2.imshow(WINDOW_NAME, _fit_to_screen(result.orig_img))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(f"Video: {settings.TRAFFIC_VIDEO_PATH}")
    print("Pencere aciliyor... cikmak icin 'q' tusuna bas.")
    main()
