"""
RoadGuardian-CV - Sürücü Uyku/Dikkat Canlı İzleyici

Bir sürücü içi kameradan (webcam) ya da videodan kare okur ve:
    1. MediaPipe Face Mesh ile yüz landmark'larını çıkarır.
    2. EAR / MAR / baş eğikliğinden uyku-yorgunluk durumunu hesaplar.
    3. Durumu HUD panelinde gösterir; ALARM'da görsel + sesli uyarı verir.

Çalıştırmak (proje kök dizininden):
    venv\\Scripts\\python run_driver.py --cam 0          (webcam)
    venv\\Scripts\\python run_driver.py --video data\\test_driver.mp4
    venv\\Scripts\\python run_driver.py --cam 0 --save output\\driver_demo.mp4
    venv\\Scripts\\python run_driver.py --cam 0 --log     (olayları output/'a CSV yaz)

Tuşlar:
    q  ->  çıkış
"""

import argparse
import sys
import threading
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from driver_module.face_mesh import FaceMeshReader  # noqa: E402
from driver_module.drowsiness import (  # noqa: E402
    DrowsinessDetector, STATE_ALARM, STATE_DROWSY,
)
from driver_module.hologram_driver import draw_driver_hud  # noqa: E402
from driver_module.driver_log import DriverLogger, DriverEvent  # noqa: E402

DISPLAY_MAX_WIDTH = 1280
WINDOW_NAME = "RoadGuardian-CV | Surucu Uyku Sensoru (cikis: q)"

# Durum şiddet sırası (kayıt için yükselen kenar tespiti).
_SEVERITY = {STATE_DROWSY: 1, STATE_ALARM: 2}


def _fit_to_screen(frame):
    h, w = frame.shape[:2]
    if w <= DISPLAY_MAX_WIDTH:
        return frame
    scale = DISPLAY_MAX_WIDTH / w
    return cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(h * scale)))


class AlarmSound:
    """ALARM'da bloklamadan sesli uyarı (Windows winsound; yoksa sessiz)."""

    def __init__(self, enabled, fps):
        self._beep = None
        if enabled:
            try:
                import winsound
                self._beep = winsound.Beep
            except ImportError:
                # Windows dışında winsound yok; sesli uyarı sessizce atlanır.
                self._beep = None
        # Beep'leri arka arkaya boğmamak için ~1 sn aralık (kare cinsinden).
        self._cooldown = max(1, int(fps))
        self._last_frame = -10 ** 9

    def trigger(self, frame_idx):
        if self._beep is None or frame_idx - self._last_frame < self._cooldown:
            return
        self._last_frame = frame_idx
        hz, ms = settings.DRIVER_ALERT_BEEP_HZ, settings.DRIVER_ALERT_BEEP_MS
        threading.Thread(target=self._beep, args=(hz, ms), daemon=True).start()


def _open_capture(args):
    """--cam verildiyse webcam, yoksa video dosyası açar. (cap, kaynak_adı)."""
    if args.cam is not None:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        return cap, f"webcam #{args.cam}"
    cap = cv2.VideoCapture(str(args.video))
    return cap, str(args.video)


def main():
    parser = argparse.ArgumentParser(
        description="RoadGuardian-CV Surucu Uyku/Dikkat Sensoru"
    )
    parser.add_argument(
        "--video", default=str(settings.DRIVER_VIDEO_PATH),
        help="Islenecek video yolu (varsayilan: surucu test videosu).",
    )
    parser.add_argument(
        "--cam", type=int, default=None,
        help="Webcam indeksi (verilirse --video yerine kamera kullanilir; orn. --cam 0).",
    )
    parser.add_argument(
        "--save", default=None,
        help="Verilirse HUD'li cikti bu .mp4 yoluna kaydedilir.",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Canli pencereyi acma.",
    )
    parser.add_argument(
        "--no-sound", action="store_true", help="ALARM'da sesli uyariyi kapat.",
    )
    parser.add_argument(
        "--log", nargs="?", const="__AUTO__", default=None,
        help="Surucu olaylarini dosyaya kaydet (CSV; uzanti .jsonl ise JSON Lines).",
    )
    args = parser.parse_args()

    cap, source = _open_capture(args)
    if not cap.isOpened():
        print(f"HATA: Kaynak acilamadi: {source}")
        if args.cam is not None:
            print("  Webcam baska bir uygulamada acik olabilir ya da indeks yanlis.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or settings.FPS
    if fps <= 1:
        fps = settings.FPS
    mirror = args.cam is not None  # Webcam'i ayna gibi goster (dogal his).

    print(f"Kaynak       : {source}")
    print(f"FPS          : {fps:.0f}")
    print("Yuz landmark modeli yukleniyor (ilk acilis biraz surebilir)...")

    reader = FaceMeshReader()
    detector = DrowsinessDetector(fps=fps)
    alarm = AlarmSound(enabled=settings.DRIVER_ALERT_SOUND and not args.no_sound, fps=fps)

    # Olay kaydı (opsiyonel).
    logger = None
    if args.log is not None:
        if args.log == "__AUTO__":
            stem = "webcam" if args.cam is not None else Path(args.video).stem
            log_path = settings.OUTPUT_DIR / f"{stem}_driver.csv"
        else:
            log_path = Path(args.log)
        logger = DriverLogger(log_path, fps=fps)
        print(f"Olay kaydi   : {log_path}")

    if not args.no_show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    writer = None
    frame_idx = 0
    prev_severity = 0
    prev_yawns = 0
    prev_nods = 0
    # ALARM kenar çerçevesi saniyede ~4 kez yanıp sönsün.
    blink_period = max(1, int(fps * 0.25))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # Video bitti ya da kamera kare veremedi.
            if mirror:
                frame = cv2.flip(frame, 1)
            frame_idx += 1
            timestamp_ms = int(frame_idx * 1000.0 / fps)

            signals = reader.process(frame, timestamp_ms)
            state = detector.update(
                signals.ear, signals.mar, signals.pitch, signals.found
            )

            # --- Kayıt: yükselen kenarlar (yeni olaylar) ---
            if logger is not None:
                severity = _SEVERITY.get(state.state, 0)
                if severity > prev_severity and severity > 0:
                    logger.log(DriverEvent(
                        frame=frame_idx, event=state.state, reason=state.reason,
                        ear=state.ear, perclos=state.perclos,
                        closed_frames=state.closed_frames,
                    ))
                prev_severity = severity
                if state.yawns > prev_yawns:
                    logger.log(DriverEvent(
                        frame=frame_idx, event="YAWN", reason="Esneme",
                        ear=state.ear, perclos=state.perclos,
                        closed_frames=state.closed_frames,
                    ))
                    prev_yawns = state.yawns
                if state.nods > prev_nods:
                    logger.log(DriverEvent(
                        frame=frame_idx, event="NOD", reason="Bas dusmesi",
                        ear=state.ear, perclos=state.perclos,
                        closed_frames=state.closed_frames,
                    ))
                    prev_nods = state.nods

            # --- Sesli + görsel uyarı ---
            if state.alarm:
                alarm.trigger(frame_idx)
            blink_on = (frame_idx // blink_period) % 2 == 0
            draw_driver_hud(frame, state, blink_on=blink_on)

            # --- Kaydet ---
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
                writer.write(frame)

            # --- Göster ---
            if not args.no_show:
                cv2.imshow(WINDOW_NAME, _fit_to_screen(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        reader.close()
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Kaydedildi: {args.save}")
        if logger is not None:
            logger.close()
            print(f"Surucu olay kaydi tamamlandi: {logger.count} olay -> {logger.path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
