"""
RoadGuardian-CV - Plaka Tanima + Hologram Canli Goruntuleyici (ANPR)

Bu betik trafik videosunu isler ve:
    1. Araclari YOLO ile takip eder (her araca kalici ID).
    2. Ozel plaka modeli + EasyOCR ile her aracin plakasini okur (onbellekli).
    3. Okunan plakayi aracin USTUNDE buyutulmus HOLOGRAM panelinde gosterir.

Calistirmak (proje kok dizininden):
    venv\\Scripts\\python traffic_module\\run_anpr.py
    venv\\Scripts\\python traffic_module\\run_anpr.py --save output/anpr_demo.mp4
    venv\\Scripts\\python traffic_module\\run_anpr.py --video data/plate_test.mp4

Tuslar:
    q  ->  cikis
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from traffic_module.tracker import TrafficTracker  # noqa: E402
from traffic_module.plate_tracker import PlateTracker  # noqa: E402
from traffic_module.plate_ocr import PlateReader, infer_country_code  # noqa: E402
from traffic_module.hologram import (  # noqa: E402
    draw_plate_hologram,
    draw_vehicle_card,
    HOLO_CYAN,
)
from traffic_module.vehicle_info import (  # noqa: E402
    type_label,
    VehicleColorTracker,
    compute_wb_scale,
)
from traffic_module.plate_log import PlateLogger, PlateEvent  # noqa: E402

DISPLAY_MAX_WIDTH = 1280
WINDOW_NAME = "RoadGuardian-CV | Plaka Hologram (cikis: q)"


def _fit_to_screen(frame):
    h, w = frame.shape[:2]
    if w <= DISPLAY_MAX_WIDTH:
        return frame
    scale = DISPLAY_MAX_WIDTH / w
    return cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(h * scale)))


def _vehicles_from_result(result):
    """YOLO takip sonucundan {id: (box, class_name)} cikartir."""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return {}
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().tolist()
    cls = boxes.cls.int().cpu().tolist()
    names = result.names
    out = {}
    for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls):
        out[int(tid)] = (
            (int(x1), int(y1), int(x2), int(y2)),
            names.get(c, str(c)),
        )
    return out


def _draw_vehicle_box(frame, box, label):
    """Arac icin sade, holografik temayla uyumlu bir kutu + etiket cizer."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), HOLO_CYAN, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 8, y2 + th + 8), HOLO_CYAN, -1)
    cv2.putText(
        frame, label, (x1 + 4, y2 + th + 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
    )


def _log_plate(logger, frame_idx, tid, rec, args, force_country,
               last_type, last_color):
    """Bir aracin plaka okumasini olay kaydina yazar (arac basina bir kez)."""
    country = infer_country_code(rec.text, default=args.country, force=force_country)
    logger.log(PlateEvent(
        frame=frame_idx,
        track_id=tid,
        plate=rec.text,
        country=country or "",
        type=last_type.get(tid, ""),
        color=last_color.get(tid, ""),
        conf=rec.conf,
        votes=rec.votes,
        locked=rec.locked,
    ))


def main():
    parser = argparse.ArgumentParser(description="RoadGuardian-CV ANPR + Hologram")
    parser.add_argument(
        "--video", default=str(settings.PLATE_VIDEO_PATH),
        help="Islenecek video yolu (varsayilan: plaka test videosu).",
    )
    parser.add_argument(
        "--save", default=None,
        help="Verilirse annotated cikti bu .mp4 yoluna kaydedilir.",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Canli pencereyi acma.",
    )
    parser.add_argument(
        "--no-card", action="store_true",
        help="Arac tip/renk bilgi kartini gosterme.",
    )
    parser.add_argument(
        "--country", default=None,
        help="Varsayilan/zorunlu ulke kodu (orn. TR, GB, DE, RU...). "
             "Varsayilan: config PLATE_COUNTRY_CODE.",
    )
    parser.add_argument(
        "--country-mode", default=settings.PLATE_COUNTRY_MODE,
        choices=["auto", "force"],
        help="auto: ulkeyi plaka biciminden tahmin et (belirsizse --country'e "
             "duser). force: her plakaya --country yaz (tek-ulkeli videolar icin).",
    )
    parser.add_argument(
        "--log", nargs="?", const="__AUTO__", default=None,
        help="Okunan plakalari dosyaya kaydet (CSV; uzanti .jsonl ise JSON Lines). "
             "Yol verilmezse output/<video>_plates.csv kullanilir.",
    )
    parser.add_argument(
        "--perf", default=settings.PERF_MODE,
        choices=list(settings.PERF_PRESETS.keys()),
        help="Performans modu: fast (en hizli) / balanced / accurate (en dogru). "
             "Model, cikarim cozunurlugu ve plaka tespit sikligini belirler.",
    )
    args = parser.parse_args()
    force_country = args.country_mode == "force"

    # Performans on ayarini coz: model + imgsz + plaka tespit araligi.
    preset = settings.PERF_PRESETS.get(args.perf, settings.PERF_PRESETS["balanced"])
    track_model = settings.MODELS_DIR / preset["track_model"]

    print(f"Video        : {args.video}")
    print(f"Perf modu    : {args.perf}  (model={preset['track_model']}, "
          f"imgsz={preset['track_imgsz']}, plaka tespit/kare={preset['plate_detect_interval']})")
    print("Modeller yukleniyor (ilk OCR calismasinda model indirilebilir)...")

    tracker = TrafficTracker(model_path=track_model, imgsz=preset["track_imgsz"])
    # Plaka okuyucu + arac-plaka onbellegi.
    plate_tracker = PlateTracker(
        reader=PlateReader(), detect_interval=preset["plate_detect_interval"]
    )

    if not args.no_show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    writer = None
    color_tracker = VehicleColorTracker()

    # Plaka olay kaydi (istege bagli). Yol verilmezse output/<video>_plates.csv.
    logger = None
    if args.log is not None:
        log_path = (
            settings.OUTPUT_DIR / f"{Path(args.video).stem}_plates.csv"
            if args.log == "__AUTO__" else Path(args.log)
        )
        logger = PlateLogger(log_path)
        print(f"Plaka kaydi : {log_path}")

    # Arac basina son gorulen tip/renk (kareden cikinca kaydi tamamlamak icin).
    last_type: dict[int, str] = {}
    last_color: dict[int, str] = {}
    frame_idx = 0

    print("Isleniyor... cikmak icin 'q'.")
    try:
        # annotate=False -> temiz kareyi alip kendi overlay'imizi ciziyoruz.
        for result in tracker.track_video(
            video_path=args.video, show=False, annotate=False
        ):
            frame_idx += 1
            frame = result.orig_img
            vehicles = _vehicles_from_result(result)
            boxes_only = {tid: b for tid, (b, _) in vehicles.items()}
            # Kare seviyesinde beyaz dengesi (renk tahmininde illuminant duzeltme).
            wb = compute_wb_scale(frame) if not args.no_card else None

            # Plakalari tespit/okuma (onbellekli, kisitli OCR).
            plate_tracker.update(frame, boxes_only)
            active_ids = set(vehicles.keys())

            # Kareden CIKAN araclari, kayit silinmeden once kaydet (kilitlenmeden
            # cikan hizli araclar da o ana kadarki en iyi okumayla yazilsin).
            if logger is not None:
                for tid in list(plate_tracker.records):
                    if tid in active_ids or logger.already_logged(tid):
                        continue
                    rec = plate_tracker.records[tid]
                    if rec.ready:
                        _log_plate(logger, frame_idx, tid, rec, args,
                                   force_country, last_type, last_color)

            plate_tracker.cleanup(active_ids)
            color_tracker.cleanup(active_ids)

            # Cizim: arac kutusu + bilgi karti (tip/renk) + plaka hologrami.
            for tid, (box, cls_name) in vehicles.items():
                _draw_vehicle_box(frame, box, f"ID:{tid} {cls_name}")
                last_type[tid] = type_label(cls_name)

                if not args.no_card:
                    # Renk en yakin (en buyuk) goruntuden hesaplanir, titremez.
                    x1, y1, x2, y2 = box
                    color_name, color_bgr = color_tracker.update(
                        tid, frame[y1:y2, x1:x2], wb_scale=wb
                    )
                    last_color[tid] = color_name
                    draw_vehicle_card(
                        frame, box, type_label(cls_name), color_name, color_bgr
                    )

                rec = plate_tracker.get(tid)
                if rec is not None:
                    # Ulke kodu: auto modda plaka BICIMINDEN belirlenir (belirsizse
                    # --country'e duser); force modda dogrudan --country kullanilir.
                    country = infer_country_code(
                        rec.text, default=args.country, force=force_country
                    )
                    draw_plate_hologram(
                        frame, box, rec.text,
                        plate_box=rec.plate_box, conf=rec.conf,
                        country_code=country,
                    )
                    # Plaka kesinlesince (kilit) bir kez kaydet.
                    if logger is not None and rec.locked:
                        _log_plate(logger, frame_idx, tid, rec, args,
                                   force_country, last_type, last_color)

            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, settings.FPS, (w, h))
                writer.write(frame)

            if not args.no_show:
                cv2.imshow(WINDOW_NAME, _fit_to_screen(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
            print(f"Kaydedildi: {args.save}")
        if logger is not None:
            logger.close()
            print(f"Plaka kaydi tamamlandi: {logger.count} arac -> {logger.path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
