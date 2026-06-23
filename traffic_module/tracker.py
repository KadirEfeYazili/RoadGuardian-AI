"""
RoadGuardian-CV - Trafik Takip Modulu (Tracker)

YOLO'nun yerlesik takip ozelligini (model.track) kullanarak araclari kareler
arasinda takip eder ve her araca benzersiz bir ID atar. Tespit edilen her arac
icin bounding box ve ID goruntu uzerine cizilir.
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Proje kokunu sys.path'e ekle ki "core" paketi import edilebilsin.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402


class TrafficTracker:
    """YOLO tabanli arac takip sinifi.

    Verilen videodaki araclari tespit eder, takip eder ve her birine kalici
    bir ID atayarak bounding box + ID gosterimi yapar.
    """

    def __init__(self, model_path: str | None = None, imgsz: int | None = None):
        self.model_path = model_path or settings.TRAFFIC_MODEL_PATH
        # Cikarim cozunurlugu: kucuk = daha hizli (CPU), buyuk = daha dogru.
        self.imgsz = imgsz
        self.model = YOLO(self.model_path)
        # Her ID'ye sabit bir renk atayarak gorsel takibi kolaylastiriyoruz.
        self._id_colors: dict[int, tuple[int, int, int]] = {}

    def _get_color(self, track_id: int) -> tuple[int, int, int]:
        """Bir ID icin tutarli (deterministik) bir BGR rengi uretir."""
        if track_id not in self._id_colors:
            # ID'den turetilen sahte-rastgele ama sabit bir renk.
            self._id_colors[track_id] = (
                (track_id * 37) % 256,
                (track_id * 91) % 256,
                (track_id * 53) % 256,
            )
        return self._id_colors[track_id]

    def annotate_frame(self, frame, result):
        """Tek bir takip sonucunu (boxes) frame uzerine cizer."""
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            # Henuz takip edilen nesne yoksa frame'i oldugu gibi dondur.
            return frame

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for (x1, y1, x2, y2), track_id, cls_id, conf in zip(
            xyxy, track_ids, class_ids, confidences
        ):
            color = self._get_color(track_id)
            class_name = self.model.names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ID + sinif etiketi
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def track_video(
        self,
        video_path: str | None = None,
        show: bool = True,
        save_path: str | None = None,
        annotate: bool = True,
    ):
        """Bir videodaki araclari takip eder; ekranda gosterir ve/veya kaydeder.

        Args:
            video_path: Islenecek video yolu. Verilmezse config'teki varsayilan
                trafik videosu kullanilir.
            show: True ise canli sonuc penceresi acilir ('q' ile cikis).
            save_path: Verilirse, annotated sonuc bu yola .mp4 olarak kaydedilir.
            annotate: False ise box+ID cizilmez (cagiranlar temiz kare uzerine
                kendi overlay'lerini -orn. plaka hologrami- cizebilir).
        """
        source = str(video_path or settings.TRAFFIC_VIDEO_PATH)

        # stream=True ile bellek dostu sekilde kare kare sonuc uretiyoruz.
        # persist=True ile ID'ler kareler arasinda korunuyor.
        track_kwargs = dict(
            source=source,
            stream=True,
            persist=True,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            classes=settings.VEHICLE_CLASSES,
            tracker=settings.TRACKER_CONFIG,
            verbose=False,
        )
        if self.imgsz:
            track_kwargs["imgsz"] = self.imgsz
        results = self.model.track(**track_kwargs)

        writer = None
        try:
            for result in results:
                frame = result.orig_img
                annotated = self.annotate_frame(frame, result) if annotate else frame

                # Cikti kaydedicisini ilk karede, gercek boyuta gore baslat.
                if save_path and writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(save_path), fourcc, settings.FPS, (w, h)
                    )
                if writer is not None:
                    writer.write(annotated)

                if show:
                    cv2.imshow("RoadGuardian-CV: Trafik Takibi", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield result
        finally:
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
