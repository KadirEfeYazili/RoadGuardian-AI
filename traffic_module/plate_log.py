"""
RoadGuardian-CV - Plaka Olay Kayit (Event Log) Modulu

Okunan/kilitlenen her plakayi kalici bir dosyaya (CSV ya da JSON Lines) yazar.
Boylece ANPR yalnizca canli gosterim degil, sonradan incelenebilen bir KAYIT
da uretir (hangi arac, hangi plaka, ne zaman, hangi karede).

Tasarim:
- Her arac (track_id) yalnizca BIR KEZ kaydedilir. Plaka KILITLENINCE
  (en guvenli okuma) yazilir; kilitlenmeden kareden cikan araclar ise ayrilirken
  o ana kadarki en iyi okumayla yazilir (hizli gecen araclar kaybolmasin).
- Dosya bicimi uzantidan secilir: ``.jsonl`` -> JSON Lines, aksi halde CSV.
- Her satir aninda diske yazilir (flush); calisma yarida kesilse de kayit kalir.
"""

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402

# CSV/JSONL satir alanlari (sira sabittir).
_FIELDS = [
    "timestamp",   # gercek dunya zamani (ISO) - canli kamera icin anlamli
    "frame",       # video kare indeksi
    "video_time",  # videodaki an (mm:ss) - kare/FPS ile hesaplanir
    "track_id",    # arac takip ID'si
    "plate",       # okunan plaka metni
    "country",     # tahmin/secili ulke kodu
    "type",        # arac tipi (OTOMOBIL/KAMYON...)
    "color",       # baskin renk
    "conf",        # OCR guveni (0..1)
    "votes",       # uzlasiyi destekleyen oy sayisi
    "locked",      # plaka kilitli (kesinlesmis) mi
]


@dataclass
class PlateEvent:
    """Kaydedilecek tek bir plaka olayi."""

    frame: int
    track_id: int
    plate: str
    country: str
    type: str
    color: str
    conf: float
    votes: int
    locked: bool

    def as_row(self, fps: float) -> dict:
        secs = self.frame / fps if fps else 0.0
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame": self.frame,
            "video_time": f"{int(secs // 60):02d}:{int(secs % 60):02d}",
            "track_id": self.track_id,
            "plate": self.plate,
            "country": self.country,
            "type": self.type,
            "color": self.color,
            "conf": round(float(self.conf), 3),
            "votes": int(self.votes),
            "locked": int(bool(self.locked)),
        }


class PlateLogger:
    """Plaka olaylarini CSV/JSONL dosyasina, arac basina bir kez yazar.

    Args:
        path: Cikti dosyasi. Uzanti ``.jsonl`` ise JSON Lines, degilse CSV.
        fps: Video kare hizi (video_time hesabi icin); verilmezse config FPS.
    """

    def __init__(self, path, fps: float | None = None):
        self.path = Path(path)
        self.fps = float(fps if fps is not None else settings.FPS)
        self.is_jsonl = self.path.suffix.lower() == ".jsonl"
        self._logged: set[int] = set()
        self._count = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # CSV ise basligi bir kez yaz (dosya yoksa/bossa).
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        if not self.is_jsonl and self.path.stat().st_size == 0:
            csv.DictWriter(self._fh, fieldnames=_FIELDS).writeheader()
            self._fh.flush()

    def already_logged(self, track_id: int) -> bool:
        return track_id in self._logged

    def log(self, event: PlateEvent) -> bool:
        """Olayi yazar. Arac daha once kaydedildiyse atlar (False doner)."""
        if event.track_id in self._logged:
            return False
        if not event.plate or len(event.plate) < settings.OCR_MIN_PLATE_CHARS:
            return False
        self._logged.add(event.track_id)
        row = event.as_row(self.fps)
        if self.is_jsonl:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            csv.DictWriter(self._fh, fieldnames=_FIELDS).writerow(row)
        self._fh.flush()
        self._count += 1
        return True

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    @property
    def count(self) -> int:
        """Yazilan toplam plaka olayi sayisi."""
        return self._count
