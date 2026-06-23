"""
RoadGuardian-CV - Plaka Takip / Onbellek Modulu

Arac takibi (TrafficTracker) ile plaka okuma (PlateReader) arasindaki kopru.

Kararlilik icin OYLAMA (voting) kullanir:
- Her arac ID'si icin okunan plaka metinleri, OCR guveniyle agirlikli olarak
  bir oy sayacinda toplanir. Ekranda EN COK OY alan metin gosterilir.
- Bu sayede arac hareket ederken tek tek hatali okumalar gosterimi titretmez;
  dogru plaka zamanla one cikip "kilitlenir".

OCR, CPU dostu olacak sekilde kisitlanir:
- Kare basina en fazla ``OCR_MAX_PER_FRAME`` plaka okunur.
- Her arac en az ``OCR_REATTEMPT_INTERVAL`` kare arayla denenir.
- "Kilitli" (kararli) plakalar daha seyrek yenilenir; OCR butcesi henuz
  okunmamis araclara ayrilir.
"""

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from traffic_module.plate_ocr import PlateReader, plate_format_score  # noqa: E402

# Arac basina saklanan max okuma sayisi (bellek sinirlama).
_MAX_READS = 60


@dataclass
class PlateRecord:
    """Bir arac ID'sine ait plaka okuma gecmisi ve KARAKTER OYLAMASI.

    Gosterilen plaka tek bir okuma degil, tum okumalarin uzlasisidir:
      1) Her okuma, ham OCR guveni VE bicim gecerliligi ile AGIRLIKLANIR:
         bilinen bir ulke plaka bicimine uyan okumalar (orn. "34KLE88") daha
         fazla oy alir; bicimsiz gurultu (orn. "02XQ") geride kalir.
      2) Okumalar UZUNLUGA gore gruplanir; toplam agirligi en yuksek uzunluk
         secilir (farkli uzunluk gurultusunu eler).
      3) O uzunluktaki okumalarda HER KARAKTER KONUMU icin en cok agirlik alan
         karakter secilir. Boylece 'AP05JEO' / 'ZP05JEO' gibi 1 harf farkli
         okumalar tek dogru plakada birlesir.

    DOGRULAMA: Plaka, ancak en az ``OCR_MIN_VOTES_TO_SHOW`` oy biriktikten ve
    makul bir uzunluga ulastiktan SONRA "gosterime hazir" (``ready``) sayilir.
    Bu, ilk hatali okumanin (gecici "02...") ekranda gosterilmesini engeller.
    """

    # (metin, ham_guven, agirlik) -> agirlik = guven * bicim bonusu
    reads: list[tuple[str, float, float]] = field(default_factory=list)
    plate_box: tuple[int, int, int, int] | None = None            # son plaka konumu
    last_attempt_frame: int = -10_000

    def add_vote(self, text: str, conf: float) -> None:
        # Bicimsel olarak gecerli okumalari odullendir (dogrulama agirligi).
        weight = conf * (1.0 + settings.OCR_VALID_FORMAT_BONUS * plate_format_score(text))
        self.reads.append((text, conf, weight))
        if len(self.reads) > _MAX_READS:
            self.reads.pop(0)

    def _length_scores(self) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for t, _c, w in self.reads:
            scores[len(t)] += w
        return scores

    def _dominant_length(self) -> int | None:
        scores = self._length_scores()
        return max(scores, key=scores.get) if scores else None

    @property
    def text(self) -> str:
        """Agirlikli karakter oylamasiyla uzlasi plaka metni."""
        dom = self._dominant_length()
        if not dom:
            return ""
        chars = []
        for i in range(dom):
            col: dict[str, float] = defaultdict(float)
            for t, _c, w in self.reads:
                if len(t) == dom:
                    col[t[i]] += w
            chars.append(max(col, key=col.get))
        return "".join(chars)

    @property
    def conf(self) -> float:
        """Uzlasi uzunlugundaki okumalarin gorulen en yuksek HAM guveni."""
        dom = self._dominant_length()
        if not dom:
            return 0.0
        group = [c for t, c, _w in self.reads if len(t) == dom]
        return max(group) if group else 0.0

    @property
    def votes(self) -> int:
        """Uzlasi uzunlugunu destekleyen okuma (oy) sayisi."""
        dom = self._dominant_length()
        if not dom:
            return 0
        return sum(1 for t, _c, _w in self.reads if len(t) == dom)

    @property
    def ready(self) -> bool:
        """Plaka ekranda gosterilmeye hazir mi? (dogrulama kapisi).

        En az ``OCR_MIN_VOTES_TO_SHOW`` oy birikmeli ve uzlasi metni makul
        uzunlukta olmali. Boylece gecici/ilk hatali okumalar gosterilmez.
        """
        return (
            self.votes >= settings.OCR_MIN_VOTES_TO_SHOW
            and len(self.text) >= settings.OCR_MIN_PLATE_CHARS
        )

    @property
    def locked(self) -> bool:
        """Baskin uzunluk yeterince ve rakibinden belirgin ustunse kararli sayilir."""
        if len(self.reads) < 3:
            return False
        scores = sorted(self._length_scores().values(), reverse=True)
        top = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        return top >= settings.OCR_LOCK_SCORE and top >= settings.OCR_LOCK_RATIO * second


@dataclass
class PlateTracker:
    """Arac ID -> plaka eslemesini ve oylama onbellegini yoneten sinif."""

    reader: PlateReader = field(default_factory=PlateReader)
    reattempt_interval: int = settings.OCR_REATTEMPT_INTERVAL
    max_per_frame: int = settings.OCR_MAX_PER_FRAME
    locked_refresh_mult: int = settings.OCR_LOCKED_REFRESH_MULT
    detect_interval: int = settings.PLATE_DETECT_INTERVAL

    def __post_init__(self):
        self.records: dict[int, PlateRecord] = {}
        self._frame_idx = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _center(box) -> tuple[float, float]:
        x1, y1, x2, y2 = box[:4]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _contains(vbox, point) -> bool:
        x1, y1, x2, y2 = vbox
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2

    @staticmethod
    def _area(box) -> float:
        x1, y1, x2, y2 = box[:4]
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    def _match_plate_to_vehicle(self, plate_box, vehicles):
        """Bir plaka kutusunu iceren en KUCUK (en spesifik) araca esler."""
        center = self._center(plate_box)
        best_id = None
        best_area = float("inf")
        for tid, vbox in vehicles.items():
            if self._contains(vbox, center):
                area = self._area(vbox)
                if area < best_area:
                    best_area = area
                    best_id = tid
        return best_id

    def _due_for_ocr(self, rec: PlateRecord) -> bool:
        """Bu arac icin tekrar OCR zamani geldi mi? (kilitliyse daha seyrek)."""
        interval = self.reattempt_interval
        if rec.locked:
            interval *= self.locked_refresh_mult
        return self._frame_idx - rec.last_attempt_frame >= interval

    # ------------------------------------------------------------------ #
    def update(self, frame, vehicles: dict[int, tuple]) -> dict[int, PlateRecord]:
        """Bir kareyi isler: plakalari tespit eder, araclara esler, oylar.

        Args:
            frame: Islenecek (temiz) BGR kare.
            vehicles: {track_id: (x1, y1, x2, y2)} -> o karedeki araclar.

        Returns:
            Guncel {track_id: PlateRecord} onbellegi (yalnizca okunmus araclar).
        """
        self._frame_idx += 1

        # CPU tasarrufu: plaka tespiti + OCR yalnizca her ``detect_interval``
        # karede bir yapilir. Aradaki karelerde mevcut (hazir) kayitlar aynen
        # cizilmeye devam eder; plaka konumu son tespitten korunur.
        if self.detect_interval > 1 and (self._frame_idx % self.detect_interval) != 0:
            return {tid: r for tid, r in self.records.items() if r.ready}

        # 1) Tum kare uzerinde plakalari tespit et.
        plate_boxes = self.reader.detect_plates(frame)

        # 2) Her plakayi bir araca esle; ID basina en guvenli plaka kutusu.
        matched: dict[int, tuple[int, int, int, int]] = {}
        plate_conf: dict[int, float] = {}
        for (x1, y1, x2, y2, conf) in plate_boxes:
            tid = self._match_plate_to_vehicle((x1, y1, x2, y2), vehicles)
            if tid is None:
                continue
            if conf >= plate_conf.get(tid, 0.0):
                matched[tid] = (x1, y1, x2, y2)
                plate_conf[tid] = conf

        # 3) OCR adaylarini sec: araligi dolmus olanlar.
        candidates = []
        for tid, pbox in matched.items():
            rec = self.records.get(tid)
            if rec is None:
                rec = PlateRecord()
                self.records[tid] = rec
            rec.plate_box = pbox  # Stem cizimi icin konumu daima guncelle.
            if self._due_for_ocr(rec):
                candidates.append((tid, pbox))

        # Oncelik: once kilitli OLMAYAN (henuz kararsiz) araclar, sonra yuksek
        # plaka guvenlileri. Boylece OCR butcesi okunmamis araclara gider.
        candidates.sort(
            key=lambda c: (self.records[c[0]].locked, -plate_conf.get(c[0], 0.0))
        )
        for tid, pbox in candidates[: self.max_per_frame]:
            rec = self.records[tid]
            rec.last_attempt_frame = self._frame_idx
            reading = self.reader.read_plate_from_box(frame, pbox)
            if reading is None:
                continue
            text, conf = reading
            rec.add_vote(text, conf)

        # 4) Yalnizca DOGRULANMIS (gosterime hazir) kayitlari dondur.
        #    ready: yeterli oy + makul uzunluk -> ilk hatali okuma gosterilmez.
        return {tid: r for tid, r in self.records.items() if r.ready}

    def get(self, track_id: int) -> PlateRecord | None:
        rec = self.records.get(track_id)
        return rec if (rec and rec.ready) else None

    def cleanup(self, active_ids: set[int]) -> None:
        """Ekrandan cikan araclarin kayitlarini temizler (bellek icin)."""
        for tid in list(self.records.keys()):
            if tid not in active_ids:
                del self.records[tid]
