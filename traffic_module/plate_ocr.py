"""
RoadGuardian-CV - Plaka Okuma (ANPR / OCR) Modulu

Iki asamali bir plaka okuma hatti saglar:

    1. Plaka TESPITI  -> ozel egitilmis YOLO modeli (models/license_plate_detector.pt)
                         tum kare uzerinde plaka kutularini bulur.
    2. Plaka OKUMA    -> EasyOCR, her plakayi buyutup on isleyerek
                         icindeki metni cikartir.

Tasarim notlari:
- CPU'da OCR pahalidir. Bu yuzden okuma "hat" (pipeline) icinde dogrudan degil,
  ``PlateTracker`` (plate_tracker.py) tarafindan arac ID basina onbelleklenerek
  ve kisitlanarak cagrilir.
- ``read_plate`` tek bir plaka kirpintisi icin (metin, guven) dondurur.
- ``detect_plates`` tum kare uzerinde plaka kutularini dondurur.
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402

# OCR sonucundan plaka disi karakterleri (bosluk, tire, nokta vb.) ayikla.
_NON_ALNUM = re.compile(r"[^A-Z0-9]")

# OCR'in rakam/harf karistirmalari (belirli konumda rakam beklerken).
_L2D = str.maketrans("OISZBG", "015286")

# --------------------------------------------------------------------------- #
# Cok-ulkeli plaka BICIM tanima
# --------------------------------------------------------------------------- #
# Her giris: (ulke_kodu, derlenmis_regex, ozgunluk, populerlik)
#   - regex      : bosluksuz/buyuk harf normalleştirilmis metne uygulanir.
#   - ozgunluk   : 0..1, bicim ne kadar AYIRT EDICI (cok genel olanlar dusuk).
#   - populerlik : esitlik bozucu (trafikte daha sik gorulen ulke kazanir).
#
# NOT: Plaka bicimleri ulkeler arasinda CAKISIR (orn. FR ve IT ayni "AB123CD"
# kalibini paylasir). Tek basina metinden %100 ayrim mumkun degildir; bu yuzden
# esitlikte once ``default`` (videonun ulkesi), sonra populerlik tercih edilir.
# Tek-ulkeli videolarda UI'dan ulke secip ``force`` kullanmak en guvenilir yoldur.
_COUNTRY_FORMATS = [
    # kod,  regex,                                          ozgunluk, populerlik
    ("TR", re.compile(r"^\d{2}[A-Z]{1,4}\d{2,4}$"),            0.80, 9),
    ("GB", re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$"),            0.95, 9),
    ("RU", re.compile(r"^[A-Z]\d{3}[A-Z]{2}\d{2,3}$"),        0.95, 8),
    ("UA", re.compile(r"^[A-Z]{2}\d{4}[A-Z]{2}$"),            0.90, 6),
    ("ES", re.compile(r"^\d{4}[A-Z]{3}$"),                    0.90, 7),
    ("BE", re.compile(r"^\d[A-Z]{3}\d{3}$"),                  0.85, 5),
    ("FR", re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$"),            0.75, 8),  # IT ortak
    ("IT", re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$"),            0.70, 7),
    ("RO", re.compile(r"^[A-Z]{1,2}\d{2,3}[A-Z]{3}$"),        0.75, 5),
    ("GR", re.compile(r"^[A-Z]{3}\d{4}$"),                    0.72, 5),
    ("CZ", re.compile(r"^\d[A-Z]\d[A-Z]\d{4}$|^\d[A-Z]{2}\d{4}$"), 0.72, 4),
    ("NL", re.compile(r"^[A-Z]{2}\d{3}[A-Z]$|^\d{2}[A-Z]{3}\d$|^\d{1,2}[A-Z]{2}\d{2,3}$"), 0.65, 6),
    ("PT", re.compile(r"^\d{2}[A-Z]{2}\d{2}$|^[A-Z]{2}\d{2}[A-Z]{2}$"), 0.60, 5),
    ("SE", re.compile(r"^[A-Z]{3}\d{2}[A-Z0-9]$|^[A-Z]{3}\d{3}$"), 0.58, 5),
    ("PL", re.compile(r"^[A-Z]{2,3}\d{4,5}$|^[A-Z]{2,3}\d{3}[A-Z]{1,2}$"), 0.55, 6),
    ("DE", re.compile(r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}$"),      0.45, 9),  # cok genel
    ("AT", re.compile(r"^[A-Z]{1,2}\d{3,5}[A-Z]{0,2}$"),      0.42, 5),
    ("CH", re.compile(r"^[A-Z]{2}\d{1,6}$"),                  0.40, 5),
    ("US", re.compile(r"^[A-Z0-9]{5,7}$"),                    0.15, 8),  # son care
]

# Bilinen ulke kodlari (UI/dogrulama icin).
KNOWN_COUNTRY_CODES = [c for c, *_ in _COUNTRY_FORMATS]


def _tolerant_variants(t: str):
    """OCR rakam/harf karisikligini telafi eden metin varyantlari uretir.

    Plaka metni ``t`` ile birlikte, belirli konumlardaki harfleri rakama ceviren
    (O->0, I->1, S->5...) makul varyantlari da dondurur ki "GXISOGJ" gibi bozuk
    okumalar yine de dogru bicime oturabilsin. Asiri varyant uretmemek icin
    yalnizca tam metin ve tam-cevrilmis metin denenir.
    """
    seen = {t}
    yield t
    swapped = t.translate(_L2D)
    if swapped not in seen:
        yield swapped


def _scan_formats(base: str, prefer: str | None = None):
    """Normalize metin icin tum ulke sablonlarini (toleransli varyantlarla) tarar.

    Returns:
        (en_iyi_ozgunluk, kod) ya da eslesme yoksa None. ``prefer`` verilirse,
        ayni en-iyi ozgunluge sahip adaylar arasinda o ulke kodu tercih edilir
        (esitlik bozucu: videonun ulkesi).
    """
    best = None  # (ozgunluk, populerlik, prefer_mi, kod)
    for vi, t in enumerate(_tolerant_variants(base)):
        penalty = 0.0 if vi == 0 else 0.08  # toleransli eslesme biraz daha zayif
        for code, rx, spec, pop in _COUNTRY_FORMATS:
            if rx.match(t):
                cand = (spec - penalty, pop, 1 if code == prefer else 0, code)
                if best is None or cand[:3] > best[:3]:
                    best = cand
    if best is None:
        return None
    return best[0], best[3]


def plate_format_score(plate_text: str) -> float:
    """Plaka metninin bilinen bir ulke BICIMINE uyma derecesi (0..1).

    Oylama dogrulamasinda kullanilir: bicimsel olarak gecerli okumalara daha
    fazla agirlik verilir. Hicbir bicime uymuyorsa 0 doner.
    """
    base = _NON_ALNUM.sub("", (plate_text or "").upper())
    if len(base) < 5:
        return 0.0
    hit = _scan_formats(base)
    return hit[0] if hit else 0.0


def infer_country_code(
    plate_text: str, default: str | None = None, force: bool = False
) -> str | None:
    """Plaka metninin BICIMINDEN ulke kodunu tahmin eder (cok ulkeli).

    Args:
        plate_text: Okunan plaka metni.
        default: Karar verilemezse / ``force`` ise kullanilacak ulke kodu
            (verilmezse ``config.PLATE_COUNTRY_CODE``).
        force: True ise bicim tahminini atla, dogrudan ``default`` doner.
            Tek-ulkeli videolarda (UI'dan ulke secilince) en guvenilirdir.

    Yontem: normalize edilmis metin, ~19 ulkenin plaka sablonuyla eslenir.
    Eslesenler arasindan en YUKSEK ozgunluk kazanir; esitlikte ``default``
    (videonun ulkesi) tercih edilir. Hicbiri eslesmezse ``default`` doner.
    Boylece "AP05JEO" -> GB, "34KLE88" -> TR, "A123BC77" -> RU.
    """
    from core.config import settings as _s

    fallback = default if default is not None else _s.PLATE_COUNTRY_CODE
    if force:
        return fallback

    base = _NON_ALNUM.sub("", (plate_text or "").upper())
    if len(base) < 5:
        return fallback

    hit = _scan_formats(base, prefer=fallback)
    return hit[1] if hit else fallback


class PlateReader:
    """Plaka tespiti (YOLO) + metin okuma (EasyOCR) yapan sinif."""

    def __init__(
        self,
        model_path: str | None = None,
        languages: list[str] | None = None,
        gpu: bool | None = None,
        plate_conf: float | None = None,
    ):
        self.plate_conf = (
            plate_conf if plate_conf is not None else settings.PLATE_CONFIDENCE_THRESHOLD
        )
        self.target_width = settings.OCR_TARGET_WIDTH
        self.min_chars = settings.OCR_MIN_PLATE_CHARS

        # --- Plaka tespit modeli ---
        from ultralytics import YOLO

        self.model = YOLO(str(model_path or settings.PLATE_MODEL_PATH))

        # --- OCR motoru (EasyOCR) ---
        # Ilk kullanimda tanima modellerini indirir; sonra yerelden yuklenir.
        import easyocr

        self.reader = easyocr.Reader(
            languages or settings.OCR_LANGUAGES,
            gpu=settings.OCR_USE_GPU if gpu is None else gpu,
        )

    # ------------------------------------------------------------------ #
    # 1) Plaka tespiti                                                   #
    # ------------------------------------------------------------------ #
    def detect_plates(self, frame) -> list[tuple[int, int, int, int, float]]:
        """Tum kare uzerinde plakalari tespit eder.

        Returns:
            (x1, y1, x2, y2, conf) demetlerinden olusan liste.
        """
        result = self.model(frame, conf=self.plate_conf, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        out: list[tuple[int, int, int, int, float]] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return out

    # ------------------------------------------------------------------ #
    # 2) Plaka okuma (OCR)                                               #
    # ------------------------------------------------------------------ #
    def _preprocess(self, plate_crop):
        """OCR oncesi kirpintiyi temizler: HEDEF GENISLIGE olcekle, gri, kontrast.

        Hiz icin: pahali ``bilateralFilter`` kaldirildi; sabit 4x buyutme yerine
        plaka ~``OCR_TARGET_WIDTH`` (px) olacak sekilde olceklenir. Boylece kucuk
        plakalar yeterince buyur, buyuk plakalar gereksiz yere buyumeyip OCR'i
        yavaslatmaz. Kontrast CLAHE ile esitlenir.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        h, w = plate_crop.shape[:2]
        if h < 4 or w < 4:
            return None

        # En-boy oranini koruyarak hedef genislige olcekle (buyut veya kucult).
        scale = self.target_width / float(w)
        scale = max(0.5, min(scale, 5.0))  # asiri olceklemeyi sinirla
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
        crop = cv2.resize(plate_crop, (new_w, new_h), interpolation=interp)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Kontrast esitle (CLAHE) - hizli ve okumayi belirgin artirir.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray

    @staticmethod
    def _clean_text(text: str) -> str:
        """OCR metnini plaka formatina yaklastir: buyuk harf + alfasayisal."""
        return _NON_ALNUM.sub("", text.upper())

    def read_plate(self, plate_crop) -> tuple[str, float] | None:
        """Tek bir plaka kirpintisindan (metin, guven) dondurur.

        Plaka tek satir olsa da EasyOCR onu birden cok kutuya bolebilir
        (orn. "34 ABC" + "123"). Bu yuzden tum parcalar SOLDAN SAGA siralanip
        birlestirilir; boylece plakanin tamami yakalanir.

        Plaka benzeri metin bulunamazsa None doner.
        """
        prepped = self._preprocess(plate_crop)
        if prepped is None:
            return None

        # allowlist: yalnizca plakada gecebilecek karakterler -> daha temiz okuma.
        detections = self.reader.readtext(
            prepped,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
        )
        if not detections:
            return None

        # Parcalari sol-x'e gore sirala, gurultu olanlari (cok dusuk guven) ele.
        frags = []
        for bbox, raw_text, conf in detections:
            cleaned = self._clean_text(raw_text)
            if not cleaned or conf < settings.OCR_FRAGMENT_MIN_CONF:
                continue
            x_left = min(p[0] for p in bbox)
            frags.append((x_left, cleaned, float(conf)))

        if not frags:
            return None
        frags.sort(key=lambda f: f[0])

        text = "".join(f[1] for f in frags)
        # Guven = karakter sayisina gore agirlikli ortalama.
        total_chars = sum(len(f[1]) for f in frags)
        conf = sum(f[2] * len(f[1]) for f in frags) / max(1, total_chars)

        if len(text) < self.min_chars:
            return None
        return text, conf

    def read_plate_from_box(
        self, frame, box: tuple[int, int, int, int]
    ) -> tuple[str, float] | None:
        """Kare + plaka kutusu verilince ilgili kirpintiyi okur."""
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        return self.read_plate(crop)
