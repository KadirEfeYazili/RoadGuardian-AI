"""
RoadGuardian-CV - Arac Bilgisi (Tip + Renk) Modulu

Arac TIPI takip modelinden (YOLO/COCO) gelir: car, motorcycle, bus, truck.
Bu modul bu tipleri Turkce etikete cevirir ve aracin BASKIN RENGINI tahmin eder.

Renk tahmini (klasik ama saglam hat):
    1. ARKA PLANI AYIKLA  -> GrabCut ile arac kutusu icindeki yol/gokyuzu/golge
       pikselleri elenir; sadece arac govdesi kalir. (Onceki hata: kutu
       icindeki arka plan -ozellikle gri yol- rengi bozuyordu.)
    2. GOVDE PIKSELLERI    -> cam (cok koyu) ve parlama (cok parlak speküler)
       pikselleri maskelenir.
    3. BASKIN RENK         -> k-means (cv2) ile en kalabalik renk kumesi alinir
       (ortanca yerine kume merkezi; karisik tonlarda daha kararli).
    4. SINIFLANDIRMA       -> Lab uzayinda akromatik (siyah/gri/gumus/beyaz) ve
       kromatik (kirmizi/mavi...) ayrimi; renkli ise HSV tonuna eslenir.

Ozel model gerektirmez; OpenCV ile calisir. Maliyet ``VehicleColorTracker``
sayesinde arac basina yalnizca en yakin (en buyuk) goruntude odenir.
"""

import cv2
import numpy as np

# COCO arac sinif adlari -> Turkce etiket.
TYPE_NAMES_TR = {
    "car": "OTOMOBIL",
    "motorcycle": "MOTOSIKLET",
    "bus": "OTOBUS",
    "truck": "KAMYON",
}

# Temel renk paleti: ad -> temsili BGR (kart uzerindeki ornek kare icin).
_COLOR_BGR = {
    "Siyah": (30, 30, 30),
    "Beyaz": (245, 245, 245),
    "Gri": (128, 128, 128),
    "Gumus": (190, 190, 190),
    "Kirmizi": (40, 40, 210),
    "Turuncu": (30, 130, 240),
    "Sari": (40, 220, 230),
    "Yesil": (60, 170, 70),
    "Mavi": (200, 110, 40),
    "Lacivert": (110, 60, 20),
    "Mor": (160, 60, 120),
}


def type_label(class_name: str) -> str:
    """COCO sinif adini Turkce arac tipine cevirir."""
    return TYPE_NAMES_TR.get(class_name, class_name.upper())


def compute_wb_scale(frame) -> "np.ndarray":
    """Tum kareden gray-world beyaz dengesi olcegi (BGR) hesaplar.

    Sahnenin renk yansimasini (orn. otoyolun soguk mavi tonu) tahmin eder.
    Gokyuzu/parlama (cok parlak) ve golge (cok koyu) pikselleri haric tutulur ki
    olcek araclarin govde tonuna gore daha dogru ciksin. KARE seviyesinde
    uygulanir (tek arac kirpintisina degil) -> arac renkleri korunur.
    """
    s = frame[::4, ::4].reshape(-1, 3).astype(np.float32)
    lum = s.mean(axis=1)
    keep = (lum > 30) & (lum < 230)
    sel = s[keep] if np.count_nonzero(keep) > 100 else s
    means = sel.mean(axis=0)
    gray = float(means.mean())
    # Olcegi sinirlama: asiri illuminant duzeltmesi renkleri ters cevirmesin.
    scale = gray / (means + 1e-6)
    return np.clip(scale, 0.6, 1.7)


def _foreground_mask(crop) -> "np.ndarray | None":
    """GrabCut ile arac govdesi (on plan) maskesi cikartir.

    Kutu icindeki arka plan (yol/gokyuzu/bariyer) elenir. Kucuk kirpintilarda
    veya basarisizlikta None doner -> cagiran merkez bolgeye geri duser.
    """
    h, w = crop.shape[:2]
    if h < 40 or w < 40:
        return None
    mask = np.zeros((h, w), np.uint8)
    # Kutu genelde araci sikica sarar; kenarlardan biraz icerisi kesin on plan.
    rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    # On plan cok kucukse (GrabCut araci bulamadi) guvenme.
    if np.count_nonzero(fg) < 0.15 * h * w:
        return None
    return fg


def _dominant_bgr(pixels) -> "np.ndarray | None":
    """Piksel kumesinin (Nx3 BGR) k-means ile en kalabalik renk merkezini verir."""
    if pixels is None or len(pixels) < 10:
        return None
    data = pixels.astype(np.float32)
    k = 3 if len(data) >= 60 else 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    return centers[int(np.argmax(counts))]


def _classify_bgr(bgr) -> str:
    """Tek bir baskin BGR rengi temel renk adina eslenir (Lab + HSV)."""
    px = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(int)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    L = lab[0]                     # 0..255
    a, b = lab[1] - 128, lab[2] - 128
    chroma = (a * a + b * b) ** 0.5
    hue, val = hsv[0], hsv[2]

    # --- Akromatik mi? (renksiz: siyah/gri/gumus/beyaz) ---
    # Lab kromasi dusukse renk yok demektir; HSV doygunlugundan daha kararli.
    if chroma < 20:
        if L < 70:
            return "Siyah"
        if L < 140:
            return "Gri"
        if L < 200:
            return "Gumus"
        return "Beyaz"

    # --- Kromatik: HSV tonuna gore (OpenCV hue 0..179) ---
    if val < 50:
        return "Siyah"
    if hue < 10 or hue >= 170:
        return "Kirmizi"
    if hue < 22:
        return "Turuncu"
    if hue < 33:
        return "Sari"
    if hue < 85:
        return "Yesil"
    if hue < 100:
        return "Mavi"
    if hue < 130:
        return "Lacivert" if L < 110 else "Mavi"
    return "Mor"


def estimate_color(vehicle_crop, wb_scale=None) -> tuple[str, tuple[int, int, int]]:
    """Arac kirpintisindan baskin rengi tahmin eder.

    Args:
        vehicle_crop: Arac BGR kirpintisi (arac kutusu; arka plan icerebilir).
        wb_scale: ``compute_wb_scale`` ile kareden hesaplanan BGR beyaz denge
            olcegi (opsiyonel; soguk/sicak isikta dogruluk icin onerilir).

    Returns:
        (renk_adi, ornek_bgr)
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "Bilinmiyor", (128, 128, 128)
    h, w = vehicle_crop.shape[:2]
    if h < 12 or w < 12:
        return "Bilinmiyor", (128, 128, 128)

    # 0) HIZ: renk analizi tam cozunurluk gerektirmez. Buyuk kirpintilari kucult
    #    (GrabCut maliyeti alanla karesel artar; 4K karelerde araç kutusu cok
    #    buyuk olabilir). ~128px genislik renk icin fazlasiyla yeterli.
    _MAX_W = 128
    if w > _MAX_W:
        scale = _MAX_W / float(w)
        vehicle_crop = cv2.resize(
            vehicle_crop, (_MAX_W, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = vehicle_crop.shape[:2]

    # 1) Beyaz dengesi (illuminant duzeltme) - kare seviyesinden gelen olcek.
    if wb_scale is not None:
        work = np.clip(
            vehicle_crop.astype(np.float32) * wb_scale, 0, 255
        ).astype(np.uint8)
    else:
        work = vehicle_crop

    # 2) Arka plani ayikla (GrabCut). Basarisizsa ust-orta govde bolgesine duser.
    fg = _foreground_mask(work)
    if fg is None:
        y1, y2 = int(h * 0.20), int(h * 0.60)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        fg = np.zeros((h, w), np.uint8)
        fg[y1:y2, x1:x2] = 255

    # 3) Govde maskesi: cam/golge (cok koyu) ve speküler parlama (cok parlak +
    #    dusuk doygunluk) pikselleri ele.
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    v_ch, s_ch = hsv[:, :, 2], hsv[:, :, 1]
    body = (fg > 0) & (v_ch > 35) & ~((v_ch > 245) & (s_ch < 40))
    if np.count_nonzero(body) < 40:
        body = fg > 0  # cok az kaldi: maskeyi gevset
    if np.count_nonzero(body) < 10:
        return "Bilinmiyor", (128, 128, 128)

    # 4) Baskin renk (k-means en kalabalik kume) + siniflandirma.
    pixels = work[body].reshape(-1, 3)
    dom = _dominant_bgr(pixels)
    if dom is None:
        return "Bilinmiyor", (128, 128, 128)
    name = _classify_bgr(dom)
    return name, _COLOR_BGR.get(name, (128, 128, 128))


class VehicleColorTracker:
    """Arac ID basina rengi EN BUYUK (en yakin) goruntuden tahmin eder.

    Arac uzaktayken kucuk/karanlik gorunur ve rengi yanlis cikar. Bu sinif
    rengi, aracin o ana kadarki EN BUYUK kutusundan hesaplar ve arac
    yaklastikca (kutu buyudukce) yeniden hesaplayip gunceller. Boylece
    gosterilen renk en guvenilir (en yakin) goruntuye dayanir ve titremez.
    GrabCut maliyeti de bu sayede arac basina seyrek odenir.
    """

    def __init__(self, grow_ratio: float = 1.15):
        self.grow_ratio = grow_ratio
        self._cache: dict[int, tuple[int, str, tuple[int, int, int]]] = {}

    def update(self, track_id: int, crop, wb_scale=None) -> tuple[str, tuple[int, int, int]]:
        area = int(crop.shape[0]) * int(crop.shape[1])
        prev = self._cache.get(track_id)
        # Ilk gorulus ya da belirgin sekilde daha yakin (buyuk) bir goruntu.
        if prev is None or area > prev[0] * self.grow_ratio:
            name, bgr = estimate_color(crop, wb_scale=wb_scale)
            if name != "Bilinmiyor" and (prev is None or area >= prev[0]):
                self._cache[track_id] = (area, name, bgr)
            elif prev is None:
                self._cache[track_id] = (area, name, bgr)
        entry = self._cache[track_id]
        return entry[1], entry[2]

    def cleanup(self, active_ids) -> None:
        for tid in [t for t in self._cache if t not in active_ids]:
            del self._cache[tid]
