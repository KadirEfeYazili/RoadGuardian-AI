"""
RoadGuardian-CV - Hologram Plaka Overlay Modulu

Tespit edilen plakayi, aracin USTUNDE havada duran, GERCEK BIR PLAKA gibi
gorunen ama holografik (yari saydam + parlayan) bir panelde gosterir:

    - Gercek plaka bicimi: yuvarlatilmis kenarli, acik zeminli plaka
    - Solda MAVI ulke kodu bandi (orn. "TR" / "GB") - AB tarzi
    - Koyu, kalin plaka metni
    - Aractan panele uzanan projeksiyon huzmesi + kaynak nokta
    - Parlak cam gobegi cerceve + kose ayraclari + kucuk guven etiketi

Cizim tek bir BGR kare uzerinde yapilir; pencere kuculmeden ONCE cizilir ki
hologram tam cozunurlukte keskin kalsin.
"""

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402

# Renk paleti (BGR).
HOLO_CYAN = (255, 255, 0)       # Parlak cam gobegi - cerceve/huzme
HOLO_GLOW = (255, 230, 120)     # Acik tint - parlamalar
PLATE_BG = (245, 248, 248)      # Plaka acik zemini
PLATE_TEXT = (32, 32, 38)       # Koyu plaka metni
EU_BLUE = (200, 70, 0)          # Ulke kodu bandi (mavi)
WHITE = (255, 255, 255)

FONT_NUM = cv2.FONT_HERSHEY_DUPLEX   # Plaka rakam/harfleri (kalin)
FONT_UI = cv2.FONT_HERSHEY_SIMPLEX   # Kucuk UI metinleri


def _round_rect(img, p1, p2, color, r, thickness):
    """Yuvarlatilmis kose dikdortgen (thickness<0 -> dolu)."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        t = thickness
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, cv2.LINE_AA)


def _band_left_rounded(img, p1, p2, color, r):
    """Sadece SOL koseleri yuvarlatilmis dolu dikdortgen (ulke kodu bandi)."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(r, (x2 - x1), (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x1 + r, y2 - r), color, -1)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)


def _corner_brackets(frame, p1, p2, color, length=16, thick=2):
    """Panel koselerine HUD tarzi L ayraclari (hologram cerceve hissi)."""
    x1, y1 = p1
    x2, y2 = p2
    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ):
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thick, cv2.LINE_AA)


def draw_plate_hologram(
    frame,
    vehicle_box,
    text,
    plate_box=None,
    conf=None,
    country_code=None,
    color=HOLO_CYAN,
):
    """Aracin ustune gercek-plaka gorunumlu holografik panel cizer.

    Args:
        frame: Uzerine cizilecek BGR kare (yerinde degistirilir).
        vehicle_box: (x1, y1, x2, y2) arac kutusu.
        text: Plaka metni.
        plate_box: (x1, y1, x2, y2) plakanin gercek konumu (huzme kaynagi).
        conf: 0..1 OCR guveni (kucuk etikette % olarak gosterilir).
        country_code: Soldaki mavi banda yazilacak ulke kodu (orn. "TR").
        color: Holografik vurgu rengi (BGR).
    """
    code = (country_code or settings.PLATE_COUNTRY_CODE).upper()
    plate = text.upper()
    vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_box]
    fh, fw = frame.shape[:2]

    # --- Olcekler: arac genisligine gore ---
    veh_w = max(40, vx2 - vx1)
    font_scale = max(0.9, min(2.4, veh_w / 165.0))
    num_thick = max(2, int(round(font_scale * 1.6)))
    (tw, th), _ = cv2.getTextSize(plate, FONT_NUM, font_scale, num_thick)

    pad = max(8, int(th * 0.5))

    # Ulke kodu bandi olcusu.
    code_scale = font_scale * 0.6
    code_thick = max(1, int(round(code_scale * 2)))
    (cw, ch), _ = cv2.getTextSize(code, FONT_NUM, code_scale, code_thick)
    band_w = max(cw + 14, int((th + 2 * pad) * 0.52))

    panel_h = th + 2 * pad
    panel_w = band_w + pad + tw + pad
    radius = max(5, int(panel_h * 0.18))

    # --- Konum: aracin ust-orta noktasinin uzerinde ---
    cx = (vx1 + vx2) // 2
    gap = 26
    px1 = cx - panel_w // 2
    py2 = vy1 - gap
    py1 = py2 - panel_h

    # Kare disina tasmayi engelle.
    px1 = max(6, min(px1, fw - panel_w - 6))
    if py1 < 22:  # ustte guven etiketine de yer birak
        py1 = 22
        py2 = py1 + panel_h
    px2 = px1 + panel_w

    # --- Projeksiyon huzmesi (plaka -> panel) ---
    if plate_box is not None:
        sx = (int(plate_box[0]) + int(plate_box[2])) // 2
        sy = int(plate_box[1])
    else:
        sx, sy = cx, vy1
    beam = frame.copy()
    cv2.line(beam, (sx, sy), (px1 + radius, py2), color, 1, cv2.LINE_AA)
    cv2.line(beam, (sx, sy), (px2 - radius, py2), color, 1, cv2.LINE_AA)
    cv2.addWeighted(beam, 0.45, frame, 0.55, 0, frame)
    cv2.circle(frame, (sx, sy), 4, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (sx, sy), 7, color, 1, cv2.LINE_AA)

    # --- Panel govdesi: ROI uzerinde yari saydam ciz (holografik) ---
    rx1, ry1 = max(0, px1), max(0, py1)
    rx2, ry2 = min(fw, px2), min(fh, py2)
    if rx2 <= rx1 or ry2 <= ry1:
        return frame
    roi = frame[ry1:ry2, rx1:rx2]
    overlay = roi.copy()

    def L(p):  # frame koord -> ROI yerel koord
        return (p[0] - rx1, p[1] - ry1)

    # Acik plaka zemini (yuvarlak), uzerine mavi ulke bandi.
    _round_rect(overlay, L((px1, py1)), L((px2, py2)), PLATE_BG, radius, -1)
    _band_left_rounded(overlay, L((px1, py1)), L((px1 + band_w, py2)), EU_BLUE, radius)
    # Band - plaka ayirici ince cizgi.
    cv2.line(overlay, L((px1 + band_w, py1 + 2)), L((px1 + band_w, py2 - 2)),
             (210, 210, 210), 1, cv2.LINE_AA)

    # Ulke kodu (banda, dikey ortalanmis, beyaz).
    code_org = L((px1 + (band_w - cw) // 2, py1 + (panel_h + ch) // 2))
    cv2.putText(overlay, code, code_org, FONT_NUM, code_scale, WHITE,
                code_thick, cv2.LINE_AA)

    # Plaka metni (koyu, kalin).
    num_org = L((px1 + band_w + pad, py1 + (panel_h + th) // 2))
    cv2.putText(overlay, plate, num_org, FONT_NUM, font_scale, PLATE_TEXT,
                num_thick, cv2.LINE_AA)

    # Holografik saydamlik: metin okunur kalsin diye yuksek alpha.
    cv2.addWeighted(overlay, 0.88, roi, 0.12, 0, roi)

    # --- Parlak cerceve + kose ayraclari (tam opak, keskin) ---
    _round_rect(frame, (px1, py1), (px2, py2), color, radius, 2)
    _corner_brackets(frame, (px1, py1), (px2, py2), color)

    # --- Kucuk guven etiketi (panel ustunde, holografik chip) ---
    if conf is not None:
        tag = f"OKUNDU  %{int(conf * 100)}"
        cv2.circle(frame, (px1 + 4, py1 - 8), 3, color, -1, cv2.LINE_AA)
        cv2.putText(frame, tag, (px1 + 12, py1 - 4), FONT_UI, 0.42, color,
                    1, cv2.LINE_AA)

    return frame


def draw_vehicle_card(
    frame,
    vehicle_box,
    type_label,
    color_name,
    color_bgr,
    accent=HOLO_CYAN,
):
    """Aracin yan kosesine tip + renk bilgisini gosteren holografik kart cizer.

    Plaka hologrami aracin USTUNDE durdugu icin bu kart, ust kosenin YANINA
    (sagina; sigmazsa soluna) yerlestirilir; boylece ikisi cakismaz.

    Args:
        frame: BGR kare (yerinde degistirilir).
        vehicle_box: (x1, y1, x2, y2) arac kutusu.
        type_label: Arac tipi (orn. "OTOMOBIL").
        color_name: Renk adi (orn. "Beyaz").
        color_bgr: Renk ornegi karesi icin BGR.
        accent: Holografik vurgu rengi.
    """
    vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_box]
    fh, fw = frame.shape[:2]

    line1 = type_label
    line2 = color_name
    (w1, h1), _ = cv2.getTextSize(line1, FONT_UI, 0.5, 1)
    (w2, h2), _ = cv2.getTextSize(line2, FONT_UI, 0.48, 1)

    swatch = 14
    pad = 8
    card_w = max(w1, swatch + 6 + w2) + pad * 2
    card_h = h1 + h2 + pad * 3

    # Konum: aracin sag-ust kosesinin yanina; sigmazsa sol tarafa.
    cx1 = vx2 + 8
    if cx1 + card_w > fw - 4:
        cx1 = vx1 - 8 - card_w
    cx1 = max(4, cx1)
    cy1 = max(4, vy1)
    cx2, cy2 = cx1 + card_w, cy1 + card_h
    if cy2 > fh - 4:
        cy2 = fh - 4
        cy1 = cy2 - card_h

    # Yari saydam koyu zemin (ROI blend).
    rx1, ry1, rx2, ry2 = max(0, cx1), max(0, cy1), min(fw, cx2), min(fh, cy2)
    if rx2 <= rx1 or ry2 <= ry1:
        return frame
    roi = frame[ry1:ry2, rx1:rx2]
    overlay = roi.copy()
    overlay[:] = (35, 25, 10)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)

    # Cerceve + sol vurgu serit.
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), accent, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (cx1, cy1), (cx1 + 3, cy2), accent, -1)

    # 1. satir: tip.
    cv2.putText(frame, line1, (cx1 + pad, cy1 + pad + h1),
                FONT_UI, 0.5, WHITE, 1, cv2.LINE_AA)
    # 2. satir: renk ornegi + ad.
    sy = cy1 + pad * 2 + h1
    cv2.rectangle(frame, (cx1 + pad, sy), (cx1 + pad + swatch, sy + h2),
                  color_bgr, -1)
    cv2.rectangle(frame, (cx1 + pad, sy), (cx1 + pad + swatch, sy + h2),
                  (220, 220, 220), 1)
    cv2.putText(frame, line2, (cx1 + pad + swatch + 6, sy + h2 - 1),
                FONT_UI, 0.48, accent, 1, cv2.LINE_AA)

    return frame
