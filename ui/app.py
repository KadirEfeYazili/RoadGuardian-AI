"""
RoadGuardian-CV - Video Secim / Yukleme Kontrol Paneli (Tkinter)

Bu masaustu panel, ANPR + hologram islemini calistirmadan once:
    - data/ klasorundeki videolari listeler,
    - bilgisayardan yeni video SECMENI (Gozat) veya data/ icine YUKLEMENI saglar,
    - ulke modunu (Otomatik / belirli ulke) ve cikti/kart seceneklerini ayarlar,
    - "Baslat" ile islemi ayri bir surecte (run.py) baslatir.

Calistirmak (proje kok dizininden):
    venv\\Scripts\\python run_ui.py
        veya
    venv\\Scripts\\pythonw ui\\app.py      (konsolsuz)

Not: Islem ayri bir surecte calisir; canli pencere acilir ('q' ile kapatilir).
Panel acik kalir, boylece pesi sira baska video calistirabilirsin.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.config import settings  # noqa: E402
from traffic_module.plate_ocr import KNOWN_COUNTRY_CODES  # noqa: E402

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
RUN_PY = ROOT / "run.py"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Ulke kodu -> okunabilir ad (panelde gostermek icin).
COUNTRY_NAMES = {
    "TR": "Turkiye", "GB": "Birlesik Krallik", "DE": "Almanya",
    "FR": "Fransa", "IT": "Italya", "ES": "Ispanya", "NL": "Hollanda",
    "BE": "Belcika", "PL": "Polonya", "RU": "Rusya", "UA": "Ukrayna",
    "RO": "Romanya", "CZ": "Cekya", "GR": "Yunanistan", "PT": "Portekiz",
    "AT": "Avusturya", "CH": "Isvicre", "SE": "Isvec", "US": "ABD",
}
AUTO_LABEL = "Otomatik (plakadan tahmin)"


class RoadGuardianUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RoadGuardian-CV | Video Kontrol Paneli")
        root.geometry("560x520")
        root.minsize(520, 480)

        self._procs: list[subprocess.Popen] = []
        self.selected_video: Path | None = None

        self._build()
        self._refresh_video_list()

    # ------------------------------------------------------------------ #
    # Arayuz kurulumu
    # ------------------------------------------------------------------ #
    def _build(self):
        pad = {"padx": 12, "pady": 6}

        title = ttk.Label(
            self.root, text="RoadGuardian-CV",
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(anchor="w", padx=12, pady=(12, 0))
        ttk.Label(
            self.root, text="Plaka Tanima + Hologram | Video sec ve baslat",
            foreground="#555",
        ).pack(anchor="w", padx=12, pady=(0, 6))

        # --- 1) Video listesi (data/) ---
        frame_list = ttk.LabelFrame(self.root, text="1) data/ icindeki videolar")
        frame_list.pack(fill="both", expand=True, **pad)

        self.listbox = tk.Listbox(frame_list, height=6, activestyle="dotbox")
        self.listbox.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        sb = ttk.Scrollbar(frame_list, orient="vertical", command=self.listbox.yview)
        sb.pack(side="left", fill="y", pady=8)
        self.listbox.config(yscrollcommand=sb.set)

        btns = ttk.Frame(frame_list)
        btns.pack(side="left", fill="y", padx=8, pady=8)
        ttk.Button(btns, text="Yenile", command=self._refresh_video_list).pack(
            fill="x", pady=2
        )
        ttk.Button(btns, text="Gozat...", command=self._browse_video).pack(
            fill="x", pady=2
        )
        ttk.Button(btns, text="data/'ya yukle...", command=self._upload_video).pack(
            fill="x", pady=2
        )

        # --- 2) Secili video ---
        self.lbl_selected = ttk.Label(
            self.root, text="Secili video: (yok)", foreground="#1a5fb4",
        )
        self.lbl_selected.pack(anchor="w", padx=14)

        # --- 3) Secenekler ---
        frame_opt = ttk.LabelFrame(self.root, text="2) Secenekler")
        frame_opt.pack(fill="x", **pad)

        row = ttk.Frame(frame_opt)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Ulke:").pack(side="left")
        self.country_var = tk.StringVar(value=AUTO_LABEL)
        choices = [AUTO_LABEL] + [
            f"{c} - {COUNTRY_NAMES.get(c, c)}" for c in KNOWN_COUNTRY_CODES
        ]
        self.country_combo = ttk.Combobox(
            row, textvariable=self.country_var, values=choices,
            state="readonly", width=32,
        )
        self.country_combo.pack(side="left", padx=8)

        # Performans modu (hiz/dogruluk).
        row2 = ttk.Frame(frame_opt)
        row2.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2, text="Performans:").pack(side="left")
        self.perf_labels = {
            "fast": "Hizli (en akici)",
            "balanced": "Dengeli (onerilen)",
            "accurate": "Dogru (en yavas)",
        }
        self.perf_var = tk.StringVar(value=self.perf_labels.get(settings.PERF_MODE))
        self.perf_combo = ttk.Combobox(
            row2, textvariable=self.perf_var,
            values=list(self.perf_labels.values()),
            state="readonly", width=28,
        )
        self.perf_combo.pack(side="left", padx=8)

        self.save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame_opt, text="Sonucu output/ klasorune .mp4 kaydet",
            variable=self.save_var,
        ).pack(anchor="w", padx=8, pady=2)

        self.card_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame_opt, text="Arac tip/renk bilgi kartini goster",
            variable=self.card_var,
        ).pack(anchor="w", padx=8, pady=(2, 8))

        # --- 4) Baslat ---
        self.btn_start = ttk.Button(
            self.root, text="▶  Baslat", command=self._start,
        )
        self.btn_start.pack(fill="x", padx=12, pady=(4, 2))

        self.status = ttk.Label(
            self.root,
            text="Hazir. Bir video sec ve Baslat'a bas. Canli pencerede 'q' ile cikilir.",
            foreground="#2a7", wraplength=520, justify="left",
        )
        self.status.pack(anchor="w", padx=14, pady=(2, 10))

    # ------------------------------------------------------------------ #
    # Video listesi / secimi
    # ------------------------------------------------------------------ #
    def _list_videos(self) -> list[Path]:
        if not DATA_DIR.exists():
            return []
        return sorted(
            p for p in DATA_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )

    def _refresh_video_list(self):
        self.listbox.delete(0, tk.END)
        self._videos = self._list_videos()
        for p in self._videos:
            size_mb = p.stat().st_size / (1024 * 1024)
            self.listbox.insert(tk.END, f"{p.name}   ({size_mb:.0f} MB)")
        if not self._videos:
            self.listbox.insert(tk.END, "(data/ bos - Gozat ya da yukle)")

    def _on_list_select(self, _event=None):
        sel = self.listbox.curselection()
        if not sel or not self._videos:
            return
        idx = sel[0]
        if idx < len(self._videos):
            self._set_selected(self._videos[idx])

    def _set_selected(self, path: Path):
        self.selected_video = path
        self.lbl_selected.config(text=f"Secili video: {path.name}")

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Video sec",
            filetypes=[("Video dosyalari", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                       ("Tum dosyalar", "*.*")],
        )
        if path:
            self._set_selected(Path(path))

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="data/'ya kopyalanacak videoyu sec",
            filetypes=[("Video dosyalari", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                       ("Tum dosyalar", "*.*")],
        )
        if not path:
            return
        src = Path(path)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dst = DATA_DIR / src.name
        if dst.exists() and not messagebox.askyesno(
            "Uzerine yaz?", f"data/ icinde '{src.name}' zaten var. Uzerine yazilsin mi?"
        ):
            return
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            messagebox.showerror("Yukleme hatasi", str(exc))
            return
        self._refresh_video_list()
        self._set_selected(dst)
        self.status.config(text=f"Yuklendi: {dst.name}", foreground="#2a7")

    # ------------------------------------------------------------------ #
    # Calistirma
    # ------------------------------------------------------------------ #
    def _start(self):
        if self.selected_video is None:
            messagebox.showwarning("Video yok", "Once bir video sec.")
            return
        if not self.selected_video.exists():
            messagebox.showerror("Bulunamadi", f"Dosya yok:\n{self.selected_video}")
            return

        cmd = [sys.executable, str(RUN_PY), "--video", str(self.selected_video)]

        choice = self.country_var.get()
        if choice == AUTO_LABEL:
            cmd += ["--country-mode", "auto"]
        else:
            code = choice.split(" - ")[0].strip()
            cmd += ["--country", code, "--country-mode", "force"]

        # Performans modu (etiketten koda).
        perf_code = next(
            (k for k, v in self.perf_labels.items() if v == self.perf_var.get()),
            settings.PERF_MODE,
        )
        cmd += ["--perf", perf_code]

        if self.save_var.get():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUTPUT_DIR / f"{self.selected_video.stem}_anpr.mp4"
            cmd += ["--save", str(out)]
        if not self.card_var.get():
            cmd += ["--no-card"]

        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT))
        except OSError as exc:
            messagebox.showerror("Baslatilamadi", str(exc))
            return
        self._procs.append(proc)
        self.status.config(
            text=(f"Baslatildi: {self.selected_video.name}\n"
                  "Canli pencere aciliyor (modeller ilk acilista biraz surebilir). "
                  "Cikmak icin pencerede 'q'."),
            foreground="#1a5fb4",
        )

    def on_close(self):
        running = [p for p in self._procs if p.poll() is None]
        if running and not messagebox.askyesno(
            "Cikis", f"{len(running)} islem hala calisiyor. Yine de kapatilsin mi?"
        ):
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")  # Windows'ta sik gorunum
    except tk.TclError:
        pass
    app = RoadGuardianUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
