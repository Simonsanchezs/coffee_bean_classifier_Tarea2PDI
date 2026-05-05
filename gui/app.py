# gui/app.py
"""
Interfaz gráfica — Coffee Bean Classifier

Flujo:
  1. Cargar imagen.
  2. Elegir descriptor (HOG / SIFT) y modelo (SVM / Red Neuronal).
  3. Clasificar → muestra la clase predicha y confianza por clase.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from config.settings import CLASSES, IMAGE_SIZE
from features.hog_extractor  import extract_hog_single
from features.sift_extractor import encode_bow_single, load_vocabulary
from models.nn_classifier    import load_nn, predict_nn
from models.svm_classifier   import load_svm, predict_svm
from preprocessing.gaussian  import apply_gaussian

# ── Paleta ────────────────────────────────────────────────────────────────────
C = {
    "bg":     "#1A1714",
    "panel":  "#252018",
    "card":   "#2E2820",
    "accent": "#C8860A",
    "gold":   "#E09A18",
    "text":   "#F0E8DC",
    "muted":  "#7A6A58",
    "ok":     "#5DBB63",
    "border": "#3A3028",
}
CLASS_COLOR = {
    "Dark":   "#4A3728",
    "Green":  "#3A6A38",
    "Light":  "#C8A060",
    "Medium": "#8A5A30",
}

PREVIEW  = 300
F_TITLE  = ("Georgia", 19, "bold")
F_LABEL  = ("Georgia", 11)
F_MONO   = ("Courier New", 10)
F_RESULT = ("Georgia", 26, "bold")
F_SEC    = ("Courier New", 8, "bold")


class CoffeeBeanApp(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title("Coffee Bean Classifier  ☕")
        self.configure(bg=C["bg"])
        self.resizable(False, False)

        self._img: Optional[np.ndarray] = None
        self._img_preprocessed: Optional[np.ndarray] = None
        self._descriptor = tk.StringVar(value="HOG")
        self._model      = tk.StringVar(value="SVM")

        self._build_ui()
        self._center(1200, 750)

    # ── Construcción UI ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Encabezado
        hdr = tk.Frame(self, bg=C["bg"], pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="☕  Coffee Bean Classifier",
                 font=F_TITLE, bg=C["bg"], fg=C["gold"]).pack()
        tk.Label(hdr, text="HOG · SIFT · SVM · Red Neuronal",
                 font=("Georgia", 9), bg=C["bg"], fg=C["muted"]).pack()

        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=4)
        self._build_image_panel(body)
        self._build_controls_panel(body)
        self._build_result_panel()

    def _build_image_panel(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg=C["card"], padx=10, pady=10)
        card.pack(side="left", fill="both")
        
        # Columna izquierda: Imagen original
        left = tk.Frame(card, bg=C["card"])
        left.pack(side="left", padx=(0, 10), anchor="n")
        tk.Label(left, text="Imagen original", font=F_LABEL,
                 bg=C["card"], fg=C["muted"]).pack(pady=(0, 6))
        self._canvas = tk.Canvas(left, width=PREVIEW, height=PREVIEW,
                                 bg=C["bg"], highlightthickness=1,
                                 highlightbackground=C["border"])
        self._canvas.pack()
        self._draw_placeholder()
        tk.Button(left, text="📂  Cargar imagen",
                  command=self._load_image,
                  **self._btn(C["accent"])).pack(fill="x", pady=(10, 0))
        
        # Columna derecha: Imagen preprocesada
        right = tk.Frame(card, bg=C["card"])
        right.pack(side="left", anchor="n")
        tk.Label(right, text="Imagen preprocesada", font=F_LABEL,
                 bg=C["card"], fg=C["muted"]).pack(pady=(0, 6))
        self._canvas_preprocessed = tk.Canvas(right, width=PREVIEW, height=PREVIEW,
                                              bg=C["bg"], highlightthickness=1,
                                              highlightbackground=C["border"])
        self._canvas_preprocessed.pack()
        self._draw_placeholder_preprocessed()
        # Espacio invisible para alineación
        tk.Frame(right, height=37, bg=C["card"]).pack(fill="x", pady=(10, 0))

    def _build_controls_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(parent, bg=C["bg"])
        panel.pack(side="left", fill="both", expand=True, padx=(18, 0))

        # Descriptor
        self._section(panel, "DESCRIPTOR DE CARACTERÍSTICAS")
        df = tk.Frame(panel, bg=C["panel"], padx=14, pady=10)
        df.pack(fill="x", pady=(4, 12))
        for val, hint in [
            ("HOG",  "Histogram of Oriented Gradients"),
            ("SIFT", "Scale-Invariant Feature Transform + BoW"),
        ]:
            row = tk.Frame(df, bg=C["panel"])
            row.pack(anchor="w", pady=3)
            tk.Radiobutton(row, text=val, variable=self._descriptor, value=val,
                           **self._radio()).pack(side="left")
            tk.Label(row, text=f"— {hint}", font=("Georgia", 9),
                     bg=C["panel"], fg=C["muted"]).pack(side="left", padx=4)

        # Modelo
        self._section(panel, "MODELO CLASIFICADOR")
        mf = tk.Frame(panel, bg=C["panel"], padx=14, pady=10)
        mf.pack(fill="x", pady=(4, 12))
        for val, hint in [
            ("SVM",          "Support Vector Machine  (kernel RBF)"),
            ("Red Neuronal", "MLP multicapa  (TensorFlow / Keras)"),
        ]:
            row = tk.Frame(mf, bg=C["panel"])
            row.pack(anchor="w", pady=3)
            tk.Radiobutton(row, text=val, variable=self._model, value=val,
                           **self._radio()).pack(side="left")
            tk.Label(row, text=f"— {hint}", font=("Georgia", 9),
                     bg=C["panel"], fg=C["muted"]).pack(side="left", padx=4)

        tk.Button(panel, text="🔍  Clasificar",
                  command=self._classify,
                  **self._btn(C["ok"], fg=C["bg"],
                              font=("Georgia", 12, "bold"))).pack(fill="x", pady=(10, 0))

        

    def _build_result_panel(self) -> None:
        rc = tk.Frame(self, bg=C["panel"], pady=12)
        rc.pack(fill="x", padx=22, pady=(4, 14))
        self._result_lbl = tk.Label(rc, text="Carga una imagen y presiona Clasificar",
                                    font=("Georgia", 12), bg=C["panel"], fg=C["muted"])
        self._result_lbl.pack()
        self._prob_lbl = tk.Label(rc, text="", font=F_MONO, bg=C["panel"], fg=C["muted"])
        self._prob_lbl.pack()

    # ── Lógica ────────────────────────────────────────────────────────────────

    def _load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")],
        )
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", f"No se pudo leer:\n{path}")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, IMAGE_SIZE)
        self._img = rgb
        self._render(rgb)
        self._result_lbl.config(text="Imagen lista — presiona Clasificar", fg=C["muted"])
        self._prob_lbl.config(text="")

    def _classify(self) -> None:
        if self._img is None:
            messagebox.showwarning("Sin imagen", "Carga una imagen primero.")
            return
        try:
            preprocessed  = apply_gaussian(self._img.copy())
            self._img_preprocessed = preprocessed
            self._render_preprocessed(preprocessed)
            features       = self._extract_features(preprocessed)
            y_pred, probs  = self._run_model(features)
            self._show_result(int(y_pred[0]), probs[0])
        except FileNotFoundError as e:
            messagebox.showerror("Modelo no encontrado", str(e))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        if self._descriptor.get() == "HOG":
            return extract_hog_single(img).reshape(1, -1).astype(np.float32)
        km = load_vocabulary()
        return encode_bow_single(img, km).reshape(1, -1).astype(np.float32)

    def _run_model(self, X: np.ndarray):
        key = self._descriptor.get().lower()
        if self._model.get() == "SVM":
            return predict_svm(load_svm(key), X)
        return predict_nn(load_nn(key), X)

    def _show_result(self, idx: int, probs: np.ndarray) -> None:
        name  = CLASSES[idx]
        conf  = probs[idx] * 100
        color = CLASS_COLOR.get(name, C["accent"])
        self._result_lbl.config(
            text=f"  {name}   {conf:.1f}%  ",
            font=F_RESULT, bg=color, fg="#FFFFFF", padx=16, pady=4,
        )
        lines = []
        for i, cls in enumerate(CLASSES):
            p   = probs[i] * 100
            bar = "▓" * int(p / 5) + "░" * (20 - int(p / 5))
            mark = "  ◀" if i == idx else ""
            lines.append(f"  {cls:<8} {bar}  {p:5.1f}%{mark}")
        self._prob_lbl.config(text="\n".join(lines), fg=C["text"])

    # ── Utilidades UI ─────────────────────────────────────────────────────────

    def _render(self, img: np.ndarray) -> None:
        pil   = Image.fromarray(img).resize((PREVIEW, PREVIEW), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._canvas.create_image(PREVIEW // 2, PREVIEW // 2, anchor="center", image=photo)
        self._canvas._photo = photo

    def _render_preprocessed(self, img: np.ndarray) -> None:
        pil   = Image.fromarray(img).resize((PREVIEW, PREVIEW), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._canvas_preprocessed.create_image(PREVIEW // 2, PREVIEW // 2, anchor="center", image=photo)
        self._canvas_preprocessed._photo = photo

    def _draw_placeholder(self) -> None:
        self._canvas.create_text(PREVIEW // 2, PREVIEW // 2,
                                 text="☕\nSin imagen", fill=C["muted"],
                                 font=("Georgia", 13), justify="center")

    def _draw_placeholder_preprocessed(self) -> None:
        self._canvas_preprocessed.create_text(PREVIEW // 2, PREVIEW // 2,
                                              text="⏳\nClasificar", fill=C["muted"],
                                              font=("Georgia", 13), justify="center")

    def _section(self, parent, text: str) -> None:
        tk.Label(parent, text=text, font=F_SEC,
                 bg=C["bg"], fg=C["accent"]).pack(anchor="w", pady=(12, 0))

    @staticmethod
    def _btn(bg, fg="#1A1714", font=("Georgia", 10, "bold")) -> dict:
        return dict(bg=bg, fg=fg, font=font, relief="flat",
                    cursor="hand2", padx=12, pady=7, bd=0, activebackground=bg)

    @staticmethod
    def _radio() -> dict:
        return dict(bg=C["panel"], fg=C["text"], selectcolor=C["bg"],
                    activebackground=C["panel"], font=F_LABEL, cursor="hand2")

    def _center(self, w: int, h: int) -> None:
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


def run_app() -> None:
    CoffeeBeanApp().mainloop()


if __name__ == "__main__":
    run_app()
