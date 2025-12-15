import cv2
import numpy as np
from typing import Optional, Tuple

from skimage import measure

#TP7 SEGMENTACION CREO 

class Segmentador:
    def _validar(self, img: np.ndarray) -> None:
        if img is None or getattr(img, "size", 0) == 0:
            raise ValueError("Imagen nula o vacía.")

    def _a_gris_u8(self, img: np.ndarray) -> np.ndarray:
        self._validar(img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img


    def umbral_50(self, img: np.ndarray) -> np.ndarray:
        #Binariza con percentil 50 casi casi mitad blacno y negro
        g = self._a_gris_u8(img)
        t = int(np.percentile(g, 50))
        _, out = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
        return out

    def umbral_bimodal(self, img: np.ndarray, win: int = 11, sigma: float = 2.0) -> np.ndarray:
        #Estima dos modas (oscura/clara) en histograma suavizado y umbraliza a mitad de camino.
 
        g = self._a_gris_u8(img)

        hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)

        # Suavizado 1D (gauss) del histograma
        win = max(3, int(win) | 1)  # impar
        x = np.arange(win) - win // 2
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k /= k.sum()
        hs = np.convolve(hist, k, mode="same")

        corte = int(np.mean(g))  # separa 
        corte = np.clip(corte, 1, 255)

        m1 = int(np.argmax(hs[:corte]))
        m2 = int(np.argmax(hs[corte:]) + corte)

        t = int((m1 + m2) / 2)
        _, out = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
        return out

    def umbral_otsu(self, img: np.ndarray) -> np.ndarray:
        #Umbral autommatco por Otsu 
        g = self._a_gris_u8(img)
        _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out

    #BORDES

    def bordes_laplaciano(self, img: np.ndarray, ksize: int = 3) -> np.ndarray:
        g = self._a_gris_u8(img)
        lap = cv2.Laplacian(g, cv2.CV_16S, ksize=ksize)
        return cv2.convertScaleAbs(lap)

    def bordes_morfologico(self, img: np.ndarray, k: int = 3) -> np.ndarray:
        #Gradiente morfologico:
        g = self._a_gris_u8(img)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(g, cv2.MORPH_GRADIENT, se)

    #CONTORNOS 

    def marching_squares(self, img: np.ndarray, level: Optional[float] = None) -> np.ndarray:
    
        g = self._a_gris_u8(img)

        if level is None:
            t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            level = float(t)

        conts = measure.find_contours(g, level=level)

        canvas = np.zeros_like(g)
        for c in conts:
            pts = np.flip(c, axis=1).astype(np.int32)  # (x,y)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=1)
        return canvas

    # -------------------- REGIONES --------------------

    def varita_magica_mask(
        self,
        img: np.ndarray,
        seed_xy: Tuple[int, int],
        tolerancia: int = 20,
        conectividad: int = 4,
    ) -> np.ndarray:
        #Flood fill tipo "varita mágica": devuelve máscara binaria (0/255).
        self._validar(img)
        h, w = img.shape[:2]
        x, y = seed_xy
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("seed_xy fuera de la imagen (x,y).")

        work = img.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)

        flags = conectividad | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE

        if work.ndim == 2:
            lo = (tolerancia,)
            up = (tolerancia,)
            new_val = 255
        else:
            lo = (tolerancia, tolerancia, tolerancia)
            up = (tolerancia, tolerancia, tolerancia)
            new_val = (0, 255, 0)

        cv2.floodFill(work, mask, (x, y), new_val, loDiff=lo, upDiff=up, flags=flags)

        return (mask[1:-1, 1:-1] * 255).astype(np.uint8)

    def overlay_mascara(self, img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        #Superpone la maskara
        self._validar(img)
        self._validar(mask)

        base = img
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        base = base.copy()

        m = (mask > 0)
        verde = np.zeros_like(base)
        verde[..., 1] = 255

        base[m] = cv2.addWeighted(base[m], 1.0 - alpha, verde[m], alpha, 0)
        return base


if __name__ == "__main__":
    x = np.linspace(0, 255, 128, dtype=np.float32)
    xv, yv = np.meshgrid(x, x)
    grad = ((xv + yv) / 2).astype(np.uint8)

    seg = Segmentador()
    b = seg.umbral_otsu(grad)
    print("OK -> umbral_otsu sobre gradiente:", b.shape, b.dtype)