import numpy as np
import cv2
from PIL import Image

def cargar_imagen_rgb(ruta):
    """Carga una imagen RGB normalizada desde archivo."""
    imagen = Image.open(ruta).convert("RGB")
    imagen = imagen.resize((300, 300))
    imagen_np = np.array(imagen, dtype=np.float32) / 255.0
    return imagen_np

def guardar_imagen_rgb(ruta, imagen_rgb):
    """Guarda una imagen RGB normalizada en archivo."""
    imagen_uint8 = (imagen_rgb * 255).astype(np.uint8)
    imagen_bgr = cv2.cvtColor(imagen_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(ruta, imagen_bgr)

def rgb_a_yiq(imagen_rgb):
    """Convierte imagen RGB normalizada a YIQ."""
    matriz = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])
    return np.dot(imagen_rgb, matriz.T)

def yiq_a_rgb(imagen_yiq):
    """Convierte imagen YIQ a RGB normalizada."""
    matriz_inv = np.array([
        [1.0, 0.956, 0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106, 1.703]
    ])
    rgb = np.dot(imagen_yiq, matriz_inv.T)
    return np.clip(rgb, 0.0, 1.0)