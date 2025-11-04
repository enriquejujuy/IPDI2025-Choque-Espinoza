import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
# Importar funciones desde módulos definidos
from funciones import cargar_imagen_rgb, guardar_imagen_rgb, rgb_a_yiq, yiq_a_rgb
from filtros import (
    filtro_plano, filtro_bartlett, filtro_gaussiano,
    filtro_laplaciano_v4, filtro_laplaciano_v8,
    filtro_sobel, filtro_pasaaltos, aplicar_filtro
)
# ============================
# Funciones de interfaz
def cargar_imagen():
    ruta = filedialog.askopenfilename()
    if ruta:
        imagen_rgb = cargar_imagen_rgb(ruta)
        imagen_yiq = rgb_a_yiq(imagen_rgb)
        canal_y = imagen_yiq[:, :, 0]
        app.imagen_y = canal_y
        app.imagen_yiq = imagen_yiq

        imagen_mostrar = Image.fromarray((canal_y * 255).astype(np.uint8))
        imagen_mostrar.thumbnail((300, 300))
        imagen_tk = ImageTk.PhotoImage(imagen_mostrar)
        panel_imagen.config(image=imagen_tk)
        panel_imagen.image = imagen_tk

def procesar_imagen():
    if hasattr(app, 'imagen_y'):
        filtro = combo_filtros.get()
        f_corte = slider_frecuencia.get()
        y_original = (app.imagen_y * 255).astype(np.uint8)

        # Aplicar filtro seleccionado
        if filtro == "Plano 3×3":
            y_filtrada = aplicar_filtro(y_original, filtro_plano(3))
        elif filtro == "Bartlett 3×3":
            y_filtrada = aplicar_filtro(y_original, filtro_bartlett(3))
        elif filtro == "Bartlett 5×5":
            y_filtrada = aplicar_filtro(y_original, filtro_bartlett(5))
        elif filtro == "Bartlett 7×7":
            y_filtrada = aplicar_filtro(y_original, filtro_bartlett(7))
        elif filtro == "Gaussiano 5×5":
            y_filtrada = aplicar_filtro(y_original, filtro_gaussiano(5))
        elif filtro == "Gaussiano 7×7":
            y_filtrada = aplicar_filtro(y_original, filtro_gaussiano(7))
        elif filtro == "Laplaciano v4":
            y_filtrada = filtro_laplaciano_v4(y_original)
        elif filtro == "Laplaciano v8":
            y_filtrada = filtro_laplaciano_v8(y_original)
        elif filtro.startswith("Sobel"):
            direccion = filtro.split(" ")[1]
            y_filtrada = filtro_sobel(y_original, direccion)
        elif filtro == "Pasaaltos (frecuencia)":
            y_filtrada = filtro_pasaaltos(y_original, f_corte)
        else:
            return

        # Mostrar canal Y filtrado en panel derecho
        imagen_filtrada = Image.fromarray(y_filtrada)
        imagen_filtrada.thumbnail((300, 300))
        imagen_tk = ImageTk.PhotoImage(imagen_filtrada)
        panel_procesada.config(image=imagen_tk)
        panel_procesada.image = imagen_tk

        # Guardar canal Y filtrado normalizado
        app.imagen_procesada = y_filtrada / 255.0

def guardar_imagen():
    if hasattr(app, 'imagen_procesada'):
        ruta = filedialog.asksaveasfilename(defaultextension=".bmp",
                                            filetypes=[("Imagen BMP", "*.bmp"),
                                                       ("PNG", "*.png"),
                                                       ("JPEG", "*.jpg")])
        if ruta:
            # Convertir canal Y normalizado a imagen en escala de grises (uint8)
            imagen_y = (app.imagen_procesada * 255).astype(np.uint8)
            imagen_pil = Image.fromarray(imagen_y, mode='L')  # 'L' = grayscale
            imagen_pil.save(ruta)
# ============================
# Interfaz gráfica principal
app = tk.Tk()
app.title("IPDI 2025 - Procesador de Imágenes")
app.geometry("800x620")

# Imagen original
frame_imagen = tk.LabelFrame(app, text="Imagen Original (Canal Y)", width=300, height=300)
frame_imagen.pack_propagate(False)
frame_imagen.place(x=20, y=20)
panel_imagen = tk.Label(frame_imagen)
panel_imagen.pack()

btn_cargar = tk.Button(app, text="Cargar Imagen", command=cargar_imagen)
btn_cargar.place(x=100, y=340)

# Imagen procesada
frame_procesada = tk.LabelFrame(app, text="Imagen Procesada (Canal Y)", width=300, height=300)
frame_procesada.pack_propagate(False)
frame_procesada.place(x=480, y=20)
panel_procesada = tk.Label(frame_procesada)
panel_procesada.pack()

btn_guardar = tk.Button(app, text="Guardar Imagen", command=guardar_imagen)
btn_guardar.place(x=560, y=340)

# Controles
frame_controles = tk.Frame(app, width=160, height=200)
frame_controles.place(x=320, y=120)

combo_filtros = ttk.Combobox(frame_controles, values=[
    "Plano 3×3",
    "Bartlett 3×3",
    "Bartlett 5×5",
    "Bartlett 7×7",
    "Gaussiano 5×5",
    "Gaussiano 7×7",
    "Laplaciano v4",
    "Laplaciano v8",
    "Sobel N",
    "Sobel S",
    "Sobel E",
    "Sobel O",
    "Sobel NE",
    "Sobel NO",
    "Sobel SE",
    "Sobel SO",
    "Pasaaltos (frecuencia)"
])
combo_filtros.set("Plano 3×3")
combo_filtros.pack(pady=10)

btn_procesar = tk.Button(frame_controles, text="Procesar Imagen", command=procesar_imagen)
btn_procesar.pack(pady=10)

label_slider = tk.Label(app, text="Frecuencia de corte (solo para pasaaltos)")
label_slider.place(x=300, y=370)

slider_frecuencia = tk.Scale(app, from_=0.01, to=1.0, resolution=0.01,
                             orient="horizontal", length=200)
slider_frecuencia.set(0.2)
slider_frecuencia.place(x=300, y=400)

app.mainloop()