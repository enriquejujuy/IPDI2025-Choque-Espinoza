import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Procesador de Imágenes")
ventana.geometry("1300x700")

# Variables Tkinter
Ymin = tk.DoubleVar(value=0.2)
Ymax = tk.DoubleVar(value=0.8)

# Variables globales
matriz_original = None
matriz_procesada = None

# Funciones base
def cargar_imagen(ruta, tamaño=(300, 300)):
    imagen = Image.open(ruta).convert("RGB")
    imagen = imagen.resize(tamaño)
    matriz = np.array(imagen).astype(float) / 255
    return matriz

def matriz_a_imagen(matriz):
    matriz = np.clip(matriz, 0, 1) * 255
    return Image.fromarray(matriz.astype(np.uint8))

def rgb_a_yiq(matriz):
    T = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.274, -0.322],
                  [0.211, -0.523, 0.312]])
    return np.dot(matriz, T.T)

def yiq_a_rgb(matriz_yiq):
    T_inv = np.array([[1.0, 0.956, 0.621],
                      [1.0, -0.272, -0.647],
                      [1.0, -1.106, 1.703]])
    rgb = np.dot(matriz_yiq, T_inv.T)
    return np.clip(rgb, 0, 1)

# Procesamiento acumulativo de luminancia
def procesar_luminancia(matriz_rgb, tipo):
    yiq = rgb_a_yiq(matriz_rgb)
    Y = yiq[:, :, 0]

    if tipo == "Raíz cuadrada":
        Y = np.sqrt(Y)
    elif tipo == "Cuadrática":
        Y = np.power(Y, 2)
    elif tipo == "Lineal a trozos":
        ymin = Ymin.get()
        ymax = Ymax.get()
        Yp = np.zeros_like(Y)
        Yp[Y < ymin] = 0
        Yp[Y > ymax] = 1
        mask = (Y >= ymin) & (Y <= ymax)
        Yp[mask] = (Y[mask] - ymin) / (ymax - ymin)
        Y = Yp

    yiq[:, :, 0] = np.clip(Y, 0, 1)
    return yiq_a_rgb(yiq)

# Mostrar imagen
def mostrar_imagen(matriz, label):
    imagen = matriz_a_imagen(matriz)
    img_tk = ImageTk.PhotoImage(imagen)
    label.config(image=img_tk)
    label.image = img_tk

# Mostrar histograma
def mostrar_histograma(matriz, canvas):
    Y = rgb_a_yiq(matriz)[:, :, 0]
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.hist(Y.flatten(), bins=20, color='orange', edgecolor='black', linewidth=1.2)
    ax.set_title("Histograma de Luminancia Y")
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    canvas.draw()

# Graficar función lineal a trozos
def graficar_funcion_lineal():
    ymin = Ymin.get()
    ymax = Ymax.get()
    Y = np.linspace(0, 1, 256)
    Yp = np.piecewise(Y,
                      [Y < ymin, Y > ymax, (Y >= ymin) & (Y <= ymax)],
                      [0, 1, lambda y: (y - ymin) / (ymax - ymin)])
    fig = canvas_funcion.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(Y, Yp, color='blue')
    ax.set_title("Transformación Lineal a Trozos")
    ax.set_xlabel("Y")
    ax.set_ylabel("Y'")
    canvas_funcion.draw()

def actualizar_ymin(val): graficar_funcion_lineal()
def actualizar_ymax(val): graficar_funcion_lineal()

# Cargar imagen
def cargar():
    global matriz_original, matriz_procesada
    ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")])
    if ruta:
        matriz_original = cargar_imagen(ruta)
        matriz_procesada = None
        mostrar_imagen(matriz_original, label_original)
        mostrar_histograma(matriz_original, canvas_hist_original)
        mostrar_imagen(matriz_original, label_procesada)
        mostrar_histograma(matriz_original, canvas_hist_procesada)

# Procesar imagen acumulativamente
def procesar():
    global matriz_procesada
    if matriz_original is None:
        return
    tipo = combo.get()
    base = matriz_procesada if matriz_procesada is not None else matriz_original.copy()
    matriz_procesada = procesar_luminancia(base.copy(), tipo)
    mostrar_imagen(matriz_procesada, label_procesada)
    mostrar_histograma(matriz_procesada, canvas_hist_procesada)

# Reiniciar procesamiento
def reiniciar():
    global matriz_procesada
    if matriz_original is None:
        return
    matriz_procesada = None
    mostrar_imagen(matriz_original, label_procesada)
    mostrar_histograma(matriz_original, canvas_hist_procesada)

# Guardar imagen
def guardar():
    if matriz_procesada is None:
        return
    imagen = matriz_a_imagen(matriz_procesada)
    ruta = filedialog.asksaveasfilename(defaultextension=".png")
    if ruta:
        imagen.save(ruta)

# Frames
frame_original = tk.Frame(ventana)
frame_original.pack(side=tk.LEFT, padx=10, pady=10)

frame_central = tk.Frame(ventana)
frame_central.pack(side=tk.LEFT, padx=10, pady=10)

frame_procesada = tk.Frame(ventana)
frame_procesada.pack(side=tk.LEFT, padx=10, pady=10)

# Imagen original
btn_cargar = tk.Button(frame_original, text="Cargar Imagen", command=cargar)
btn_cargar.pack()

label_original = tk.Label(frame_original)
label_original.pack()

canvas_hist_original = FigureCanvasTkAgg(plt.Figure(figsize=(4, 2)), master=frame_original)
canvas_hist_original.get_tk_widget().pack()

# Controles centrales
combo = ttk.Combobox(frame_central, values=["Raíz cuadrada", "Cuadrática", "Lineal a trozos"])
combo.set("Raíz cuadrada")
combo.pack(pady=10)

btn_procesar = tk.Button(frame_central, text="Procesar Imagen", command=procesar)
btn_procesar.pack(pady=10)

btn_reiniciar = tk.Button(frame_central, text="Reiniciar Procesamiento", command=reiniciar)
btn_reiniciar.pack(pady=10)

tk.Label(frame_central, text="Ymin").pack()
slider_ymin = tk.Scale(frame_central, from_=0.0, to=1.0, resolution=0.01,
                       orient=tk.HORIZONTAL, variable=Ymin, command=actualizar_ymin)
slider_ymin.pack()

tk.Label(frame_central, text="Ymax").pack()
slider_ymax = tk.Scale(frame_central, from_=0.0, to=1.0, resolution=0.01,
                       orient=tk.HORIZONTAL, variable=Ymax, command=actualizar_ymax)
slider_ymax.pack()

canvas_funcion = FigureCanvasTkAgg(plt.Figure(figsize=(4, 2)), master=frame_central)
canvas_funcion.get_tk_widget().pack()
graficar_funcion_lineal()

btn_guardar = tk.Button(frame_central, text="Guardar Imagen", command=guardar)
btn_guardar.pack(pady=10)

btn_salir = tk.Button(frame_central, text="  Salir  ", command=ventana.quit)
btn_salir.pack(pady=10)

# Imagen procesada
label_procesada = tk.Label(frame_procesada)
label_procesada.pack()

canvas_hist_procesada = FigureCanvasTkAgg(plt.Figure(figsize=(4, 2)), master=frame_procesada)
canvas_hist_procesada.get_tk_widget().pack()

# Ejecutar
ventana.mainloop()