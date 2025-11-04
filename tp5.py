import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
from funciones import cargar_imagen_rgb, rgb_a_yiq

# ============================
# Elemento estructurante
def crear_elemento_estructurante(tam, forma):
    if tam % 2 == 0:
        tam += 1
    centro = tam // 2
    estructura = np.zeros((tam, tam), dtype=np.uint8)

    if forma == "Cuadrado":
        estructura[:, :] = 1
    elif forma == "Cruz":
        estructura[centro, :] = 1
        estructura[:, centro] = 1
    elif forma == "Disco":
        for i in range(tam):
            for j in range(tam):
                if (i - centro)**2 + (j - centro)**2 <= centro**2:
                    estructura[i, j] = 1
    return estructura

# ============================
# Procesamiento
def binarizar(imagen, umbral):
    return (imagen >= umbral).astype(np.uint8) * 255

def erosion(imagen, estructura):
    tam = estructura.shape[0]
    pad = tam // 2
    padded = np.pad(imagen, pad, mode='edge')
    resultado = np.zeros_like(imagen)
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            vecindad = padded[i:i+tam, j:j+tam]
            resultado[i, j] = np.min(vecindad[estructura == 1])
    return resultado

def dilatacion(imagen, estructura):
    tam = estructura.shape[0]
    pad = tam // 2
    padded = np.pad(imagen, pad, mode='edge')
    resultado = np.zeros_like(imagen)
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            vecindad = padded[i:i+tam, j:j+tam]
            resultado[i, j] = np.max(vecindad[estructura == 1])
    return resultado

# ============================
# Interfaz
def cargar_imagen():
    ruta = filedialog.askopenfilename()
    if ruta:
        imagen_rgb = cargar_imagen_rgb(ruta)
        imagen_yiq = rgb_a_yiq(imagen_rgb)
        canal_y = imagen_yiq[:, :, 0]
        app.imagen_y = canal_y
        frame_imagen.config(text="Imagen Original (Canal Y)")
        btn_copiar.config(state="disabled")

        imagen_mostrar = Image.fromarray((canal_y * 255).astype(np.uint8))
        imagen_mostrar.thumbnail((350, 350))
        imagen_tk = ImageTk.PhotoImage(imagen_mostrar)
        panel_imagen.config(image=imagen_tk)
        panel_imagen.image = imagen_tk

def procesar_imagen():
    if hasattr(app, 'imagen_y'):
        operacion = combo_operaciones.get()
        umbral = slider_umbral.get()
        tam = int(combo_tamanio.get())
        forma = combo_forma.get()
        estructura = crear_elemento_estructurante(tam, forma)
        y_original = (app.imagen_y * 255).astype(np.uint8)

        if operacion == "Binarizar":
            y_filtrada = binarizar(app.imagen_y, umbral)
        elif operacion == "Erosión":
            y_filtrada = erosion(y_original, estructura)
        elif operacion == "Dilatación":
            y_filtrada = dilatacion(y_original, estructura)
        elif operacion == "Apertura":
            y_filtrada = apertura(y_original, estructura)
        elif operacion == "Cierre":
            y_filtrada = cierre(y_original, estructura)
        elif operacion == "Borde Exterior":
            y_filtrada = borde_exterior(y_original, estructura)
        elif operacion == "Borde Interior":
            y_filtrada = borde_interior(y_original, estructura)
        elif operacion == "Gradiente":
            y_filtrada = gradiente_morfologico(y_original, estructura)

        else:
            return

        imagen_filtrada = Image.fromarray(y_filtrada)
        imagen_filtrada.thumbnail((350, 350))
        imagen_tk = ImageTk.PhotoImage(imagen_filtrada)
        panel_procesada.config(image=imagen_tk)
        panel_procesada.image = imagen_tk

        app.imagen_procesada = y_filtrada / 255.0
        btn_copiar.config(state="normal")

def guardar_imagen():
    if hasattr(app, 'imagen_procesada'):
        ruta = filedialog.asksaveasfilename(defaultextension=".bmp",
                                            filetypes=[("Imagen BMP", "*.bmp"),
                                                       ("PNG", "*.png"),
                                                       ("JPEG", "*.jpg")])
        if ruta:
            imagen_y = (app.imagen_procesada * 255).astype(np.uint8)
            imagen_pil = Image.fromarray(imagen_y, mode='L')
            imagen_pil.save(ruta)

def copiar_a_original():
    if hasattr(app, 'imagen_procesada'):
        app.imagen_y = app.imagen_procesada.copy()
        imagen_mostrar = Image.fromarray((app.imagen_y * 255).astype(np.uint8))
        imagen_mostrar.thumbnail((350, 350))
        imagen_tk = ImageTk.PhotoImage(imagen_mostrar)
        panel_imagen.config(image=imagen_tk)
        panel_imagen.image = imagen_tk
        frame_imagen.config(text="Imagen Original (Modificada)")
        btn_copiar.config(state="disabled")
def apertura(imagen, estructura):
    erosionada = erosion(imagen, estructura)
    return dilatacion(erosionada, estructura)

def cierre(imagen, estructura):
    dilatada = dilatacion(imagen, estructura)
    return erosion(dilatada, estructura)
def borde_exterior(imagen, estructura):
    dilatada = dilatacion(imagen, estructura)
    return dilatada - imagen

def borde_interior(imagen, estructura):
    erosionada = erosion(imagen, estructura)
    return imagen - erosionada

def gradiente_morfologico(imagen, estructura):
    dilatada = dilatacion(imagen, estructura)
    erosionada = erosion(imagen, estructura)
    return dilatada - erosionada

def salir_app():
    app.destroy()

# ============================
# Ventana principal
app = tk.Tk()
app.title("IPDI 2025 - Operaciones Morfologicas")
app.geometry("1000x620")

# Imagen original
frame_imagen = tk.LabelFrame(app, text="Imagen Original (Canal Y)", width=350, height=350)
frame_imagen.pack_propagate(False)
frame_imagen.place(x=50, y=20)
panel_imagen = tk.Label(frame_imagen)
panel_imagen.pack()

btn_cargar = tk.Button(app, text="Cargar Imagen", command=cargar_imagen)
btn_cargar.place(x=150, y=380)

# Imagen procesada
frame_procesada = tk.LabelFrame(app, text="Imagen Procesada (Canal Y)", width=350, height=350)
frame_procesada.pack_propagate(False)
frame_procesada.place(x=600, y=20)
panel_procesada = tk.Label(frame_procesada)
panel_procesada.pack()

btn_guardar = tk.Button(app, text="Guardar Imagen", command=guardar_imagen)
btn_guardar.place(x=700, y=380)

# Controles centrales entre imágenes
frame_central = tk.Frame(app, width=180, height=300)
frame_central.place(x=445, y=100)

combo_operaciones = ttk.Combobox(frame_central, values=[
    "Binarizar",
    "Erosión",
    "Dilatación",
    "Apertura",
    "Cierre",
    "Borde Exterior",
    "Borde Interior",
    "Gradiente"
], width=12)
combo_operaciones.set("Binarizar")
combo_operaciones.pack(pady=5)

btn_procesar = tk.Button(frame_central, text="Filtrar Imagen", command=procesar_imagen)
btn_procesar.pack(pady=5)

btn_copiar = tk.Button(frame_central, text=" <- Copiar ", command=copiar_a_original, state="disabled")
btn_copiar.pack(pady=5)

ttk.Label(frame_central, text="Tamaño Estructura").pack(pady=5)
combo_tamanio = ttk.Combobox(frame_central, values=["3", "5", "7"], width=5)
combo_tamanio.set("3")
combo_tamanio.pack()

ttk.Label(frame_central, text="Forma Estructura").pack(pady=5)
combo_forma = ttk.Combobox(frame_central, values=["Cuadrado", "Cruz", "Disco"], width=10)
combo_forma.set("Cuadrado")
combo_forma.pack()

# Slider y botón salir
label_slider = tk.Label(app, text="Umbral de binarización (0 a 1)")
label_slider.place(x=400, y=460)

slider_umbral = tk.Scale(app, from_=0.0, to=1.0, resolution=0.01,
                         orient="horizontal", length=200)
slider_umbral.set(0.5)
slider_umbral.place(x=400, y=490)

btn_salir = tk.Button(app, text="Salir", command=salir_app,
                      width=15, height=2, font=("Arial", 12, "bold"))
btn_salir.place(x=425, y=540)

app.mainloop()