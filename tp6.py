import tkinter
from tkinter import HORIZONTAL, VERTICAL, Scale, filedialog, StringVar, OptionMenu
from tkinter import simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import imageio.v2 as imageio
import numpy as np
import math
from skimage.morphology import skeletonize
from skimage import data, measure
from skimage.util import invert

# Variables globales
imagen = None
imagenProcesada = None
canvasOriginal = None
canvasProcesado = None
canvasHistogramaOriginal = None
canvasHistogramaProcesado = None

# Metodos
def cargarImagen(canvas, frame):
    global imagen
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    
    if ruta:
        etiqueta["text"] = "Imagen importada correctamente!"
        
        imagenOriginal = imageio.imread(ruta)
        imagenOriginal = np.clip(imagenOriginal / 255., 0., 1.)
        
        if len(imagenOriginal.shape) == 3 and imagenOriginal.shape[2] == 3:
            imagen = rgb2Gray(imagenOriginal)            
        else:
            imagen = imagenOriginal 
        
        mostrarImagen(imagen, canvas, frame)
    else:
        etiqueta["text"] = "No se ha cargado ninguna imagen."
        
        mostrarImagen(imagen, canvas, frame)

def mostrarImagen(imagen, canvas, frame):
    global canvasOriginal, canvasHistogramaOriginal

    # Limpiar canvas anteriores si existen
    if canvas:
        canvas.get_tk_widget().destroy()
    if canvasHistogramaOriginal:
        canvasHistogramaOriginal.get_tk_widget().destroy()

    # Mostrar la imagen en el canvas
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(imagen, cmap="gray")
    ax.axis('off')

    canvasOriginal = FigureCanvasTkAgg(fig, master=frame)
    canvasOriginal.get_tk_widget().pack(expand=True)
    canvasOriginal.draw()

    # Mostrar el histograma en el canvas correspondiente
    canvasHistogramaOriginal = mostrarHistograma(imagen, histogramaA)

def copiarImagen(imagenProcesada):
    global canvasOriginal, frameImagenOriginal, imagen
    
    if imagenProcesada is not None:
        imagen = imagenProcesada
        
        mostrarImagen(imagen, canvasOriginal, frameImagenOriginal)
    else:
        etiqueta["text"] = "No hay ninguna imagen procesada para copiar!"

        
def guardarImagen(imagenProcesada):
    if imagenProcesada is not None:
        ruta_guardado = filedialog.asksaveasfilename(
            defaultextension=".png", 
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("GIF", "*.gif")],
            title="Guardar imagen procesada"
        )
        
        if ruta_guardado:
            imagen_guardar = np.clip(imagenProcesada * 255, 0, 255).astype(np.uint8)
            
            imageio.imwrite(ruta_guardado, imagen_guardar)
            etiqueta["text"] = f"Imagen procesada guardada en {ruta_guardado}!"
        else:
            etiqueta["text"] = "No se ha guardado la imagen procesada."
    else:
        etiqueta["text"] = "No hay ninguna imagen procesada para guardar."

def procesarImagen(frame):
    global canvasProcesado, imagenProcesada, imagen

    if imagen is not None:  # Asegurarse de que la imagen ha sido cargada
        # Realizar la operación según selección
        if tipoOperacion.get() == "Binarizar 50-50":
            fig = binarizar(imagen)
        elif tipoOperacion.get() == "Binarizar Moda":
            fig = binarizarModa(imagen)
        elif tipoOperacion.get() == "Binarizar Otsu":
            fig = binarizarOtsu(imagen)
        elif tipoOperacion.get() == "Borde Laplaciano":
            fig = bordeLaplaciano(imagen)
        elif tipoOperacion.get() == "Borde Morfologico":
            fig = bordeMorfologico(imagen)
        elif tipoOperacion.get() == "Marching Squares":
            fig = marchingSquares(imagen)
        else:
            etiqueta["text"] = "No es posible realizar esa selección!"
            return

        # Limpiar el canvas previo
        if canvasProcesado:
            canvasProcesado.get_tk_widget().destroy()

        # Mostrar el resultado en el canvas
        canvasProcesado = FigureCanvasTkAgg(fig, master=frame)
        canvasProcesado.get_tk_widget().pack(expand=True)
        canvasProcesado.draw()

    else:
        etiqueta["text"] = "Primero debes cargar las imágenes!"
                
def rgb2Gray(imagen):
    return np.dot(imagen[..., :3], [0.2989, 0.5870, 0.1140])

def mostrarHistograma(imagen, frame):
    hist, bins = np.histogram(imagen.flatten(), bins=10, range=(0, 1))
    
    hist_normalized = hist / np.sum(hist)

    fig, ax = plt.subplots(figsize=(4, 4))
    hist, bins = np.histogram(imagen.flatten(), bins=10, range=(0, 1))
    hist_normalized = hist / np.sum(hist)
    
    ax.bar(bins[:-1], hist_normalized, width=(bins[1] - bins[0]), edgecolor='black')
    ax.set_title('Histograma')
    ax.set_xlabel('Luminancia')
    ax.set_ylabel('Frecuencia (%)')
    
    canvasHistograma = FigureCanvasTkAgg(fig, master=frame)
    canvasHistograma.get_tk_widget().pack(expand=True)
    canvasHistograma.draw()
    
    return canvasHistograma

# Funciones de procesamiento de imágenes
def binarizar(imagen):
    global imagenProcesada
    
    umbral = np.median(imagen)
    imBin = np.where(imagen >= umbral, 1, 0)
    
    imagenProcesada = imBin
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(imBin, cmap="gray")
    ax.axis('off')
    
    return fig

def binarizarModa(imagen):
    global imagenProcesada
    
    histograma, bins = np.histogram(imagen.flatten(), bins=256, range=[0, 1])

    moda_oscura = np.argmax(histograma[:128])
    moda_clara = np.argmax(histograma[128:]) + 128

    umbral_moda = (moda_oscura + moda_clara) / 2 / 255  # Normalizar entre 0 y 1

    binarizada_modas = np.zeros(imagen.shape)
    binarizada_modas = np.where(imagen >= umbral_moda, 1, 0)
    
    imagenProcesada = binarizada_modas
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(binarizada_modas, cmap="gray")
    ax.axis('off')
    
    return fig

def binarizarOtsu(imagen):
    global imagenProcesada
    
    hist, bins = np.histogram(imagen, bins=100, range=(0, 1))
    pixel_numb = imagen.shape[0] * imagen.shape[1]
    prom_pond = 1 / pixel_numb
    final_thresh, final_value = -1, -1
    intensity = np.linspace(0, 1, 100)

    for x in range(1, 100):
        pcb = np.sum(hist[:x])
        pcf = np.sum(hist[x:])
        wb, wf = pcb * prom_pond, pcf * prom_pond
        mub = np.sum(intensity[:x] * hist[:x]) / pcb if pcb > 0 else 0
        muf = np.sum(intensity[x:] * hist[x:]) / pcf if pcf > 0 else 0
        value = wb * wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh, final_value = x / 100, value

    binarizada_otsu = np.where(imagen > final_thresh, 1, 0)
    
    imagenProcesada = binarizada_otsu

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(binarizada_otsu, cmap="gray")
    ax.axis('off')
    
    return fig
    
def convolucionLaplaciana(imagen, kernel):

    conv = np.zeros((imagen.shape[0] - kernel.shape[0] + 1, imagen.shape[1] - kernel.shape[1] + 1))

    for y in range(conv.shape[0]):
        for x in range(conv.shape[1]):
            # Suma de productos entre la región de la imagen y el kernel
            valor = np.sum(imagen[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel)
            # Coerción de los valores: si es menor que 0, se ajusta a 0; si es mayor que 1, se ajusta a 1
            conv[y, x] = np.clip(valor, 0, 1)
    return conv 

def bordeLaplaciano(imagen_binaria):
    global imagenProcesada
    
    laplaceV8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolucion con kernel laplaciano V8
    imagen_convolucionada_v8 = convolucionLaplaciana(imagen_binaria, laplaceV8)

    imagenProcesada = imagen_convolucionada_v8

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(imagen_convolucionada_v8, cmap="gray")
    ax.axis('off')
    
    return fig

def erosion(imagen_binaria):
  elemento_estructurante = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])

  filas, columnas = imagen_binaria.shape
  filas_est, columnas_est = elemento_estructurante.shape
  imagen_erosionada = np.zeros((filas, columnas))

  for i in range(filas_est // 2, filas - filas_est // 2):
    for j in range(columnas_est // 2, columnas - columnas_est // 2):
      vecindario = imagen_binaria[i - filas_est // 2: i + filas_est // 2 + 1,
                                j - columnas_est // 2: j + columnas_est // 2 + 1]
      if np.array_equal(vecindario * elemento_estructurante, elemento_estructurante):
        imagen_erosionada[i, j] = 1

  return imagen_erosionada

def bordeMorfologico(imagen_binaria):
    global imagenProcesada
        
    imagen_erosionada = erosion(imagen_binaria)
    imagen_borde = imagen_binaria - imagen_erosionada
    
    imagenProcesada = imagen_borde

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(imagen_borde, cmap="gray")
    ax.axis('off')
    
    return fig

def marchingSquares(imagen_binaria):
    global imagenProcesada
    
    # Encuentra los contornos en la imagen binaria
    contornos = measure.find_contours(imagen_binaria, level=0.5)
    imagenProcesada = imagen_binaria  # Guardamos la imagen binaria como procesada

    # Crear figura para mostrar los contornos
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(imagen_binaria, cmap="gray")  # Muestra la imagen binaria
    for contorno in contornos:
        # Graficar cada contorno con un color y grosor
        ax.plot(contorno[:, 1], contorno[:, 0], linewidth=2, color='red')
    ax.axis('off')
    
    return fig
    
# Crear la ventana principal
ventana = tkinter.Tk()
ventana.resizable(False, False)

# Etiqueta para mostrar mensajes de estado
etiqueta = tkinter.Label(ventana, text="")
etiqueta.grid(padx=5, pady=5, row=0, columnspan=4)

# Crear un marco para la imagen A
tituloA = tkinter.Label(ventana, text="Imagen Original")
frameImagenOriginal = tkinter.Frame(ventana, width=250, height=250, bg="white")
histogramaA = tkinter.Frame(ventana, width=200, height=200, bg="white")
botonCargarA = tkinter.Button(ventana, text="Cargar Imagen", command=lambda:cargarImagen(canvasOriginal, frameImagenOriginal))

tituloA.grid(padx=5, pady=5, row=1, column=0)
frameImagenOriginal.grid(padx=10, pady=10, row=2, column=0)
histogramaA.grid(padx=10, pady=10, row=2, column=1)
botonCargarA.grid(pady=5, row=3, column=0)

# Crear un marco para los Procesar y Copiar
tituloB = tkinter.Label(ventana, text="Controles")
controlesVariables = tkinter.Frame(ventana, width=250, height=250, bg="white")

botonProcesar = tkinter.Button(controlesVariables, text="Procesar Imagen -->", command=lambda:procesarImagen(frameProcesada))
botonCopiar = tkinter.Button(controlesVariables, text="<-- Copiar Imagen", command=lambda:copiarImagen(imagenProcesada))
slider = Scale(controlesVariables, from_=0, to=1, orient=HORIZONTAL, label="Umbral", resolution=0.1)
slider.set(0)

tituloB.grid(padx=5, pady=5, row=1, column=2)
controlesVariables.grid(padx=10, pady=10, row=2, column=2)

botonProcesar.pack(pady=5)
botonCopiar.pack(pady=5)
slider.pack(side="top", padx=20, pady=10)

# Crear un marco para la imagen Procesada
tituloProcesado = tkinter.Label(ventana, text="Imagen Procesada")
frameProcesada = tkinter.Frame(ventana, width=200, height=200, bg="white")
botonGuardar = tkinter.Button(ventana, text="Guardar Imagen", command=lambda:guardarImagen(imagenProcesada))

tituloProcesado.grid(padx=5, pady=5, row=1, column=3)
frameProcesada.grid(padx=1, pady=1, row=2, column=3)
botonGuardar.grid(pady=5, row=3, column=3)

# Crear un marco para los desplegables
frameDesplegables = tkinter.Frame(ventana)
frameDesplegables.grid(padx=5, pady=5, row=4, columnspan=4)

# Menú desplegable para seleccionar el tipo de operación
tipoOperacion = StringVar(value="Binarizar 50-50")
opcionesOperacion = ["Binarizar Moda",
                     "Binarizar Otsu",
                     "Borde Laplaciano",
                     "Borde Morfologico",
                     "Marching Squares"]
menuOperacion = OptionMenu(frameDesplegables, tipoOperacion, *opcionesOperacion)
menuOperacion.grid(padx=5, pady=5, row=0, column=0)

ventana.mainloop()
