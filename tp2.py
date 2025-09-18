import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

# Conversión entre RGB y YIQ
def rgb_to_yiq(img):
    arr = np.asarray(img).astype(np.float32) / 255
    matrix = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.274, -0.322],
                       [0.211, -0.523, 0.312]])
    yiq = arr @ matrix.T
    return yiq

def yiq_to_rgb(yiq):
    matrix = np.array([[1.0, 0.956, 0.621],
                       [1.0, -0.272, -0.647],
                       [1.0, -1.106, 1.703]])
    rgb = yiq @ matrix.T
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)

# Operaciones YIQ
def yiq_sum(img1, img2):
    Y1, I1, Q1 = img1[:,:,0], img1[:,:,1], img1[:,:,2]
    Y2, I2, Q2 = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    Y = Y1 + Y2
    I = (Y1 * I1 + Y2 * I2) / (Y1 + Y2 + 1e-5)
    Q = (Y1 * Q1 + Y2 * Q2) / (Y1 + Y2 + 1e-5)
    return np.stack([Y, I, Q], axis=-1)

def yiq_sub(img1, img2):
    return img1 - img2

def yiq_div(img1, img2):
    return img1 / (img2 + 1e-5)

def yiq_abs_sub(img1, img2):
    return np.abs(img1 - img2)

def yiq_if_lighter(img1, img2):
    return np.where(img1[:,:,0:1] > img2[:,:,0:1], img1, img2)

def yiq_if_darker(img1, img2):
    return np.where(img1[:,:,0:1] < img2[:,:,0:1], img1, img2)

# Operaciones RGB
def rgb_sum(img1, img2):
    return img1 + img2

def rgb_sub(img1, img2):
    return img1 - img2

# Procesamiento principal
def process_images():
    if img1_arr is None or img2_arr is None:
        return
    op = operation_var.get()
    mode = mode_var.get()

    img1_resized = img1_arr.resize((300, 300))
    img2_resized = img2_arr.resize((300, 300))
    arr1 = np.array(img1_resized).astype(np.float32)
    arr2 = np.array(img2_resized).astype(np.float32)

    if "YIQ" in op:
        yiq1 = rgb_to_yiq(arr1)
        yiq2 = rgb_to_yiq(arr2)

        if op == "suma en YIQ":
            result = yiq_sum(yiq1, yiq2)
        elif op == "resta en YIQ":
            result = yiq_sub(yiq1, yiq2)
        elif op == "cociente en YIQ":
            result = yiq_div(yiq1, yiq2)
        elif op == "resta absoluta en YIQ":
            result = yiq_abs_sub(yiq1, yiq2)
        elif op == "if ligther en YIQ":
            result = yiq_if_lighter(yiq1, yiq2)
        elif op == "if darker en YIQ":
            result = yiq_if_darker(yiq1, yiq2)

        if "if" not in op:
            if mode == "Clampeada":
                result = np.clip(result, -1, 1)
            elif mode == "Promediada":
                Y_avg = result[:, :, 0].mean()
                I_avg = result[:, :, 1].mean()
                Q_avg = result[:, :, 2].mean()
                result[:, :, 0] = (result[:, :, 0] + Y_avg) / 2
                result[:, :, 1] = (result[:, :, 1] + I_avg) / 2
                result[:, :, 2] = (result[:, :, 2] + Q_avg) / 2

        result_rgb = yiq_to_rgb(result)

    else:
        if op == "suma en RGB":
            result_rgb = rgb_sum(arr1, arr2)
        elif op == "resta en RGB":
            result_rgb = rgb_sub(arr1, arr2)

        if mode == "Clampeada":
            result_rgb = np.clip(result_rgb, 0, 255)
        elif mode == "Promediada":
            R_avg = result_rgb[:, :, 0].mean()
            G_avg = result_rgb[:, :, 1].mean()
            B_avg = result_rgb[:, :, 2].mean()
            result_rgb[:, :, 0] = (result_rgb[:, :, 0] + R_avg) / 2
            result_rgb[:, :, 1] = (result_rgb[:, :, 1] + G_avg) / 2
            result_rgb[:, :, 2] = (result_rgb[:, :, 2] + B_avg) / 2

        result_rgb = result_rgb.astype(np.uint8)

    show_result(Image.fromarray(result_rgb), op)

def show_result(img, label):
    global result_photo
    result_photo = ImageTk.PhotoImage(img)
    result_label.config(image=result_photo)
    result_text.config(text=f"Resultado: {label}")

def load_image1():
    global img1_arr, img1_photo
    path = filedialog.askopenfilename()
    if path:
        img1_arr = Image.open(path).convert("RGB")
        img1_arr.thumbnail((300, 300))
        img1_photo = ImageTk.PhotoImage(img1_arr)
        img1_label.config(image=img1_photo)

def load_image2():
    global img2_arr, img2_photo
    path = filedialog.askopenfilename()
    if path:
        img2_arr = Image.open(path).convert("RGB")
        img2_arr.thumbnail((300, 300))
        img2_photo = ImageTk.PhotoImage(img2_arr)
        img2_label.config(image=img2_photo)

def save_result():
    if result_photo:
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            result_photo._PhotoImage__photo.write(path)

# Interfaz
root = tk.Tk()
root.title("Procesador de Imágenes IPDI 2025")
root.geometry("1000x600")
root.resizable(False, False)

img1_arr = None
img2_arr = None
img1_photo = None
img2_photo = None
result_photo = None

# Frame superior para imágenes
image_frame = tk.Frame(root)
image_frame.pack(pady=20)

img1_label = tk.Label(image_frame, text="Imagen 1")
img1_label.grid(row=0, column=0, padx=20)

img2_label = tk.Label(image_frame, text="Imagen 2")
img2_label.grid(row=0, column=1, padx=20)

result_label = tk.Label(image_frame, text="Resultado")
result_label.grid(row=0, column=2, padx=20)

# Frame medio para opciones
options_frame = tk.Frame(root)
options_frame.pack(pady=10)

tk.Button(options_frame, text="Cargar Imagen 1", command=load_image1).grid(row=0, column=0, padx=10)
tk.Button(options_frame, text="Cargar Imagen 2", command=load_image2).grid(row=0, column=1, padx=10)

operation_var = tk.StringVar()
operation_menu = ttk.Combobox(options_frame, textvariable=operation_var, state="readonly", width=25)
operation_menu['values'] = [
    "suma en RGB", "resta en RGB",
    "suma en YIQ", "resta en YIQ", "cociente en YIQ",
    "resta absoluta en YIQ", "if ligther en YIQ", "if darker en YIQ"
]
operation_menu.grid(row=0, column=2, padx=10)
operation_menu.current(0)

mode_var = tk.StringVar()
mode_menu = ttk.Combobox(options_frame, textvariable=mode_var, state="readonly", width=15)
mode_menu['values'] = ["Clampeada", "Promediada"]
mode_menu.grid(row=0, column=3, padx=10)
mode_menu.current(0)

tk.Button(options_frame, text="Procesar", command=process_images).grid(row=0, column=4, padx=10)

# Frame inferior para resultado y acciones
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=20)

result_text = tk.Label(bottom_frame, text="Resultado:")
result_text.pack()

# Botones de acción: Guardar y Salir
action_frame = tk.Frame(root)
action_frame.pack(pady=10)

tk.Button(action_frame, text="Guardar", command=save_result, width=15).pack(side=tk.LEFT, padx=20)
tk.Button(action_frame, text="Salir", command=root.quit, width=15).pack(side=tk.RIGHT, padx=20)

# Ejecutar la interfaz
root.mainloop()