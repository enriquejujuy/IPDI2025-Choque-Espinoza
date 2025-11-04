import numpy as np

# Filtros pasabajos
def filtro_plano(n):
    return np.ones((n, n), dtype=np.float32) / (n * n)

def filtro_bartlett(n):
    centro = n // 2
    bartlett_1d = np.array([n - abs(i - centro) for i in range(n)], dtype=np.float32)
    kernel = np.outer(bartlett_1d, bartlett_1d)
    return kernel / kernel.sum()

def filtro_gaussiano(n, sigma=1.0):
    ax = np.linspace(-(n // 2), n // 2, n)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()

# Filtros Laplaciano
def filtro_laplaciano_v4(imagen):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float32)
    return aplicar_filtro(imagen, kernel)

def filtro_laplaciano_v8(imagen):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return aplicar_filtro(imagen, kernel)

# Filtros Sobel
sobel_kernels = {
    "N":  np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]], dtype=np.float32),
    "S":  np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float32),
    "E":  np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]], dtype=np.float32),
    "O":  np.array([[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]], dtype=np.float32),
    "NE": np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]], dtype=np.float32),
    "NO": np.array([[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]], dtype=np.float32),
    "SE": np.array([[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]], dtype=np.float32),
    "SO": np.array([[-2, -1,  0], [-1,  0,  1], [ 0,  1,  2]], dtype=np.float32)
}

def filtro_sobel(imagen, direccion):
    kernel = sobel_kernels.get(direccion)
    if kernel is not None:
        return aplicar_filtro(imagen, kernel)
    return imagen

# Filtro pasaaltos en frecuencia
def filtro_pasaaltos(imagen, f_corte):
    f = np.fft.fft2(imagen)
    fshift = np.fft.fftshift(f)
    filas, columnas = imagen.shape
    crow, ccol = filas // 2, columnas // 2
    mask = np.zeros((filas, columnas), dtype=np.float32)

    for u in range(filas):
        for v in range(columnas):
            d = np.sqrt((u - crow)**2 + (v - ccol)**2)
            d_norm = d / np.sqrt(crow**2 + ccol**2)
            if d_norm >= f_corte:
                mask[u, v] = 1

    fshift_filtrado = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back).astype(np.uint8)

# Convolución manual
def aplicar_filtro(imagen, kernel):
    filas, columnas = imagen.shape
    k_size = kernel.shape[0]
    offset = k_size // 2
    salida = np.zeros_like(imagen, dtype=np.float32)

    for i in range(offset, filas - offset):
        for j in range(offset, columnas - offset):
            region = imagen[i - offset:i + offset + 1, j - offset:j + offset + 1]
            salida[i, j] = np.sum(region * kernel)

    return np.clip(salida, 0, 255).astype(np.uint8)