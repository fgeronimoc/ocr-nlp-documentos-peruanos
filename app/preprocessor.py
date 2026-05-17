"""
preprocessor.py
---------------
Módulo de preprocesamiento de imágenes para OCR.
Aplica técnicas de limpieza para mejorar la calidad
del texto extraído.
"""

import cv2
import numpy as np
from PIL import Image


def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga una imagen desde ruta y la convierte a array numpy."""
    img = Image.open(ruta).convert("RGB")
    return np.array(img)


def convertir_gris(imagen: np.ndarray) -> np.ndarray:
    """Convierte imagen a escala de grises."""
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)


def redimensionar(imagen: np.ndarray, escala: float = 1.5) -> np.ndarray:
    """Redimensiona la imagen para mejorar resolución del OCR."""
    alto, ancho = imagen.shape[:2]
    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)
    return cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_CUBIC)


def eliminar_ruido(imagen: np.ndarray) -> np.ndarray:
    """Elimina ruido con filtro gaussiano."""
    return cv2.GaussianBlur(imagen, (3, 3), 0)


def binarizar(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica umbralización adaptativa (binarización).
    Convierte la imagen a blanco y negro para mejorar
    el contraste del texto.
    """
    return cv2.adaptiveThreshold(
        imagen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )


def mejorar_contraste(imagen: np.ndarray) -> np.ndarray:
    """Mejora el contraste usando ecualización de histograma."""
    return cv2.equalizeHist(imagen)


def preprocesar_completo(imagen: np.ndarray) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento:
    1. Redimensionar (escala x1.5)
    2. Convertir a gris
    3. Eliminar ruido
    4. Mejorar contraste
    5. Binarizar

    Retorna imagen lista para OCR como array numpy.
    """
    img = redimensionar(imagen)
    img = convertir_gris(img)
    img = eliminar_ruido(img)
    img = mejorar_contraste(img)
    img = binarizar(img)
    return img


def preprocesar_desde_bytes(imagen_bytes: bytes) -> np.ndarray:
    """
    Recibe bytes de imagen (desde Streamlit file_uploader)
    y retorna imagen preprocesada.
    """
    import io
    pil_img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    arr = np.array(pil_img)
    return preprocesar_completo(arr)
