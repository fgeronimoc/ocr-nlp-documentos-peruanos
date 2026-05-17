"""
preprocessor.py
---------------
Módulo de preprocesamiento de imágenes para OCR.
Usa PIL como base y OpenCV opcionalmente si está disponible.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io

try:
    import cv2
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False


def preprocesar_completo(imagen: np.ndarray) -> np.ndarray:
    """
    Pipeline de preprocesamiento usando PIL (siempre disponible).
    Si OpenCV está instalado, aplica pasos adicionales.
    """
    pil_img = Image.fromarray(imagen).convert("L")  # escala de grises

    # Mejorar contraste
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)

    # Eliminar ruido suave
    pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))

    # Redimensionar x1.5
    w, h = pil_img.size
    pil_img = pil_img.resize((int(w * 1.5), int(h * 1.5)), Image.LANCZOS)

    return np.array(pil_img)


def preprocesar_desde_bytes(imagen_bytes: bytes) -> np.ndarray:
    """
    Recibe bytes de imagen (desde Streamlit file_uploader)
    y retorna imagen preprocesada.
    """
    pil_img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    arr = np.array(pil_img)
    return preprocesar_completo(arr)
