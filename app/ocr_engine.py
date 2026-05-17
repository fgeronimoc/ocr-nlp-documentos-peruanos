"""
ocr_engine.py
-------------
Módulo de extracción de texto usando pytesseract.
Más liviano que EasyOCR, ideal para despliegue en la nube.
"""

import pytesseract
import numpy as np
from PIL import Image
import io


def extraer_texto(imagen: np.ndarray) -> str:
    """
    Extrae texto de una imagen numpy array.
    Usa configuración optimizada para español.
    """
    pil_img = Image.fromarray(imagen)
    texto = pytesseract.image_to_string(
        pil_img,
        lang="spa",
        config="--psm 3"   # detección automática de layout
    )
    return texto.strip()


def extraer_texto_con_confianza(imagen: np.ndarray, umbral: float = 30) -> str:
    """
    Extrae texto filtrando por umbral de confianza (0-100).
    Retorna solo el texto con confianza >= umbral.
    """
    pil_img = Image.fromarray(imagen)
    datos = pytesseract.image_to_data(
        pil_img,
        lang="spa",
        config="--psm 3",
        output_type=pytesseract.Output.DICT
    )

    texto_filtrado = []
    for i, palabra in enumerate(datos["text"]):
        try:
            confianza = int(datos["conf"][i])
        except (ValueError, TypeError):
            continue
        if confianza >= umbral and palabra.strip():
            texto_filtrado.append(palabra.strip())

    return " ".join(texto_filtrado)


def extraer_desde_bytes(imagen_bytes: bytes, umbral: float = 30) -> str:
    """
    Extrae texto directamente desde bytes de imagen
    (compatible con st.file_uploader de Streamlit).
    """
    pil_img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    arr = np.array(pil_img)
    return extraer_texto_con_confianza(arr, umbral=umbral)
