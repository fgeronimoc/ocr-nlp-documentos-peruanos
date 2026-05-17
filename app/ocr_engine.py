"""
ocr_engine.py
-------------
Módulo de extracción de texto usando EasyOCR.
Soporta imágenes JPG, PNG y páginas de PDF.
"""

import easyocr
import numpy as np
from PIL import Image
import io


# Inicializar lector una sola vez (es costoso en memoria)
# Idiomas: español + inglés (para capturar términos técnicos)
_reader = None


def obtener_lector():
    """Carga el lector EasyOCR (singleton para no recargar en cada uso)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["es", "en"], gpu=False)
    return _reader


def extraer_texto(imagen: np.ndarray, detalle: bool = False) -> str:
    """
    Extrae texto de una imagen numpy array ya preprocesada.

    Parámetros:
        imagen  : array numpy (gris o RGB)
        detalle : si True, retorna lista con (bbox, texto, confianza)

    Retorna:
        str con el texto completo extraído
    """
    reader = obtener_lector()
    resultados = reader.readtext(imagen)

    if detalle:
        return resultados

    # Unir solo el texto detectado
    texto = " ".join([res[1] for res in resultados])
    return texto


def extraer_texto_con_confianza(imagen: np.ndarray, umbral: float = 0.3) -> str:
    """
    Extrae texto filtrando por un umbral mínimo de confianza.

    Parámetros:
        imagen  : array numpy
        umbral  : confianza mínima (0 a 1), default 0.3

    Retorna:
        str con el texto de alta confianza
    """
    reader = obtener_lector()
    resultados = reader.readtext(imagen)

    texto_filtrado = [
        res[1]
        for res in resultados
        if res[2] >= umbral
    ]

    return " ".join(texto_filtrado)


def extraer_desde_bytes(imagen_bytes: bytes) -> str:
    """
    Extrae texto directamente desde bytes de imagen
    (compatible con st.file_uploader de Streamlit).
    """
    pil_img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    arr = np.array(pil_img)
    return extraer_texto(arr)


def extraer_desde_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extrae texto de un PDF (página por página).
    Requiere pdf2image y poppler instalados.
    """
    from pdf2image import convert_from_bytes

    paginas = convert_from_bytes(pdf_bytes)
    texto_total = []

    for i, pagina in enumerate(paginas):
        arr = np.array(pagina.convert("RGB"))
        texto_pagina = extraer_texto(arr)
        texto_total.append(f"--- Página {i+1} ---\n{texto_pagina}")

    return "\n\n".join(texto_total)
