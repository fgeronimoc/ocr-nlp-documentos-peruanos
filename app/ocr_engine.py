"""
ocr_engine.py
-------------
Módulo de extracción de texto usando pytesseract.
Incluye corrección automática de ángulo (deskew) para imágenes torcidas.
"""

import re
import pytesseract
import numpy as np
from PIL import Image
import io


def auto_rotar(pil_img: Image.Image) -> Image.Image:
    """
    Detecta y corrige la orientación de la imagen usando pytesseract OSD.
    Maneja rotaciones de 90, 180 y 270 grados.
    Para inclinaciones pequeñas usa detección por proyección.
    Retorna la imagen corregida (o la original si falla).
    """
    try:
        osd = pytesseract.image_to_osd(pil_img, config="--psm 0 -c min_characters_to_try=5")
        angulo = int(re.search(r"Rotate:\s*(\d+)", osd).group(1))
        if angulo != 0:
            pil_img = pil_img.rotate(-angulo, expand=True, fillcolor="white")
    except Exception:
        pass  # Si OSD falla (imagen muy pequeña o sin texto claro), continúa sin rotar

    return pil_img


def _deskew_fino(pil_img: Image.Image) -> Image.Image:
    """
    Corrige inclinaciones pequeñas (<15°) usando análisis de proyección horizontal.
    Convierte a escala de grises, busca el ángulo que maximiza la varianza de proyecciones.
    """
    try:
        gris = np.array(pil_img.convert("L"))
        # Binarizar
        umbral_bin = 128
        binaria = (gris < umbral_bin).astype(np.float32)

        # Buscar ángulo óptimo entre -15 y +15 grados
        mejor_angulo = 0
        mejor_score = -1
        for angulo in range(-15, 16, 1):
            rotada = pil_img.rotate(angulo, fillcolor="white")
            arr = np.array(rotada.convert("L"))
            binaria_r = (arr < umbral_bin).astype(np.float32)
            proyeccion = binaria_r.sum(axis=1)
            score = proyeccion.var()
            if score > mejor_score:
                mejor_score = score
                mejor_angulo = angulo

        if abs(mejor_angulo) > 1:  # Solo corregir si la inclinación es significativa
            pil_img = pil_img.rotate(mejor_angulo, expand=True, fillcolor="white")
    except Exception:
        pass

    return pil_img


def extraer_texto(imagen: np.ndarray) -> str:
    """
    Extrae texto de una imagen numpy array.
    Aplica corrección automática de ángulo antes del OCR.
    """
    pil_img = Image.fromarray(imagen)
    pil_img = auto_rotar(pil_img)
    texto = pytesseract.image_to_string(
        pil_img,
        lang="spa",
        config="--psm 3"
    )
    return texto.strip()


def _ocr_con_psm(pil_img: Image.Image, psm: int, umbral: float) -> str:
    """Ejecuta OCR con un PSM específico y retorna texto filtrado por confianza."""
    datos = pytesseract.image_to_data(
        pil_img,
        lang="spa",
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT
    )
    palabras = []
    for i, palabra in enumerate(datos["text"]):
        try:
            confianza = int(datos["conf"][i])
        except (ValueError, TypeError):
            continue
        if confianza >= umbral and palabra.strip():
            palabras.append(palabra.strip())
    return " ".join(palabras)


def extraer_texto_con_confianza(imagen: np.ndarray, umbral: float = 30, deskew: bool = False) -> str:
    """
    Extrae texto filtrando por umbral de confianza (0-100).
    Prueba PSM 3, 6 y 11 y elige el que extrae más palabras.
    Aplica corrección de ángulo solo si deskew=True.
    """
    pil_img = Image.fromarray(imagen)
    if deskew:
        pil_img = auto_rotar(pil_img)
        pil_img = _deskew_fino(pil_img)

    # Probar PSM 3 (auto), PSM 6 (bloque uniforme), PSM 11 (texto disperso)
    textos = []
    for psm in [3, 6, 11]:
        try:
            texto = _ocr_con_psm(pil_img, psm, umbral)
            if texto:
                textos.append(texto)
        except Exception:
            continue

    if not textos:
        return ""

    # Elegir el resultado con más palabras
    return max(textos, key=lambda t: len(t.split()))


def extraer_desde_bytes(imagen_bytes: bytes, umbral: float = 30) -> str:
    """
    Extrae texto directamente desde bytes de imagen
    (compatible con st.file_uploader de Streamlit).
    """
    pil_img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    arr = np.array(pil_img)
    return extraer_texto_con_confianza(arr, umbral=umbral)
