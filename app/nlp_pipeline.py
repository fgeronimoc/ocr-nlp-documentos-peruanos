"""
nlp_pipeline.py
---------------
Módulo NLP modular para análisis de texto extraído por OCR.
Cada tipo de documento tiene su propia función de análisis.
Para agregar un nuevo tipo: añadir función analizar_<tipo>()
y registrarla en PIPELINES.
"""

import re
import unicodedata
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64

try:
    from wordcloud import WordCloud
    WORDCLOUD_DISPONIBLE = True
except ImportError:
    WORDCLOUD_DISPONIBLE = False

# --- Descargar solo stopwords (no requiere punkt) ---
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS_ES = set(stopwords.words("spanish"))


# ============================================================
# LIMPIEZA GENERAL DE TEXTO
# ============================================================

def limpiar_texto(texto: str) -> str:
    """
    Limpieza básica NLP:
    - Convertir a minúsculas
    - Eliminar caracteres especiales
    - Normalizar espacios
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-zA-Z0-9\s.,:/%-]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto: str, eliminar_stopwords: bool = True) -> list:
    """
    Tokeniza usando regex — sin dependencias de archivos NLTK externos.
    Captura palabras con letras (incluyendo ñ y vocales acentuadas).
    """
    tokens = re.findall(r"\b[a-záéíóúüñ]{2,}\b", texto.lower())
    if eliminar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES]
    return tokens


def palabras_frecuentes(texto: str, n: int = 10) -> list:
    """Retorna las n palabras más frecuentes (sin stopwords)."""
    tokens = tokenizar(texto)
    return Counter(tokens).most_common(n)


# ============================================================
# ANÁLISIS POR TIPO DE DOCUMENTO
# ============================================================

def analizar_boleta(texto: str) -> dict:
    """
    Análisis NLP específico para boletas peruanas.
    Extrae: RUC, fecha, montos (S/), empresa emisora,
    palabras clave y nube de palabras.
    """
    resultado = {
        "tipo": "Boleta / Factura",
        "entidades": {},
        "palabras_frecuentes": [],
        "resumen": "",
        "wordcloud_base64": None
    }

    # --- Extraer RUC (11 dígitos, tolerante a espacios por OCR) ---
    # Busca secuencias de 11 dígitos, con o sin prefijo RUC/R.U.C.
    ruc_raw = re.findall(r"(?:r\.?u\.?c\.?[\s:]*)?(\d[\s]{0,2}\d[\s]{0,2}\d[\s]{0,2}\d[\s]{0,2}\d[\s]{0,2}\d[\s]{0,2}\d[\s]{0,2}\d[\s]{0,2}\d[\s]{0,2}\d[\s]{0,2}\d)", texto, re.IGNORECASE)
    ruc = [re.sub(r"\s", "", r) for r in ruc_raw if len(re.sub(r"\s", "", r)) == 11]
    # También buscar 11 dígitos directos
    ruc += re.findall(r"\b\d{11}\b", texto)
    ruc = list(set(ruc))
    resultado["entidades"]["RUC"] = ruc if ruc else ["No detectado"]

    # --- Extraer montos con contexto (IGV, subtotal, total) ---
    PATRON_MONTO = r"\d{1,3}(?:[.,]\d{3})*[.,]\d{2}"

    def extraer_monto_contextual(patron_contexto, texto, ventana=80):
        """Busca un monto numérico dentro de 'ventana' caracteres después de la palabra clave."""
        match_kw = re.search(patron_contexto, texto, re.IGNORECASE)
        if match_kw:
            fragmento = texto[match_kw.start(): match_kw.start() + ventana]
            match_m = re.search(PATRON_MONTO, fragmento)
            if match_m:
                return match_m.group(0)
        return None

    # Subtotal / Gravada (precio sin IGV)
    subtotal = extraer_monto_contextual(r"gravad[ao]|subtotal|valor\s*venta", texto)
    # IGV
    igv = extraer_monto_contextual(r"i\.?\s*g\.?\s*v\.?|igv|impuesto\s*gral", texto)
    # Total — también busca al revés: monto seguido de la palabra total
    total = extraer_monto_contextual(r"total(?!\s*gravad|\s*venta)", texto)
    if not total:
        # Buscar patrón: monto seguido de "total" (algunos formatos invierten el orden)
        match_inv = re.search(PATRON_MONTO + r"\s{0,10}(?:soles|sol)?\s{0,5}(?=total)", texto, re.IGNORECASE)
        if match_inv:
            total = match_inv.group(0)

    # Si no encuentra con contexto, busca todos los montos como respaldo
    montos_genericos = re.findall(
        r"(?:s[\s/.]?\s?)?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}",
        texto, re.IGNORECASE
    )

    resultado["entidades"]["Subtotal (sin IGV)"] = subtotal if subtotal else "No detectado"
    resultado["entidades"]["IGV (18%)"] = igv if igv else "No detectado"
    resultado["entidades"]["Total"] = total if total else "No detectado"
    resultado["entidades"]["Todos los montos"] = montos_genericos if montos_genericos else ["No detectado"]

    # --- Extraer fechas (múltiples formatos) ---
    fechas = re.findall(
        r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"
        r"|\d{1,2}\s+de\s+\w+\s+de\s+\d{4}"
        r"|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b",
        texto, re.IGNORECASE
    )
    resultado["entidades"]["Fechas"] = fechas if fechas else ["No detectada"]

    # --- Extraer serie de comprobante ---
    # Formatos: B001-000567, F003-000282, 001-000567, N° 001-000567
    serie = re.findall(
        r"\b(?:[BFbf]\d{3}|\d{3})\s*[-–]\s*\d{4,}\b"
        r"|[Nn][°º\.]\s*\d{3}\s*[-–]\s*\d+",
        texto
    )
    resultado["entidades"]["Serie/Número"] = serie if serie else ["No detectada"]

    # --- Palabras frecuentes ---
    resultado["palabras_frecuentes"] = palabras_frecuentes(texto, n=10)

    # --- Resumen simple ---
    lineas = [l.strip() for l in texto.split("\n") if len(l.strip()) > 5]
    resultado["resumen"] = " | ".join(lineas[:5]) if lineas else "Sin contenido suficiente"

    # --- Nube de palabras ---
    resultado["wordcloud_base64"] = generar_wordcloud(texto)

    return resultado


def analizar_noticia(texto: str) -> dict:
    """
    Análisis NLP para noticias o artículos periodísticos.
    Aplica: resumen extractivo, palabras clave, nube de palabras.
    """
    resultado = {
        "tipo": "Noticia / Artículo",
        "entidades": {},
        "palabras_frecuentes": [],
        "resumen": "",
        "wordcloud_base64": None
    }

    # --- Resumen extractivo (top 3 oraciones por longitud y palabras clave) ---
    oraciones = sent_tokenize(texto)
    palabras_importantes = ["perú", "lima", "gobierno", "presidente", "ministerio",
                             "economía", "salud", "educación", "proyecto"]
    puntajes = []
    for oracion in oraciones:
        score = len(oracion.split())
        for palabra in palabras_importantes:
            if palabra in oracion.lower():
                score += 5
        puntajes.append((score, oracion))

    mejores = sorted(puntajes, key=lambda x: x[0], reverse=True)[:3]
    resultado["resumen"] = ". ".join([o for _, o in mejores]) + "."

    # --- Palabras frecuentes ---
    resultado["palabras_frecuentes"] = palabras_frecuentes(texto, n=10)

    # --- Nube de palabras ---
    resultado["wordcloud_base64"] = generar_wordcloud(texto)

    return resultado


def analizar_generico(texto: str) -> dict:
    """
    Análisis genérico para cualquier documento no categorizado.
    Aplica: palabras frecuentes y nube de palabras.
    """
    return {
        "tipo": "Documento genérico",
        "entidades": {},
        "palabras_frecuentes": palabras_frecuentes(texto, n=10),
        "resumen": texto[:300] + "..." if len(texto) > 300 else texto,
        "wordcloud_base64": generar_wordcloud(texto)
    }


# ============================================================
# NUBE DE PALABRAS
# ============================================================

def generar_wordcloud(texto: str):
    """
    Genera una nube de palabras y la retorna como
    imagen base64 para mostrar en Streamlit.
    Retorna None si wordcloud no está instalado.
    """
    if not WORDCLOUD_DISPONIBLE:
        return None

    tokens = tokenizar(texto)
    if not tokens:
        return None

    texto_limpio = " ".join(tokens)

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Blues",
        max_words=80,
        collocations=False
    ).generate(texto_limpio)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_b64


# ============================================================
# REGISTRO DE PIPELINES (agregar nuevos tipos aquí)
# ============================================================

PIPELINES = {
    "Boleta / Factura": analizar_boleta,
    "Noticia / Artículo": analizar_noticia,
    "Otro documento": analizar_generico,
}


def analizar(texto: str, tipo_documento: str) -> dict:
    """
    Punto de entrada principal del módulo NLP.
    Selecciona el pipeline según el tipo de documento.
    """
    funcion = PIPELINES.get(tipo_documento, analizar_generico)
    return funcion(texto)
