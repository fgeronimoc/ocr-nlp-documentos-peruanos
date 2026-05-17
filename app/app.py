"""
app.py
------
Aplicación Streamlit: OCR + NLP para Documentos Peruanos
Proyecto de Examen de Medio Curso - PLN
"""

import streamlit as st
import numpy as np
import pandas as pd
import base64
import io
from PIL import Image

from preprocessor import preprocesar_completo
from ocr_engine import extraer_texto, extraer_texto_con_confianza
from nlp_pipeline import analizar, limpiar_texto, PIPELINES

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="OCR + NLP Perú",
    page_icon="🇵🇪",
    layout="wide"
)

st.title("🇵🇪 OCR + NLP para Documentos Peruanos")
st.markdown(
    "Extrae texto automáticamente desde imágenes o documentos "
    "y aplica análisis de lenguaje natural."
)

# ============================================================
# BARRA LATERAL - Configuración
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    tipo_documento = st.selectbox(
        "Tipo de documento",
        options=list(PIPELINES.keys()),
        help="Selecciona el tipo para aplicar el análisis NLP correcto."
    )

    umbral_confianza = st.slider(
        "Umbral de confianza OCR",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Filtra texto con baja confianza. 0 = incluir todo."
    )

    aplicar_preprocesamiento = st.checkbox(
        "Aplicar preprocesamiento de imagen",
        value=False,
        help="Útil para fotos de papel arrugado u oscuras. Para imágenes digitales limpias, déjalo desactivado."
    )

    st.markdown("---")
    st.markdown("**Tipos de documentos soportados:**")
    for tipo in PIPELINES.keys():
        st.markdown(f"- {tipo}")
    st.markdown("---")
    st.caption("Proyecto PLN — Examen Medio Curso")

# ============================================================
# SUBIDA DE ARCHIVO
# ============================================================

st.header("📤 Cargar Documento")

archivo = st.file_uploader(
    "Sube una imagen o PDF escaneado",
    type=["jpg", "jpeg", "png"],
    help="Formatos soportados: JPG, PNG"
)

if archivo is not None:

    extension = archivo.name.split(".")[-1].lower()
    contenido = archivo.read()

    col1, col2 = st.columns(2)

    # ----------------------------------------------------------
    # COLUMNA 1: Imagen original
    # ----------------------------------------------------------
    with col1:
        st.subheader("Imagen original")
        img_original = Image.open(io.BytesIO(contenido))
        st.image(img_original, use_column_width=True)

    # ----------------------------------------------------------
    # COLUMNA 2: Imagen preprocesada
    # ----------------------------------------------------------
    with col2:
        st.subheader("Imagen preprocesada")
        if aplicar_preprocesamiento:
            arr = np.array(Image.open(io.BytesIO(contenido)).convert("RGB"))
            img_proc = preprocesar_completo(arr)
            st.image(img_proc, use_column_width=True, clamp=True)
        else:
            st.image(Image.open(io.BytesIO(contenido)), use_column_width=True)

    # ----------------------------------------------------------
    # BOTÓN: Procesar
    # ----------------------------------------------------------
    st.markdown("---")
    if st.button("🔍 Extraer texto y analizar", type="primary"):

        # --- OCR ---
        with st.spinner("Extrayendo texto con EasyOCR..."):
            try:
                arr = np.array(Image.open(io.BytesIO(contenido)).convert("RGB"))
                if aplicar_preprocesamiento:
                    arr = preprocesar_completo(arr)
                texto_crudo = extraer_texto_con_confianza(arr, umbral=umbral_confianza)
            except Exception as e:
                st.error(f"Error en OCR: {e}")
                st.stop()

        # --- Limpieza NLP ---
        with st.spinner("Limpiando texto..."):
            texto_limpio = limpiar_texto(texto_crudo)

        # --- Análisis NLP ---
        with st.spinner("Aplicando análisis NLP..."):
            resultado = analizar(texto_limpio, tipo_documento)

        # ==========================================================
        # SECCIÓN: TEXTO EXTRAÍDO
        # ==========================================================
        st.header("📝 Texto extraído (OCR)")
        tab1, tab2 = st.tabs(["Texto crudo", "Texto limpio"])
        with tab1:
            st.text_area("Salida del OCR", texto_crudo, height=200)
            st.download_button(
                "⬇️ Descargar texto crudo",
                data=texto_crudo,
                file_name="texto_crudo.txt",
                mime="text/plain"
            )
        with tab2:
            st.text_area("Texto preprocesado (NLP)", texto_limpio, height=200)
            st.download_button(
                "⬇️ Descargar texto limpio",
                data=texto_limpio,
                file_name="texto_limpio.txt",
                mime="text/plain"
            )

        # ==========================================================
        # SECCIÓN: ANÁLISIS NLP
        # ==========================================================
        st.header(f"🧠 Análisis NLP — {resultado['tipo']}")

        # --- Entidades detectadas ---
        if resultado["entidades"]:
            st.subheader("📌 Entidades detectadas")
            for entidad, valores in resultado["entidades"].items():
                if isinstance(valores, list):
                    st.markdown(f"**{entidad}:** {', '.join(str(v) for v in valores)}")
                else:
                    st.markdown(f"**{entidad}:** {valores}")

        # --- Resumen ---
        if resultado["resumen"]:
            st.subheader("📋 Resumen / Información principal")
            st.info(resultado["resumen"])

        # --- Palabras frecuentes ---
        if resultado["palabras_frecuentes"]:
            st.subheader("📊 Palabras más frecuentes")
            df_palabras = pd.DataFrame(
                resultado["palabras_frecuentes"],
                columns=["Palabra", "Frecuencia"]
            )
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(df_palabras, use_container_width=True)
            with col_b:
                st.bar_chart(df_palabras.set_index("Palabra"))

        # --- Nube de palabras ---
        if resultado["wordcloud_base64"]:
            st.subheader("☁️ Nube de palabras")
            img_bytes = base64.b64decode(resultado["wordcloud_base64"])
            st.image(img_bytes, use_column_width=True)
            st.download_button(
                "⬇️ Descargar nube de palabras",
                data=img_bytes,
                file_name="nube_palabras.png",
                mime="image/png"
            )

        # --- Descargar todo como CSV ---
        st.markdown("---")
        st.subheader("⬇️ Descargar resultados completos")
        datos_exportar = {
            "tipo_documento": [tipo_documento],
            "texto_crudo": [texto_crudo[:500]],
            "texto_limpio": [texto_limpio[:500]],
            "resumen": [resultado["resumen"]],
        }
        for ent, val in resultado["entidades"].items():
            datos_exportar[ent] = [str(val)]

        df_export = pd.DataFrame(datos_exportar)
        csv = df_export.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️ Descargar resultados en CSV",
            data=csv,
            file_name="resultados_analisis.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Sube una imagen o PDF para comenzar el análisis.")
    st.markdown("""
    **Ejemplos de documentos que puedes usar:**
    - 📄 Foto de una boleta de compra peruana
    - 🧾 Factura electrónica escaneada
    - 📰 Recorte de periódico o noticia impresa
    - 📋 Formulario o comunicado institucional
    """)
