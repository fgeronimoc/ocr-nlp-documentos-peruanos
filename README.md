# Proyecto OCR + NLP para Documentos Peruanos

**Curso:** Procesamiento de Lenguaje Natural  
**Entrega:** Lunes 25 de mayo de 2026  
**Tema:** Extracción y análisis inteligente de texto en documentos peruanos usando OCR y NLP

---

## Descripción

Aplicación inteligente capaz de extraer texto automáticamente desde imágenes o documentos escaneados usando OCR, y aplicar técnicas de NLP para analizar, resumir, clasificar o interpretar la información extraída en el contexto peruano.

**Caso de uso principal:** Boletas y facturas peruanas (formato SUNAT)  
**Arquitectura:** Modular — fácilmente extensible a otros tipos de documentos

---

## Estructura del Proyecto

```
Proyecto_OCR_NLP_Documentos_Peruanos/
│
├── dataset/
│   ├── raw/                  # Imágenes originales (JPG, PNG, PDF)
│   │   ├── boletas/
│   │   ├── facturas/
│   │   └── otros/
│   └── preprocessed/         # Imágenes ya procesadas listas para OCR
│
├── notebook/
│   └── OCR_NLP_Documentos_Peruanos.ipynb   # Notebook principal (Google Colab)
│
├── app/
│   ├── app.py                # Aplicación Streamlit principal
│   ├── ocr_engine.py         # Módulo OCR (EasyOCR)
│   ├── preprocessor.py       # Módulo preprocesamiento de imágenes
│   ├── nlp_pipeline.py       # Módulo NLP (entidades, resumen, nube de palabras)
│   └── requirements.txt      # Dependencias del proyecto
│
├── assets/
│   └── logo.png              # Logo o imágenes de la interfaz (opcional)
│
├── outputs/
│   └── resultados_ejemplo.csv  # Resultados de pruebas guardados
│
├── informe/
│   └── Informe_Proyecto.docx   # Informe breve del proyecto
│
└── README.md                 # Este archivo
```

---

## Pipeline del Sistema

```
[Imagen / PDF]
      ↓
[Preprocesamiento]  → escala de grises, binarización, eliminación de ruido
      ↓
[OCR - EasyOCR]     → extracción de texto crudo
      ↓
[Limpieza NLP]      → minúsculas, stopwords, tokenización
      ↓
[Análisis NLP]      → entidades (RUC, fechas, montos), nube de palabras, resumen
      ↓
[Visualización]     → Streamlit app
```

---

## Tipos de Documentos Soportados

| Tipo           | Estado     | Análisis NLP aplicado              |
|----------------|------------|------------------------------------|
| Boletas SUNAT  | ✅ Activo  | Entidades: RUC, montos, fechas     |
| Facturas       | ✅ Activo  | Entidades: empresa, IGV, total     |
| Recetas médicas| 🔜 Próximo | Extracción de medicamentos/dosis   |
| Noticias       | 🔜 Próximo | Resumen automático + clasificación |
| Formularios    | 🔜 Próximo | Extracción clave-valor             |

---

## Instalación

```bash
pip install -r app/requirements.txt
```

## Ejecutar la aplicación

```bash
cd app
streamlit run app.py
```

---

## Entregables

- [x] Repositorio GitHub
- [ ] Notebook Google Colab (`notebook/OCR_NLP_Documentos_Peruanos.ipynb`)
- [ ] Aplicación Streamlit desplegada
- [ ] Código fuente completo
- [ ] Informe breve (`informe/Informe_Proyecto.docx`)
- [ ] Exposición grupal

---

## Rúbrica (Total: 20 pts)

| Criterio                        | Puntaje |
|---------------------------------|---------|
| Contexto peruano / problema     | 2       |
| Recolección de datos            | 2       |
| Preprocesamiento de imágenes    | 3       |
| Implementación OCR              | 3       |
| Limpieza y preprocesamiento NLP | 3       |
| Aplicación de NLP               | 4       |
| Desarrollo app Streamlit        | 2       |
| Presentación y exposición       | 1       |
| **TOTAL**                       | **20**  |
