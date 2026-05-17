# Instrucciones para armar el Dataset

## ¿Qué imágenes necesitas recolectar?

El dataset debe tener **mínimo 10-15 imágenes** para que el proyecto tenga sustento.
Mientras más imágenes tengas, mejor puntaje en el criterio "Recolección de Datos" (2 pts).

---

## Carpeta: `raw/boletas/`

Coloca aquí fotos o capturas de pantalla de **boletas de compra peruanas**.

### ¿Cómo conseguirlas?
- Toma foto con tu celular a boletas físicas que tengas en casa
- Captura pantalla de boletas electrónicas que hayas recibido por correo
- Descarga imágenes de ejemplo de boletas SUNAT del sitio oficial

### ¿Qué debe tener la boleta idealmente?
- RUC del emisor (11 dígitos)
- Fecha de emisión
- Serie y número (ej: B001-00001234)
- Lista de productos o servicios
- Montos en soles (S/)
- IGV (18%)
- Total a pagar

### Formatos aceptados
- JPG, JPEG, PNG
- Mínimo 300 DPI para buena calidad de OCR
- Si tienes PDF escaneado, también funciona

---

## Carpeta: `raw/facturas/`

Igual que boletas pero facturas (llevan RUC del cliente también).

---

## Carpeta: `raw/otros/`

Cualquier otro documento que quieras probar:
- Recetas médicas
- Noticias de periódico
- Formularios
- Comunicados

---

## Naming de archivos sugerido

```
boleta_001.jpg
boleta_002.jpg
factura_001.jpg
...
```

---

## Checklist antes de empezar el notebook

- [ ] Al menos 5 boletas en `raw/boletas/`
- [ ] Al menos 3 facturas en `raw/facturas/`
- [ ] Imágenes legibles (no muy borrosas ni oscuras)
- [ ] Variedad: distintos comercios, fechas, montos
