# GUÍA PASO A PASO: Sistema de Análisis de Fútbol con YOLO

## 📋 REQUISITOS PREVIOS

- Python 3.8 o superior
- Video de fútbol (formato: .mp4, .avi, .mov)
- Al menos 4GB de RAM
- (Opcional) GPU con CUDA para procesamiento más rápido

---

## 🚀 PASO 1: INSTALACIÓN

### 1.1 Crear entorno virtual (recomendado)
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

### 1.2 Instalar dependencias
```bash
pip install -r requirements.txt
```

### 1.3 Verificar instalación
```bash
python -c "import cv2; import ultralytics; print('✓ Instalación exitosa')"
```

---

## 📁 PASO 2: PREPARAR TU VIDEO

1. Coloca tu video de fútbol en la misma carpeta que el script
2. Renómbralo a `futbol_video.mp4` o edita la variable `VIDEO_PATH` en el código

**Formato recomendado:**
- Resolución: 720p o 1080p
- FPS: 25-30
- Duración: 30 segundos
- Vista: Preferiblemente desde una cámara fija con vista amplia de la cancha

---

## ⚙️ PASO 3: CONFIGURACIÓN

### 3.1 Ajustar parámetros en football_analysis.py

```python
# Línea ~220 en main()
VIDEO_PATH = "futbol_video.mp4"      # Tu video de entrada
OUTPUT_VIDEO = "futbol_analizado.mp4"  # Video procesado
CONFIDENCE = 0.5                      # Umbral de confianza (0.3-0.7)
```

### 3.2 Calibración de la cancha

Si tu video tiene una vista diferente, ajusta estos valores en la clase:

```python
# Línea ~23-24
self.FIELD_LENGTH_METERS = 105  # Longitud de la cancha en metros
self.FIELD_WIDTH_METERS = 68    # Ancho de la cancha en metros
```

---

## ▶️ PASO 4: EJECUTAR EL ANÁLISIS

```bash
python football_analysis.py
```

### ¿Qué hace el script?

1. **Carga el modelo YOLO** (primera vez descargará ~6MB)
2. **Procesa cada frame** del video
3. **Detecta y trackea:**
   - Jugadores (cuadros verdes)
   - Balón (cuadro rojo)
4. **Calcula en tiempo real:**
   - Distancia recorrida por cada jugador
   - Velocidad promedio y máxima
5. **Genera:**
   - Video anotado: `futbol_analizado.mp4`
   - Reporte estadístico: `statistics.txt`
   - Mapa de calor: `heatmap_todos.png`

---

## 📊 PASO 5: REVISAR RESULTADOS

### 5.1 Video Analizado
- Abre `futbol_analizado.mp4`
- Verás cuadros alrededor de jugadores y balón
- Cada jugador tiene un ID único
- Sobre cada jugador aparece:
  - Distancia recorrida (metros)
  - Velocidad (km/h)

### 5.2 Reporte de Estadísticas
- Abre `statistics.txt`
- Contiene métricas detalladas por jugador:
  - Distancia total
  - Tiempo visible
  - Velocidad promedio
  - Velocidad máxima

### 5.3 Mapa de Calor
- Abre `heatmap_todos.png`
- Visualiza dónde se movió cada jugador
- Diferentes colores = diferentes jugadores

---

## 🎯 PASO 6: OPTIMIZACIÓN

### Mejorar precisión de detección

**Si no detecta jugadores:**
```python
CONFIDENCE = 0.3  # Bajar umbral (más detecciones, más falsos positivos)
```

**Si hay muchos falsos positivos:**
```python
CONFIDENCE = 0.7  # Subir umbral (menos detecciones, más precisión)
```

### Usar modelo más preciso

Editar línea ~14:
```python
# Modelos disponibles (de más ligero a más preciso):
model_name='yolov8n.pt'  # Nano (más rápido, menos preciso)
model_name='yolov8s.pt'  # Small
model_name='yolov8m.pt'  # Medium
model_name='yolov8l.pt'  # Large
model_name='yolov8x.pt'  # Extra Large (más preciso, más lento)
```

### Mejorar calibración de distancias

Para mayor precisión, puedes:
1. Detectar las líneas de la cancha
2. Usar perspectiva inversa
3. Implementar homografía

---

## 🔧 TROUBLESHOOTING

### Error: "No se encontró el video"
- Verifica que `futbol_video.mp4` esté en la carpeta correcta
- Verifica la ruta en `VIDEO_PATH`

### Error: "No module named 'ultralytics'"
```bash
pip install ultralytics --break-system-packages
```

### Video muy lento
- Usa un modelo más ligero (yolov8n.pt)
- Reduce la resolución del video
- Usa GPU si está disponible

### No detecta el balón
- El balón es difícil de detectar (pequeño, rápido)
- Ajusta `CONFIDENCE` más bajo
- Considera entrenar un modelo personalizado

---

## 📈 SIGUIENTES PASOS

### Mejoras sugeridas:

1. **Entrenamiento personalizado:**
   - Anota tu propio dataset de fútbol
   - Entrena YOLO específicamente para tu caso

2. **Detección de equipos:**
   - Clasificar jugadores por color de camiseta
   - Usar clustering de colores

3. **Análisis táctico:**
   - Formaciones de equipo
   - Mapas de pases
   - Zonas de presión

4. **Calibración avanzada:**
   - Detección automática de líneas de cancha
   - Transformación de perspectiva

5. **Métricas adicionales:**
   - Aceleración
   - Sprints
   - Tiempo de posesión del balón

---

## 📚 RECURSOS ADICIONALES

- Documentación YOLO: https://docs.ultralytics.com/
- OpenCV Python: https://docs.opencv.org/
- Tutorial tracking: https://github.com/ultralytics/ultralytics

---

## ❓ PREGUNTAS FRECUENTES

**P: ¿Puedo usar videos más largos?**
R: Sí, pero el procesamiento tomará más tiempo. Para videos >5min considera procesamiento por lotes.

**P: ¿Funciona con cualquier deporte?**
R: Sí, pero las métricas de distancia necesitarían recalibrarse para cada campo/cancha.

**P: ¿Necesito GPU?**
R: No es obligatorio, pero acelera el procesamiento significativamente.

**P: ¿Las distancias son precisas?**
R: Son aproximadas. Para precisión profesional se necesita calibración con puntos de referencia reales.
