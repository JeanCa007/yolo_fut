# ⚽ Sistema de Análisis de Fútbol con YOLO

Sistema de computer vision para analizar videos de fútbol, detectar jugadores y balón, y calcular métricas de rendimiento como distancia recorrida y velocidad.

## 🎯 Características

- ✅ Detección automática de jugadores
- ✅ Detección del balón
- ✅ Tracking de jugadores entre frames
- ✅ Cálculo de distancia recorrida
- ✅ Cálculo de velocidad (promedio y máxima)
- ✅ Video anotado con métricas en tiempo real
- ✅ Reporte estadístico detallado
- ✅ Mapas de calor de movimiento

## 📁 Archivos del Proyecto

```
├── requirements.txt           # Dependencias Python
├── GUIA_INSTALACION.md       # Guía paso a paso completa
├── football_analysis.py       # Script principal (completo)
├── simple_analysis.py         # Script simplificado (para empezar)
└── README.md                  # Este archivo
```

## 🚀 Inicio Rápido

### 1.  Crear Ambiente

python -m venv yolo_fut

# Ejecutar Ambiente 

.\yolo_fut\Scripts\activate.bat

# Instalar Dependencias

pip install -r requirements.txt

### 2. Preparar video
- Coloca tu video de fútbol en esta carpeta
- Nómbralo `futbol_video.mp4` (o cambia el nombre en el script)

### 3. Ejecutar

**Opción A: Script simple (recomendado para empezar)**
```bash
python simple_analysis.py
```

**Opción B: Script completo (con más funciones)**
```bash
python football_analysis.py
```

**Opción Streamlit: Script completo (con más funciones)**
```bash
streamlit run app.py
```

## 📊 Resultados

El sistema genera:

1. **Video anotado** (`resultado.mp4` o `futbol_analizado.mp4`)
   - Cuadros alrededor de jugadores (verde) y balón (rojo)
   - ID único por jugador
   - Distancia y velocidad en tiempo real

2. **Reporte de estadísticas** (`statistics.txt`)
   - Métricas por jugador
   - Distancia total
   - Velocidad promedio y máxima

3. **Mapa de calor** (`heatmap_todos.png`)
   - Visualización de posiciones
   - Trayectorias de movimiento

## ⚙️ Configuración

### Ajustar parámetros

En el script, modifica estas variables:

```python
VIDEO_INPUT = "tu_video.mp4"    # Tu video de entrada
VIDEO_OUTPUT = "resultado.mp4"  # Nombre del resultado
CONFIDENCE = 0.5                # 0.3 = más detecciones, 0.7 = más precisión
```

### Modelos YOLO disponibles

```python
'yolov8n.pt'  # Nano - Más rápido (recomendado)
'yolov8s.pt'  # Small
'yolov8m.pt'  # Medium
'yolov8l.pt'  # Large
'yolov8x.pt'  # Extra Large - Más preciso pero más lento
```

## 🎬 Ejemplo de Uso

```bash
# 1. Activar entorno virtual (si lo usas)
source venv/bin/activate

# 2. Ejecutar análisis
python simple_analysis.py

# 3. Esperar procesamiento (depende del video)
# Para 30s @ 30fps = ~900 frames ≈ 2-5 minutos

# 4. Ver resultados
# - resultado.mp4 (video anotado)
# - Estadísticas en consola
```

## 📈 Resultados Esperados

Para un video de 30 segundos:

- **Jugadores detectados**: Variable (depende de la vista)
- **Tracking**: IDs únicos mantenidos entre frames
- **Distancia**: Típicamente 50-200 metros por jugador
- **Velocidad**: 5-25 km/h (promedio), hasta 30+ km/h (sprints)

## 🔧 Troubleshooting

| Problema | Solución |
|----------|----------|
| No detecta jugadores | Bajar `CONFIDENCE` a 0.3 |
| Muchos falsos positivos | Subir `CONFIDENCE` a 0.7 |
| Procesamiento muy lento | Usar modelo `yolov8n.pt` |
| No detecta el balón | Normal, es difícil (muy pequeño) |
| Error de instalación | Ver `GUIA_INSTALACION.md` |

## 📚 Mejoras Futuras

- [ ] Clasificación de equipos por color
- [ ] Detección de eventos (goles, faltas, etc.)
- [ ] Calibración automática de cancha
- [ ] Análisis táctico (formaciones, mapas de pases)
- [ ] Exportar datos a CSV/JSON
- [ ] Interfaz gráfica (GUI)
- [ ] Entrenar modelo personalizado

## 🤝 Contribuir

Ideas para mejorar:

1. **Calibración avanzada**: Detectar líneas de cancha para mayor precisión
2. **Clustering de equipos**: Usar K-means en colores HSV
3. **Detección de eventos**: Usar cambios bruscos en trayectorias
4. **Base de datos**: Guardar estadísticas en SQLite

## 📖 Recursos

- [YOLO Ultralytics Docs](https://docs.ultralytics.com/)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Computer Vision in Sports](https://paperswithcode.com/task/sports-video-analysis)

## ⚠️ Limitaciones

- **Precisión de distancias**: Aproximada, mejora con calibración
- **Detección de balón**: Difícil (pequeño, movimiento rápido)
- **Vista de cámara**: Funciona mejor con vista aérea/lateral fija
- **Oclusiones**: Jugadores solapados pueden perder tracking

## 📝 Licencia

Proyecto educativo - Úsalo libremente para aprender y mejorar.

---

**Desarrollado con ❤️ para análisis deportivo**

¿Preguntas? Revisa `GUIA_INSTALACION.md` para más detalles.
