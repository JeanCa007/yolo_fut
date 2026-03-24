# 🚀 MEJORAS AVANZADAS

Guía de optimizaciones y características avanzadas para tu sistema de análisis de fútbol.

---

## 1. CALIBRACIÓN PRECISA DE LA CANCHA

### Problema
La calibración actual es aproximada (asume que el frame completo = cancha completa).

### Solución: Detección de Líneas

```python
def detect_field_lines(frame):
    """Detecta líneas de la cancha para calibración precisa"""
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar detección de bordes
    edges = cv2.Canny(gray, 50, 150)
    
    # Detectar líneas con transformada de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                            minLineLength=100, maxLineGap=10)
    
    # Filtrar líneas horizontales y verticales
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 10:  # Línea horizontal
                horizontal_lines.append(line)
            elif abs(abs(angle) - 90) < 10:  # Línea vertical
                vertical_lines.append(line)
    
    return horizontal_lines, vertical_lines

# Uso en tu código:
h_lines, v_lines = detect_field_lines(first_frame)
# Usar estas líneas para calcular pixels_per_meter con mayor precisión
```

---

## 2. CLASIFICACIÓN DE EQUIPOS POR COLOR

### Detectar a qué equipo pertenece cada jugador

```python
def classify_team_by_color(frame, box, num_clusters=2):
    """
    Clasifica jugador en equipo basándose en color de camiseta
    
    Args:
        frame: Frame del video
        box: Bounding box del jugador [x1, y1, x2, y2]
        num_clusters: Número de equipos (típicamente 2)
    
    Returns:
        team_id: ID del equipo (0, 1, ...)
    """
    from sklearn.cluster import KMeans
    
    x1, y1, x2, y2 = map(int, box)
    
    # Extraer región del jugador
    player_roi = frame[y1:y2, x1:x2]
    
    # Convertir a HSV (mejor para colores)
    hsv = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
    
    # Obtener región superior (torso/camiseta)
    height = y2 - y1
    torso = hsv[int(height*0.2):int(height*0.6), :]
    
    # Reshape para clustering
    pixels = torso.reshape(-1, 3)
    
    # K-means para encontrar color dominante
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    # Aquí puedes comparar con colores conocidos de equipos
    # O usar clustering global para todos los jugadores
    
    return dominant_color

# Integrar en tu código:
# Dentro del loop de detección
if class_id == PERSON_ID:
    team_color = classify_team_by_color(frame, box)
    # Asignar color según equipo
    if is_team_a(team_color):
        color = (255, 0, 0)  # Azul
    else:
        color = (0, 255, 255)  # Amarillo
```

---

## 3. DETECCIÓN DE EVENTOS

### Detectar goles, tiros, pases, etc.

```python
class EventDetector:
    def __init__(self):
        self.events = []
        self.ball_speed_threshold = 50  # km/h
        self.goal_area = None  # Definir coordenadas de áreas
    
    def detect_shot(self, ball_positions, frame_num):
        """Detecta tiros (balón a alta velocidad)"""
        
        if len(ball_positions) < 2:
            return False
        
        # Calcular velocidad del balón
        pos1 = ball_positions[-2]
        pos2 = ball_positions[-1]
        
        distance = np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
        # Convertir a velocidad real
        speed_kmh = (distance / self.pixels_per_meter) * self.fps * 3.6
        
        if speed_kmh > self.ball_speed_threshold:
            self.events.append({
                'type': 'SHOT',
                'frame': frame_num,
                'speed': speed_kmh
            })
            return True
        
        return False
    
    def detect_goal(self, ball_position):
        """Detecta si el balón está en el área de gol"""
        
        if self.goal_area is None:
            return False
        
        x, y = ball_position
        # Verificar si está dentro del área de gol
        # (necesitas definir goal_area según tu video)
        
        return False  # Implementar lógica
    
    def export_events(self, filename='events.json'):
        """Exporta eventos detectados a JSON"""
        import json
        
        with open(filename, 'w') as f:
            json.dump(self.events, f, indent=2)
```

---

## 4. MAPA DE CALOR PROFESIONAL

### Versión mejorada con densidad y zonas

```python
def create_advanced_heatmap(player_tracks, frame_shape, output='heatmap.png'):
    """
    Crea mapa de calor profesional con densidad de kernel
    """
    from scipy.stats import gaussian_kde
    
    height, width = frame_shape[:2]
    
    # Crear canvas
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Para cada jugador
    for track_id, positions in player_tracks.items():
        if len(positions) < 10:
            continue
        
        x_coords = np.array([p[0] for p in positions])
        y_coords = np.array([p[1] for p in positions])
        
        # Estimación de densidad con kernel gaussiano
        try:
            values = np.vstack([x_coords, y_coords])
            kernel = gaussian_kde(values)
            
            # Evaluar en una grilla
            xx, yy = np.mgrid[0:width:100j, 0:height:100j]
            positions_grid = np.vstack([xx.ravel(), yy.ravel()])
            density = np.reshape(kernel(positions_grid).T, xx.shape)
            
            # Añadir al heatmap
            heatmap += cv2.resize(density.T, (width, height))
        except:
            continue
    
    # Normalizar
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    
    # Aplicar colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Guardar
    cv2.imwrite(output, heatmap_color)
    
    return heatmap_color
```

---

## 5. ANÁLISIS TÁCTICO

### Formaciones y posicionamiento

```python
def analyze_formation(player_positions, team_id):
    """
    Analiza la formación del equipo en un momento dado
    """
    
    if len(player_positions) < 3:
        return "Insuficientes jugadores"
    
    # Extraer posiciones X, Y
    x_positions = [p[0] for p in player_positions]
    y_positions = [p[1] for p in player_positions]
    
    # Clustering vertical (líneas del equipo)
    from sklearn.cluster import DBSCAN
    
    y_array = np.array(y_positions).reshape(-1, 1)
    clustering = DBSCAN(eps=50, min_samples=2).fit(y_array)
    
    n_lines = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    
    formations = {
        3: "3-X-X",
        4: "4-X-X o X-4-X",
        5: "Posible 4-3-3 o 4-4-2"
    }
    
    return formations.get(n_lines, f"{n_lines} líneas detectadas")


def calculate_team_compactness(player_positions):
    """
    Calcula qué tan compacto está un equipo
    """
    
    if len(player_positions) < 2:
        return 0
    
    x_coords = [p[0] for p in player_positions]
    y_coords = [p[1] for p in player_positions]
    
    # Desviación estándar como medida de dispersión
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)
    
    compactness = np.sqrt(x_std**2 + y_std**2)
    
    return compactness
```

---

## 6. EXPORTAR DATOS A CSV/JSON

```python
def export_data_to_csv(player_tracks, fps, pixels_per_meter, filename='data.csv'):
    """Exporta todos los datos a CSV para análisis posterior"""
    
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['player_id', 'frame', 'time_s', 'x', 'y', 
                        'distance_m', 'speed_kmh'])
        
        # Para cada jugador
        for player_id, positions in player_tracks.items():
            total_distance = 0
            
            for i, (x, y, frame) in enumerate(positions):
                time_s = frame / fps
                
                # Calcular distancia incremental
                if i > 0:
                    prev_x, prev_y = positions[i-1][:2]
                    pixel_dist = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                    distance_m = pixel_dist / pixels_per_meter
                    total_distance += distance_m
                    
                    # Velocidad instantánea
                    time_diff = 1 / fps
                    speed_kmh = (distance_m / time_diff) * 3.6
                else:
                    distance_m = 0
                    speed_kmh = 0
                
                writer.writerow([player_id, frame, time_s, x, y, 
                               total_distance, speed_kmh])
    
    print(f"Datos exportados a {filename}")
```

---

## 7. INTERFAZ GRÁFICA (GUI)

### Usando Streamlit

```python
# Archivo: app.py

import streamlit as st
import cv2
from football_analysis import FootballAnalyzer

st.title("⚽ Análisis de Fútbol con YOLO")

# Upload video
uploaded_file = st.file_uploader("Sube tu video de fútbol", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Guardar temporalmente
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    # Parámetros
    confidence = st.slider("Confianza", 0.1, 1.0, 0.5, 0.1)
    
    if st.button("Analizar"):
        with st.spinner("Procesando..."):
            analyzer = FootballAnalyzer("temp_video.mp4")
            analyzer.process_video(confidence_threshold=confidence)
            analyzer.generate_statistics_report()
        
        st.success("¡Análisis completado!")
        
        # Mostrar video
        st.video("futbol_analizado.mp4")
        
        # Mostrar estadísticas
        with open("statistics.txt", "r") as f:
            st.text(f.read())
```

Ejecutar con:
```bash
pip install streamlit
streamlit run app.py
```

---

## 8. ENTRENAR MODELO PERSONALIZADO

### Para detectar balón específicamente

```python
# 1. Anotar tu propio dataset
# Usa herramientas como: LabelImg, CVAT, Roboflow

# 2. Organizar dataset
"""
dataset/
  ├── images/
  │   ├── train/
  │   ├── val/
  ├── labels/
      ├── train/
      ├── val/
"""

# 3. Crear archivo de configuración
# dataset.yaml
"""
path: ./dataset
train: images/train
val: images/val

nc: 2  # Número de clases
names: ['jugador', 'balon']
"""

# 4. Entrenar
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Partir del modelo pre-entrenado
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='futbol_detector'
)

# 5. Usar modelo personalizado
model = YOLO('runs/detect/futbol_detector/weights/best.pt')
```

---

## 9. OPTIMIZACIÓN DE RENDIMIENTO

### Para videos largos

```python
# Procesamiento multi-threading
from concurrent.futures import ThreadPoolExecutor

def process_frame_batch(frames, model):
    """Procesa un lote de frames en paralelo"""
    results = model(frames, stream=True)
    return list(results)

# En tu código principal:
def process_video_fast(video_path, batch_size=8):
    cap = cv2.VideoCapture(video_path)
    
    frame_buffer = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_buffer.append(frame)
        
        # Procesar en lotes
        if len(frame_buffer) >= batch_size:
            results = process_frame_batch(frame_buffer, model)
            # Procesar resultados...
            frame_buffer = []
```

---

## 10. DASHBOARD DE ANÁLISIS

### Crear panel interactivo

```python
# Usar Plotly para gráficos interactivos

import plotly.graph_objects as go

def create_dashboard(player_tracks, fps):
    """Crea dashboard interactivo con métricas"""
    
    fig = go.Figure()
    
    # Para cada jugador, crear línea de velocidad
    for player_id, positions in player_tracks.items():
        times = [p[2]/fps for p in positions]
        
        # Calcular velocidades
        speeds = []
        for i in range(1, len(positions)):
            # ... calcular velocidad ...
            speeds.append(speed)
        
        fig.add_trace(go.Scatter(
            x=times[1:],
            y=speeds,
            name=f'Jugador #{player_id}',
            mode='lines'
        ))
    
    fig.update_layout(
        title='Velocidad de Jugadores en el Tiempo',
        xaxis_title='Tiempo (s)',
        yaxis_title='Velocidad (km/h)'
    )
    
    fig.write_html('dashboard.html')
    print("Dashboard guardado en dashboard.html")
```

---

## 🎯 ORDEN RECOMENDADO DE IMPLEMENTACIÓN

1. ✅ **Script básico funcionando** (Ya lo tienes)
2. 🔄 **Calibración precisa** (Mejora mediciones)
3. 🔄 **Clasificación de equipos** (Análisis por equipo)
4. 🔄 **Exportar a CSV** (Análisis posterior)
5. 🔄 **Mapas de calor avanzados** (Visualización)
6. 🔄 **Detección de eventos** (Tiros, pases)
7. 🔄 **Dashboard interactivo** (Presentación)
8. 🔄 **GUI con Streamlit** (Usabilidad)
9. 🔄 **Modelo personalizado** (Precisión)
10. 🔄 **Análisis táctico** (Insights avanzados)

---

¡Éxito con tu proyecto! 🚀
