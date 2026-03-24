"""
Sistema de Análisis de Fútbol con YOLO
Detecta jugadores, balón y calcula distancia recorrida y velocidad
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt


class FootballAnalyzer:
    def __init__(self, video_path, model_name='yolov8n.pt'):
        """
        Inicializa el analizador de fútbol
        
        Args:
            video_path: Ruta al video de fútbol
            model_name: Modelo YOLO a usar (yolov8n.pt es el más ligero)
        """
        self.video_path =  r"C:\Users\jalvarez\Downloads\yolo_fut\futbol_video.mp4.mp4" 
        self.model = YOLO(model_name)
        
        # IDs de clases COCO que nos interesan
        self.PERSON_CLASS_ID = 0  # Jugadores
        self.SPORTS_BALL_CLASS_ID = 32  # Balón
        
        # Diccionarios para tracking
        self.player_tracks = defaultdict(list)  # {track_id: [(x, y, frame), ...]}
        self.ball_tracks = []
        
        # Parámetros de la cancha (ajustar según tu video)
        self.FIELD_LENGTH_METERS = 105  # Longitud estándar de cancha en metros
        self.FIELD_WIDTH_METERS = 68   # Ancho estándar de cancha en metros
        
        # Variables para cálculo de distancias
        self.pixels_per_meter = None
        self.fps = None
        
    def calibrate_field(self, frame):
        """
        Calibra los píxeles por metro basándose en el tamaño del frame
        Nota: Esto es una aproximación simple. Para mejor precisión, 
        se necesitaría detectar las líneas de la cancha.
        """
        height, width = frame.shape[:2]
        
        # Asumimos que el frame captura aproximadamente toda la cancha
        # Esta es una simplificación, ajustar según tu video específico
        self.pixels_per_meter = width / self.FIELD_LENGTH_METERS
        
        print(f"Calibración: {self.pixels_per_meter:.2f} píxeles por metro")
        
    def calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana entre dos puntos en metros"""
        if self.pixels_per_meter is None:
            return 0
        
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + 
                                (point2[1] - point1[1])**2)
        return pixel_distance / self.pixels_per_meter
    
    def calculate_speed(self, distance_meters, time_seconds):
        """Calcula velocidad en km/h"""
        if time_seconds == 0:
            return 0
        meters_per_second = distance_meters / time_seconds
        return meters_per_second * 3.6  # Convertir a km/h
    
    def process_video(self, output_path='output_analyzed.mp4', 
                     confidence_threshold=0.5):
        """
        Procesa el video completo, detecta objetos y calcula métricas
        
        Args:
            output_path: Ruta para guardar el video procesado
            confidence_threshold: Umbral de confianza para detecciones
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # Obtener propiedades del video
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {self.fps} FPS")
        print(f"Total frames: {total_frames}")
        
        # Configurar writer para video de salida
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        frame_count = 0
        
        # Leer primer frame para calibración
        ret, first_frame = cap.read()
        if ret:
            self.calibrate_field(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Volver al inicio
        
        print("\nProcesando video...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Realizar detección con tracking
            results = self.model.track(frame, persist=True, 
                                      conf=confidence_threshold,
                                      classes=[self.PERSON_CLASS_ID, 
                                              self.SPORTS_BALL_CLASS_ID])
            
            # Procesar detecciones
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Obtener IDs de tracking si están disponibles
                track_ids = None
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Procesar cada detección
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Jugadores
                    if class_id == self.PERSON_CLASS_ID:
                        color = (0, 255, 0)  # Verde
                        label = "Jugador"
                        
                        # Guardar tracking si hay ID
                        if track_ids is not None:
                            track_id = track_ids[i]
                            self.player_tracks[track_id].append(
                                (center_x, center_y, frame_count)
                            )
                            label = f"Jugador #{track_id}"
                            
                            # Calcular distancia recorrida
                            if len(self.player_tracks[track_id]) > 1:
                                total_distance = self.calculate_total_distance(track_id)
                                
                                # Calcular velocidad promedio
                                time_elapsed = frame_count / self.fps
                                avg_speed = self.calculate_speed(total_distance, time_elapsed)
                                
                                # Mostrar métricas
                                cv2.putText(frame, f"Dist: {total_distance:.1f}m", 
                                          (int(x1), int(y1) - 35),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                cv2.putText(frame, f"Vel: {avg_speed:.1f}km/h", 
                                          (int(x1), int(y1) - 15),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Balón
                    elif class_id == self.SPORTS_BALL_CLASS_ID:
                        color = (0, 0, 255)  # Rojo
                        label = "Balon"
                        self.ball_tracks.append((center_x, center_y, frame_count))
                    
                    else:
                        continue
                    
                    # Dibujar bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), 
                                (int(x2), int(y2)), color, 2)
                    
                    # Dibujar label
                    cv2.putText(frame, f"{label} {conf:.2f}", 
                              (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Dibujar punto central
                    cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Información del frame
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Escribir frame procesado
            out.write(frame)
            frame_count += 1
            
            # Mostrar progreso
            if frame_count % 30 == 0:
                print(f"Procesado: {frame_count}/{total_frames} frames")
        
        # Liberar recursos
        cap.release()
        out.release()
        
        print(f"\n✓ Video procesado guardado en: {output_path}")
        
    def calculate_total_distance(self, track_id):
        """Calcula la distancia total recorrida por un jugador"""
        positions = self.player_tracks[track_id]
        total_distance = 0
        
        for i in range(1, len(positions)):
            point1 = positions[i-1][:2]
            point2 = positions[i][:2]
            total_distance += self.calculate_distance(point1, point2)
        
        return total_distance
    
    def generate_statistics_report(self, output_file='statistics.txt'):
        """Genera reporte con estadísticas de los jugadores"""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("REPORTE DE ANÁLISIS DE FÚTBOL")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        if not self.player_tracks:
            report_lines.append("No se detectaron jugadores con tracking válido.")
            return
        
        # Duración del video
        max_frame = max(max(track[-1][2] for track in self.player_tracks.values()))
        duration_seconds = max_frame / self.fps
        
        report_lines.append(f"Duración del video: {duration_seconds:.2f} segundos")
        report_lines.append(f"FPS: {self.fps}")
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("ESTADÍSTICAS POR JUGADOR")
        report_lines.append("-" * 60)
        report_lines.append("")
        
        # Estadísticas por jugador
        for track_id in sorted(self.player_tracks.keys()):
            positions = self.player_tracks[track_id]
            
            if len(positions) < 2:
                continue
            
            # Calcular métricas
            total_distance = self.calculate_total_distance(track_id)
            
            # Tiempo que estuvo visible
            frames_visible = len(positions)
            time_visible = frames_visible / self.fps
            
            # Velocidad promedio
            avg_speed = self.calculate_speed(total_distance, time_visible)
            
            # Velocidad máxima (entre frames consecutivos)
            max_speed = 0
            for i in range(1, len(positions)):
                point1 = positions[i-1][:2]
                point2 = positions[i][:2]
                frame_diff = positions[i][2] - positions[i-1][2]
                time_diff = frame_diff / self.fps
                
                if time_diff > 0:
                    distance = self.calculate_distance(point1, point2)
                    speed = self.calculate_speed(distance, time_diff)
                    max_speed = max(max_speed, speed)
            
            # Agregar al reporte
            report_lines.append(f"Jugador #{track_id}:")
            report_lines.append(f"  • Distancia total recorrida: {total_distance:.2f} metros")
            report_lines.append(f"  • Tiempo visible: {time_visible:.2f} segundos")
            report_lines.append(f"  • Velocidad promedio: {avg_speed:.2f} km/h")
            report_lines.append(f"  • Velocidad máxima: {max_speed:.2f} km/h")
            report_lines.append("")
        
        # Estadísticas del balón
        if self.ball_tracks:
            report_lines.append("-" * 60)
            report_lines.append("ESTADÍSTICAS DEL BALÓN")
            report_lines.append("-" * 60)
            report_lines.append(f"  • Detecciones: {len(self.ball_tracks)} frames")
            report_lines.append("")
        
        # Escribir reporte
        report_text = "\n".join(report_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\n✓ Reporte guardado en: {output_file}")
    
    def visualize_heatmap(self, track_id=None, output_file='heatmap.png'):
        """
        Genera un mapa de calor de las posiciones de los jugadores
        
        Args:
            track_id: ID específico del jugador (None = todos los jugadores)
            output_file: Archivo de salida para el mapa de calor
        """
        plt.figure(figsize=(12, 8))
        
        if track_id is None:
            # Todos los jugadores
            for tid, positions in self.player_tracks.items():
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                plt.scatter(x_coords, y_coords, alpha=0.3, s=10, 
                          label=f'Jugador #{tid}')
        else:
            # Jugador específico
            if track_id in self.player_tracks:
                positions = self.player_tracks[track_id]
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                plt.scatter(x_coords, y_coords, alpha=0.5, s=20)
                plt.title(f'Mapa de Calor - Jugador #{track_id}')
        
        plt.xlabel('X (píxeles)')
        plt.ylabel('Y (píxeles)')
        plt.legend()
        plt.gca().invert_yaxis()  # Invertir Y para que coincida con imagen
        plt.grid(True, alpha=0.3)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Mapa de calor guardado en: {output_file}")


def main():
    """Función principal"""
    
    print("=" * 60)
    print("SISTEMA DE ANÁLISIS DE FÚTBOL CON YOLO")
    print("=" * 60)
    print()
    
    # Configuración
    VIDEO_PATH = r"C:\Users\jalvarez\Downloads\yolo_fut\futbol_video.mp4.mp4"  # Cambiar por la ruta de tu video
    OUTPUT_VIDEO = "futbol_analizado.mp4"
    CONFIDENCE = 0.5  # Ajustar umbral de confianza (0.3 - 0.7)
    
    print(f"Video de entrada: {VIDEO_PATH}")
    print(f"Video de salida: {OUTPUT_VIDEO}")
    print(f"Umbral de confianza: {CONFIDENCE}")
    print()
    
    # Inicializar analizador
    print("Cargando modelo YOLO...")
    analyzer = FootballAnalyzer(VIDEO_PATH)
    
    # Procesar video
    analyzer.process_video(output_path=OUTPUT_VIDEO, 
                          confidence_threshold=CONFIDENCE)
    
   
    
    # Generar mapas de calor
    analyzer.visualize_heatmap(output_file='heatmap_todos.png')

     # Generar estadísticas
    analyzer.generate_statistics_report()
    
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETADO")
    print("=" * 60)
    

if __name__ == "__main__":
    main()
