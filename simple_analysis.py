"""
Script SIMPLE de Análisis de Fútbol
Versión básica para comenzar rápidamente
"""

import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import os


def main():
    # ==================== CONFIGURACIÓN ====================
    VIDEO_INPUT = r"C:\Users\jalvarez\Downloads\yolo_fut\futbol_video.mp4.mp4"       # Tu video de entrada
    VIDEO_OUTPUT = "resultado.mp4"         # Video con anotaciones
    CONFIDENCE = 0.5                       # Confianza mínima (0-1)
    
    # Clases de YOLO que nos interesan
    PERSON_ID = 0        # Jugadores
    BALL_ID = 32         # Balón deportivo
    
    # ==================== INICIALIZACIÓN ====================
    print("Cargando modelo YOLO...")
    model = YOLO('yolov8n.pt')  # Modelo más ligero
    
    print("Abriendo video...")
    cap = cv2.VideoCapture(VIDEO_INPUT)


    # Propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    print(f"Total: {total_frames} frames ({total_frames/fps:.1f} segundos)")
    
    # Crear video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))
    
    # Para calcular distancias (aproximación simple)
    # Asumimos que el ancho del frame ~ 105 metros (longitud de cancha)
    PIXELS_PER_METER = width / 105
    
    # Diccionario para guardar posiciones por jugador
    player_positions = defaultdict(list)
    
    # ==================== PROCESAMIENTO ====================
    print("\nProcesando video...")
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar y trackear objetos
        results = model.track(
            frame,
            persist=True,           # Mantener IDs entre frames
            conf=CONFIDENCE,        # Confianza mínima
            classes=[PERSON_ID, BALL_ID]  # Solo personas y balón
        )
        
        # Si hay detecciones
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # IDs de tracking (si están disponibles)
            track_ids = None
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            # Procesar cada detección
            for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Si es un jugador
                if class_id == PERSON_ID:
                    color = (0, 255, 0)  # Verde
                    label = "Jugador"
                    
                    # Si tenemos ID de tracking
                    if track_ids is not None:
                        track_id = track_ids[i]
                        player_positions[track_id].append((center_x, center_y))
                        label = f"J{track_id}"
                        
                        # Calcular distancia recorrida
                        if len(player_positions[track_id]) > 1:
                            positions = player_positions[track_id]
                            
                            # Distancia total en píxeles
                            total_pixels = 0
                            for j in range(1, len(positions)):
                                dx = positions[j][0] - positions[j-1][0]
                                dy = positions[j][1] - positions[j-1][1]
                                total_pixels += np.sqrt(dx**2 + dy**2)
                            
                            # Convertir a metros
                            distance_m = total_pixels / PIXELS_PER_METER
                            
                            # Velocidad (metros/segundo -> km/h)
                            time_s = len(positions) / fps
                            speed_kmh = (distance_m / time_s) * 3.6 if time_s > 0 else 0
                            
                            # Mostrar métricas
                            cv2.putText(frame, f"{distance_m:.1f}m", 
                                      (int(x1), int(y1)-25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, f"{speed_kmh:.1f}km/h", 
                                      (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Si es el balón
                elif class_id == BALL_ID:
                    color = (0, 0, 255)  # Rojo
                    label = "BALON"
                    # Círculo más grande para el balón
                    cv2.circle(frame, (center_x, center_y), 10, color, -1)
                
                else:
                    continue
                
                # Dibujar cuadro
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            color, 2)
                
                # Etiqueta
                cv2.putText(frame, label, (int(x1), int(y2)+20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Punto central
                cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        # Info del frame
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Guardar frame
        out.write(frame)
        frame_num += 1
        
        # Mostrar progreso cada 30 frames
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progreso: {progress:.1f}% ({frame_num}/{total_frames})")
    
    # ==================== FINALIZACIÓN ====================
    cap.release()
    out.release()
    
    print(f"\n✅ COMPLETADO!")
    print(f"Video guardado: {VIDEO_OUTPUT}")
    
    # ==================== ESTADÍSTICAS ====================
    print("\n" + "="*50)
    print("ESTADÍSTICAS")
    print("="*50)
    
    if player_positions:
        for track_id in sorted(player_positions.keys()):
            positions = player_positions[track_id]
            
            if len(positions) < 2:
                continue
            
            # Calcular distancia total
            total_pixels = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_pixels += np.sqrt(dx**2 + dy**2)
            
            distance_m = total_pixels / PIXELS_PER_METER
            time_s = len(positions) / fps
            speed_avg = (distance_m / time_s) * 3.6 if time_s > 0 else 0
            
            print(f"\nJugador #{track_id}:")
            print(f"  Distancia: {distance_m:.2f} metros")
            print(f"  Tiempo visible: {time_s:.2f} segundos")
            print(f"  Velocidad promedio: {speed_avg:.2f} km/h")
    else:
        print("\nNo se detectaron jugadores con tracking.")
    
    print("\n" + "="*50)
    print("¡Revisa el video de salida!")


if __name__ == "__main__":
    main()
