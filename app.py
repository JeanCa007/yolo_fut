import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

st.set_page_config(page_title="Análisis de Fútbol", layout="wide")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


def process_video(
    video_path,
    confidence=0.5,
    realtime=True,
    save_output=False,
    output_path="resultado_streamlit.mp4"
):
    PERSON_ID = 0
    BALL_ID = 32

    model = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("No se pudo abrir el video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    PIXELS_PER_METER = width / 105 if width > 0 else 1

    player_positions = defaultdict(list)

    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_box = st.empty()

    stop_button_placeholder = st.empty()

    frame_num = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            conf=confidence,
            classes=[PERSON_ID, BALL_ID],
            verbose=False
        )

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            track_ids = None
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if class_id == PERSON_ID:
                    color = (0, 255, 0)
                    label = "Jugador"

                    if track_ids is not None:
                        track_id = track_ids[i]
                        player_positions[track_id].append((center_x, center_y))
                        label = f"J{track_id}"

                        if len(player_positions[track_id]) > 1:
                            positions = player_positions[track_id]

                            total_pixels = 0
                            for j in range(1, len(positions)):
                                dx = positions[j][0] - positions[j - 1][0]
                                dy = positions[j][1] - positions[j - 1][1]
                                total_pixels += np.sqrt(dx**2 + dy**2)

                            distance_m = total_pixels / PIXELS_PER_METER
                            time_s = len(positions) / fps
                            speed_kmh = (distance_m / time_s) * 3.6 if time_s > 0 else 0

                            cv2.putText(
                                frame,
                                f"{distance_m:.1f}m",
                                (x1, max(20, y1 - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                            cv2.putText(
                                frame,
                                f"{speed_kmh:.1f}km/h",
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )

                elif class_id == BALL_ID:
                    color = (0, 0, 255)
                    label = "BALON"
                    cv2.circle(frame, (center_x, center_y), 10, color, -1)

                else:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, min(height - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                cv2.circle(frame, (center_x, center_y), 4, color, -1)

        cv2.putText(
            frame,
            f"Frame: {frame_num}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        if save_output and out is not None:
            out.write(frame)

        progress = int(((frame_num + 1) / max(total_frames, 1)) * 100)
        progress_bar.progress(min(progress, 100))

        elapsed = time.time() - start_time
        current_fps = (frame_num + 1) / elapsed if elapsed > 0 else 0

        status_text.text(
            f"Procesando frame {frame_num + 1}/{total_frames} | FPS procesamiento: {current_fps:.2f}"
        )

        if realtime:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        if frame_num % 20 == 0:
            resumen = []
            for track_id in sorted(player_positions.keys())[:10]:
                positions = player_positions[track_id]
                if len(positions) < 2:
                    continue

                total_pixels = 0
                for j in range(1, len(positions)):
                    dx = positions[j][0] - positions[j - 1][0]
                    dy = positions[j][1] - positions[j - 1][1]
                    total_pixels += np.sqrt(dx**2 + dy**2)

                distance_m = total_pixels / PIXELS_PER_METER
                time_s = len(positions) / fps
                speed_avg = (distance_m / time_s) * 3.6 if time_s > 0 else 0

                resumen.append(
                    f"Jugador {track_id}: {distance_m:.1f} m | {speed_avg:.1f} km/h"
                )

            if resumen:
                stats_box.markdown("### Estadísticas parciales\n" + "\n".join(resumen))

        frame_num += 1

    cap.release()
    if out is not None:
        out.release()

    final_stats = []
    for track_id in sorted(player_positions.keys()):
        positions = player_positions[track_id]
        if len(positions) < 2:
            continue

        total_pixels = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_pixels += np.sqrt(dx**2 + dy**2)

        distance_m = total_pixels / PIXELS_PER_METER
        time_s = len(positions) / fps
        speed_avg = (distance_m / time_s) * 3.6 if time_s > 0 else 0

        final_stats.append({
            "jugador": int(track_id),
            "distancia_m": round(distance_m, 2),
            "tiempo_visible_s": round(time_s, 2),
            "velocidad_prom_kmh": round(speed_avg, 2),
        })

    return {
        "output_path": output_path if save_output else None,
        "stats": final_stats
    }


def main():
    st.title("⚽ Análisis de Fútbol con Computer Vision CREO")

    st.sidebar.header("Configuración")

    confidence = st.sidebar.slider(
        "Confianza mínima",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    realtime = st.sidebar.toggle("Mostrar en tiempo real", value=True)
    save_output = st.sidebar.toggle("Guardar video procesado", value=True)

    uploaded_file = st.file_uploader(
        "Sube un video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Procesar video", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name

            st.info("Procesando video...")

            result = process_video(
                video_path=temp_video_path,
                confidence=confidence,
                realtime=realtime,
                save_output=save_output,
                output_path="resultado_streamlit.mp4"
            )

            if result is not None:
                st.success("Proceso completado")

                if result["stats"]:
                    st.subheader("Estadísticas finales")
                    st.dataframe(result["stats"], use_container_width=True)
                else:
                    st.warning("No se detectaron jugadores con tracking.")

                if result["output_path"] and os.path.exists(result["output_path"]):
                    st.subheader("Video procesado")
                    with open(result["output_path"], "rb") as f:
                        st.download_button(
                            "Descargar video procesado",
                            data=f,
                            file_name="resultado_streamlit.mp4",
                            mime="video/mp4"
                        )

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)


if __name__ == "__main__":
    main()