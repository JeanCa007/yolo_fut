"""
Script de prueba para verificar que todo está instalado correctamente
"""

import sys


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    
    print("=" * 60)
    print("VERIFICACIÓN DE DEPENDENCIAS")
    print("=" * 60)
    print()
    
    errors = []
    
    # OpenCV
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV no está instalado")
        errors.append("opencv-python")
    
    # Ultralytics (YOLO)
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics (YOLO): Instalado")
    except ImportError:
        print("❌ Ultralytics no está instalado")
        errors.append("ultralytics")
    
    # NumPy
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy no está instalado")
        errors.append("numpy")
    
    # Matplotlib
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib no está instalado")
        errors.append("matplotlib")
    
    # SciPy
    try:
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
    except ImportError:
        print("❌ SciPy no está instalado")
        errors.append("scipy")
    
    print()
    
    if errors:
        print("=" * 60)
        print("❌ FALTAN DEPENDENCIAS")
        print("=" * 60)
        print()
        print("Instala las dependencias faltantes con:")
        print(f"pip install {' '.join(errors)}")
        print()
        print("O instala todas con:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("=" * 60)
        print("✅ TODAS LAS DEPENDENCIAS ESTÁN INSTALADAS")
        print("=" * 60)
        return True


def test_yolo_model():
    """Prueba que YOLO puede cargar un modelo"""
    
    print()
    print("=" * 60)
    print("PRUEBA DEL MODELO YOLO")
    print("=" * 60)
    print()
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        print("Cargando modelo YOLOv8n...")
        print("(Primera vez descargará ~6MB)")
        
        model = YOLO('yolov8n.pt')
        print("✅ Modelo cargado exitosamente")
        
        # Crear imagen de prueba
        print("\nProbando detección con imagen de prueba...")
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        results = model(test_image, verbose=False)
        print("✅ Detección funcionando correctamente")
        
        print()
        print("=" * 60)
        print("✅ YOLO ESTÁ LISTO PARA USAR")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error al probar YOLO: {e}")
        return False


def check_video_file():
    """Verifica si existe un video de prueba"""
    
    print()
    print("=" * 60)
    print("VERIFICACIÓN DE VIDEO")
    print("=" * 60)
    print()
    
    import os
    
    video_files = [
        "futbol_video.mp4",
        "futbol.mp4",
        "video.mp4"
    ]
    
    found = False
    for video in video_files:
        if os.path.exists(video):
            print(f"✅ Video encontrado: {video}")
            found = True
            break
    
    if not found:
        print("⚠️  No se encontró ningún video de prueba")
        print()
        print("Para probar el sistema:")
        print("1. Coloca tu video en esta carpeta")
        print("2. Nómbralo 'futbol_video.mp4'")
        print("3. O edita VIDEO_INPUT en el script")
    
    return found


def main():
    """Ejecuta todas las verificaciones"""
    
    print()
    print("🔍 VERIFICACIÓN DEL SISTEMA DE ANÁLISIS DE FÚTBOL")
    print()
    
    # Verificar dependencias
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print()
        print("⚠️  Primero instala las dependencias faltantes")
        sys.exit(1)
    
    # Probar YOLO
    yolo_ok = test_yolo_model()
    
    if not yolo_ok:
        print()
        print("⚠️  Hay problemas con YOLO")
        sys.exit(1)
    
    # Verificar video
    video_ok = check_video_file()
    
    # Resumen final
    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print()
    
    if deps_ok and yolo_ok:
        print("✅ Sistema listo para usar")
        print()
        
        if video_ok:
            print("🎬 Puedes ejecutar el análisis:")
            print("   python simple_analysis.py")
        else:
            print("⚠️  Agrega un video para comenzar")
            print("   Luego ejecuta: python simple_analysis.py")
    else:
        print("❌ Hay problemas que resolver")
    
    print()


if __name__ == "__main__":
    main()
