import numpy as np
import cv2 # Necesario para el procesamiento de imágenes
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Importa la lógica separada del sistema difuso
from fuzzy_system import initialize_fuzzy_system, evaluate_quality 

app = Flask(__name__)
# Habilitar CORS para permitir llamadas desde el frontend (index.html)
CORS(app) 

# Carpeta donde se guardarán temporalmente las imágenes subidas
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Crear la carpeta si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Inicializar el sistema difuso una sola vez al iniciar el servidor
CALIDAD_SIMULADOR = None
try:
    CALIDAD_SIMULADOR = initialize_fuzzy_system()
    print("Sistema de Lógica Difusa cargado con éxito.")
except Exception as e:
    print(f"Error al inicializar el sistema difuso: {e}")

# --- Funciones de Análisis de Imagen con OpenCV ---

def calculate_metrics(image_path):
    """
    Calcula Nitidez, Contraste y Exposición (brillo) de una imagen usando OpenCV.
    
    Los resultados se escalan de 0 a 10 para coincidir con el universo de entrada del sistema difuso.
    """
    # Leer la imagen en escala de grises para cálculos de nitidez y contraste
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Leer la imagen a color para el cálculo de brillo/exposición
    img_bgr = cv2.imread(image_path)
    
    if img_gray is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {image_path}")

    # 1. NITIDEZ (Sharpness) - Se usa la Varianza del Laplaciano
    # Mayor varianza = Mayor nitidez
    laplacian_variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    # Los valores típicos de nitidez varían, se necesita un mapeo:
    # 0 (desenfocado) a 1000+ (muy nítido). Usamos un mapeo logarítmico para comprimir.
    nitidez_val = np.log1p(laplacian_variance) * (10 / np.log1p(300)) # Escala aproximada a 0-10
    nitidez_val = np.clip(nitidez_val, 0, 10) # Limitar a [0, 10]
    
    # 2. CONTRASTE (Contrast) - Se usa la desviación estándar de la intensidad
    # Mayor desviación estándar = Mayor contraste
    contrast_std = np.std(img_gray)
    # Rango de 0 a 255. 
    contraste_val = (contrast_std / 50) * 10 # Escala: 50 es un buen punto medio.
    contraste_val = np.clip(contraste_val, 0, 10)

    # 3. EXPOSICIÓN (Exposure/Brightness) - Se usa el valor medio de los píxeles
    # Convertir a HSV y usar el canal V (Value/Brillo)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_value = np.mean(img_hsv[:,:,2])
    # Rango de 0 a 255. Un valor de 127 es 'correcta' (5.0 en el sistema difuso)
    # Se ajusta para que el valor 127 sea el centro (5.0) y 255 sea el máximo (10.0)
    exposicion_val = (mean_value / 255) * 10 
    
    return float(nitidez_val), float(contraste_val), float(exposicion_val)

@app.route('/upload_and_evaluate', methods=['POST'])
def upload_and_evaluate_fuzzy_api():
    """
    Endpoint que recibe una imagen subida, calcula sus métricas
    (nitidez, contraste, exposicion) y devuelve la calificación de calidad estética.
    """
    if CALIDAD_SIMULADOR is None:
        return jsonify({'error': 'Motor de Lógica Difusa no disponible.', 'calidad': 50.0}), 503

    # Verificar si se subió un archivo
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No se encontró el archivo de imagen.', 'calidad': 50.0}), 400

    file = request.files['imageFile']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío.', 'calidad': 50.0}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # 1. Guardar la imagen temporalmente
            file.save(filepath)
            
            # 2. Calcular las métricas con OpenCV
            nitidez_val, contraste_val, exposicion_val = calculate_metrics(filepath)

            # 3. Ejecutar la lógica difusa
            calidad_final = evaluate_quality(nitidez_val, contraste_val, exposicion_val, CALIDAD_SIMULADOR)

            # 4. Devolver resultado como JSON (y también las métricas calculadas)
            return jsonify({
                'calidad': calidad_final,
                'nitidez_calc': round(nitidez_val, 2),
                'contraste_calc': round(contraste_val, 2),
                'exposicion_calc': round(exposicion_val, 2),
                'status': 'success'
            })

        except Exception as e:
            print(f"Error procesando la solicitud: {e}")
            return jsonify({
                'calidad': 50.0, 
                'error': f'Error interno: {str(e)}',
                'status': 'error'
            }), 500
        finally:
            # 5. Limpiar: Eliminar el archivo temporal
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    print("Iniciando servidor de Lógica Difusa en http://127.0.0.1:5000")
    # Para el desarrollo, app.run es suficiente. Para producción, usar un WSGI como Gunicorn.
    app.run(debug=True)