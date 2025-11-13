from flask import Flask, request, jsonify
from flask_cors import CORS
# Importa la lógica separada del sistema difuso
from fuzzy_system import initialize_fuzzy_system, evaluate_quality 

app = Flask(__name__)
# Habilitar CORS para permitir llamadas desde el frontend (index.html)
CORS(app) 

# Inicializar el sistema difuso una sola vez al iniciar el servidor
CALIDAD_SIMULADOR = None
try:
    CALIDAD_SIMULADOR = initialize_fuzzy_system()
    print("Sistema de Lógica Difusa cargado con éxito.")
except Exception as e:
    print(f"Error al inicializar el sistema difuso: {e}")

@app.route('/evaluate', methods=['POST'])
def evaluate_fuzzy_api():
    """
    Endpoint que recibe los inputs (nitidez, contraste, exposicion) y 
    devuelve la calificación de calidad estética.
    """
    if CALIDAD_SIMULADOR is None:
        return jsonify({'error': 'Motor de Lógica Difusa no disponible.', 'calidad': 50.0}), 503

    try:
        data = request.json
        
        # Obtener y validar/convertir datos de entrada
        nitidez_val = float(data.get('nitidez'))
        contraste_val = float(data.get('contraste'))
        exposicion_val = float(data.get('exposicion'))

        # Ejecutar la lógica difusa
        calidad_final = evaluate_quality(nitidez_val, contraste_val, exposicion_val, CALIDAD_SIMULADOR)

        # Devolver resultado como JSON
        return jsonify({
            'calidad': calidad_final,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error procesando la solicitud: {e}")
        return jsonify({
            'calidad': 50.0, 
            'error': 'Error interno del servidor al evaluar la lógica.',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("Iniciando servidor de Lógica Difusa en http://127.0.0.1:5000")
    app.run(debug=True)