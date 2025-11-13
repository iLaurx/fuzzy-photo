import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Definición del Sistema Difuso (Mamdani) ---

def initialize_fuzzy_system():
    # Universos del Discurso
    universo_entrada = np.arange(0, 10.1, 0.1)
    universo_salida = np.arange(0, 100.1, 1)

    # Antecedentes (Entradas)
    nitidez = ctrl.Antecedent(universo_entrada, 'nitidez')
    contraste = ctrl.Antecedent(universo_entrada, 'contraste')
    exposicion = ctrl.Antecedent(universo_entrada, 'exposicion')

    # Consecuente (Salida)
    calidad_estetica = ctrl.Consequent(universo_salida, 'calidad_estetica', defuzzify_method='centroid')

    # Funciones de Pertenencia (MFs)
    # Nitidez
    nitidez['baja'] = fuzz.trimf(nitidez.universe, [0, 0, 5])
    nitidez['media'] = fuzz.trimf(nitidez.universe, [0, 5, 10])
    nitidez['alta'] = fuzz.trimf(nitidez.universe, [5, 10, 10])
    # Contraste
    contraste['bajo'] = fuzz.trimf(contraste.universe, [0, 0, 5])
    contraste['normal'] = fuzz.trimf(contraste.universe, [0, 5, 10])
    contraste['alto'] = fuzz.trimf(contraste.universe, [5, 10, 10])
    # Exposición
    exposicion['oscura'] = fuzz.trimf(exposicion.universe, [0, 0, 5])
    exposicion['correcta'] = fuzz.trimf(exposicion.universe, [0, 5, 10])
    exposicion['brillante'] = fuzz.trimf(exposicion.universe, [5, 10, 10])
    # Calidad Estetica (Salida)
    calidad_estetica['pobre'] = fuzz.trapmf(calidad_estetica.universe, [0, 0, 30, 50])
    calidad_estetica['aceptable'] = fuzz.trimf(calidad_estetica.universe, [30, 60, 90])
    calidad_estetica['excelente'] = fuzz.trapmf(calidad_estetica.universe, [70, 90, 100, 100])

    # Reglas de Inferencia
    rule1 = ctrl.Rule(nitidez['baja'] | contraste['bajo'], calidad_estetica['pobre'])
    rule2 = ctrl.Rule(nitidez['alta'] & contraste['alto'], calidad_estetica['excelente'])
    rule3 = ctrl.Rule(exposicion['oscura'] | exposicion['brillante'], calidad_estetica['pobre'])
    rule4 = ctrl.Rule(nitidez['media'] & contraste['normal'] & exposicion['correcta'], calidad_estetica['aceptable'])
    rule5 = ctrl.Rule(nitidez['alta'] & contraste['normal'] & exposicion['correcta'], calidad_estetica['excelente'])
    rule6 = ctrl.Rule(nitidez['media'] | contraste['normal'], calidad_estetica['aceptable'])
    rule7 = ctrl.Rule(nitidez['baja'] & exposicion['oscura'], calidad_estetica['pobre'])
    rule8 = ctrl.Rule(contraste['alto'] & exposicion['brillante'], calidad_estetica['pobre'])

    # Sistema de Control y Simulación (Se usa para evaluar los inputs)
    calidad_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
    calidad_simulador = ctrl.ControlSystemSimulation(calidad_ctrl)
    
    return calidad_simulador

def evaluate_quality(nitidez_val, contraste_val, exposicion_val, simulador):
    """
    Ejecuta el sistema difuso con los valores de entrada.
    """
    simulador.input['nitidez'] = nitidez_val
    simulador.input['contraste'] = contraste_val
    simulador.input['exposicion'] = exposicion_val
    
    simulador.compute()
    calidad_final = simulador.output['calidad_estetica']
    
    return float(calidad_final)