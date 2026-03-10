

def ablacion_iterativa_completa( modelos_ordenados,  X_prueba_dict, y_prueba, entrenar_y_evaluar_fn, min_modelos=1, label_encoder = None):
    """
    Realiza una ablación iterativa exhaustiva eliminando un modelo en cada paso,
    independientemente de si mejora el Macro-F1, hasta dejar `min_modelos`.

    Parámetros:
        modelos_ordenados: lista de tuplas (modelo, f1_val, tipo, nombre)
        y_val: etiquetas de validación (no se usa directamente)
        X_prueba_dict: dict de conjuntos de prueba por tipo
        y_prueba: etiquetas verdaderas del conjunto de prueba
        entrenar_y_evaluar_fn: función que entrena y evalúa un conjunto de modelos
        min_modelos: mínimo número de modelos a dejar antes de detenerse (default=1)

    Retorna:
        DataFrame con el historial del proceso
    """

    modelos_actuales = deepcopy(modelos_ordenados)
    historial = []

    resultado_base, modelo_base,_ = entrenar_y_evaluar_fn(modelos_actuales, X_prueba_dict, y_prueba)
    f1_base = resultado_base["Macro_F1_Score"].iloc[-1]
    mejor_modelos = deepcopy(modelos_actuales)
    mejor_score = f1_base

    paso = 1
    mejor_f1_actual = -1

    while len(modelos_actuales) > min_modelos:
        print(f"\nPaso {paso}: analizando impacto de cada eliminación teniendo {len(modelos_actuales)} modelos.")
        mejor_idx = -1
        mejor_delta = -float('inf')

        for i in range(len(modelos_actuales)):
            subset = modelos_actuales[:i] + modelos_actuales[i+1:]
            resultado,_,_ = entrenar_y_evaluar_fn(subset, X_prueba_dict, y_prueba)
            f1_actual = resultado["Macro_F1_Score"].iloc[-1]
            delta = f1_actual - f1_base
            print(f"   - Eliminando {modelos_actuales[i][3]}_{modelos_actuales[i][2]} → F1 = {f1_actual:.4f} (ΔF1 = {delta:+.4f}), buscando superar F1 = {mejor_f1_actual:.4f}")

            if f1_actual >= mejor_f1_actual:
                mejor_f1_actual = f1_actual
                mejor_idx = i
                mejor_delta = delta

        eliminado = modelos_actuales[mejor_idx]
        modelos_actuales.pop(mejor_idx)

        if mejor_f1_actual > mejor_score:
            mejor_score = mejor_f1_actual
            mejor_modelos = deepcopy(modelos_actuales)
            print(f"Nuevo mejor score encontrado: {mejor_score:.4f} con {len(mejor_modelos)} modelos.")

        f1_base = mejor_f1_actual
        paso += 1

    # Re-evaluar el mejor conjunto para devolver atención
    return entrenar_y_evaluar_fn(mejor_modelos,  X_prueba_dict, y_prueba,label_encoder,print_classification = True)
    
