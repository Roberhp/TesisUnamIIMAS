import os
import joblib
import optuna
import numpy as np

from typing import Callable, Any, Tuple
from scipy.sparse import vstack, issparse

from src.utils.metrics import evaluate_model
from src.data.dataset import DatasetSplit
from src.config.settings import SEED


def run_optuna_model(
    model_class: Any,
    suggest_params_fn: Callable,
    dataset: DatasetSplit,
    model_path: str,
    n_trials: int = 30,
    fixed_params: dict | None = None,
) -> Tuple[Any, float | None]:
    """
    Ejecuta búsqueda de hiperparámetros con Optuna usando DatasetSplit.

    Flujo:
    - Si el modelo ya existe en model_path, lo carga.
    - Si no existe, ejecuta Optuna usando:
        Train → fit
        Val   → evaluación
    - Después de seleccionar los mejores hiperparámetros,
      reentrena el modelo final usando Train + Val combinados.
    - Guarda el modelo entrenado en disco.
    """

    X_train = dataset.X_train
    y_train = dataset.y_train
    X_val = dataset.X_val
    y_val = dataset.y_val

    # ------------------------------------------------------
    # 1️ Si el modelo ya existe, cargar y devolver
    # ------------------------------------------------------
    if os.path.exists(model_path):
        print(f"[INFO] Modelo encontrado en {model_path}. Cargando...")
        best_model = joblib.load(model_path)
        return best_model, None

    # ------------------------------------------------------
    # 2️ Ejecutar búsqueda con Optuna (reproducible)
    # ------------------------------------------------------
    def objective(trial):
        params = suggest_params_fn(trial)

        if fixed_params:
            params.update(fixed_params)

        model = model_class(**params)
        model.fit(X_train, y_train)

        f1 = evaluate_model(model, X_val, y_val)
        return f1

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    if fixed_params:
        best_params.update(fixed_params)

    best_score = study.best_value

    print(f"[INFO] Mejor F1 validación: {best_score:.4f}")
    print(f"[INFO] Mejores hiperparámetros: {best_params}")

    # ------------------------------------------------------
    # 3️ Reentrenar modelo final con Train + Val combinados
    # ------------------------------------------------------
    if issparse(X_train):
        X_full = vstack([X_train, X_val])
    else:
        X_full = np.concatenate([X_train, X_val], axis=0)

    y_full = np.concatenate([y_train, y_val], axis=0)

    best_model = model_class(**best_params)
    best_model.fit(X_full, y_full)

    # ------------------------------------------------------
    # 4️ Guardar modelo final
    # ------------------------------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)

    print(f"[INFO] Modelo guardado en {model_path}")

    return best_model, best_score