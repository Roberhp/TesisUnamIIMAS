# ===============================
# Standard library
# ===============================
import os
from typing import Dict, Any

# ===============================
# Local imports
# ===============================
from src.data.dataset import DatasetSplit
from src.tuning.optuna_runner import run_optuna_model
from src.tuning.model_configs import MODEL_CONFIGS
from src.config.settings import MODELS_DIR


# ==========================================================
# Internal helper
# ==========================================================

def _train_family_with_optuna(
    model_name: str,
    features: Dict[str, Dict[str, Any]],
    dataset: DatasetSplit,
    n_trials: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """
    Entrena una familia de modelos usando Optuna.

    Para cada representación lingüística:
    - Si el modelo ya existe en disco → lo carga
    - Si no existe → ejecuta Optuna
    - Reentrena con Train + Val
    - Guarda el modelo automáticamente

    Retorna estructura:

    {
        feature_name: {
            "model": modelo_entrenado,
            "val_f1_score": best_val_score,
            "feature_name": feature_name,
            "model_name": model_name
        }
    }
    """

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Modelo '{model_name}' no definido en MODEL_CONFIGS"
        )

    # Asegurar que el directorio de modelos exista
    os.makedirs(MODELS_DIR, exist_ok=True)

    config = MODEL_CONFIGS[model_name]

    model_class = config["model_class"]
    suggest_fn = config["suggest_fn"]
    fixed_params = config.get("fixed_params", {}) or {}

    trained_family: Dict[str, Dict[str, Any]] = {}

    for feature_name, splits in features.items():

        # Dataset específico para esta representación
        feature_dataset = DatasetSplit(
            X_train=splits["train"],
            X_val=splits["val"],
            X_test=None,  # no se usa aquí
            y_train=dataset.y_train,
            y_val=dataset.y_val,
            y_test=None,
            n_classes=dataset.n_classes,
            class_names=dataset.class_names,
            label_encoder=dataset.label_encoder,
        )

        model_path = os.path.join(
            MODELS_DIR,
            f"{model_name}_{feature_name}.pkl",
        )

        best_model, best_score = run_optuna_model(
            model_class=model_class,
            suggest_params_fn=suggest_fn,
            dataset=feature_dataset,
            model_path=model_path,
            n_trials=n_trials,
            fixed_params=fixed_params,
        )

        trained_family[feature_name] = {
            "model": best_model,
            "val_f1_score": best_score,
            "feature_name": feature_name,
            "model_name": model_name,
        }

    return trained_family


# ==========================================================
# Public API
# ==========================================================

def train_model_family(
    model_name: str,
    features: Dict[str, Dict[str, Any]],
    dataset: DatasetSplit,
    n_trials: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """
    Punto de entrada genérico para entrenar cualquier familia
    definida en MODEL_CONFIGS.
    """
    return _train_family_with_optuna(
        model_name=model_name,
        features=features,
        dataset=dataset,
        n_trials=n_trials,
    )


# ==========================================================
# Wrappers explícitos (opcionales)
# ==========================================================

def train_random_forest(
    features: Dict[str, Dict[str, Any]],
    dataset: DatasetSplit,
    n_trials: int = 30,
):
    return train_model_family("random_forest", features, dataset, n_trials)


def train_logreg(
    features: Dict[str, Dict[str, Any]],
    dataset: DatasetSplit,
    n_trials: int = 30,
):
    return train_model_family("logreg", features, dataset, n_trials)


def train_knn(
    features: Dict[str, Dict[str, Any]],
    dataset: DatasetSplit,
    n_trials: int = 30,
):
    return train_model_family("knn", features, dataset, n_trials)


def train_mlp(
    features: Dict[str, Dict[str, Any]],
    dataset: DatasetSplit,
    n_trials: int = 30,
):
    return train_model_family("mlp", features, dataset, n_trials)


def train_xgb(
    features: Dict[str, Dict[str, Any]],
    dataset: DatasetSplit,
    n_trials: int = 30,
):
    return train_model_family("xgb", features, dataset, n_trials)