# ===============================
# Standard library
# ===============================
from typing import Dict, Any

# ===============================
# Third-party
# ===============================
import pandas as pd

# ===============================
# Local imports
# ===============================
from src.data.loader import load_csv_dataset
from src.data.dataset import separa_datos
from src.data.encoding import encode_labels
from src.data.preprocessing import preprocess_dataset
from src.features.vectorizers import build_all_features
from src.models.classical_models import train_model_family
from src.utils.metrics import evaluate_model
from src.config.settings import DATA_PATH, LABEL_COLUMN, TEXT_COLUMN
from src.models.fan_trainer import train_and_evaluate_fan


# ==========================================================
# principal flujo de entrenamiento
# ==========================================================

def train_pipeline(n_trials: int = 30) -> Dict[str, Any]:
    """
    Pipeline completo de entrenamiento.

    Flujo:
    - Carga datos
    - Split train/val/test
    - Preprocesamiento
    - Encoding de etiquetas
    - Construcción de features
    - Entrenamiento de modelos con Optuna
    - Evaluación en test
    """

    print("* Cargando dataset...")
    df = load_csv_dataset(DATA_PATH)

    print("* Creando splits...")
    dataset = separa_datos(
            df,
            text_col=TEXT_COLUMN,
            label_col=LABEL_COLUMN,
        )

    print("* Preprocesando texto...")
    dataset = preprocess_dataset(dataset)

    print("* Codificando etiquetas...")
    dataset = encode_labels(dataset)

    print("* Generando representaciones lingüísticas...")
    features, vectorizers = build_all_features(
        dataset,
        n_topics=dataset.n_classes,
    )

    # Guardar artifacts de vectorización y metadata
    from src.artifacts.artifacts_manager import save_vectorizers, save_metadata

    save_vectorizers(vectorizers)

    save_metadata({
        "n_classes": dataset.n_classes,
        "class_names": dataset.class_names,
    })

    print("* Entrenando modelos clásicos con Optuna...")
    results: Dict[str, Dict[str, Any]] = {}

    from src.artifacts.artifacts_manager import save_classical_model

    for model_name in ["random_forest", "logreg", "knn", "mlp", "xgb"]:

        trained_family = train_model_family(
            model_name=model_name,
            features=features,
            dataset=dataset,
            n_trials=n_trials,
        )

        results[model_name] = trained_family

        # Guardar cada modelo entrenado
        for feature_name, info in trained_family.items():
            save_classical_model(
                model=info["model"],
                model_name=model_name,
                feature_name=feature_name,
            )

    print("* Evaluando en test...")
    final_scores = evaluate_on_test(results, features, dataset)

    print("* Construyendo ensamblado FAN...")
    print("  - Tamaño y_train:", len(dataset.y_train))
    print("  - Tamaño y_val:", len(dataset.y_val))
    print("  - Clases:", dataset.n_classes)
    print("  - Modelos entrenados:", {k: list(v.keys()) for k, v in results.items()})
    print("  > Entrando a train_and_evaluate_fan()...")

    fan_results = train_and_evaluate_fan(
        trained_models=results,
        features=features,
        dataset=dataset,
    )

    print("  > train_and_evaluate_fan() terminó correctamente")

    from src.artifacts.artifacts_manager import save_fan_model

    print("* Guardando modelo FAN...")

    save_fan_model(
        fan=fan_results["fan_model"],
        n_models=fan_results["n_models"],
        n_classes=fan_results["n_classes"],
    )

    return {
        "models": results,
        "test_scores": final_scores,
        "fan_results": fan_results,
    }


# ==========================================================
# Evaluation on test set
# ==========================================================

def evaluate_on_test(
    trained_models: Dict[str, Dict[str, Any]],
    features: Dict[str, Dict[str, Any]],
    dataset,
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa cada combinación modelo + representación en el conjunto de test.
    """

    test_scores = {}

    for model_name, feature_dict in trained_models.items():

        test_scores[model_name] = {}

        for feature_name, info in feature_dict.items():

            model = info["model"]
            X_test = features[feature_name]["test"]
            y_test = dataset.y_test

            f1 = evaluate_model(model, X_test, y_test)

            test_scores[model_name][feature_name] = f1

            print(
                f"Modelo: {model_name} | Feature: {feature_name} | Test F1: {f1:.4f}"
            )

    return test_scores


# ==========================================================
# Script execution
# ==========================================================

if __name__ == "__main__":
    train_pipeline()
