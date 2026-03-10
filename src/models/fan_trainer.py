# ===============================
# Standard library
# ===============================
from typing import Dict, Any

# ===============================
# Third-party
# ===============================
import torch
import numpy as np
from sklearn.metrics import f1_score

# ===============================
# Local imports
# ===============================
from src.models.fan import FeatureAttentionNetwork


# ==========================================================
# Public API
# ==========================================================

def train_and_evaluate_fan(
    trained_models: Dict[str, Dict[str, Any]],
    features: Dict[str, Dict[str, Any]],
    dataset,
    epochs: int = 30,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """
    Entrena FAN usando Train + Val
    Evalúa únicamente en Test
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Predicciones
    train_preds = _collect_predictions(trained_models, features, "train")
    val_preds = _collect_predictions(trained_models, features, "val")
    test_preds = _collect_predictions(trained_models, features, "test")

    # Train + Val
    X_train_full = np.concatenate([train_preds, val_preds], axis=0)
    y_train_full = np.concatenate([dataset.y_train, dataset.y_val], axis=0)

    X_test = test_preds
    y_test = dataset.y_test

    # Tensores
    X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_full, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    n_models = X_train_tensor.shape[1]
    n_classes = X_train_tensor.shape[2]

    fan = FeatureAttentionNetwork(
        n_models=n_models,
        n_classes=n_classes,
    ).to(device)
    print("Entrando a train_fan()")
    _train_fan(fan, X_train_tensor, y_train_tensor, epochs, lr)

    macro_f1 = _evaluate_fan(fan, X_test_tensor, y_test_tensor)

    print(f"FAN Macro F1 (test): {macro_f1:.4f}")

    return {
        "fan_test_macro_f1": macro_f1,
        "fan_model": fan,
        "n_models": n_models,
        "n_classes": n_classes,
    }


# ==========================================================
# Helpers
# ==========================================================

def _collect_predictions(
    trained_models: Dict[str, Dict[str, Any]],
    features: Dict[str, Dict[str, Any]],
    split: str,
) -> np.ndarray:

    preds_list = []

    for model_name, feature_dict in trained_models.items():
        for feature_name, info in feature_dict.items():

            model = info["model"]
            X = features[feature_name][split]

            print(f"    -> Generando predict_proba | Modelo: {model_name} | Feature: {feature_name} | Split: {split}")

            # Si es matriz sparse (TF-IDF), no convertir a tensor aún
            if hasattr(X, "toarray"):
                X_input = X
            else:
                X_input = X

            preds = model.predict_proba(X_input)
            preds = np.asarray(preds, dtype=np.float32)
            preds_list.append(preds)

            
    print("    -> Shapes individuales:")
    for p in preds_list:
        print("       ", p.shape, type(p))

    print("    -> Intentando hacer np.stack...")
    stacked = np.stack(preds_list, axis=1).astype(np.float32)
    print("    -> Stack completado. Shape final:", stacked.shape)

    return stacked
    #return np.stack(preds_list, axis=1)


def _train_fan(
    fan: FeatureAttentionNetwork,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
):

    optimizer = torch.optim.Adam(fan.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 1024
    dataset_size = X_train.shape[0]

    for epoch in range(epochs):
        fan.train()
        total_loss = 0.0

        for i in range(0, dataset_size, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            logits, _ = fan(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"      Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")


def _evaluate_fan(
    fan: FeatureAttentionNetwork,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> float:

    fan.eval()
    with torch.no_grad():
        logits, _ = fan(X_test)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    y_true = y_test.cpu().numpy()

    return f1_score(y_true, preds, average="macro")


# ==========================================================
# Model persistence
# ==========================================================

def save_fan_model(
    fan: FeatureAttentionNetwork,
    path: str,
) -> None:
    """
    Guarda únicamente el state_dict del modelo FAN.
    """
    torch.save(fan.state_dict(), path)


def load_fan_model(
    path: str,
    n_models: int,
    n_classes: int,
) -> FeatureAttentionNetwork:
    """
    Carga un modelo FAN desde un archivo state_dict.
    """

    model = FeatureAttentionNetwork(
        n_models=n_models,
        n_classes=n_classes,
    )

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model