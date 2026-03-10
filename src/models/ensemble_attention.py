# ===============================
# Third-party imports
# ===============================
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# ===============================
# Local imports
# ===============================
from src.models.fan import FeatureAttentionNetwork
from src.utils.model_utils import predict_model


# ==========================================================
# Internal helpers
# ==========================================================

def _generate_meta_features(
    modelos_ordenados: List[Tuple[Any, float, str, str]],
    X_prueba_dict: Dict[str, Any],
) -> torch.Tensor:
    """
    Genera el tensor de meta-features (predicciones de modelos base).

    Retorna:
        Tensor de forma (n_samples, n_models, n_classes)
    """
    predicciones = []

    for model, _, tipo, _ in modelos_ordenados:
        X = X_prueba_dict[tipo]
        pred = predict_model(model, tipo, X)
        predicciones.append(pred)

    pred_np = np.array(predicciones)  # (n_models, n_samples, n_classes)
    pred_tensor = torch.tensor(pred_np, dtype=torch.float32).permute(1, 0, 2)

    return pred_tensor


def _train_attention_network(
    pred_tensor: torch.Tensor,
    y_prueba: np.ndarray,
    n_epochs: int = 5,
    lr: float = 0.01,
) -> Tuple[FeatureAttentionNetwork, torch.Tensor]:
    """
    Entrena la red de atención sobre las predicciones base.

    Retorna:
        - Modelo entrenado
        - Pesos finales de atención
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_tensor = torch.tensor(y_prueba, dtype=torch.long)

    dataset = TensorDataset(pred_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    net = FeatureAttentionNetwork(
        n_models=pred_tensor.shape[1],
        n_classes=pred_tensor.shape[2],
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        net.train()
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = net(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    net.eval()
    with torch.no_grad():
        _, final_attention = net(pred_tensor.to(device))

    return net, final_attention


def _evaluate_attention_network(
    model: FeatureAttentionNetwork,
    pred_tensor: torch.Tensor,
    y_prueba: np.ndarray,
    label_encoder=None,
    print_classification: bool = False,
) -> Dict[str, Any]:
    """
    Evalúa la red de atención entrenada.
    """
    device = next(model.parameters()).device

    with torch.no_grad():
        logits, _ = model(pred_tensor.to(device))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    macro_f1 = f1_score(y_prueba, y_pred, average="macro")
    macro_precision = precision_score(y_prueba, y_pred, average="macro")
    macro_recall = recall_score(y_prueba, y_pred, average="macro")

    if print_classification:
        print("\nEvaluación del ensamble con atención:")
        if label_encoder:
            print(
                classification_report(
                    y_prueba,
                    y_pred,
                    target_names=label_encoder.classes_,
                )
            )
        else:
            print(classification_report(y_prueba, y_pred))

        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall:    {macro_recall:.4f}")
        print(f"Macro F1-Score:  {macro_f1:.4f}")

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "y_pred": y_pred,
    }


# ==========================================================
# Public API
# ==========================================================

def entrenar_y_evaluar_simple(
    modelos_ordenados: List[Tuple[Any, float, str, str]],
    X_prueba_dict: Dict[str, Any],
    y_prueba: np.ndarray,
    label_encoder=None,
    print_classification: bool = False,
) -> Tuple[pd.DataFrame, FeatureAttentionNetwork, List]:
    """
    Entrena y evalúa el ensamble con atención sobre modelos clásicos.

    Retorna:
        - DataFrame con métricas
        - Modelo de atención entrenado
        - Lista de modelos utilizados
    """
    model_names = [f"{m[3]}_{m[2]}" for m in modelos_ordenados]

    # 1️⃣ Generar meta-features
    pred_tensor = _generate_meta_features(modelos_ordenados, X_prueba_dict)

    # 2️⃣ Entrenar red de atención
    net, final_attention = _train_attention_network(
        pred_tensor, y_prueba
    )

    # 3️⃣ Evaluar
    metrics = _evaluate_attention_network(
        net,
        pred_tensor,
        y_prueba,
        label_encoder=label_encoder,
        print_classification=print_classification,
    )

    results_df = pd.DataFrame(
        {
            "Modelos": [model_names],
            "Macro_F1_Score": [metrics["macro_f1"]],
            "Macro_Precision": [metrics["macro_precision"]],
            "Macro_Recall": [metrics["macro_recall"]],
            "Attention_Weights": [final_attention.cpu().numpy()],
        }
    )

    return results_df, net, modelos_ordenados