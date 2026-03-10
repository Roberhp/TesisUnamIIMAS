import torch
import torch.nn as nn


class FeatureAttentionNetwork(nn.Module):
    def __init__(self, n_models: int, n_classes: int):
        super().__init__()

        self.n_models = n_models
        self.n_classes = n_classes

        # Pesos de atención aprendibles
        self.attention_weights = nn.Parameter(
            torch.randn(n_models)
        )

        # Clasificador final
        self.classifier = nn.Linear(n_classes, n_classes)

    def forward(self, features: torch.Tensor):
        """
        features shape:
        (batch_size, n_models, n_classes)
        """

        # Normalizar pesos
        weights = torch.softmax(self.attention_weights, dim=0)

        # Combinación ponderada
        combined = torch.einsum("bmk,m->bk", features, weights)

        logits = self.classifier(combined)

        return logits, weights