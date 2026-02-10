import torch
import torch.nn as nn


class FeatureAttentionNetwork(nn.Module):
    def __init__(self, n_models: int, n_classes: int):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(n_models))
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, features: torch.Tensor):
        weights = torch.softmax(self.attention_weights, dim=0)
        features = torch.einsum("mnk,n->mk", features, weights)
        return self.classifier(features), weights