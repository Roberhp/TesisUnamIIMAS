import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.fan import FeatureAttentionNetwork



def train_classical_model(model, X_train, y_train):
    """
    Entrena un modelo clásico de sklearn.
    """
    model.fit(X_train, y_train)
    return model


def train_fan(
    features_train,
    y_train,
    features_val,
    y_val,
    n_classes,
    n_epochs=50,
    lr=1e-3,
):
    """
    Entrena una Feature Attention Network.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_models = features_train.shape[1]

    model = FeatureAttentionNetwork(
        n_models=n_models,
        n_classes=n_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(features_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    X_val = torch.tensor(features_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        logits, _ = model(X_train)
        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        # validación simple
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val)
            val_loss = criterion(val_logits, y_val)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | train_loss={loss.item():.4f} | val_loss={val_loss.item():.4f}")

    return model


def evaluate_fan(model, features_test, y_test):
    device = next(model.parameters()).device

    X_test = torch.tensor(features_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(X_test)
        preds = torch.argmax(logits, dim=1)

    return (preds.cpu().numpy() == y_test.cpu().numpy()).mean()
