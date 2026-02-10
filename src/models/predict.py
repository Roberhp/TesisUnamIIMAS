
# Función general de predicción
def predict_model(model, tipo, X):
    if tipo == "token":
        pred = model.predict(X)
        return pred  # ya es softmax
    else:
        return model.predict_proba(X)