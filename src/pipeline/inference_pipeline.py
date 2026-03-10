# ===============================
# Standard library
# ===============================
from typing import List, Dict, Any

# ===============================
# Third-party
# ===============================
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

# ===============================
# Local imports
# ===============================
from src.features.text_cleaning import preprocess_text
from src.artifacts.artifacts_manager import (
    load_classical_model,
    load_fan_model,
    load_vectorizer,
)
from src.models.fan import FeatureAttentionNetwork


# ==========================================================
# Inference Pipeline
# ==========================================================

class InferencePipeline:
    def __init__(self):

        # Cargar vectorizadores
        self.tfidf_vectorizer = load_vectorizer("tfidf")
        self.lda_vectorizer = load_vectorizer("lda_vectorizer")
        self.lda_model = load_vectorizer("lda_model")
        self.vader_analyzer = load_vectorizer("vader_analyzer")

        # Cargar modelos clásicos
        self.classical_models = self._load_classical_models()

        # Cargar modelo FAN
        self.fan_model: FeatureAttentionNetwork = load_fan_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fan_model = self.fan_model.to(self.device)

    # ==========================================================
    # Public API
    # ==========================================================

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predice clases para una lista de textos.
        """

        processed = [preprocess_text(t) for t in texts]

        features = self._build_features(processed)

        classical_preds = self._collect_classical_predictions(features)

        fan_input = torch.tensor(classical_preds, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, _ = self.fan_model(fan_input)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy().tolist()

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _load_classical_models(self) -> Dict[str, Dict[str, Any]]:

        model_names = ["random_forest", "logreg", "knn", "mlp", "xgb"]
        feature_names = ["tfidf", "lda", "vader"]

        models = {}

        for model_name in model_names:
            models[model_name] = {}

            for feature_name in feature_names:
                models[model_name][feature_name] = load_classical_model(
                    model_name=model_name,
                    feature_name=feature_name,
                )

        return models

    def _build_features(self, texts: List[str]) -> Dict[str, Any]:

        # TF-IDF
        X_tfidf = self.tfidf_vectorizer.transform(texts)

        # LDA
        X_counts = self.lda_vectorizer.transform(texts)
        X_lda = self.lda_model.transform(X_counts)

        # VADER
        X_vader = np.array([
            [
                s["neg"],
                s["neu"],
                s["pos"],
                s["compound"],
            ]
            for s in [self.vader_analyzer.polarity_scores(t) for t in texts]
        ])

        return {
            "tfidf": X_tfidf,
            "lda": X_lda,
            "vader": X_vader,
        }

    def _collect_classical_predictions(
        self,
        features: Dict[str, Any],
    ) -> np.ndarray:
        """
        Genera tensor (batch, n_models, n_classes)
        """

        preds_list = []

        for model_name, feature_dict in self.classical_models.items():
            for feature_name, model in feature_dict.items():

                X = features[feature_name]
                probs = model.predict_proba(X)
                preds_list.append(probs)

        return np.stack(preds_list, axis=1)

# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(title="Mental Health Detection API")

pipeline = InferencePipeline()


class TextRequest(BaseModel):
    texts: List[str]


@app.post("/predict")
def predict(request: TextRequest):
    """
    Endpoint para predicción usando el modelo FAN ensamblado.
    """
    predictions = pipeline.predict(request.texts)

    return {
        "predictions": predictions
    }