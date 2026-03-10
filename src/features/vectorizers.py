from typing import Dict, Any, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer

from src.data.dataset import DatasetSplit
from src.config.settings import SEED

# ==========================================================
# Tipos estructurados
# ==========================================================

FeatureDict = Dict[str, Dict[str, Any]]
VectorizerDict = Dict[str, Any]


# ==========================================================
# TF-IDF
# ==========================================================

def build_tfidf_features(
    dataset: DatasetSplit,
    max_df: float = 0.95,
    min_df: float = 0.01,
) -> Tuple[FeatureDict, VectorizerDict]:
    """
    Genera representación TF-IDF para train/val/test.
    """

    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        lowercase=False,  # ya hicimos preprocessing
    )

    X_train = vectorizer.fit_transform(dataset.X_train)
    X_val = vectorizer.transform(dataset.X_val)
    X_test = vectorizer.transform(dataset.X_test)

    features = {
        "tfidf": {
            "train": X_train,
            "val": X_val,
            "test": X_test,
        }
    }

    return features, {"tfidf": vectorizer}


# ==========================================================
# LDA
# ==========================================================

def build_lda_features(
    dataset: DatasetSplit,
    n_topics: int,
    max_df: float = 0.95,
    min_df: float = 2,
) -> Tuple[FeatureDict, VectorizerDict]:
    """
    Genera representación LDA (distribución de tópicos).
    """

    count_vectorizer = CountVectorizer(
        max_df=max_df,
        min_df=min_df,
        lowercase=False,
    )

    X_train_counts = count_vectorizer.fit_transform(dataset.X_train)
    X_val_counts = count_vectorizer.transform(dataset.X_val)
    X_test_counts = count_vectorizer.transform(dataset.X_test)

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=SEED,
    )

    X_train = lda_model.fit_transform(X_train_counts)
    X_val = lda_model.transform(X_val_counts)
    X_test = lda_model.transform(X_test_counts)

    features = {
        "lda": {
            "train": X_train,
            "val": X_val,
            "test": X_test,
        }
    }

    return features, {
        "lda_vectorizer": count_vectorizer,
        "lda_model": lda_model,
    }


# ==========================================================
# VADER
# ==========================================================

def _vader_vectorize(
    texts,
    analyzer: SentimentIntensityAnalyzer,
) -> np.ndarray:
    """
    Convierte textos en vectores de sentimiento VADER.
    """

    scores = [
        analyzer.polarity_scores(text)
        for text in texts
    ]

    # Convertimos dict → vector ordenado
    return np.array([
        [s["neg"], s["neu"], s["pos"], s["compound"]]
        for s in scores
    ])


def build_vader_features(
    dataset: DatasetSplit,
) -> Tuple[FeatureDict, VectorizerDict]:
    """
    Genera representación basada en VADER.
    """

    analyzer = SentimentIntensityAnalyzer()

    X_train = _vader_vectorize(dataset.X_train, analyzer)
    X_val = _vader_vectorize(dataset.X_val, analyzer)
    X_test = _vader_vectorize(dataset.X_test, analyzer)

    features = {
        "vader": {
            "train": X_train,
            "val": X_val,
            "test": X_test,
        }
    }

    return features, {"vader_analyzer": analyzer}


# ==========================================================
# Builder general
# ==========================================================

def build_all_features(
    dataset: DatasetSplit,
    n_topics: int,
) -> Tuple[FeatureDict, VectorizerDict]:
    """
    Construye todas las representaciones disponibles
    y las combina en un solo diccionario estructurado.
    """

    all_features: FeatureDict = {}
    all_vectorizers: VectorizerDict = {}

    tfidf_features, tfidf_vec = build_tfidf_features(dataset)
    lda_features, lda_vec = build_lda_features(dataset, n_topics)
    vader_features, vader_vec = build_vader_features(dataset)

    all_features.update(tfidf_features)
    all_features.update(lda_features)
    all_features.update(vader_features)

    all_vectorizers.update(tfidf_vec)
    all_vectorizers.update(lda_vec)
    all_vectorizers.update(vader_vec)

    return all_features, all_vectorizers