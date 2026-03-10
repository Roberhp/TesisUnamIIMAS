import pandas as pd
from typing import Iterable
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def _vader_vectorize(texts: Iterable[str], analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Convierte una lista de textos en features de sentimiento usando VADER.
    """
    features = [analyzer.polarity_scores(text) for text in texts]
    return pd.DataFrame(features)


def process_vader(X_train, X_val, X_test):
    """
    Genera features de sentimiento (VADER) para train, validation y test.
    """
    analyzer = SentimentIntensityAnalyzer()

    X_train_vader = _vader_vectorize(X_train, analyzer)
    X_val_vader = _vader_vectorize(X_val, analyzer)
    X_test_vader = _vader_vectorize(X_test, analyzer)

    return X_train_vader, X_val_vader, X_test_vader
