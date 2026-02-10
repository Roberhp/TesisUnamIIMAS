import numpy as np
from typing import Iterable, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.config.settings import SEED, STOP_WORDS


def process_lda(
    X_train: Iterable[str],
    X_val: Iterable[str],
    X_test: Iterable[str],
    n_topics: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CountVectorizer, LatentDirichletAllocation]:
    """
    Genera features de tópicos usando LDA.
    Ajusta el vectorizador y el modelo SOLO con el conjunto de entrenamiento.
    """
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words=STOP_WORDS
    )

    # Fit en train
    X_train_counts = vectorizer.fit_transform(X_train)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=SEED
    )
    X_train_lda = lda.fit_transform(X_train_counts)

    # Transform en validation
    X_val_counts = vectorizer.transform(X_val)
    X_val_lda = lda.transform(X_val_counts)

    # Transform en test
    X_test_counts = vectorizer.transform(X_test)
    X_test_lda = lda.transform(X_test_counts)

    return X_train_lda, X_val_lda, X_test_lda, vectorizer, lda
