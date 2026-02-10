from sklearn.feature_extraction.text import TfidfVectorizer
from src.config.settings import STOP_WORDS


def process_tfidf(X_train, X_val, X_test):
    """
    Genera features TF-IDF.
    El vectorizador se ajusta SOLO con el conjunto de entrenamiento.
    """
    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=0.01,
        stop_words=STOP_WORDS,
        lowercase=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer
