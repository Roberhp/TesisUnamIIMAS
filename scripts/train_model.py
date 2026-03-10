import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from features.text_cleaning import preprocess_text
from src.data.encoding import encode_labels
from src.features.linguistic import process_tfidf
from src.features.sentiment import process_vader
from src.features.topics import process_lda
from models.training_fan_ import train_classical_model, train_fan
from src.utils.metrics import evaluate_model
from src.config.settings import SEED


def main():
    # ============================
    # 1. Cargar datos
    # ============================
    df = pd.read_csv("data/dataset.csv")  # ruta ejemplo

    texts = df["text"].astype(str)
    labels = df["label"]

    # ============================
    # 2. Preprocesamiento
    # ============================
    texts = texts.apply(preprocess_text)

    # ============================
    # 3. Split manual (ejemplo simple)
    # ============================
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )

    # ============================
    # 4. Codificar etiquetas
    # ============================
    y_train, y_val, y_test, label_encoder = encode_labels(
        y_train, y_val, y_test
    )

    n_classes = len(label_encoder.classes_)

    # ============================
    # 5. Features
    # ============================
    Xtr_tfidf, Xval_tfidf, Xte_tfidf, _ = process_tfidf(
        X_train, X_val, X_test
    )

    Xtr_vader, Xval_vader, Xte_vader = process_vader(
        X_train, X_val, X_test
    )

    Xtr_lda, Xval_lda, Xte_lda, _, _ = process_lda(
        X_train, X_val, X_test, n_topics=10
    )

    # ============================
    # 6. Baseline clásico
    # ============================
    clf = LogisticRegression(max_iter=1000)
    clf = train_classical_model(clf, Xtr_tfidf, y_train)

    f1 = evaluate_model(clf, Xte_tfidf, y_test)
    print(f"F1 TF-IDF baseline: {f1:.4f}")

    # ============================
    # 7. Preparar features para FAN
    # ============================
    Xtr_fan = np.stack(
        [Xtr_tfidf.toarray(), Xtr_vader, Xtr_lda], axis=1
    )
    Xval_fan = np.stack(
        [Xval_tfidf.toarray(), Xval_vader, Xval_lda], axis=1
    )
    Xte_fan = np.stack(
        [Xte_tfidf.toarray(), Xte_vader, Xte_lda], axis=1
    )

    # ============================
    # 8. Entrenar FAN
    # ============================
    fan_model = train_fan(
        Xtr_fan,
        y_train,
        Xval_fan,
        y_val,
        n_classes=n_classes,
        n_epochs=50,
    )

    # ============================
    # 9. Evaluar FAN
    # ============================
    fan_acc = (fan_model(Xte_fan)[0].argmax(1).cpu().numpy() == y_test).mean()
    print(f"Accuracy FAN: {fan_acc:.4f}")


if __name__ == "__main__":
    main()