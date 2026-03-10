import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from features.text_cleaning import preprocess_text
from src.data.labels import encode_labels
from src.features.linguistic import process_tfidf
from src.utils.metrics import evaluate_model
from src.config.settings import SEED


def main():

    df = pd.read_csv("data/raw/Combined_Data.csv")  
    

    texts = df["statement"].astype(str)
    labels = df["status"]

    texts = texts.apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED
    )

    y_train, y_test, _, label_encoder = encode_labels(
        y_train, y_test, y_test
    )

    Xtr_tfidf, _, Xte_tfidf, _ = process_tfidf(
        X_train, X_test, X_test
    )

    print("Shape train:", Xtr_tfidf.shape)
    print("Shape test:", Xte_tfidf.shape)

    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr_tfidf, y_train)

    f1 = evaluate_model(clf, Xte_tfidf, y_test)
    print("F1 score:", f1)


if __name__ == "__main__":
    main()