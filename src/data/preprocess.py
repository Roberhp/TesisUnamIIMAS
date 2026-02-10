import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config.settings import STOP_WORDS


# Inicializar lematizador
lemmatizer = WordNetLemmatizer()


def _lemmatize_text(text: str) -> str:
    """
    Lematiza el texto y elimina stopwords.
    """
    words = text.split()
    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.lower() not in STOP_WORDS
    ]
    return " ".join(lemmatized_words)


def preprocess_text(text: str) -> str:
    """
    Limpieza básica de texto:
    - lowercase
    - elimina URLs, HTML, puntuación y números
    - aplica lematización y stopwords
    """
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)

    text = _lemmatize_text(text)
    return text