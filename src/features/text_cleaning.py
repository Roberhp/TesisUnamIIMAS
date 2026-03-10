# ===============================
# Standard library
# ===============================
import re
import string

# ===============================
# Third-party
# ===============================
from nltk.stem import WordNetLemmatizer

# ===============================
# Local imports
# ===============================
from src.config.settings import STOP_WORDS


# Singleton lemmatizer
_lemmatizer = WordNetLemmatizer()


# ==========================================================
# Text-level preprocessing utilities
# ==========================================================

def _lemmatize_text(text: str) -> str:
    """
    Lematiza el texto y elimina stopwords.
    """
    words = text.split()

    lemmatized_words = [
        _lemmatizer.lemmatize(word)
        for word in words
        if word.lower() not in STOP_WORDS
    ]

    return " ".join(lemmatized_words)


def preprocess_text(text: str) -> str:
    """
    Limpieza básica de texto:
    - lowercase
    - elimina URLs
    - elimina HTML
    - elimina puntuación
    - elimina números
    - aplica lematización y remoción de stopwords
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)

    text = _lemmatize_text(text)

    return text.strip()
