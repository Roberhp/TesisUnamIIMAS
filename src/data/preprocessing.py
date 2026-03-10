# ===============================
# Standard library
# ===============================
from copy import deepcopy
from typing import Callable

# ===============================
# Third-party
# ===============================
import pandas as pd

# ===============================
# Local imports
# ===============================
from src.data.dataset import DatasetSplit
from src.features.text_cleaning import preprocess_text


# ==========================================================
# Dataset-level preprocessing
# ==========================================================

def preprocess_dataset(
    dataset: DatasetSplit,
    text_processor: Callable[[str], str] = preprocess_text,
) -> DatasetSplit:
    """
    Aplica el preprocesamiento de texto a todas las particiones
    del DatasetSplit (train, val, test).

    Esta función NO define cómo se limpia el texto,
    solo aplica la transformación a nivel estructural.

    Parameters
    ----------
    dataset : DatasetSplit
        Objeto con los splits originales.
    text_processor : Callable
        Función que transforma un string.

    Returns
    -------
    DatasetSplit
        Nuevo DatasetSplit con texto procesado.
    """

    new_dataset = deepcopy(dataset)

    if isinstance(new_dataset.X_train, pd.Series):
        new_dataset.X_train = new_dataset.X_train.apply(text_processor)

    if isinstance(new_dataset.X_val, pd.Series):
        new_dataset.X_val = new_dataset.X_val.apply(text_processor)

    if isinstance(new_dataset.X_test, pd.Series):
        new_dataset.X_test = new_dataset.X_test.apply(text_processor)

    return new_dataset