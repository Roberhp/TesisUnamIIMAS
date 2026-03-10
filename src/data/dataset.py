

from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import SEED


@dataclass
class DatasetSplit:
    """
    Contenedor estructurado para los datos de entrenamiento,
    validación y prueba.
    """
    X_train: Any
    X_val: Any
    X_test: Any
    y_train: Any
    y_val: Any
    y_test: Any
    n_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    label_encoder: Optional[object] = None


def separa_datos(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.3,
    val_size: float = 0.2,
) -> DatasetSplit:
    """
    Divide el dataset en entrenamiento, validación y prueba.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio.
    text_col : str
        Nombre de la columna de texto.
    label_col : str
        Nombre de la columna de etiqueta.
    test_size : float
        Proporción del conjunto de prueba.
    val_size : float
        Proporción del conjunto de validación
        respecto al conjunto de entrenamiento.

    Returns
    -------
    DatasetSplit
        Objeto estructurado con los splits.
    """

    if text_col not in df.columns:
        raise ValueError(f"La columna '{text_col}' no existe en el DataFrame.")

    if label_col not in df.columns:
        raise ValueError(f"La columna '{label_col}' no existe en el DataFrame.")

    X = df[text_col]
    y = df[label_col]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=SEED,
        stratify=y_train,
    )

    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )