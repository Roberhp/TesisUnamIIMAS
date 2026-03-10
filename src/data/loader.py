import pandas as pd
from pathlib import Path
from typing import List


REQUIRED_COLUMNS: List[str] = []  


def load_csv_dataset(
    path: str,
    required_columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Carga un dataset CSV desde disco y valida que contenga
    las columnas requeridas.

    Parameters
    ----------
    path : str
        Ruta al archivo CSV.
    required_columns : List[str]
        Lista de columnas obligatorias.

    Returns
    -------
    pd.DataFrame
        DataFrame cargado y validado.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    df = pd.read_csv(file_path)

    if required_columns:
        _validate_columns(df, required_columns)

    return df


def _validate_columns(
    df: pd.DataFrame,
    required_columns: List[str],
) -> None:
    """
    Verifica que existan las columnas necesarias en el DataFrame.
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en el dataset: {missing}"
        )


def clean_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    Elimina filas inválidas (NaN o vacías) en las columnas
    de texto y etiqueta.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.
    text_col : str
        Nombre de la columna de texto.
    label_col : str
        Nombre de la columna de etiqueta.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio.
    """

    if text_col not in df.columns:
        raise ValueError(f"La columna '{text_col}' no existe en el DataFrame.")

    if label_col not in df.columns:
        raise ValueError(f"La columna '{label_col}' no existe en el DataFrame.")

    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    df = df[df[label_col].astype(str).str.strip() != ""]

    return df