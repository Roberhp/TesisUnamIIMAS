from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.data.dataset import DatasetSplit


def encode_labels(
    dataset: DatasetSplit,
) -> DatasetSplit:
    """
    Codifica las etiquetas categóricas a valores numéricos.

    Parameters
    ----------
    dataset : DatasetSplit
        Objeto con los splits originales.

    Returns
    -------
    Tuple[DatasetSplit, LabelEncoder]
        Nuevo DatasetSplit con etiquetas codificadas
        y el LabelEncoder ajustado.
    """

    label_encoder = LabelEncoder()

    # Ajustar solo con entrenamiento
    y_train_enc = label_encoder.fit_transform(dataset.y_train)

    # Transformar val y test
    y_val_enc = label_encoder.transform(dataset.y_val)
    y_test_enc = label_encoder.transform(dataset.y_test)

    encoded_dataset = DatasetSplit(
        X_train=dataset.X_train,
        X_val=dataset.X_val,
        X_test=dataset.X_test,
        y_train=y_train_enc,
        y_val=y_val_enc,
        y_test=y_test_enc,
        n_classes=len(label_encoder.classes_),
        class_names=list(label_encoder.classes_),
        label_encoder = label_encoder
    )

    return encoded_dataset


