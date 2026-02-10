from sklearn.preprocessing import LabelEncoder


def transforma_etiquetas (y_ent, y_val, y_prueba):
    label_encoder = LabelEncoder()
    y_ent = label_encoder.fit_transform(y_ent)
    y_val = label_encoder.transform(y_val)
    y_prueba = label_encoder.transform(y_prueba)
    return y_ent, y_val, y_prueba, label_encoder