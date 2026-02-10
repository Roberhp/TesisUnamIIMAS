
def separa_datos(datos, col_texto ='statement', col_clasificacion='status',
    seed=42):
    X = datos[col_texto]
    y = datos[col_clasificacion]
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_ent, X_val, y_ent, y_val = train_test_split(X_ent, y_ent, test_size=0.2, random_state=seed)
    return X_ent, X_val, X_prueba, y_ent, y_val,y_prueba
