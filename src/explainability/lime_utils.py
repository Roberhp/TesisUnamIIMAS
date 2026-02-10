


from src.data.preprocess import preprocess_text


def predict_proba_aligned(texts, model, vectorizer, representation, label_encoder, lda_model=None):
    processed_texts = [preprocess_text(t) for t in texts] # Se aplica preprocesamiento individualmente

    # Transformación de features
    if representation == "tfidf":
        X = vectorizer.transform(processed_texts)
    elif representation == "vader":
        X = vectorizer(processed_texts)
    elif representation == "lda":
        X_counts = vectorizer.transform(processed_texts)
        X = lda_model.transform(X_counts)
    else:
        raise ValueError("Representación no válida")

    # Probabilidades originales del modelo
    proba = model.predict_proba(X)

    # Alinear orden de clases
    aligned = np.zeros_like(proba)
    for i, class_label in enumerate(label_encoder.classes_):
        class_idx = label_encoder.transform([class_label])[0]
        aligned[:, i] = proba[:, class_idx]

    return aligned
    
def contruye_predictor_LIME (caracteristica, texto, modelo, vectorizers, label_encoder):
    # se construye el predictor que LIME usara para evaluar el modelo sobre cada variante
    if caracteristica == "tfidf":
        predictor = lambda texts: predict_proba_aligned(texts, modelo, vectorizers["tfidf"], "tfidf", label_encoder)
        preprocessed_text = preprocess_text(texto)
        X_text = vectorizers["tfidf"].transform([preprocessed_text])
    
    elif caracteristica == "vader":
        predictor = lambda texts: predict_proba_aligned(texts, modelo, vectorizers["vader"], "vader", label_encoder)
        preprocessed_text = preprocess_text(texto)
        X_text = vectorizers["vader"]([preprocessed_text])
    
    elif caracteristica == "lda":
        count_vectorizer, lda_model_obj = vectorizers["lda"]
        predictor = lambda texts: predict_proba_aligned(texts, modelo, count_vectorizer, "lda", label_encoder, lda_model=lda_model_obj)
        preprocessed_text = preprocess_text(texto)
        X_counts = count_vectorizer.transform([preprocessed_text])
        X_text = lda_model_obj.transform(X_counts)

    else:
        X_text= None
        predictor = None
    
    return X_text, predictor