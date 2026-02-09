# ───── Librerías estándar
import re
import string
import random
import os

import psutil
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, mutual_info_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D,
    Bidirectional, LSTM, Dropout, Dense
)

from xgboost import XGBClassifier


import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
import pandas as pd



nltk.download('wordnet')
nltk.download('stopwords')  
nltk.download('vader_lexicon')


SEED = 42
np.random.seed(SEED)


# Inicializar lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # Puedes cambiarlo según el idioma



# Inicializar lematizador y eliminar de stopwors
lemmatizer = WordNetLemmatizer()
def lematizar_texto(texto):
    palabras = texto.split() 
    palabras_lemmatizadas = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra.lower() not in stop_words]
    return ' '.join(palabras_lemmatizadas)

def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    # Aplicar lematización al texto
    lematizar_texto(text)
    return text

def process_lda(X_ent, X_val, X_prueba, n_topics):
    # Vectorizar y aplicar LDA al conjunto de entrenamiento
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X_ent_counts = vectorizer.fit_transform(X_ent)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=SEED)
    X_ent_lda = lda.fit_transform(X_ent_counts)

    # Transformar el conjunto de validación y prueba usando el vectorizador y el modelo LDA ajustado
    X_val_counts = vectorizer.transform(X_val)
    X_val_lda = lda.transform(X_val_counts)

    X_prueba_counts = vectorizer.transform(X_prueba)
    X_prueba_lda = lda.transform(X_prueba_counts)

    return X_ent_lda, X_val_lda, X_prueba_lda, vectorizer, lda 


def process_vader(X_ent, X_val, X_prueba):
    analyzer = SentimentIntensityAnalyzer()
    
    X_ent_vader_df = vader_vectorizer(X_ent)
    X_val_vader_df = vader_vectorizer(X_val)
    X_prueba_vader_df = vader_vectorizer(X_prueba)
    
    return X_ent_vader_df, X_val_vader_df, X_prueba_vader_df, vader_vectorizer

def vader_vectorizer(texts):
    analyzer = SentimentIntensityAnalyzer()
    features = [analyzer.polarity_scores(text) for text in texts]
    return pd.DataFrame(features)


def process_tfidf(X_ent, X_val, X_prueba):
    # Ajustar el vectorizador solo con el conjunto de entrenamiento
    vectorizador = TfidfVectorizer(max_df=0.95, min_df=0.01, stop_words='english', lowercase=True)
    
    X_ent_tfidf = vectorizador.fit_transform(X_ent)
    X_val_tfidf = vectorizador.transform(X_val)
    X_prueba_tfidf = vectorizador.transform(X_prueba)
    
    return X_ent_tfidf, X_val_tfidf, X_prueba_tfidf, vectorizador#, y_ent, y_val, y_prueba, label_encoder


def separa_datos(datos, col_texto ='statement', col_clasificacion='status'):
    X = datos[col_texto]
    y = datos[col_clasificacion]
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_ent, X_val, y_ent, y_val = train_test_split(X_ent, y_ent, test_size=0.2, random_state=SEED)
    return X_ent, X_val, X_prueba, y_ent, y_val,y_prueba

def transforma_etiquetas (y_ent, y_val, y_prueba):
    label_encoder = LabelEncoder()
    y_ent = label_encoder.fit_transform(y_ent)
    y_val = label_encoder.transform(y_val)
    y_prueba = label_encoder.transform(y_prueba)
    return y_ent, y_val, y_prueba, label_encoder


# Función general de predicción
def predict_model(model, tipo, X):
    if tipo == "token":
        pred = model.predict(X)
        return pred  # ya es softmax
    else:
        return model.predict_proba(X)


def ablacion_iterativa_completa( modelos_ordenados,  X_prueba_dict, y_prueba, entrenar_y_evaluar_fn, min_modelos=1, label_encoder = None):
    """
    Realiza una ablación iterativa exhaustiva eliminando un modelo en cada paso,
    independientemente de si mejora el Macro-F1, hasta dejar `min_modelos`.

    Parámetros:
        modelos_ordenados: lista de tuplas (modelo, f1_val, tipo, nombre)
        y_val: etiquetas de validación (no se usa directamente)
        X_prueba_dict: dict de conjuntos de prueba por tipo
        y_prueba: etiquetas verdaderas del conjunto de prueba
        entrenar_y_evaluar_fn: función que entrena y evalúa un conjunto de modelos
        min_modelos: mínimo número de modelos a dejar antes de detenerse (default=1)

    Retorna:
        DataFrame con el historial del proceso
    """

    modelos_actuales = deepcopy(modelos_ordenados)
    historial = []

    resultado_base, modelo_base,_ = entrenar_y_evaluar_fn(modelos_actuales, X_prueba_dict, y_prueba)
    f1_base = resultado_base["Macro_F1_Score"].iloc[-1]
    mejor_modelos = deepcopy(modelos_actuales)
    mejor_score = f1_base

    paso = 1
    mejor_f1_actual = -1

    while len(modelos_actuales) > min_modelos:
        print(f"\nPaso {paso}: analizando impacto de cada eliminación teniendo {len(modelos_actuales)} modelos.")
        mejor_idx = -1
        mejor_delta = -float('inf')

        for i in range(len(modelos_actuales)):
            subset = modelos_actuales[:i] + modelos_actuales[i+1:]
            resultado,_,_ = entrenar_y_evaluar_fn(subset, X_prueba_dict, y_prueba)
            f1_actual = resultado["Macro_F1_Score"].iloc[-1]
            delta = f1_actual - f1_base
            print(f"   - Eliminando {modelos_actuales[i][3]}_{modelos_actuales[i][2]} → F1 = {f1_actual:.4f} (ΔF1 = {delta:+.4f}), buscando superar F1 = {mejor_f1_actual:.4f}")

            if f1_actual >= mejor_f1_actual:
                mejor_f1_actual = f1_actual
                mejor_idx = i
                mejor_delta = delta

        eliminado = modelos_actuales[mejor_idx]
        modelos_actuales.pop(mejor_idx)

        if mejor_f1_actual > mejor_score:
            mejor_score = mejor_f1_actual
            mejor_modelos = deepcopy(modelos_actuales)
            print(f"Nuevo mejor score encontrado: {mejor_score:.4f} con {len(mejor_modelos)} modelos.")

        f1_base = mejor_f1_actual
        paso += 1

    # Re-evaluar el mejor conjunto para devolver atención
    return entrenar_y_evaluar_fn(mejor_modelos,  X_prueba_dict, y_prueba,label_encoder,print_classification = True)
    

def entrenar_y_evaluar_simple(modelos_ordenados, X_prueba_dict, y_prueba, label_encoder=None, print_classification = False):
    """
    Entrena un modelo de atención con un conjunto de modelos base y devuelve las métricas de evaluación.

    Parámetros:
        modelos_ordenados: lista de tuplas (modelo, f1, tipo, nombre)
        X_prueba_dict: dict con X por tipo de modelo
        y_prueba: etiquetas reales del conjunto de prueba
        label_encoder: (opcional) para nombres de clases en el reporte

    Retorna:
        DataFrame con F1, precisión, recall y pesos de atención del conjunto actual
    """
    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

    model_names = [f"{m[3]}_{m[2]}" for m in modelos_ordenados]

    predicciones = []
    for model, _, tipo, _ in modelos_ordenados:
        X = X_prueba_dict[tipo]
        pred = predict_model(model, tipo, X)
        
        predicciones.append(pred)

    pred_np = np.array(predicciones)
    pred_tensor = torch.tensor(pred_np, dtype=torch.float32).permute(1, 0, 2)
    y_tensor = torch.tensor(y_prueba, dtype=torch.long)

    dataset = TensorDataset(pred_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    net = FeatureAttentionNetwork(n_models=pred_tensor.shape[1], n_classes=pred_tensor.shape[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        net.train()
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out, _ = net(features)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

    net.eval()
    with torch.no_grad():
        outputs, final_attention = net(pred_tensor.to(device))
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        macro_f1 = f1_score(y_prueba, y_pred, average='macro')
        macro_precision = precision_score(y_prueba, y_pred, average='macro')
        macro_recall = recall_score(y_prueba, y_pred, average='macro')

    # (Opcional) Reporte por clases
    if (print_classification):
        print(f"\nEvaluación ({len(modelos_ordenados)} modelos):")
        if label_encoder:
            print(classification_report(y_prueba, y_pred, target_names=label_encoder.classes_))
        else:
            print(classification_report(y_prueba, y_pred))

        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall:    {macro_recall:.4f}")
        print(f"Macro F1-Score:  {macro_f1:.4f}")

    return pd.DataFrame({
        "Modelos": [model_names],
        "Macro_F1_Score": [macro_f1],
        "Macro_Precision": [macro_precision],
        "Macro_Recall": [macro_recall],
        "Attention_Weights": [final_attention.cpu().numpy()]
    }), net  , modelos_ordenados


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')


def train_random_forest_fijo(X_train_lda, X_test_lda , X_train_vader, X_test_vader, X_train_tfidf, X_test_tfidf, 
    y_train_enc, y_test_enc, modelos_entrenados):
    

    from sklearn.ensemble import RandomForestClassifier
    name = "random_forest"

    # LDA
    rf_lda = RandomForestClassifier(
        max_depth=15,
        min_samples_split=3,
        n_estimators=151,
        n_jobs=-1,
        random_state=SEED
    )
    rf_lda.fit(X_train_lda, y_train_enc)
    f1_lda = evaluate_model(rf_lda, X_test_lda, y_test_enc)
    modelos_entrenados.append((rf_lda, f1_lda, "lda", name))
    print("finaliza el entrenamiento con lda")


    # VADER
    rf_vader = RandomForestClassifier(
        max_depth=15,
        n_estimators=157,
        n_jobs=-1,
        random_state=SEED
    )
    rf_vader.fit(X_train_vader, y_train_enc)
    f1_vader = evaluate_model(rf_vader, X_test_vader, y_test_enc)
    modelos_entrenados.append((rf_vader, f1_vader, "vader", name))
    print("finaliza el entrenamiento con vader")

    # TF-IDF
    rf_tfidf = RandomForestClassifier(
        max_depth=15,
        n_estimators=133,
        n_jobs=-1,
        random_state=SEED
    )
    rf_tfidf.fit(X_train_tfidf, y_train_enc)
    f1_tfidf = evaluate_model(rf_tfidf, X_test_tfidf, y_test_enc)
    modelos_entrenados.append((rf_tfidf, f1_tfidf, "tfidf", name))
    print("finaliza el entrenamiento con tfidf")

    return modelos_entrenados



def train_logreg_models_fijo (X_train_lda, X_test_lda,
                        X_train_vader, X_test_vader,
                        X_train_tfidf, X_test_tfidf,
                        y_train_enc, y_test_enc,
                        modelos_entrenados):
    from sklearn.linear_model import LogisticRegression
    
    name = "logreg"

    # LDA
    lr_lda = LogisticRegression(C=1.226, solver='saga', max_iter=1000)
    lr_lda.fit(X_train_lda, y_train_enc)
    f1_lda = evaluate_model(lr_lda, X_test_lda, y_test_enc)
    modelos_entrenados.append((lr_lda, f1_lda, "lda", name))
    print("finaliza el entrenamiento con lda")

    # VADER
    lr_vader = LogisticRegression(C=2.063, max_iter=1000)
    lr_vader.fit(X_train_vader, y_train_enc)
    f1_vader = evaluate_model(lr_vader, X_test_vader, y_test_enc)
    modelos_entrenados.append((lr_vader, f1_vader, "vader", name))
    print("finaliza el entrenamiento con vader")

    # TF-IDF
    lr_tfidf = LogisticRegression(C=2.546, solver='saga', max_iter=1000)
    lr_tfidf.fit(X_train_tfidf, y_train_enc)
    f1_tfidf = evaluate_model(lr_tfidf, X_test_tfidf, y_test_enc)

    modelos_entrenados.append((lr_tfidf, f1_tfidf, "tfidf", name))

    print("finaliza el entrenamiento con tfidf")

    return modelos_entrenados


def train_knn_models_fijo (X_train_lda, X_test_lda,
                        X_train_vader, X_test_vader,
                        X_train_tfidf, X_test_tfidf,
                        y_train_enc, y_test_enc,
                        modelos_entrenados):

    from sklearn.neighbors import KNeighborsClassifier

    name = "knn"

    knn_lda = KNeighborsClassifier (n_neighbors=15, p=1, weights='distance')
    knn_lda.fit(X_train_lda, y_train_enc)
    best_f1_score_lda = evaluate_model(knn_lda,X_test_lda,y_test_enc)
    modelos_entrenados.append((knn_lda, best_f1_score_lda, "lda", name))

    print(f"finaliza el entrenamiento con lda  + {name}")




    knn_vader = KNeighborsClassifier (n_neighbors=15, p=1, weights='distance')
    knn_vader.fit(X_train_vader, y_train_enc)
    best_f1_score_vader = evaluate_model(knn_vader, X_test_vader, y_test_enc)
    modelos_entrenados.append((knn_vader, best_f1_score_vader, "vader", name))

    print(f"finaliza el entrenamiento con vader  + {name}")


    knn_tfidf = KNeighborsClassifier (n_neighbors=3, p=1, weights='distance')
    knn_tfidf.fit(X_train_tfidf, y_train_enc)
    best_f1_score_tfidf = evaluate_model(knn_tfidf, X_test_tfidf, y_test_enc)
    modelos_entrenados.append((knn_tfidf, best_f1_score_tfidf, "tfidf", name))

    print(f"finaliza el entrenamiento con tfidf + {name}")


    return modelos_entrenados


def train_mlp_models_fijo (X_train_lda, X_test_lda,
                        X_train_vader, X_test_vader,
                        X_train_tfidf, X_test_tfidf,
                        y_train_enc, y_test_enc,
                        modelos_entrenados):

    from sklearn.neural_network import MLPClassifier

    name = "mlp"

    mlp_lda = MLPClassifier(activation='tanh', alpha=0.0016609660563125484, early_stopping=True, 
                            hidden_layer_sizes=(50, 50), max_iter=150, random_state=SEED) 
    mlp_lda.fit(X_train_lda, y_train_enc)
    best_f1_score_lda = evaluate_model(mlp_lda,X_test_lda,y_test_enc)
    modelos_entrenados.append((mlp_lda, best_f1_score_lda, "lda", name))

    print(f"finaliza el entrenamiento con lda  + {name}")


    mlp_vader = MLPClassifier(activation='tanh', alpha=0.005296578633529615,
              early_stopping=True, hidden_layer_sizes=(50, 50),
              learning_rate='adaptive', max_iter=150, random_state=SEED)
    mlp_vader.fit(X_train_vader, y_train_enc)
    best_f1_score_vader = evaluate_model(mlp_vader, X_test_vader, y_test_enc)
    modelos_entrenados.append((mlp_vader, best_f1_score_vader, "vader", name))

    print(f"finaliza el entrenamiento con vader  + {name}")


    mlp_tfidf = MLPClassifier(alpha=0.0001111079131541661, early_stopping=True,
              hidden_layer_sizes=(50, 50), max_iter=150, random_state=SEED)
    mlp_tfidf.fit(X_train_tfidf, y_train_enc)
    best_f1_score_tfidf = evaluate_model(mlp_tfidf, X_test_tfidf, y_test_enc)
    modelos_entrenados.append((mlp_tfidf, best_f1_score_tfidf, "tfidf", name))

    print(f"finaliza el entrenamiento con tfidf + {name}")

    return modelos_entrenados



    
def train_xgb_models_fijo(X_train_lda, X_test_lda,
                     X_train_vader, X_test_vader,
                     X_train_tfidf, X_test_tfidf,
                     y_train_enc, y_test_enc,
                     modelos_entrenados):


    from xgboost import XGBClassifier
    import numpy as np  # Necesario para manejar el parámetro `missing=np.nan`

    name = "xgb"

    # LDA
    xgb_lda = XGBClassifier(
        colsample_bytree=0.7411532267971501,
        gamma=0.1043403189305191,
        learning_rate=0.20203486162318796,
        max_depth=8,
        min_child_weight=4,
        n_estimators=62,
        eval_metric='mlogloss',
        missing=np.nan
    )
    xgb_lda.fit(X_train_lda, y_train_enc)
    f1_lda = evaluate_model(xgb_lda, X_test_lda, y_test_enc)
    modelos_entrenados.append((xgb_lda, f1_lda, "lda", name))
    print(f"finaliza el entrenamiento con lda + {name}")

    # VADER
    xgb_vader = XGBClassifier(
        colsample_bytree=0.927841302345394,
        gamma=0.42112101743563324,
        learning_rate=0.12542834437974032,
        max_depth=8,
        min_child_weight=4,
        n_estimators=138,
        eval_metric='mlogloss',
        missing=np.nan
    )
    xgb_vader.fit(X_train_vader, y_train_enc)
    f1_vader = evaluate_model(xgb_vader, X_test_vader, y_test_enc)
    modelos_entrenados.append((xgb_vader, f1_vader, "vader", name))
    print(f"finaliza el entrenamiento con vader + {name}")

    # TF-IDF
    xgb_tfidf = XGBClassifier(
        colsample_bytree=0.8188128244256361,
        gamma=0.9515363753250188,
        learning_rate=0.24971909269237794,
        max_depth=8,
        min_child_weight=5,
        n_estimators=136,
        eval_metric='mlogloss',
        missing=np.nan
    )
    xgb_tfidf.fit(X_train_tfidf, y_train_enc)
    f1_tfidf = evaluate_model(xgb_tfidf, X_test_tfidf, y_test_enc)
    modelos_entrenados.append((xgb_tfidf, f1_tfidf, "tfidf", name))
    print(f"finaliza el entrenamiento con tfidf + {name}")

    return modelos_entrenados



# Red de atención
class FeatureAttentionNetwork(nn.Module):
    def __init__(self, n_models, n_classes):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(n_models))
        self.classifier = nn.Linear(n_classes, n_classes)

    def forward(self, features):
        weights = torch.softmax(self.attention_weights, dim=0)
        weighted = torch.einsum('mnk,n->mk', features, weights)
        return self.classifier(weighted), weights



def print_weights (final_attention, model_names) :
    print("Modelos ordenados por peso de atención:")
    pesos_finales = final_attention.cpu().numpy()
    ordenados = sorted(zip(model_names, pesos_finales), key=lambda x: x[1], reverse=True)
    for name, weight in ordenados:
        print(f"Modelo: {name}, Peso: {weight:.4f}")

def print_weight_evol (final_attention, model_names,attention_weights_history ,top_n):
    # Mostrar pesos finales ordenados
    print(" Modelos ordenados por peso de atención:")
    pesos_finales = final_attention.cpu().numpy()
    ordenados = sorted(zip(model_names, pesos_finales), key=lambda x: x[1], reverse=True)
    for name, weight in ordenados:
        print(f"* Modelo: {name}, Peso: {weight:.4f}")

    # Gráfico de evolución
    attention_array = np.array(attention_weights_history)
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(model_names):
        plt.plot(attention_array[:, i], label=name)
    plt.title(f"Evolución de Pesos de Atención - Top {top_n}")
    plt.xlabel("Época")
    plt.ylabel("Peso de Atención")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def class_report (label_encoder, y_true, y_pred):
    print("\nReporte de Clasificación")
    if label_encoder:
        target_names = label_encoder.classes_
    else:
        target_names = [str(i) for i in np.unique(y_true)]
    print(classification_report(y_true, y_pred, target_names=target_names))





def predict_proba_aligned(texts, model, vectorizer, representation, label_encoder, lda_model=None):
    processed_texts = [util.preprocess_text(t) for t in texts] # Se aplica preprocesamiento individualmente

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
        preprocessed_text = util.preprocess_text(texto)
        X_text = vectorizers["tfidf"].transform([preprocessed_text])
    
    elif caracteristica == "vader":
        predictor = lambda texts: predict_proba_aligned(texts, modelo, vectorizers["vader"], "vader", label_encoder)
        preprocessed_text = util.preprocess_text(texto)
        X_text = vectorizers["vader"]([preprocessed_text])
    
    elif caracteristica == "lda":
        count_vectorizer, lda_model_obj = vectorizers["lda"]
        predictor = lambda texts: predict_proba_aligned(texts, modelo, count_vectorizer, "lda", label_encoder, lda_model=lda_model_obj)
        preprocessed_text = util.preprocess_text(texto)
        X_counts = count_vectorizer.transform([preprocessed_text])
        X_text = lda_model_obj.transform(X_counts)

    else:
        X_text= None
        predictor = None
    
    return X_text, predictor