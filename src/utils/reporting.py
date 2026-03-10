import matplotlib.pyplot as plt
from sklearn.metrics import  classification_report


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

